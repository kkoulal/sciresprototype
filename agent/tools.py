"""
Qdrant + Neo4j tool functions exposed to the LLM agent.

Each function returns a formatted string. The QA node parses <tool_call>
blocks from the LLM output, dispatches here, and feeds results back as
the next user turn.
"""
import os
import json
import httpx
from qdrant_client import QdrantClient, models
from neo4j import GraphDatabase
from online.config import QDRANT, EMBED, NEO4J


def _neo4j():
    return GraphDatabase.driver(NEO4J["uri"], auth=(NEO4J["user"], NEO4J["password"]))

NOT_REPORTED = "Not explicitly reported in the provided text."

_SUMMARY_FIELDS = [
    "doi",
    "domain_classification",
    "type_etude",
    "context_problematique",
    "research_gap_lacune",
    "resultats_principaux",
    "conclusions_principales",
    "limitations_mentionnees",
    "directions_futures",
    "entites_nommees_specifiques_domaine",
    "implications_pratiques_theoriques",
]

_FULL_FIELDS = _SUMMARY_FIELDS + [
    "authors",
    "question_de_recherche",
    "hypotheses",
    "objectifs_primaires",
    "objectifs_secondaires",
    "design_etude",
    "population_echantillon",
    "taille_echantillon_n",
    "methodes_collecte_donnees",
    "techniques_instruments",
    "valeurs_numeriques_cles",
    "methodes_analyse_statistique",
]


def _embed(text: str):
    r = httpx.post(
        f"{EMBED['url']}/embed_both",
        json={"inputs": [text]},
        timeout=EMBED["timeout"],
    )
    r.raise_for_status()
    v = r.json()
    dense = v["dense"][0]
    sparse_raw = v["sparse"][0]
    sparse = models.SparseVector(
        indices=[x["index"] for x in sparse_raw],
        values=[x["value"] for x in sparse_raw],
    )
    return dense, sparse


def _enrich_meta(dois: list[str]) -> dict[str, dict]:
    """Batch-fetch title, year, fwci, cited_by_count from Neo4j for a list of DOIs."""
    if not dois:
        return {}
    driver = _neo4j()
    try:
        with driver.session() as session:
            records = session.run(
                """
                UNWIND $dois AS doi
                MATCH (p:SimPaper {doi: doi})
                RETURN doi, p.title AS title, p.year AS year,
                       p.fwci AS fwci, p.cited_by_count AS citations
                """,
                dois=dois,
            )
            result = {r["doi"]: dict(r) for r in records}
    except Exception:
        result = {}
    finally:
        driver.close()
    return result


def _fmt(p: dict, fields=None, meta: dict = None) -> str:
    if fields is None:
        fields = _SUMMARY_FIELDS
    doi = p.get("doi", "N/A")
    m = (meta or {}).get(doi, {})
    header_parts = [f"DOI: {doi}"]
    if m.get("title"):
        header_parts.append(f'Title: {m["title"]}')
    if m.get("year"):
        header_parts.append(f'Year: {m["year"]}')
    if m.get("fwci") is not None:
        header_parts.append(f'FWCI: {m["fwci"]:.2f}')
    if m.get("citations") is not None:
        header_parts.append(f'Citations: {m["citations"]}')
    lines = [" | ".join(header_parts)]
    for field in fields:
        if field == "doi":
            continue
        val = p.get(field, "")
        if val and val != NOT_REPORTED:
            lines.append(f"{field}: {val}")
    return "\n".join(lines)


_DOI_RE       = __import__("re").compile(r"\b10\.\d{4,9}/\S+\b")
_ACRONYM_RE   = __import__("re").compile(r"\b[A-Z]{2,6}(?:-\d+)?\b")
_KNOWN_BENCHMARKS = {
    "mnist", "cifar", "cifar-10", "cifar-100", "imagenet", "coco", "pascal voc",
    "usps", "lfw", "modelnet", "kitti", "voc", "ade20k", "cityscapes", "wikitext",
}


def _query_lexicality(query: str) -> float:
    """
    Score 0.0 (pure conceptual) → 1.0 (pure lexical). Drives hybrid weighting.
    Lexical signals: DOIs, dataset names, capitalized acronyms, quoted exact terms.
    """
    q = query.strip()
    score = 0.0
    if _DOI_RE.search(q):
        score += 0.7
    acronyms = _ACRONYM_RE.findall(q)
    if acronyms:
        score += min(0.4, 0.1 * len(acronyms))
    ql = q.lower()
    if any(b in ql for b in _KNOWN_BENCHMARKS):
        score += 0.4
    if '"' in q:
        score += 0.3
    return min(score, 1.0)


def search_papers(
    query: str,
    top_k: int = 15,
    domain_filter: str = None,
    rank_by: str = "relevance",
) -> str:
    """
    Hybrid semantic + keyword search over all papers, with query-adaptive weighting.
    - Lexical queries (DOIs, dataset names, acronyms) lean sparse.
    - Conceptual queries lean dense.
    Args:
        rank_by: "relevance" (default), "fwci" (rerank top hits by field impact),
                 or "recent" (rerank by year desc among top hits).
    """
    client = QdrantClient(host=QDRANT["host"], port=QDRANT["port"])
    dense, sparse = _embed(query)
    lex = _query_lexicality(query)

    # Allocate prefetch budget proportional to query character.
    # Pure conceptual → dense 3x candidates; pure lexical → sparse 3x.
    pool = top_k * 6
    dense_limit  = max(top_k, int(pool * (1.0 - lex)))
    sparse_limit = max(top_k, int(pool * lex)) if lex > 0 else top_k * 2

    qfilter = None
    if domain_filter:
        qfilter = models.Filter(must=[
            models.FieldCondition(
                key="domain_classification",
                match=models.MatchText(text=domain_filter),
            )
        ])

    # Over-fetch when reranking so we have headroom
    fetch_k = top_k * 3 if rank_by in ("fwci", "recent") else top_k

    resp = client.query_points(
        collection_name=QDRANT["collection"],
        prefetch=[
            models.Prefetch(query=dense,  using="dense",  limit=dense_limit),
            models.Prefetch(query=sparse, using="sparse", limit=sparse_limit),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=qfilter,
        with_payload=_SUMMARY_FIELDS,
        limit=fetch_k,
    )

    docs = [pt.payload for pt in resp.points if pt.payload]
    if not docs:
        return "No papers found."

    meta = _enrich_meta([d.get("doi") for d in docs if d.get("doi")])

    if rank_by == "fwci":
        docs.sort(key=lambda d: meta.get(d.get("doi"), {}).get("fwci") or 0, reverse=True)
        docs = docs[:top_k]
    elif rank_by == "recent":
        docs.sort(key=lambda d: meta.get(d.get("doi"), {}).get("year") or 0, reverse=True)
        docs = docs[:top_k]

    weight_note = f"lexical={lex:.2f}, dense_cand={dense_limit}, sparse_cand={sparse_limit}, rank={rank_by}"
    parts = [f"search_papers('{query}') → {len(docs)} results [{weight_note}]:\n"]
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] {_fmt(d, meta=meta)}\n---")
    return "\n".join(parts)


def filter_by_entity(entity: str, top_k: int = 20) -> str:
    """
    Find all papers that mention a specific concept, method, algorithm,
    dataset, or technique. Good for mapping which papers use a given approach.
    """
    client = QdrantClient(host=QDRANT["host"], port=QDRANT["port"])

    points, _ = client.scroll(
        collection_name=QDRANT["collection"],
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="entites_nommees_specifiques_domaine",
                    match=models.MatchText(text=entity),
                )
            ]
        ),
        limit=top_k,
        with_payload=[
            "doi",
            "domain_classification",
            "entites_nommees_specifiques_domaine",
            "resultats_principaux",
            "directions_futures",
            "limitations_mentionnees",
        ],
        with_vectors=False,
    )

    if not points:
        return f"No papers found mentioning '{entity}'."

    parts = [f"filter_by_entity('{entity}') → {len(points)} papers:\n"]
    for pt in points:
        p = pt.payload or {}
        doi = p.get("doi", "N/A")
        domain = p.get("domain_classification", "")[:70]
        results = p.get("resultats_principaux", "")
        future = p.get("directions_futures", "")
        lim = p.get("limitations_mentionnees", "")

        meta = _enrich_meta([doi])
        m = meta.get(doi, {})
        header = f"DOI: {doi}"
        if m.get("title"):    header += f' | Title: {m["title"]}'
        if m.get("year"):     header += f' | Year: {m["year"]}'
        if m.get("fwci"):     header += f' | FWCI: {m["fwci"]:.2f}'
        lines = [header, f"Domain: {domain}"]
        if results and results != NOT_REPORTED:
            lines.append(f"Results: {results[:200]}")
        if future and future != NOT_REPORTED:
            lines.append(f"Future directions: {future[:180]}")
        if lim and lim != NOT_REPORTED:
            lines.append(f"Limitations: {lim[:160]}")
        parts.append("\n".join(lines) + "\n---")

    return "\n".join(parts)


def find_papers_with_gaps(keywords: str, top_k: int = 15) -> str:
    """
    Semantic search focused on research gaps, limitations, and future directions.
    Use this to find what problems remain unsolved and what future work exists
    in a specific area — essential for cross-paper connection and scenario reasoning.
    """
    client = QdrantClient(host=QDRANT["host"], port=QDRANT["port"])
    dense, sparse = _embed(keywords + " research gap open problem limitation future work direction")

    resp = client.query_points(
        collection_name=QDRANT["collection"],
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=top_k * 3),
            models.Prefetch(query=sparse, using="sparse", limit=top_k * 3),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=[
            "doi",
            "domain_classification",
            "research_gap_lacune",
            "directions_futures",
            "limitations_mentionnees",
            "conclusions_principales",
        ],
        limit=top_k,
    )

    docs = [pt.payload for pt in resp.points if pt.payload]
    if not docs:
        return "No papers found for this gap/future direction query."

    meta = _enrich_meta([d.get("doi") for d in docs if d.get("doi")])
    parts = [f"find_papers_with_gaps('{keywords}') → {len(docs)} papers:\n"]
    for d in docs:
        doi = d.get("doi", "N/A")
        gap = d.get("research_gap_lacune", "")
        future = d.get("directions_futures", "")
        lim = d.get("limitations_mentionnees", "")
        domain = d.get("domain_classification", "")[:60]

        lines = [f"DOI: {doi}", f"Domain: {domain}"]
        if gap and gap != NOT_REPORTED:
            lines.append(f"Gap: {gap[:220]}")
        if future and future != NOT_REPORTED:
            lines.append(f"Future: {future[:220]}")
        if lim and lim != NOT_REPORTED:
            lines.append(f"Limitations: {lim[:160]}")
        parts.append("\n".join(lines) + "\n---")

    return "\n".join(parts)


def get_paper_details(doi: str) -> str:
    """
    Retrieve the complete metadata for one specific paper by its DOI.
    Use this after finding a paper via search to get its full content,
    including hypotheses, methodology, all results, and implications.
    """
    client = QdrantClient(host=QDRANT["host"], port=QDRANT["port"])

    points, _ = client.scroll(
        collection_name=QDRANT["collection"],
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="doi",
                    match=models.MatchValue(value=doi),
                )
            ]
        ),
        limit=1,
        with_payload=_FULL_FIELDS,
        with_vectors=False,
    )

    if not points or not points[0].payload:
        return f"Paper not found for DOI: {doi}"

    return f"get_paper_details('{doi}'):\n{_fmt(points[0].payload, fields=_FULL_FIELDS)}"


def get_domain_synthesis(domain_name: str) -> str:
    """
    Retrieve the pre-computed Layer C synthesis document for a research domain.
    Each synthesis covers: dominant methods, consensus findings, active debates,
    common limitations, emerging trends, and top future directions — derived from
    all papers in that domain. Use this first when the user asks about a field
    or domain rather than a specific paper.
    """
    index_path = os.path.join(os.path.dirname(__file__), "..", "..", "domain_syntheses", "index.json")
    syntheses_dir = os.path.join(os.path.dirname(__file__), "..", "..", "domain_syntheses")

    if not os.path.exists(index_path):
        return (
            "Domain syntheses not built yet. "
            "Run build_domain_syntheses.py first, then build_world_model.py."
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    # Fuzzy match: find best domain in index
    query_lower = domain_name.lower().strip()
    matched_domain = None
    best_score = 0

    for name in index:
        name_lower = name.lower()
        if query_lower == name_lower:
            matched_domain = name
            break
        # partial match scoring
        if query_lower in name_lower or name_lower in query_lower:
            score = len(set(query_lower.split()) & set(name_lower.split()))
            if score > best_score:
                best_score = score
                matched_domain = name

    if not matched_domain:
        available = ", ".join(sorted(index.keys()))
        return (
            f"Domain '{domain_name}' not found in synthesis index.\n"
            f"Available domains: {available}"
        )

    fpath = os.path.join(syntheses_dir, index[matched_domain])
    if not os.path.exists(fpath):
        return f"Synthesis file for '{matched_domain}' not found on disk."

    with open(fpath, "r", encoding="utf-8") as f:
        doc = json.load(f)

    synthesis = doc.get("synthesis", "")
    n_papers  = doc.get("total_papers_in_corpus", "?")
    used      = doc.get("papers_used_for_synthesis", "?")

    # Cap at 4000 chars to stay within agent context budget
    if len(synthesis) > 4000:
        synthesis = synthesis[:4000] + "\n…[synthesis truncated]"

    return (
        f"get_domain_synthesis('{matched_domain}'):\n"
        f"Total papers in corpus: {n_papers} | Papers used for synthesis: {used}\n\n"
        f"{synthesis}"
    )


def find_papers_sharing_entities(doi: str, top_k: int = 10) -> str:
    """
    Graph traversal: given a paper DOI, find papers that share the most
    entity/method nodes in the knowledge graph. Returns shared entity names
    and the gap/future text of each related paper — ideal for finding
    methodologically similar work and their open problems.
    """
    driver = _neo4j()
    query = """
    MATCH (p:SimPaper {doi: $doi})-[:USES]->(e:SimEntity)<-[:USES]-(related:SimPaper)
    WHERE related.doi <> $doi
    RETURN related.doi                        AS doi,
           count(e)                           AS shared_count,
           collect(e.name)[..6]               AS shared_entities,
           related.research_gap_lacune        AS gap,
           related.directions_futures         AS future,
           related.domain_classification      AS domain,
           related.year                       AS year
    ORDER BY shared_count DESC
    LIMIT $top_k
    """
    try:
        with driver.session() as session:
            records = list(session.run(query, doi=doi, top_k=top_k))
        driver.close()
    except Exception as e:
        driver.close()
        return f"Graph query failed: {e}"

    if not records:
        return (
            f"No papers found sharing entities with DOI {doi}. "
            "The paper may not be in the graph yet — run build_knowledge_graph.py first."
        )

    NOT_R = "Not explicitly reported in the provided text."
    lines = [f"find_papers_sharing_entities('{doi}') → {len(records)} related papers:\n"]
    for r in records:
        lines.append(f"DOI: {r['doi']} | Year: {r['year']} | Domain: {(r['domain'] or '')[:60]}")
        lines.append(f"  Shared entities ({r['shared_count']}): {', '.join(r['shared_entities'])}")
        gap = r.get("gap") or ""
        fut = r.get("future") or ""
        if gap and gap != NOT_R:
            lines.append(f"  Gap: {gap[:220]}")
        if fut and fut != NOT_R:
            lines.append(f"  Future: {fut[:180]}")
        lines.append("---")
    return "\n".join(lines)


def get_entity_neighborhood(entity_name: str, top_k: int = 15) -> str:
    """
    Graph traversal: given a concept or method name, find what other concepts
    always co-appear with it (CO_OCCURS edges), and which papers use it.
    Use this to understand the methodological ecosystem around a technique
    and discover which papers are central to it.
    """
    driver = _neo4j()
    query = """
    MATCH (e:SimEntity)
    WHERE e.name CONTAINS toLower($name)
    OPTIONAL MATCH (e)-[r:CO_OCCURS]-(other:SimEntity)
    WITH e,
         collect({name: other.name, count: r.count})[..$top_k] AS co_occurs
    OPTIONAL MATCH (p:SimPaper)-[:USES]->(e)
    RETURN e.name       AS canonical,
           e.frequency  AS frequency,
           e.aliases    AS aliases,
           co_occurs,
           collect(p.doi)[..10] AS used_in_papers
    LIMIT 1
    """
    try:
        with driver.session() as session:
            result = session.run(query, name=entity_name.lower(), top_k=top_k).single()
        driver.close()
    except Exception as e:
        driver.close()
        return f"Graph query failed: {e}"

    if not result:
        return f"Entity '{entity_name}' not found in the knowledge graph."

    co = sorted(result["co_occurs"] or [], key=lambda x: x.get("count", 0), reverse=True)
    lines = [
        f"get_entity_neighborhood('{entity_name}'):",
        f"Canonical name : {result['canonical']}",
        f"Appears in     : {result['frequency']} papers",
        f"Aliases seen   : {', '.join(result['aliases'] or [])}",
        "",
        f"Top co-occurring concepts (appear together in same papers):",
    ]
    for c in co[:top_k]:
        lines.append(f"  • {c['name']} (co-occurs {c['count']}x)")
    lines.append(f"\nSample papers using this entity:")
    for doi in result["used_in_papers"] or []:
        lines.append(f"  {doi}")
    return "\n".join(lines)


def find_future_realized(doi: str, top_k: int = 10) -> str:
    """
    Graph traversal: given a paper DOI, find papers that REALIZED its future
    directions — i.e. papers whose contributions match what this paper said
    needed to be done next. Uses FUTURE_REALIZED_IN edges built by
    build_semantic_edges.py. Ideal for tracing how research evolved over time.
    """
    driver = _neo4j()
    query = """
    MATCH (a:SimPaper {doi: $doi})-[r:FUTURE_REALIZED_IN]->(b:SimPaper)
    RETURN b.doi AS doi, b.title AS title, b.year AS year, b.fwci AS fwci,
           r.score AS score, r.semantic_sim AS semantic_sim,
           b.resultats_principaux AS results,
           b.domain_classification AS domain
    ORDER BY r.score DESC
    LIMIT $top_k
    """
    try:
        with driver.session() as session:
            records = list(session.run(query, doi=doi, top_k=top_k))
        driver.close()
    except Exception as e:
        driver.close()
        return f"Graph query failed: {e}"

    if not records:
        return f"No FUTURE_REALIZED_IN edges found for {doi}. Run build_semantic_edges.py first."

    NOT_R = "Not explicitly reported in the provided text."
    lines = [f"find_future_realized('{doi}') → {len(records)} papers that built on its future directions:\n"]
    for r in records:
        title = r["title"] or ""
        year  = r["year"] or "?"
        fwci  = f'{r["fwci"]:.2f}' if r["fwci"] else "?"
        score = f'{r["score"]:.3f}' if r["score"] else "?"
        lines.append(f"DOI: {r['doi']} | {title} | Year: {year} | FWCI: {fwci} | Score: {score}")
        res = r.get("results") or ""
        if res and res != NOT_R:
            lines.append(f"  Contribution: {res[:220]}")
        lines.append("---")
    return "\n".join(lines)


def find_unrealized_futures(topic: str, top_k: int = 10) -> str:
    """
    Find research papers whose proposed future directions have NOT been built on
    by any later paper in the corpus (no outgoing FUTURE_REALIZED_IN edge).
    These are the open frontiers — gaps that have been articulated but not closed.

    The `topic` is matched semantically against the future-direction text so the
    LLM can scope to an area (e.g. "transformer interpretability" or "medical AI").
    Returns the highest-impact (FWCI) papers whose futures remain open.
    """
    client = QdrantClient(host=QDRANT["host"], port=QDRANT["port"])
    dense, sparse = _embed(topic + " future work open problem next step")

    resp = client.query_points(
        collection_name=QDRANT["collection"],
        prefetch=[
            models.Prefetch(query=dense,  using="dense",  limit=top_k * 8),
            models.Prefetch(query=sparse, using="sparse", limit=top_k * 8),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=["doi", "directions_futures", "domain_classification"],
        limit=top_k * 6,
    )

    candidate_dois = [pt.payload.get("doi") for pt in resp.points
                      if pt.payload and pt.payload.get("doi")]
    if not candidate_dois:
        return f"No candidate papers found for topic '{topic}'."

    # Ask Neo4j which of these candidates have NO outgoing FUTURE_REALIZED_IN edge
    driver = _neo4j()
    query = """
    UNWIND $dois AS doi
    MATCH (p:SimPaper {doi: doi})
    WHERE NOT (p)-[:FUTURE_REALIZED_IN]->()
      AND p.directions_futures IS NOT NULL
      AND p.directions_futures <> ''
    RETURN p.doi                  AS doi,
           p.title                AS title,
           p.year                 AS year,
           p.fwci                 AS fwci,
           p.cited_by_count       AS citations,
           p.domain_classification AS domain,
           p.directions_futures   AS future
    ORDER BY coalesce(p.fwci, 0.0) DESC, coalesce(p.cited_by_count, 0) DESC
    LIMIT $top_k
    """
    try:
        with driver.session() as session:
            records = list(session.run(query, dois=candidate_dois, top_k=top_k))
    except Exception as e:
        driver.close()
        return f"Graph query failed: {e}"
    driver.close()

    if not records:
        return (
            f"No unrealized futures found for '{topic}'. "
            "Either every candidate has a FUTURE_REALIZED_IN edge already, "
            "or no candidate has directions_futures set."
        )

    lines = [f"find_unrealized_futures('{topic}') → {len(records)} open frontiers:\n"]
    for r in records:
        title = (r["title"] or "")[:80]
        year  = r["year"] or "?"
        fwci  = f'{r["fwci"]:.2f}' if r["fwci"] is not None else "?"
        cites = r["citations"] or 0
        dom   = (r["domain"] or "")[:50]
        lines.append(f"DOI: {r['doi']} | {title} | Year: {year} | FWCI: {fwci} | Cites: {cites}")
        lines.append(f"  Domain: {dom}")
        lines.append(f"  Unrealized future: {(r['future'] or '')[:300]}")
        lines.append("---")
    return "\n".join(lines)


def bridge_domains(domain_a: str, domain_b: str, top_k: int = 10) -> str:
    """
    Find SimEntity nodes (methods, concepts) that are used in BOTH domain A and
    domain B but rarely by the same paper — i.e. methodology transfer opportunities.
    A method that's "owned" by domain A but appears occasionally in domain B is
    a candidate for cross-domain innovation.

    Returns entities ranked by their bridge potential: paper count in each domain,
    and a small sample of bridge papers (papers that touch both).
    """
    driver = _neo4j()
    query = """
    MATCH (e:SimEntity)<-[:USES]-(p:SimPaper)
    WITH e,
         sum(CASE WHEN toLower(p.domain_classification) CONTAINS toLower($a) THEN 1 ELSE 0 END) AS in_a,
         sum(CASE WHEN toLower(p.domain_classification) CONTAINS toLower($b) THEN 1 ELSE 0 END) AS in_b,
         count(p) AS total
    WHERE in_a >= 2 AND in_b >= 2
    WITH e, in_a, in_b, total,
         (in_a + in_b) * 1.0 / (1 + abs(in_a - in_b)) AS bridge_score
    OPTIONAL MATCH (e)<-[:USES]-(bridge:SimPaper)
    WHERE toLower(bridge.domain_classification) CONTAINS toLower($a)
       OR toLower(bridge.domain_classification) CONTAINS toLower($b)
    WITH e, in_a, in_b, total, bridge_score,
         collect(DISTINCT bridge.doi)[..3] AS sample_dois
    RETURN e.name        AS entity,
           e.frequency   AS frequency,
           in_a, in_b, total,
           bridge_score,
           sample_dois
    ORDER BY bridge_score DESC, total DESC
    LIMIT $top_k
    """
    try:
        with driver.session() as session:
            records = list(session.run(query, a=domain_a, b=domain_b, top_k=top_k))
    except Exception as e:
        driver.close()
        return f"Graph query failed: {e}"
    driver.close()

    if not records:
        return (
            f"No bridging entities found between '{domain_a}' and '{domain_b}'. "
            "Either the domains share no methods or domain names don't match the corpus."
        )

    lines = [
        f"bridge_domains('{domain_a}', '{domain_b}') → {len(records)} candidate bridges:\n",
        "Methods used in BOTH domains — candidates for cross-domain transfer.\n",
    ]
    for r in records:
        lines.append(
            f"Entity: {r['entity']} | freq: {r['frequency']} | "
            f"{domain_a}: {r['in_a']} papers | {domain_b}: {r['in_b']} papers | "
            f"bridge_score: {r['bridge_score']:.2f}"
        )
        for doi in r["sample_dois"] or []:
            lines.append(f"  • Sample bridge paper: {doi}")
        lines.append("---")
    return "\n".join(lines)


def find_benchmark_papers(dataset_name: str, top_k: int = 10) -> str:
    """
    Given a benchmark dataset name (e.g. "CIFAR-10", "ImageNet", "MNIST"), return
    the highest-impact papers that evaluated on it. Uses EVALUATED_ON edges between
    SimPaper and SimDataset nodes. Useful for "what's state of the art on X?".
    """
    driver = _neo4j()
    canonical = (
        dataset_name.strip().lower()
        .replace('-', ' ').replace('_', ' ')
    )
    query = """
    MATCH (ds:SimDataset)
    WHERE ds.name CONTAINS $canon
       OR $canon CONTAINS ds.name
    MATCH (p:SimPaper)-[:EVALUATED_ON]->(ds)
    RETURN p.doi AS doi, p.title AS title, p.year AS year, p.fwci AS fwci,
           p.cited_by_count AS citations,
           p.resultats_principaux AS results,
           ds.name AS matched_dataset
    ORDER BY coalesce(p.fwci, 0.0) DESC, coalesce(p.cited_by_count, 0) DESC
    LIMIT $top_k
    """
    try:
        with driver.session() as session:
            records = list(session.run(query, canon=canonical, top_k=top_k))
    except Exception as e:
        driver.close()
        return f"Graph query failed: {e}"
    driver.close()

    if not records:
        return (
            f"No papers found evaluating on '{dataset_name}'. "
            "Available SimDatasets are limited to known benchmarks. Try filter_by_entity instead."
        )

    NOT_R = "Not explicitly reported in the provided text."
    lines = [f"find_benchmark_papers('{dataset_name}') → {len(records)} papers:\n"]
    seen_ds = set()
    for r in records:
        seen_ds.add(r["matched_dataset"])
        title = (r["title"] or "")[:80]
        year  = r["year"] or "?"
        fwci  = f'{r["fwci"]:.2f}' if r["fwci"] is not None else "?"
        cites = r["citations"] or 0
        lines.append(f"DOI: {r['doi']} | {title} | Year: {year} | FWCI: {fwci} | Cites: {cites}")
        res = r.get("results") or ""
        if res and res != NOT_R:
            lines.append(f"  Results: {res[:200]}")
        lines.append("---")
    lines.append(f"Matched SimDataset names: {', '.join(sorted(seen_ds))}")
    return "\n".join(lines)


def entity_trend(entity_name: str, window: int = 6) -> str:
    """
    Temporal trend of a method/concept. Returns paper count and average FWCI
    per year for the last `window` years. Shows whether a method is rising
    (more recent papers, higher impact) or declining.
    """
    driver = _neo4j()
    query = """
    MATCH (e:SimEntity)
    WHERE e.name CONTAINS toLower($name)
    WITH e ORDER BY e.frequency DESC LIMIT 1
    MATCH (p:SimPaper)-[:USES]->(e)
    WHERE p.year IS NOT NULL
    WITH e, p.year AS yr, p.fwci AS fwci
    RETURN e.name AS canonical,
           e.frequency AS total_freq,
           yr,
           count(*) AS papers,
           avg(coalesce(fwci, 0.0)) AS avg_fwci,
           sum(CASE WHEN coalesce(fwci, 0.0) > 5.0 THEN 1 ELSE 0 END) AS high_impact
    ORDER BY yr DESC
    LIMIT $window
    """
    try:
        with driver.session() as session:
            records = list(session.run(query, name=entity_name.lower(), window=window))
    except Exception as e:
        driver.close()
        return f"Graph query failed: {e}"
    driver.close()

    if not records:
        return f"Entity '{entity_name}' has no time-stamped papers in the graph."

    canonical = records[0]["canonical"]
    total = records[0]["total_freq"]
    lines = [
        f"entity_trend('{entity_name}') → matched: {canonical} (total papers: {total})",
        "",
        f"{'Year':<6} {'Papers':<8} {'Avg FWCI':<10} {'High-impact (FWCI>5)'}",
        "-" * 50,
    ]
    for r in sorted(records, key=lambda x: x["yr"]):
        lines.append(
            f"{r['yr']!s:<6} {r['papers']!s:<8} {r['avg_fwci']:.2f}{'':<5} {r['high_impact']}"
        )
    return "\n".join(lines)


def anticipate_scenario(topic: str, top_k_gaps: int = 5, top_k_methods: int = 5) -> str:
    """
    Orchestration tool — runs the full scenario-anticipation chain in one call.

    1. Find unrealized future directions for the topic.
    2. For each gap, surface candidate methods from the corpus that semantically
       align with the gap text (could plausibly attack it).
    3. Return the (gap_paper, gap_text, candidate_methods, candidate_papers) bundle.

    This is the high-leverage tool for "what new research scenarios could I run?".
    The LLM should call this when the user wants forward-looking ideas, then cite
    DOIs and methods from the result.
    """
    client = QdrantClient(host=QDRANT["host"], port=QDRANT["port"])
    driver = _neo4j()

    # ── Step 1: gather candidate gap papers ──────────────────────────────────
    dense, sparse = _embed(topic + " future work open problem direction")
    resp = client.query_points(
        collection_name=QDRANT["collection"],
        prefetch=[
            models.Prefetch(query=dense,  using="dense",  limit=top_k_gaps * 10),
            models.Prefetch(query=sparse, using="sparse", limit=top_k_gaps * 10),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=["doi", "directions_futures", "domain_classification"],
        limit=top_k_gaps * 8,
    )
    candidate_dois = [pt.payload.get("doi") for pt in resp.points
                      if pt.payload and pt.payload.get("doi")]
    if not candidate_dois:
        driver.close()
        return f"No gap candidates found for topic '{topic}'."

    # ── Step 2: filter to those with no FUTURE_REALIZED_IN ────────────────────
    gap_query = """
    UNWIND $dois AS doi
    MATCH (p:SimPaper {doi: doi})
    WHERE NOT (p)-[:FUTURE_REALIZED_IN]->()
      AND p.directions_futures IS NOT NULL
      AND p.directions_futures <> ''
    RETURN p.doi AS doi, p.title AS title, p.year AS year, p.fwci AS fwci,
           p.directions_futures AS future,
           p.domain_classification AS domain
    ORDER BY coalesce(p.fwci, 0.0) DESC
    LIMIT $top_k
    """
    try:
        with driver.session() as session:
            gap_papers = list(session.run(gap_query, dois=candidate_dois, top_k=top_k_gaps))
    except Exception as e:
        driver.close()
        return f"Graph query failed: {e}"

    if not gap_papers:
        driver.close()
        return f"No unrealized gaps for '{topic}'. Try a broader topic."

    # ── Step 3: for each gap, find candidate methods via semantic search ──────
    lines = [
        f"anticipate_scenario('{topic}') → {len(gap_papers)} scenario hypotheses:\n",
        "Each hypothesis = (open gap paper) + (candidate methods that could attack it).",
        "Use these as starting points for new research scenarios.\n",
        "=" * 70,
    ]

    for i, gp in enumerate(gap_papers, 1):
        future_text = gp["future"] or ""
        dense_g, sparse_g = _embed(future_text[:500])
        method_resp = client.query_points(
            collection_name=QDRANT["collection"],
            prefetch=[
                models.Prefetch(query=dense_g,  using="dense",  limit=top_k_methods * 4),
                models.Prefetch(query=sparse_g, using="sparse", limit=top_k_methods * 4),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=["doi", "entites_nommees_specifiques_domaine",
                          "resultats_principaux", "domain_classification"],
            limit=top_k_methods + 5,
        )
        method_candidates = [pt.payload for pt in method_resp.points
                             if pt.payload and pt.payload.get("doi") != gp["doi"]][:top_k_methods]

        method_dois = [m.get("doi") for m in method_candidates if m.get("doi")]
        meta = _enrich_meta(method_dois) if method_dois else {}

        gap_title = (gp["title"] or "")[:80]
        gap_year  = gp["year"] or "?"
        gap_fwci  = f'{gp["fwci"]:.2f}' if gp["fwci"] is not None else "?"

        lines.append(f"\n### Scenario #{i}")
        lines.append(f"GAP from: {gp['doi']} | {gap_title} | Year: {gap_year} | FWCI: {gap_fwci}")
        lines.append(f"Domain: {(gp['domain'] or '')[:60]}")
        lines.append(f"Open direction: {future_text[:280]}")
        lines.append(f"\nCandidate methods (papers that could attack this gap):")
        for m in method_candidates:
            mdoi  = m.get("doi", "?")
            md    = meta.get(mdoi, {})
            title = (md.get("title") or "")[:70]
            fwci  = f'{md["fwci"]:.2f}' if md.get("fwci") is not None else "?"
            entities = (m.get("entites_nommees_specifiques_domaine") or "")[:100]
            lines.append(f"  • {mdoi} | {title} | FWCI: {fwci}")
            if entities:
                lines.append(f"      Methods: {entities}")
        lines.append("-" * 70)

    driver.close()
    return "\n".join(lines)


TOOLS = {
    "search_papers": search_papers,
    "filter_by_entity": filter_by_entity,
    "find_papers_with_gaps": find_papers_with_gaps,
    "get_paper_details": get_paper_details,
    "find_papers_sharing_entities": find_papers_sharing_entities,
    "get_entity_neighborhood": get_entity_neighborhood,
    "get_domain_synthesis": get_domain_synthesis,
    "find_future_realized": find_future_realized,
    "find_unrealized_futures": find_unrealized_futures,
    "bridge_domains": bridge_domains,
    "find_benchmark_papers": find_benchmark_papers,
    "entity_trend": entity_trend,
    "anticipate_scenario": anticipate_scenario,
}
