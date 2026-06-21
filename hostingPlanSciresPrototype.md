# hostingPlanSciresPrototype.md — Deploy the standalone agent to a self-contained VPS

Goal: run `sciresVprototype` (the Digital Twin Brain agent) on **one VPS** that carries its
**own data** — the Neo4j Sim subgraph and the Qdrant vectors — so it depends on nothing
external except the Claude brain (OpenRouter). After this, the demo is reachable from
anywhere and needs zero per-session setup.

---

## 1. Architecture (everything on one VPS)

```
        ┌──────────────────── VPS (always-on) ────────────────────┐
Browser ─┤ Caddy (HTTPS + basic-auth, the only public port)        │
         │   ├─ app        Streamlit app_agent.py        :8501     │
         │   ├─ bge-m3     /embed_both (CPU)             :8080     │
         │   ├─ qdrant     simulation_papers collection  :6333     │
         │   └─ neo4j      SimPaper/SimEntity subgraph    :7687     │
         └──────────────────────────────────────────────────────────┘
                              │
                      Claude API (brain) ← OpenRouter, nothing to host
```

**No MinIO.** Unlike the v1 RAG, the agent reads paper fields directly from Qdrant payloads
([agent/nodes/retrieve.py](agent/nodes/retrieve.py)) and walks Neo4j for relationships
([agent/nodes/graph_enrich.py](agent/nodes/graph_enrich.py), [agent/tools.py](agent/tools.py)).
It never fetches markdown/figures from MinIO, so that service is dropped entirely.

### What moves vs what stays

| Data | Source (local) | To VPS? | How |
|------|----------------|---------|-----|
| Neo4j **Sim subgraph** (`SimPaper`/`SimEntity`/`SimDataset` + internal edges) | mixed local Neo4j | ✅ | APOC export → import |
| Neo4j **OpenAlex 250M graph** | local Neo4j | ❌ | stays local (agent never queries it) |
| Qdrant **`simulation_papers`** | local Qdrant | ✅ | snapshot → restore |
| MinIO buckets | local | ❌ | not used by the agent |

The agent's complete Neo4j surface (verified): nodes `SimPaper`, `SimEntity`, `SimDataset`;
relationships `USES`, `FUTURE_REALIZED_IN`, `CO_OCCURS`, `EVALUATED_ON`. OpenAlex-derived
values (`fwci`, etc.) are already materialized as properties on the Sim nodes, so the big
graph is not needed at serving time.

---

## 2. VPS sizing

Small — the Sim graph is ~3.6k papers and there's no MinIO/GPU.

| Resource | Recommended | Why |
|----------|-------------|-----|
| RAM | **8–16 GB** | neo4j ~2–3 GB + qdrant ~2–3 GB + bge-m3 ~3 GB + app ~0.5 GB + OS |
| vCPU | **2–4** | BGE-M3 query embedding + Qdrant search (CPU) |
| Disk | **40–60 GB SSD** | Qdrant snapshot + Neo4j data + bge-m3 model (~2.3 GB) + OS |
| GPU | **none** | one short query per request → CPU BGE-M3 is fine |

Best value: Hetzner CPX31/CPX41, or any DO/Lightsail equivalent. No GPU tier needed.

---

## 3. STEP A — Measure locally (right-size before paying)

```cypher
// Neo4j Sim subgraph size
MATCH (n) WHERE n:SimPaper OR n:SimEntity OR n:SimDataset
RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label;
```
```powershell
# Qdrant collection size
curl http://localhost:6333/collections/simulation_papers
```

---

## 4. STEP B — Export the Neo4j Sim subgraph

The export is **label-driven**, so cross-edges to the OpenAlex graph (`SAME_WORK_AS`,
`IN_OA_TOPIC`, `IN_OA_CONCEPT`) are excluded automatically — their other endpoint isn't a
Sim node.

**B.1 — enable APOC file export** (local `neo4j.conf`): `apoc.export.file.enabled=true`
(APOC is already installed).

**B.2 — note the real key properties:**
```cypher
SHOW CONSTRAINTS;
```

**B.3 — export Sim nodes + only their internal relationships:**
```cypher
MATCH (a)-[r]->(b)
WHERE (a:SimPaper OR a:SimEntity OR a:SimDataset)
  AND (b:SimPaper OR b:SimEntity OR b:SimDataset)
WITH collect(DISTINCT r) AS rels
MATCH (n) WHERE n:SimPaper OR n:SimEntity OR n:SimDataset
WITH rels, collect(DISTINCT n) AS nodes
CALL apoc.export.cypher.data(nodes, rels, 'sim_graph.cypher',
     {format:'cypher-shell', useOptimizations:{type:'UNWIND_BATCH', unwindBatchSize:1000}})
YIELD file, nodes AS nodeCount, relationships AS relCount
RETURN file, nodeCount, relCount;
```
The file lands in your local Neo4j `import/` dir. It's small (single-digit MB). Copy it out.

---

## 5. STEP C — Export the Qdrant collection (snapshot)

```bash
curl -X POST http://localhost:6333/collections/simulation_papers/snapshots
curl http://localhost:6333/collections/simulation_papers/snapshots        # get the name
curl -o simulation_papers.snapshot \
  http://localhost:6333/collections/simulation_papers/snapshots/<SNAPSHOT_NAME>
```

---

## 6. STEP D — Required code edit (container networking)

[agent/nodes/retrieve.py](agent/nodes/retrieve.py) **hardcodes** `http://localhost:8080/embed_both`.
Inside the app container, `localhost` is the app itself — not the `bge-m3` service. Change it
to read the configured URL so it resolves to the compose service:

```python
# retrieve.py — replace the hardcoded URL
from online.config import QDRANT, EMBED            # add EMBED to the import
...
r = httpx.post(f"{EMBED['url']}/embed_both", json={"inputs": [sq.query]}, timeout=30.0)
```

(Other nodes already use `cfg.EMBED["url"]` / `cfg.QDRANT` / `cfg.NEO4J`, so they're fine.)

---

## 7. STEP E — Provision the VPS

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER     # re-login
# Caddy (auto-TLS reverse proxy)
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy
mkdir -p ~/sciresVprototype && cd ~/sciresVprototype
```

Copy the project code up (from your machine):
```powershell
scp -r C:\Users\hp\Desktop\khalilKoulal\sciresVprototype\* user@<VPS_IP>:~/sciresVprototype/
```

---

## 8. STEP F — `docker-compose.yml`

Create `~/sciresVprototype/docker-compose.yml`:

```yaml
services:
  neo4j:
    image: neo4j:5-community
    restart: unless-stopped
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_server_memory_heap_max__size: 2G
      NEO4J_server_memory_pagecache_size: 1G
    volumes:
      - neo4j_data:/data
      - ./import:/var/lib/neo4j/import     # sim_graph.cypher goes here
    ports: ["127.0.0.1:7687:7687", "127.0.0.1:7474:7474"]

  qdrant:
    image: qdrant/qdrant:latest
    restart: unless-stopped
    volumes:
      - qdrant_data:/qdrant/storage
      - ./snapshots:/qdrant/snapshots      # simulation_papers.snapshot goes here
    ports: ["127.0.0.1:6333:6333"]

  bge-m3:
    build: ./bge
    restart: unless-stopped
    ports: ["127.0.0.1:8080:8080"]

  app:
    build: ./app
    restart: unless-stopped
    env_file: .env
    depends_on: [neo4j, qdrant, bge-m3]
    ports: ["127.0.0.1:8501:8501"]

volumes:
  neo4j_data:
  qdrant_data:
```

Only **Caddy** is internet-facing; every service binds to `127.0.0.1`.

### F.1 — BGE-M3 image (`./bge/`)

`bge/Dockerfile`:
```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir FlagEmbedding fastapi "uvicorn[standard]"
COPY bge_server.py /app/bge_server.py
WORKDIR /app
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)"
CMD ["uvicorn", "bge_server:app", "--host", "0.0.0.0", "--port", "8080"]
```
Copy the project's `bge_server.py` into `./bge/` (it loads on `cpu` when there's no GPU).

### F.2 — App image (`./app/`)

Put `app_agent.py`, `agent/`, `online/`, `requirements.txt` under `./app/`. `app/Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app_agent.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

---

## 9. STEP G — Config (service names + key)

Edit `app/online/config.py`:
```python
QDRANT = {"host": "qdrant", "port": 6333, "collection": "simulation_papers", "top_k": 8}
NEO4J  = {"uri": "bolt://neo4j:7687", "user": "neo4j", "password": "<NEO4J_PASSWORD>"}
EMBED  = {"url": "http://bge-m3:8080", "timeout": 60}
LLM    = {"url": "https://openrouter.ai/api/v1", "model": "anthropic/claude-sonnet-4.6",
          "api_key": os.getenv("OPENROUTER_API_KEY"), "max_tokens": 8192, "timeout": 300}
```

`~/sciresVprototype/.env`:
```
NEO4J_PASSWORD=<choose-strong>
OPENROUTER_API_KEY=sk-or-v1-...
```

> Make sure §6's `retrieve.py` edit is in the copied `app/agent/nodes/retrieve.py`, otherwise
> embedding calls go to the app container's own `localhost:8080` and fail.

---

## 10. STEP H — Import the data on the VPS

```bash
cd ~/sciresVprototype
mkdir -p import snapshots
# upload from local:
#   sim_graph.cypher            -> ./import/
#   simulation_papers.snapshot  -> ./snapshots/

docker compose up -d neo4j qdrant bge-m3        # infra first

# Neo4j: load the Sim subgraph
docker compose exec neo4j cypher-shell -u neo4j -p <NEO4J_PASSWORD> \
  -f /var/lib/neo4j/import/sim_graph.cypher
# recreate the index the agent relies on (adjust key prop to SHOW CONSTRAINTS output)
docker compose exec neo4j cypher-shell -u neo4j -p <NEO4J_PASSWORD> \
  "CREATE CONSTRAINT simpaper_doi IF NOT EXISTS FOR (p:SimPaper) REQUIRE p.doi IS UNIQUE;
   CREATE INDEX simentity_name IF NOT EXISTS FOR (e:SimEntity) ON (e.name);"

# Qdrant: restore the collection
curl -X POST "http://localhost:6333/collections/simulation_papers/snapshots/recover" \
  -H 'Content-Type: application/json' \
  -d '{"location":"file:///qdrant/snapshots/simulation_papers.snapshot"}'
```

Verify counts match local:
```bash
docker compose exec neo4j cypher-shell -u neo4j -p <NEO4J_PASSWORD> "MATCH (p:SimPaper) RETURN count(p);"
curl http://localhost:6333/collections/simulation_papers     # points == local
```

---

## 11. STEP I — Caddy (HTTPS + auth)

`/etc/caddy/Caddyfile` (Streamlit needs websockets — Caddy proxies them automatically):
```
demo.<your-domain> {
    basicauth { demo <BCRYPT_HASH> }     # caddy hash-password
    reverse_proxy localhost:8501
}
```
```bash
sudo systemctl reload caddy        # auto-provisions Let's Encrypt TLS
```

---

## 12. STEP J — Launch + smoke test

```bash
docker compose up -d --build
docker compose ps                  # all healthy?

# internal checks
curl localhost:8080/health         # BGE-M3 "Healthy"
curl localhost:6333/collections    # simulation_papers present
curl -I localhost:8501             # Streamlit 200
```

Open `https://demo.<your-domain>` (basic-auth), ask a relationship question
(e.g. *"what methods build on agent-based epidemic models?"*) and confirm the simulate path
fills Parameters / Agent Rules / Model Brief (not empty) and graph enrichment fires:
```bash
docker compose exec neo4j cypher-shell -u neo4j -p <NEO4J_PASSWORD> \
 "MATCH (p:SimPaper)-[:USES]->(e:SimEntity)<-[:USES]-(s:SimPaper) RETURN count(*);"
```

---

## 13. STEP K — Snapshot the VPS

Once it works, take a **provider snapshot**. Next demo = restore snapshot →
`docker compose up -d` → done. Power off between events to save cost.

---

## 14. Cost

- VPS ~€15–28/mo (no GPU); power off between demos to reduce.
- Claude via OpenRouter: pay-per-token; a demo is cents–dollars.
- No GPU, no MinIO.

---

## 15. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Embedding calls fail / empty retrieval | §6 edit missing | `retrieve.py` must use `EMBED['url']`, not `localhost:8080` |
| Empty simulate result | LLM JSON truncated | already fixed (`build_package` 16000 / `extract_params` 8000) — keep those |
| Graph enrich returns nothing | index missing / wrong key | re-run §10 index creation; check `SHOW CONSTRAINTS` |
| 401 / empty Claude answer | OpenRouter key/model | check `.env`; confirm model slug `anthropic/claude-sonnet-4.6` |
| Streamlit blank behind Caddy | websockets blocked | Caddy `reverse_proxy` handles ws by default — confirm the Caddyfile host matches |
| Neo4j import OOM | heap too small | bump `NEO4J_server_memory_heap_max__size` for the import |

---

## 16. Checklist

- [ ] §3 measure local sizes → finalize disk
- [ ] §4 export `sim_graph.cypher`
- [ ] §5 snapshot `simulation_papers.snapshot`
- [ ] §6 apply the `retrieve.py` URL edit
- [ ] §7 provision VPS (Docker + Caddy), upload code
- [ ] §8 `docker-compose.yml` + `bge/` + `app/` images
- [ ] §9 config service names + `.env`
- [ ] §10 import Neo4j + Qdrant; counts match
- [ ] §11 Caddy TLS + auth
- [ ] §12 smoke test (simulate fills, graph enrich returns rows)
- [ ] §13 VPS snapshot
