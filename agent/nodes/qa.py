"""
QA node — corpus-aware, tool-calling research assistant.

Architecture:
  1. corpus_brain.json (built by build_corpus_brain.py) is injected into the
     system prompt so the LLM knows the shape of all 3,600 papers.
  2. The LLM emits <tool_call> blocks to query Qdrant iteratively (up to 5 rounds).
  3. Results are fed back as user turns so DeepSeek can keep reasoning.
  4. On the final round the LLM synthesizes everything it retrieved.
"""
import json
import os
import re
from agent.state import AgentState
from agent.llm import call_llm
from agent.tools import TOOLS

# ── Corpus brain (Layer A index) ─────────────────────────────────────────────

_BRAIN: dict | None = None
_BRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "corpus_brain.json")

# ── World model (Layer D) ────────────────────────────────────────────────────

_WORLD_MODEL: dict | None = None
_WORLD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "world_model.json")


def _load_world_model() -> str:
    global _WORLD_MODEL
    if _WORLD_MODEL is not None:
        return _WORLD_MODEL.get("world_model", "")
    try:
        with open(_WORLD_MODEL_PATH, "r", encoding="utf-8") as f:
            _WORLD_MODEL = json.load(f)
        return _WORLD_MODEL.get("world_model", "")
    except FileNotFoundError:
        _WORLD_MODEL = {}
        return ""


def _load_brain() -> dict:
    global _BRAIN
    if _BRAIN is not None:
        return _BRAIN
    try:
        with open(_BRAIN_PATH, "r", encoding="utf-8") as f:
            _BRAIN = json.load(f)
    except FileNotFoundError:
        _BRAIN = {}
    return _BRAIN


def _format_memory(mem: dict) -> str:
    """Format session memory for injection into the system prompt."""
    if not mem:
        return ""
    parts = ["─────────────────────────────────────────",
             "## Session Memory (what this user has been working on)\n"]
    topics = mem.get("topics") or []
    if topics:
        parts.append(f"**Topics explored:** {', '.join(topics[-10:])}")
    domains = mem.get("domains") or []
    if domains:
        parts.append(f"**Domains touched:** {', '.join(domains[-6:])}")
    sims = mem.get("simulations") or []
    if sims:
        parts.append(f"**Simulations built:** {', '.join(sims[-5:])}")
    notes = (mem.get("notes") or "").strip()
    if notes:
        parts.append(f"**User notes:** {notes}")
    parts += [
        "",
        "Use this to:",
        "- Skip re-explaining concepts they already know",
        "- Connect current question to their past work",
        "- Suggest follow-ups that build on prior sessions",
        "─────────────────────────────────────────",
    ]
    return "\n".join(parts)


def _format_brain(b: dict) -> str:
    if not b:
        return "(Corpus brain not found — run build_corpus_brain.py first.)"

    lines = [
        f"## Corpus Overview: {b.get('total_papers', '?')} peer-reviewed papers\n",
        "### Domains (paper count):",
    ]
    for domain, count in list(b.get("domains", {}).items())[:20]:
        lines.append(f"  - {domain}: {count}")

    lines.append("\n### Key Concepts & Methods found across the corpus:")
    lines.append("  " + ", ".join(b.get("top_entities", [])[:50]))

    lines.append("\n### Study Types:")
    for stype, count in list(b.get("study_types", {}).items())[:8]:
        lines.append(f"  - {stype}: {count}")

    lines.append("\n### Sample Research Gaps in the corpus:")
    for g in b.get("sample_gaps", [])[:10]:
        lines.append(f"  • {g}")

    lines.append("\n### Sample Future Directions in the corpus:")
    for fv in b.get("sample_futures", [])[:10]:
        lines.append(f"  • {fv}")

    return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a tool-using research agent. Your ONLY way to retrieve information is by emitting fenced ```tool_call``` blocks. You CANNOT answer from memory or from the corpus summary below — you MUST call tools first.

═════════════════════════════════════════
## MANDATORY FIRST ACTION

Your first round of every conversation MUST contain at least one fenced ```tool_call``` block. NO prose. NO meta-narration like "I will use X" or "Here is my plan". Just emit the blocks.

CORRECT round-1 output for "what do we know about computer vision?":

```tool_call
{{"tool": "get_domain_synthesis", "args": {{"domain_name": "Computer Vision"}}}}
```

```tool_call
{{"tool": "search_papers", "args": {{"query": "computer vision deep learning advances", "top_k": 10, "rank_by": "fwci"}}}}
```

```tool_call
{{"tool": "entity_trend", "args": {{"entity_name": "transformer", "window": 6}}}}
```

WRONG (these patterns will be REJECTED):
- "I will use get_domain_synthesis to retrieve…"
- "Here's my plan: 1. Call X. 2. Call Y."
- "1. Domain Overview: Materials Science…" (answering from memory)
- "Use get_domain_synthesis with domain_name Computer Vision" (no fences)

═════════════════════════════════════════
## Corpus context (use this ONLY for picking which tool to call — DO NOT cite it directly)

{corpus_overview}

{world_model_section}
─────────────────────────────────────────
## Tool catalogue

### Discovery & retrieval
- **search_papers** `{{"query": "...", "top_k": 15, "rank_by": "relevance|fwci|recent"}}` — hybrid search; weighting auto-adapts to query (lexical vs conceptual). `rank_by=fwci` surfaces high-impact papers; `rank_by=recent` surfaces latest work.
- **filter_by_entity** `{{"entity": "...", "top_k": 20}}` — papers explicitly mentioning a method/concept in their entities field.
- **find_papers_with_gaps** `{{"keywords": "...", "top_k": 15}}` — papers whose gaps/limitations/future work match a topic.
- **get_paper_details** `{{"doi": "10.xxxx/yyyy"}}` — full metadata for one paper.

### Knowledge graph (Neo4j)
- **find_papers_sharing_entities** `{{"doi": "...", "top_k": 10}}` — methodologically similar papers.
- **get_entity_neighborhood** `{{"entity_name": "...", "top_k": 15}}` — concepts that co-occur with a method.
- **find_future_realized** `{{"doi": "...", "top_k": 10}}` — papers that BUILT ON a paper's stated future directions.
- **find_benchmark_papers** `{{"dataset_name": "CIFAR-10", "top_k": 10}}` — highest-impact papers evaluated on a benchmark.

### Trend & overview
- **get_domain_synthesis** `{{"domain_name": "Machine Learning"}}` — pre-computed synthesis: dominant methods, consensus, debates, trends, future directions.
- **entity_trend** `{{"entity_name": "transformer", "window": 6}}` — yearly paper count + avg FWCI for a method (is it rising or declining?).

### Frontier discovery (the high-leverage tools)
- **find_unrealized_futures** `{{"topic": "...", "top_k": 10}}` — gaps that have been articulated but no paper has closed. Each result = an open frontier.
- **bridge_domains** `{{"domain_a": "Materials Science", "domain_b": "Machine Learning", "top_k": 10}}` — methods used by BOTH domains rarely simultaneously — cross-domain transfer opportunities.
- **anticipate_scenario** `{{"topic": "...", "top_k_gaps": 5, "top_k_methods": 5}}` — orchestration: combines unrealized gaps with candidate methods that could attack each gap. Use when the user asks "what should I research / build / explore next?".

─────────────────────────────────────────
## Query understanding — classify FIRST, then route

Before any tool call, silently classify the user's intent. Pick the closest:

| Intent | Triggers | Open with |
|--------|----------|-----------|
| **Overview** | "what is X about", "summarize Y field", "trends in Z" | `get_domain_synthesis` THEN `entity_trend` for top methods |
| **Specific paper** | DOI present, "tell me about paper X", "what does this paper say" | `get_paper_details` THEN `find_papers_sharing_entities` |
| **Method lookup** | "papers using X", "X applications" | `filter_by_entity` AND `search_papers` (cross-check) |
| **Benchmark / SOTA** | "state of the art on X", "best results on dataset Y" | `find_benchmark_papers` THEN `search_papers rank_by=fwci` |
| **Gap analysis** | "what's missing", "open problems", "limitations of Y" | `find_papers_with_gaps` THEN `find_unrealized_futures` |
| **Scenario / what next** | "what to build", "new ideas", "research directions", "anticipate" | `anticipate_scenario` THEN cite specific scenario hypotheses |
| **Cross-domain** | "apply X to Y", "X meets Y", "transfer Y into Z" | `bridge_domains` THEN `search_papers` for sample bridge papers |
| **Temporal** | "rising methods", "recent advances", "how has X evolved" | `entity_trend` AND `search_papers rank_by=recent` |
| **Tracing** | "what came after paper X", "did anyone build on Y" | `find_future_realized` THEN `find_papers_sharing_entities` |

If the question is ambiguous, run 2-3 angles in parallel in round 1.

─────────────────────────────────────────
## Reasoning protocol (every query)

1. **Round 1 — Cast wide.** Emit 2-4 tool calls covering different angles. Lead with the intent's "Open with" tools.
2. **Round 2 — Drill down.** Based on returned DOIs and entities, follow up with graph-traversal tools (`find_papers_sharing_entities`, `get_entity_neighborhood`, `find_future_realized`).
3. **Round 3 — Triangulate.** If you've cited papers, cross-check with `find_papers_with_gaps` or `bridge_domains` to enrich or challenge them.
4. **Final round — Answer.** No tool_call blocks. Write a structured answer that:
   - Opens with a 1-sentence direct answer to the user's question.
   - Lists 3-7 cited evidence points, each with `[Title (DOI, Year, FWCI=X)]`.
   - If forward-looking: end with 2-3 concrete scenario hypotheses naming specific methods and DOIs.

─────────────────────────────────────────
## Final-answer rules (only when you have enough evidence and stop calling tools)

- **Cite every claim** with `[Title (DOI, Year, FWCI=X)]`. Never write "studies show" without a DOI.
- **High-FWCI papers carry more weight** in your synthesis — surface them first.
- **If papers contradict, say so explicitly.** Show both sides with DOIs.
- **For scenario questions, name specific methods + specific papers** as candidates. Do not give generic advice.
- **NEVER output "Suggested Next Steps" or "Refine your search".** You have the tools — keep calling them.
- **Avoid restating the prompt.** Get straight to evidence.
- **Use 2-3 tool rounds minimum** for any substantive question.
"""


# ── Tool call parsing & execution ─────────────────────────────────────────────

_TOOL_PATTERN = re.compile(r"```tool_call\s*(.*?)\s*```", re.DOTALL)
# Fallback A: prose-style "Use TOOL_NAME {json args}" or "TOOL_NAME({json args})"
_PROSE_TOOL_PATTERN = re.compile(
    r"\b(search_papers|filter_by_entity|find_papers_with_gaps|get_paper_details|"
    r"find_papers_sharing_entities|get_entity_neighborhood|get_domain_synthesis|"
    r"find_future_realized|find_unrealized_futures|bridge_domains|"
    r"find_benchmark_papers|entity_trend|anticipate_scenario)\b\s*"
    r"\(?\s*(\{[^\{\}]*(?:\{[^\{\}]*\}[^\{\}]*)*\})",
    re.DOTALL,
)


def _extract_balanced_json_objects(text: str) -> list[str]:
    """Walk the text and return every top-level balanced {...} substring."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            in_str = False
            esc = False
            start = i
            while i < n:
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            out.append(text[start:i+1])
                            i += 1
                            break
                i += 1
        else:
            i += 1
    return out


def _parse_calls(text: str) -> list[dict]:
    """
    Find tool calls in three formats (in priority order):
      1. Fenced ```tool_call``` blocks
      2. Bare JSON objects with "tool" and "args" keys, anywhere in the text
      3. Prose-style "TOOL_NAME({json args})" patterns
    """
    calls = []
    seen_sigs: set[str] = set()

    def _add(call: dict) -> bool:
        if not isinstance(call, dict):
            return False
        if "tool" not in call or "args" not in call:
            return False
        sig = json.dumps(call, sort_keys=True)
        if sig in seen_sigs:
            return False
        calls.append(call)
        seen_sigs.add(sig)
        return True

    # Format 1: fenced blocks
    for m in _TOOL_PATTERN.finditer(text):
        try:
            _add(json.loads(m.group(1).strip()))
        except json.JSONDecodeError:
            pass

    # Format 2: bare JSON tool-call objects (catches the case where the model
    # emits valid {"tool": ..., "args": ...} but skips the fence)
    if not calls:
        for obj_str in _extract_balanced_json_objects(text):
            try:
                obj = json.loads(obj_str)
                _add(obj)
            except json.JSONDecodeError:
                pass

    # Format 3: prose-style "Use TOOL_NAME {args}"
    if not calls:
        for m in _PROSE_TOOL_PATTERN.finditer(text):
            try:
                _add({"tool": m.group(1), "args": json.loads(m.group(2).strip())})
            except json.JSONDecodeError:
                pass

    return calls


def _run_call(call: dict) -> str:
    name = call.get("tool", "")
    args = call.get("args", {})
    fn = TOOLS.get(name)
    if fn is None:
        return f"[ERROR] Unknown tool: '{name}'. Valid tools: {list(TOOLS)}"
    try:
        return fn(**args)
    except TypeError as e:
        return f"[ERROR] Bad args for {name}: {e}"
    except Exception as e:
        return f"[ERROR] {name} failed: {e}"


# ── Node ──────────────────────────────────────────────────────────────────────

MAX_ROUNDS = 5


def qa_node(state: AgentState) -> AgentState:
    query = state["raw_scenario"]
    placeholder = state.get("ui_placeholder")
    brain = _load_brain()
    world_model_text = _load_world_model()

    if world_model_text:
        world_model_section = (
            "─────────────────────────────────────────\n"
            "## Scientific World Model (pre-computed global prior)\n\n"
            + world_model_text[:3000]   # cap to avoid overloading context
            + ("\n…[world model truncated]" if len(world_model_text) > 3000 else "")
            + "\n─────────────────────────────────────────\n"
        )
    else:
        world_model_section = ""

    memory_section = _format_memory(state.get("session_memory") or {})
    system_content = _SYSTEM.format(
        corpus_overview=_format_brain(brain),
        world_model_section=world_model_section,
    )
    if memory_section:
        system_content = memory_section + "\n\n" + system_content

    # Inject last 6 turns of chat history (3 exchanges) for context continuity
    history = state.get("chat_history") or []
    history_trimmed = history[-6:] if len(history) > 6 else history

    messages = [
        {"role": "system", "content": system_content},
        *history_trimmed,
        {"role": "user", "content": query},
    ]

    all_docs: list = []
    seen_dois: set = set()  # cross-round working memory: dedupe paper mentions

    def _dedupe_result(text: str) -> str:
        """Mark already-seen DOIs and collapse their entries to a 1-line stub."""
        if not text:
            return text
        doi_re = re.compile(r"\b10\.\d{4,9}/\S+?\b")
        out_lines: list[str] = []
        block: list[str] = []
        block_dois: set[str] = set()

        def _flush():
            if not block:
                return
            dois_in_block = block_dois & seen_dois
            if dois_in_block and len(block) > 2:
                # Collapse: keep only header line + a "[seen]" marker
                doi = next(iter(dois_in_block))
                out_lines.append(f"{block[0]}")
                out_lines.append(f"  [already retrieved in this query — see earlier round for full details on {doi}]")
            else:
                out_lines.extend(block)
            # Remember every DOI we just emitted
            seen_dois.update(d.rstrip(".,;:)") for d in block_dois)

        for line in text.split("\n"):
            if line.strip().startswith("---") or line.strip() == "":
                _flush()
                block = []
                block_dois = set()
                out_lines.append(line)
                continue
            block.append(line)
            for d in doi_re.findall(line):
                block_dois.add(d.rstrip(".,;:)"))
        _flush()
        return "\n".join(out_lines)

    for round_num in range(MAX_ROUNDS):
        round_label = f"Round {round_num + 1}/{MAX_ROUNDS}"

        def _stream(text: str, label: str = round_label) -> None:
            if not placeholder:
                return
            display = text
            if "<think>" in display and "</think>" not in display:
                display = (
                    display.replace("<think>", f"*({label}: reasoning…)*\n\n```text\n")
                    + "\n```"
                )
            else:
                display = display.replace("<think>", "*(reasoning)*\n<!--").replace(
                    "</think>", "-->\n"
                )
            placeholder.markdown(display + "▌")

        response = call_llm(
            messages, temperature=0.1, max_tokens=4096, stream_callback=_stream
        )

        calls = _parse_calls(response)

        # Retry once on round 1 if the model produced prose instead of tool calls.
        # DeepSeek-R1 sometimes "plans" instead of acting — a forceful reminder
        # usually flips it on the second try.
        if not calls and round_num == 0:
            if placeholder:
                placeholder.markdown("*(no tool calls detected — forcing retry)*▌")
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response did not contain any ```tool_call``` blocks. "
                    "You MUST call tools — do not answer from memory or describe a plan. "
                    "Emit one or more fenced tool_call blocks NOW. Example:\n\n"
                    "```tool_call\n"
                    '{"tool": "get_domain_synthesis", "args": {"domain_name": "Computer Vision"}}\n'
                    "```\n\n"
                    "No prose. Just the blocks."
                ),
            })
            response = call_llm(
                messages, temperature=0.0, max_tokens=2048, stream_callback=_stream
            )
            calls = _parse_calls(response)

        if not calls:
            # Still no tool blocks after retry → treat response as final answer
            if placeholder:
                placeholder.markdown(response)
            state["final_answer"] = response
            state["retrieved_docs"] = {"qa_query": all_docs}
            return state

        # Append LLM turn and execute tools
        messages.append({"role": "assistant", "content": response})

        results_parts: list[str] = []
        for call in calls:
            result = _run_call(call)
            deduped = _dedupe_result(result)
            results_parts.append(
                f"### Result: {call.get('tool')} {call.get('args', {})}\n{deduped}"
            )
            # Collect any retrieved docs for state
            all_docs.append({"tool": call.get("tool"), "result_preview": deduped[:300]})

        # Truncate tool results to avoid blowing past 16k context
        combined = "\n\n".join(results_parts)
        if len(combined) > 6000:
            combined = combined[:6000] + "\n…[truncated to fit context]"

        if round_num < MAX_ROUNDS - 1:
            follow_up = (
                f"Tool results:\n\n{combined}\n\n"
                "Analyze these results. Call more tools if you need additional evidence, "
                "or write your final answer if you have enough to respond fully. "
                "Cite papers by title and DOI. Do not suggest next steps — answer directly."
            )
        else:
            follow_up = (
                f"Tool results:\n\n{combined}\n\n"
                "Final round. Write your complete answer now using the retrieved evidence. "
                "Cite every paper by title and DOI. No tool calls. No suggested next steps."
            )

        messages.append({"role": "user", "content": follow_up})

    # Force final synthesis if we exhausted rounds
    messages.append(
        {
            "role": "user",
            "content": (
                "Provide your final synthesis now based on all retrieved evidence."
            ),
        }
    )

    def _final_stream(text: str) -> None:
        if placeholder:
            placeholder.markdown(text + "▌")

    final = call_llm(
        messages, temperature=0.1, max_tokens=4096, stream_callback=_final_stream
    )

    if placeholder:
        placeholder.markdown(final)

    state["final_answer"] = final
    state["retrieved_docs"] = {"qa_query": all_docs}
    return state
