# sciresVprototype — standalone Scires agent (Digital Twin Brain)

The relationship-aware RAG agent extracted from `ProductSimulation`, runnable on its own.
It answers questions / builds simulation packages grounded in your paper corpus, using:

- **Qdrant** (`simulation_papers`) for semantic + sparse retrieval
- **Neo4j** (`SimPaper`/`SimEntity` graph) for relationship expansion
- **BGE-M3** (`:8080`) for query embedding
- **Claude Sonnet via OpenRouter** as the brain (key is in `online/config.py`)

```
app_agent.py            # Streamlit UI (entry point)
agent/                  # LangGraph agent: graph, nodes, tools, prompts, llm client
online/config.py        # endpoints + OpenRouter key (the only shared config)
bge_server.py           # local BGE-M3 embedding server (optional — see below)
requirements.txt
```

---

## ⚠️ Important: this project does NOT include the data

The corpus lives in **Docker services** (Qdrant + Neo4j + MinIO) that belong to the original
`scires` project. This prototype only contains *code* — it connects to those services on
`localhost`. So **keep the scires Docker stack running** while you use this:

```powershell
cd C:\Users\hp\Desktop\BusinessProjects\scires
docker-compose up -d        # Qdrant :6333, Neo4j :7687, MinIO :9000 (with the data)
```

If you ever want this to be *fully* independent of `scires`, you'd need to migrate the
Qdrant collection + Neo4j graph into its own services (see `ProductSimulation/HostingPlan.md`).

---

## Setup

```powershell
cd C:\Users\hp\Desktop\khalilKoulal\sciresVprototype

# (optional) virtual env
python -m venv venv ; .\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run

**1. Start the data services** (from the scires folder — see warning above).

**2. Start BGE-M3 on :8080.** Either:
- run it locally here:
  ```powershell
  pip install FlagEmbedding fastapi "uvicorn[standard]"
  uvicorn bge_server:app --host 0.0.0.0 --port 8080
  ```
- or tunnel a Thunder/remote BGE-M3 to `localhost:8080` (forward tunnel).

  Wait for `BGE-M3 ready` before continuing.

**3. Launch the agent UI:**
```powershell
streamlit run app_agent.py
```
Opens at http://localhost:8501.

---

## Config

Everything is in [online/config.py](online/config.py):

| Key | What |
|-----|------|
| `LLM` | OpenRouter URL + `anthropic/claude-sonnet-4.6` + API key (`OPENROUTER_API_KEY` env overrides) |
| `QDRANT` | `localhost:6333`, collection `simulation_papers` |
| `NEO4J` | `bolt://localhost:7687` (user `neo4j`) |
| `EMBED` | `http://localhost:8080` (BGE-M3) |

Prefer setting the key via env so it isn't hardcoded:
```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

---

## Notes

- The agent makes **many** Claude calls per query (router → parse → plan → retrieve →
  graph_enrich → extract → validate → build). A full `simulate` run costs more than a single
  `qa` question.
- `build_package` and `extract_params` use higher `max_tokens` (16000 / 8000) so large
  JSON outputs don't truncate — that's the fix for the "empty result" issue.
- This is a **code copy**. Edits here do not affect the original `ProductSimulation` and
  vice-versa.
