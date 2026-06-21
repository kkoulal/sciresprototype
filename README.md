# Digital Twin Brain

A conversational AI agent that turns a corpus of scientific papers into simulation-ready outputs. Ask questions grounded in thousands of papers, or walk through an AI-guided interview to build a full physics simulation of a physical object — including AI-generated CAD geometry and cloud FEA via SimScale.

---

## What it does

The app has two modes, selectable from the chat interface:

**Q&A** — Ask any research question. The agent searches a Qdrant vector store and a Neo4j knowledge graph, retrieves the most relevant paper excerpts, and synthesises an answer grounded in the literature.

**Simulate** — Describe a policy scenario or engineering problem in plain language. The agent:
1. Parses your scenario into a structured specification
2. Plans targeted sub-queries to the paper corpus
3. Extracts parameters, effect sizes, and behavioral rules from matched papers
4. Reconciles conflicting evidence across sources
5. Produces a downloadable **Simulation Package** (JSON) with parameters, agent rules, calibration targets, uncertainty report, and a model brief

For physical objects, a second pipeline then:
- Interviews you conversationally to capture geometry, material, loads, and boundary conditions
- Generates a CadQuery (Python) STEP file automatically — no manual CAD required
- Submits the geometry to **SimScale** for cloud FEA/CFD
- Interprets the stress or frequency results in plain language
- Optionally generates a photorealistic 3D render via **Meshy AI**

---

## Architecture

```
app_agent.py          # Streamlit UI — entry point
agent/
  graph.py            # LangGraph agent graph
  nodes/              # Individual reasoning nodes (parse, retrieve, extract, validate, build …)
  llm.py              # LLM client (OpenRouter / Claude)
  prompts.py          # All prompt templates
  object_agent.py     # Conversational object-interview agent
  geometry_agent.py   # AI CAD generation (CadQuery → STEP → STL)
  results_agent.py    # SimScale result interpretation
online/
  config.py           # All service endpoints — reads from .env
services/
  simulation.py       # SimScale API client
  meshy.py            # Meshy AI client
bge_server.py         # Local BGE-M3 embedding server
```

**Services used at runtime:**

| Service | Purpose | Default address |
|---|---|---|
| Qdrant | Vector search over the paper corpus | `localhost:6333` |
| Neo4j | Knowledge graph for relationship expansion | `bolt://localhost:7687` |
| MinIO | Object storage for simulation artefacts | `localhost:9000` |
| BGE-M3 | Query embedding | `localhost:8080` |
| Claude (OpenRouter) | LLM brain — all reasoning and synthesis | cloud |
| SimScale | Cloud FEA / CFD | cloud |
| Meshy AI | Text-to-3D visual render (optional) | cloud |

---

## Setup

### 1. Prerequisites

- Python 3.10+
- Docker (for Qdrant, Neo4j, MinIO)
- An [OpenRouter](https://openrouter.ai) API key with access to Claude

### 2. Clone and install

```bash
git clone https://github.com/kkoulal/sciresprototype.git
cd sciresprototype
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
OPENROUTER_API_KEY=sk-or-v1-...

MINIO_ACCESS_KEY=your_minio_key
MINIO_SECRET_KEY=your_minio_secret

NEO4J_PASSWORD=your_neo4j_password

SIMSCALE_API_KEY=your_simscale_key     # required for physics simulation
MESHY_API_KEY=your_meshy_key           # optional — for visual 3D renders
```

The remaining values (`MINIO_ENDPOINT`, `NEO4J_URI`, `QDRANT_HOST`, `EMBED_URL`) default to `localhost` and can be left as-is for a local setup.

### 4. Start data services

Bring up Qdrant, Neo4j, and MinIO using Docker Compose (or run them individually):

```bash
docker run -d -p 6333:6333 qdrant/qdrant
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/<your_password> neo4j
docker run -d -p 9000:9000 -e MINIO_ROOT_USER=<key> -e MINIO_ROOT_PASSWORD=<secret> minio/minio server /data
```

### 5. Start the embedding server

The BGE-M3 server embeds your queries locally before they hit Qdrant:

```bash
pip install FlagEmbedding fastapi "uvicorn[standard]"
uvicorn bge_server:app --host 0.0.0.0 --port 8080
```

Wait for `BGE-M3 ready` in the log before continuing. Alternatively, if you have a remote BGE-M3 instance, set `EMBED_URL` in `.env` to point at it.

### 6. Launch the app

```bash
streamlit run app_agent.py
```

Opens at [http://localhost:8501](http://localhost:8501).

---

## Usage

**Asking a question:**
Type any research question in the chat box and press Enter. The agent searches the corpus and returns a synthesised answer with paper references.

**Running a simulation:**
Type something like:
> *"Simulate the effect of a 10% fuel tax on urban CO₂ emissions in a mid-sized European city over 10 years"*

The agent will parse your scenario, retrieve relevant papers, extract parameters, and produce a downloadable simulation package.

**Simulating a physical object:**
Ask to simulate something physical, e.g.:
> *"Simulate an aluminium cantilever beam under a 500 N point load"*

The agent will ask you questions one at a time to capture geometry, material, and loading, then generate a CAD model, run FEA on SimScale, and explain the stress results.

**Sidebar:**
The session sidebar tracks explored topics, scientific domains, and simulation titles built in the current session. You can add free-text notes that are passed to the agent as context.

---

## Notes

- Each `simulate` run makes several LLM calls in sequence (router → parse → plan → retrieve × N → extract → validate → build). A complex scenario costs more than a simple Q&A.
- The simulation package JSON is self-contained and can be fed directly into agent-based modelling tools or used as a calibrated parameter set.
- SimScale and Meshy API keys are optional — the app degrades gracefully if they are not set; you still get the simulation package and AI-generated CAD files.
- The `.env` file is gitignored. Never commit it.

---

## License

MIT
