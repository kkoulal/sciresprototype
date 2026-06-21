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

---

## License

MIT
