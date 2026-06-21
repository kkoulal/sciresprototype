"""
Online RAG server configuration.

All secrets and environment-specific values are read from a .env file at the
project root (loaded automatically below). Copy .env.example → .env and fill
in your keys before running the app.
"""

import os
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_env_path)


# ── MinIO (object storage) ─────────────────────────────────────────────────────
MINIO = {
    "endpoint":        os.getenv("MINIO_ENDPOINT",   "localhost:9000"),
    "access_key":      os.getenv("MINIO_ACCESS_KEY"),
    "secret_key":      os.getenv("MINIO_SECRET_KEY"),
    "secure":          False,
    "extracts_bucket": "simulation-extracts",
}

# ── Qdrant ─────────────────────────────────────────────────────────────────────
QDRANT = {
    "host":       os.getenv("QDRANT_HOST", "localhost"),
    "port":       6333,
    "collection": "simulation_papers",
    "top_k":      8,
}

# ── BGE-M3 embedding server ────────────────────────────────────────────────────
EMBED = {
    "url":     os.getenv("EMBED_URL", "http://localhost:8080"),
    "timeout": 60,
}

# ── Neo4j knowledge graph ──────────────────────────────────────────────────────
NEO4J = {
    "uri":      os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
    "user":     os.getenv("NEO4J_USER",     "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD"),
}

# ── LLM — Claude via OpenRouter ───────────────────────────────────────────────
LLM = {
    "url":        os.getenv("LLM_URL",        "https://openrouter.ai/api"),
    "model":      os.getenv("LLM_MODEL",      "anthropic/claude-sonnet-4-6"),
    "model_fast": os.getenv("LLM_MODEL_FAST", "gemini-3.1-flash-lite-preview"),
    "api_key":    os.getenv("OPENROUTER_API_KEY"),
    "max_tokens": 8192,
    "timeout":    300,
}

# ── Gemini (Google AI — OpenAI-compatible endpoint) ────────────────────────────
GEMINI = {
    "api_key":  os.getenv("GEMINI_API_KEY"),
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
}

# Back-compat alias
VLLM = LLM

# ── Meshy AI (text-to-3D) ─────────────────────────────────────────────────────
MESHY = {
    "api_key":         os.getenv("MESHY_API_KEY"),
    "base_url":        "https://api.meshy.ai",
    "timeout":         60,
    "poll_interval_s": 5,
    "poll_timeout_s":  600,
}

# ── SimScale (cloud FEA/CFD) ──────────────────────────────────────────────────
SIMSCALE = {
    "api_key":         os.getenv("SIMSCALE_API_KEY"),
    "base_url":        "https://api.simscale.com/v0",
    "timeout":         60,
    "poll_interval_s": 10,
    "poll_timeout_s":  3600,
}

# Active solver: "simscale" | "omniverse" | "ansys" | "fenics"
SIMULATION_SOLVER = os.getenv("SIMULATION_SOLVER", "simscale")
