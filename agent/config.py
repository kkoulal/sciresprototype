import os
import sys

# Add parent to path so we can import from online.config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from online.config import LLM, QDRANT, MESHY, SIMSCALE, SIMULATION_SOLVER, GEMINI

AGENT_LLM_URL       = LLM["url"]
AGENT_LLM_MODEL     = LLM["model"]
AGENT_LLM_MODEL_FAST = LLM["model_fast"]
AGENT_LLM_KEY       = LLM["api_key"]
AGENT_LLM_TIMEOUT   = LLM["timeout"]

MESHY_API_KEY        = MESHY["api_key"]
MESHY_BASE_URL       = MESHY["base_url"]
MESHY_TIMEOUT        = MESHY["timeout"]
MESHY_POLL_INTERVAL  = MESHY["poll_interval_s"]
MESHY_POLL_TIMEOUT   = MESHY["poll_timeout_s"]

SIMSCALE_API_KEY     = SIMSCALE["api_key"]
SIMSCALE_BASE_URL    = SIMSCALE["base_url"]
SIMSCALE_TIMEOUT     = SIMSCALE["timeout"]
SIMSCALE_POLL_INTERVAL = SIMSCALE["poll_interval_s"]
SIMSCALE_POLL_TIMEOUT  = SIMSCALE["poll_timeout_s"]

ACTIVE_SOLVER = SIMULATION_SOLVER

GEMINI_API_KEY  = GEMINI["api_key"]
GEMINI_BASE_URL = GEMINI["base_url"]

AGENT_MAX_RETRIEVAL_ROUNDS = 3
AGENT_MIN_RETRIEVAL_SCORE = 0.5

# Simulation environments supported
SUPPORTED_SIM_ENVS = ["agent_based", "system_dynamics", "microsimulation", "discrete_event", "generic"]
