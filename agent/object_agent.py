"""
Shape Interview Agent.

A standalone (non-LangGraph) multi-turn LLM agent that collects the physical
object description through conversation and synthesizes it into a PhysicalObject.

Typical call from Streamlit:
    response, draft, confirmed = step(messages, sim_context)
    - response: str  → display to user
    - draft: PhysicalObject | None  → live preview card
    - confirmed: bool → if True, interview is done
"""

import json
import re
from typing import List, Dict, Tuple, Optional

from agent.llm import call_llm
from agent.config import AGENT_LLM_MODEL, AGENT_LLM_MODEL_FAST
from agent.models import PhysicalObject, SimulationPackage
from agent.prompts import SHAPE_INTERVIEW_SYSTEM_PROMPT, MESHY_PROMPT_BUILDER_PROMPT, render_prompt


_CONFIRM_MARKERS = {"yes", "correct", "confirmed", "confirm", "looks good", "good", "ok", "okay", "approve", "approved", "perfect", "great", "done", "proceed"}


def _extract_object_draft(text: str) -> Optional[PhysicalObject]:
    """
    Parse the object draft JSON block from the agent response.
    Accepts: ```object_draft, ```json, or plain ``` fences.
    Falls back to scanning for a bare { } block containing "name" + "shape_type".
    """
    # Priority 1: explicit object_draft fence
    patterns = [
        r"```object_draft\s*(.*?)\s*```",  # preferred
        r"```json\s*(.*?)\s*```",           # Gemini often uses ```json
        r"```\s*(\{.*?\})\s*```",           # plain ``` fence with object
    ]
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and ("name" in data or "shape_type" in data):
                    return PhysicalObject(**{k: v for k, v in data.items() if v not in (None, "", [], {})})
            except Exception:
                continue

    # Priority 2: bare JSON object anywhere in the text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict) and ("name" in data or "shape_type" in data):
                return PhysicalObject(**{k: v for k, v in data.items() if v not in (None, "", [], {})})
        except Exception:
            pass

    return None


def _is_confirmed(user_message: str) -> bool:
    words = set(re.sub(r"[^a-z\s]", "", user_message.lower()).split())
    return bool(words & _CONFIRM_MARKERS)


def _find_prior_draft(messages: List[Dict[str, str]]) -> Optional[PhysicalObject]:
    """Return the most recent parseable object_draft from prior assistant messages."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            draft = _extract_object_draft(msg.get("content", ""))
            if draft:
                return draft
    return None


def synthesize_object(
    messages: List[Dict[str, str]],
    sim_package: Optional[SimulationPackage] = None,
) -> Optional[PhysicalObject]:
    """
    Guaranteed fallback: ask the LLM to distill the whole interview into a
    PhysicalObject JSON, filling sensible engineering defaults for anything not
    discussed. Ensures confirmation always yields a usable object even if the
    interview never emitted a clean ```object_draft``` block.
    """
    convo = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content") != "start"
    )
    prompt = (
        "From the shape interview below, output ONLY a JSON object describing the "
        "physical object with these keys:\n"
        '  "name", "shape_type" (one of box|cylinder|sphere|custom), '
        '"shape_description", "dimensions" (object in METERS, keys like length_m, '
        'width_m, height_m, radius_m, thickness_m), "material", "material_properties" '
        '(E in Pa, nu, rho in kg/m^3), "boundary_conditions" (list of strings), '
        '"applied_loads" (list of {type, value, unit}), "environment" (object), '
        '"visual_style".\n'
        "Use reasonable engineering defaults for anything not stated. "
        "Respond with ONLY the JSON object.\n\nInterview:\n" + convo
    )
    data = call_llm(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
        model=AGENT_LLM_MODEL_FAST,
        extract_json=True,
    )
    if not isinstance(data, dict) or not data:
        return None
    try:
        return PhysicalObject(**{k: v for k, v in data.items() if v not in (None, "", [], {})})
    except Exception:
        return None


def build_system_prompt(sim_package: Optional[SimulationPackage]) -> str:
    """Inject a short sim-package summary into the system prompt context."""
    if not sim_package:
        ctx = "No simulation package provided."
    else:
        params_preview = json.dumps(dict(list(sim_package.parameters.items())[:6]), indent=2)
        ctx = f"Simulation parameters (excerpt):\n{params_preview}\n\nModel brief:\n{sim_package.model_brief[:400]}"
    return render_prompt(SHAPE_INTERVIEW_SYSTEM_PROMPT, sim_context=ctx)


def step(
    messages: List[Dict[str, str]],
    sim_package: Optional[SimulationPackage] = None,
) -> Tuple[str, Optional[PhysicalObject], bool]:
    """
    Run one turn of the shape interview.

    Args:
        messages: Full conversation so far (role/content dicts, excluding system).
                  The last entry must be the latest user message.
        sim_package: Optional context from Phase 1.

    Returns:
        (agent_response, current_draft, is_confirmed)
    """
    system_msg = {"role": "system", "content": build_system_prompt(sim_package)}
    full_messages = [system_msg] + messages

    _CONFIRM_MSG = "Great! Your object is confirmed. Click **Generate CAD Solid** in Step 2 to proceed."

    # Check if the user just confirmed the draft shown in the previous turn
    last_user = messages[-1]["content"] if messages else ""
    if _is_confirmed(last_user):
        draft = _find_prior_draft(messages) or synthesize_object(messages, sim_package)
        if draft:
            return _CONFIRM_MSG, draft, True

    response = call_llm(
        full_messages,
        temperature=0.4,
        max_tokens=1024,
        model=AGENT_LLM_MODEL_FAST,
    )

    # If the LLM emitted the OBJECT_CONFIRMED marker, treat it as confirmation.
    # Always resolve a draft (prior block, current response, or synthesized) so the
    # downstream UI (Step 2) always has a PhysicalObject to work with.
    if "OBJECT_CONFIRMED" in response:
        draft = (
            _find_prior_draft(messages)
            or _extract_object_draft(response)
            or synthesize_object(messages, sim_package)
        )
        if draft:
            return _CONFIRM_MSG, draft, True

    draft = _extract_object_draft(response)
    # Strip raw JSON/marker blocks — show only the prose to the user
    display_response = re.sub(r"```(?:object_draft|json).*?```", "", response, flags=re.DOTALL)
    display_response = re.sub(r"OBJECT_CONFIRMED", "", display_response).strip()

    return display_response, draft, False


def build_meshy_prompt(obj: PhysicalObject) -> str:
    """
    Call Haiku to turn a PhysicalObject into a rich Meshy AI text prompt.
    Falls back to a simple template if LLM fails.
    """
    dims_str = ", ".join(f"{k}: {v}" for k, v in obj.dimensions.items()) or "unspecified"
    messages = [
        {
            "role": "user",
            "content": render_prompt(
                MESHY_PROMPT_BUILDER_PROMPT,
                name=obj.name,
                shape_type=obj.shape_type,
                shape_description=obj.shape_description,
                dimensions=dims_str,
                material=obj.material,
                visual_style=obj.visual_style,
            ),
        }
    ]
    result = call_llm(
        messages,
        temperature=0.6,
        max_tokens=256,
        model=AGENT_LLM_MODEL_FAST,   # Haiku — simple text synthesis
    )
    if not result or result.startswith("Error"):
        # Safe fallback
        return (
            f"A {obj.visual_style} 3D model of a {obj.name}. "
            f"{obj.shape_description}. "
            f"Material: {obj.material}. "
            f"Dimensions: {dims_str}. High detail, engineering accuracy."
        )
    return result.strip()
