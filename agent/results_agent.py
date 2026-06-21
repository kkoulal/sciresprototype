"""
Results Interpretation Agent.

Turns a finished SimulationResult into a plain-language engineering interpretation
the researcher can act on (safety factor, resonance risk, recommendations). Uses the
fast LLM, grounded in the object, material, boundary conditions, and whatever numeric
results were extracted.
"""

import json
from typing import Optional

from agent.llm import call_llm
from agent.config import AGENT_LLM_MODEL_FAST
from agent.models import PhysicalObject, SimulationResult
from agent.prompts import RESULTS_INTERPRETATION_PROMPT, render_prompt


def interpret_results(
    physical_object: PhysicalObject,
    sim_result: SimulationResult,
    analysis_type: str = "static",
) -> str:
    """Return a Markdown interpretation of the simulation results."""
    # Collect any numeric results we have (often empty → viewer-based guidance).
    nums = {}
    if sim_result.max_von_mises_stress_pa:
        nums["max_von_mises_stress_pa"] = sim_result.max_von_mises_stress_pa
    if sim_result.max_displacement_m:
        nums["max_displacement_m"] = sim_result.max_displacement_m
    if sim_result.natural_frequencies_hz:
        nums["natural_frequencies_hz"] = sim_result.natural_frequencies_hz
    numeric_str = json.dumps(nums) if nums else "none extracted (read peak values from the viewer legend)"

    prompt = render_prompt(
        RESULTS_INTERPRETATION_PROMPT,
        analysis_type=analysis_type,
        name=physical_object.name,
        shape_type=physical_object.shape_type,
        shape_description=physical_object.shape_description,
        material=physical_object.material,
        material_properties=json.dumps(physical_object.material_properties or {}),
        boundary_conditions="; ".join(physical_object.boundary_conditions or []) or "fixed support",
        applied_loads=json.dumps(physical_object.applied_loads or []),
        numeric_results=numeric_str,
    )
    out = call_llm(
        [{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=900,
        model=AGENT_LLM_MODEL_FAST,
    )
    if not out or out.startswith("Error"):
        return (
            "Could not generate an interpretation. Open the SimScale viewer to inspect "
            "the result fields directly."
        )
    return out.strip()
