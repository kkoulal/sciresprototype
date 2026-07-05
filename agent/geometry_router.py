"""
Geometry Router + dual-validation orchestrator.

The heart of the Scientific Physical Model Compiler geometry stage:

    ScientificObjectSpec
          │
      route by shape_class
          │
   ┌──────┴────────┐
   │               │
 parametric_cad  procedural_organic
 (LLM→CadQuery,  (LLM→Geometry IR JSON,
  B-rep STEP)     deterministic SDF→mesh)
   │               │
   └──────┬────────┘
          ▼
   NUMERIC AUDITOR  (dimensions/volume/watertight vs spec — dominant)
   VISION CRITIC    (multi-view render → similarity + corrections)
          │
     fail → feed corrections back, regenerate (max N iterations)
          │
          ▼
   GeometryResult (sim-ready file + preview + full validation report)
"""

import json
import os
from typing import List, Optional

from pydantic import BaseModel, Field

from agent.llm import call_llm
from agent.config import AGENT_LLM_MODEL
from agent.models import PhysicalObject
from agent.spec import ScientificObjectSpec
from agent.numeric_auditor import audit_geometry, NumericAuditReport
from agent.vision_critic import critique, VisionVerdict


class GeometryResult(BaseModel):
    route: str = ""                       # parametric_cad | procedural_organic
    passed: bool = False
    sim_path: str = ""                    # what the solver imports (STEP mm, or STL m)
    preview_stl: str = ""                 # mm STL for the in-app 3D viewer
    source: str = ""                      # CadQuery code or Geometry IR JSON
    numeric: Optional[NumericAuditReport] = None
    vision: Optional[VisionVerdict] = None
    iterations: int = 0
    history: List[str] = Field(default_factory=list)   # human-readable audit trail


# ── Organic route: LLM → Geometry IR ──────────────────────────────────────────

_IR_PROMPT = """You are a computational geometry engineer. Build a Geometry IR (JSON)
for the organic object below. The IR is evaluated as smooth implicit surfaces (SDF)
— ideal for biological/organic forms.

IR format (all coordinates/sizes in MILLIMETERS):
{
  "units": "mm",
  "primitives": [
    {"id":"...","type":"capsule_chain","points":[[x,y,z],...],"radii":[r_per_point,...]},
    {"id":"...","type":"sphere","center":[x,y,z],"radius":r},
    {"id":"...","type":"ellipsoid","center":[x,y,z],"radii":[rx,ry,rz]},
    {"id":"...","type":"torus","center":[x,y,z],"axis":[x,y,z],"major_radius":R,"minor_radius":r},
    {"id":"...","type":"box","center":[x,y,z],"size":[sx,sy,sz]}
  ],
  "operations": [
    {"type":"smooth_union","targets":["id2"],"k":blend_radius_mm},
    {"type":"union","targets":["id3"]},
    {"type":"subtract","targets":["cavity_id"]},
    {"type":"shell","thickness":t}
  ]
}
Rules:
- capsule_chain is your main tool: a smooth tube along a polyline with per-point radii
  (arteries, branches, bones, roots). Vary radii for tapering/bulges/narrowings.
  A NARROWING (stenosis) = SMALLER radii mid-chain in the SAME chain. NEVER add a
  separate primitive for a narrowing — a unioned primitive creates a BULGE (the
  opposite). Diameter differences between branches: use clearly different radii.
- Model the SOLID exterior only. Do NOT attempt hollow walls, shells, or inner lumens
  unless the spec explicitly requires a shell.
- Keep it SIMPLE: maximum 6 primitives total. One chain per structure (one trunk
  chain with varying radii — not separate proximal/stenosis/distal chains).
- Operations apply in order to the running result (first primitive starts it).
- smooth_union k = 2-5 mm gives natural organic blending at junctions.
- Everything must CONNECT into ONE body (branch start points must touch the parent).
- Match the spec's overall bounding box.
- Respond ONLY with the JSON.

Specification:
<<SPEC>>
<<FEEDBACK>>
"""


def _build_organic_ir(
    spec: ScientificObjectSpec,
    feedback: List[str],
    previous_ir: Optional[dict] = None,
) -> dict:
    fb = ""
    if feedback and previous_ir is not None:
        # EDIT mode: give the model its own previous IR so it fixes only what's wrong
        fb = ("\nYour previous IR is below. It was rejected. Apply ONLY these "
              "corrections and keep everything else the same:\n- "
              + "\n- ".join(feedback)
              + "\n\nPrevious IR:\n" + json.dumps(previous_ir, indent=1))
    elif feedback:
        fb = ("\nPrevious attempt was rejected. Corrections to apply:\n- "
              + "\n- ".join(feedback))
    prompt = (_IR_PROMPT
              .replace("<<SPEC>>", spec.model_dump_json(indent=1))
              .replace("<<FEEDBACK>>", fb))
    data = call_llm(
        [{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4000,             # IR JSON must never truncate
        model=AGENT_LLM_MODEL,
        extract_json=True,
    )
    if not isinstance(data, dict) or not data.get("primitives"):
        raise RuntimeError("Organic IR generation returned no primitives")
    return data


def _run_organic(spec, feedback, dest_dir, previous_ir=None) -> tuple:
    """Returns (sim_path, preview_stl, source_json, ir_dict)."""
    from agent.organic_geometry import evaluate_ir
    ir = _build_organic_ir(spec, feedback, previous_ir=previous_ir)
    mesh = evaluate_ir(ir, resolution=128)

    os.makedirs(dest_dir, exist_ok=True)
    safe = "".join(c for c in spec.name if c.isalnum() or c in "-_")[:40] or "organic"
    preview = os.path.join(dest_dir, f"{safe}_preview.stl")
    mesh.export(preview)                       # mm — for viewer + audits

    sim = os.path.join(dest_dir, f"{safe}_sim.stl")
    mesh_m = mesh.copy()
    mesh_m.apply_scale(0.001)                  # mm → m (SimScale STL import uses meters)
    mesh_m.export(sim)
    return sim, preview, json.dumps(ir, indent=1), ir


# ── Parametric route: spec-driven CadQuery (with feedback) ────────────────────

def _run_parametric(spec, physical_object, feedback, dest_dir,
                    previous_code: Optional[str] = None) -> tuple:
    """Returns (sim_path=STEP, preview_stl, source_code)."""
    from agent.prompts import CADQUERY_CODE_PROMPT, render_prompt
    from agent.geometry_agent import generate_step, _strip_code_fences

    dims = ", ".join(f"{k}: {v}" for k, v in spec.geometry.overall_dimensions_mm.items())
    comps = "; ".join(
        f"{c.id}: {c.description} [{c.position_hint}] "
        + ", ".join(f"{k}={v}mm" for k, v in c.dimensions_mm.items())
        for c in spec.geometry.components
    )
    prompt = render_prompt(
        CADQUERY_CODE_PROMPT,
        name=spec.name,
        shape_type=spec.geometry.shape_class,
        shape_description=spec.geometry.description + " | Components: " + comps,
        dimensions=dims + " (ALREADY in millimeters — do NOT multiply again)",
        features=f"symmetry: {spec.geometry.symmetry}",
    )
    prompt += ("\n\nCRITICAL CONNECTIVITY RULE: after all unions, `result` must be ONE "
               "connected body. Every part must OVERLAP its neighbour by at least 1-2 mm "
               "before .union() — compute each translate so solids intersect, never just touch.")
    if feedback and previous_code:
        # EDIT mode: fix only what the audits flagged, keep the rest
        prompt += ("\n\nYour previous code is below. It was rejected. Apply ONLY these "
                   "corrections and keep everything else the same:\n- "
                   + "\n- ".join(feedback)
                   + "\n\nPrevious code:\n" + previous_code)
    elif feedback:
        prompt += ("\n\nPrevious attempt was rejected. Corrections to apply:\n- "
                   + "\n- ".join(feedback))

    code = _strip_code_fences(call_llm(
        [{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=3000, model=AGENT_LLM_MODEL,
    ) or "")
    step, stl, used_code, _fb = generate_step(physical_object, dest_dir=dest_dir, code=code)
    return step, stl, used_code


# ── Orchestrator ───────────────────────────────────────────────────────────────

def generate_validated_geometry(
    spec: ScientificObjectSpec,
    physical_object: PhysicalObject,
    dest_dir: str = "sim_geometry",
    max_iterations: int = 3,
    progress_cb=None,
) -> GeometryResult:
    """
    Generate → numeric audit → vision critique → correct → repeat.
    Returns the best candidate (numeric pass dominates, then vision score).
    """
    def _log(msg):
        if progress_cb:
            progress_cb(msg)

    route = spec.geometry.shape_class
    if route not in ("parametric_cad", "procedural_organic"):
        route = "parametric_cad"

    best: Optional[GeometryResult] = None
    feedback: List[str] = []
    history: List[str] = []
    prev_ir: Optional[dict] = None
    prev_code: Optional[str] = None

    for it in range(1, max_iterations + 1):
        _log(f"Iteration {it}/{max_iterations}: generating ({route})...")
        try:
            if route == "procedural_organic":
                sim_path, preview, source, prev_ir = _run_organic(
                    spec, feedback, dest_dir, previous_ir=prev_ir)
            else:
                sim_path, preview, source = _run_parametric(
                    spec, physical_object, feedback, dest_dir, previous_code=prev_code)
                prev_code = source
        except Exception as e:
            history.append(f"iter {it}: generation failed — {e}")
            feedback = [f"The previous attempt crashed with: {str(e)[:300]}. Simplify."]
            continue

        _log(f"Iteration {it}: numeric audit...")
        numeric = audit_geometry(preview, spec)

        _log(f"Iteration {it}: vision critique...")
        # SDF/marching-cubes renders can't score like glossy CAD — slightly lower gate.
        threshold = 6.0 if route == "procedural_organic" else 6.5
        try:
            vision = critique(preview, spec, pass_threshold=threshold)
        except Exception as e:
            vision = VisionVerdict(score=5.0, passed=True,
                                   summary=f"vision unavailable: {str(e)[:120]}")

        cand = GeometryResult(
            route=route,
            passed=numeric.passed and vision.passed,
            sim_path=sim_path, preview_stl=preview, source=source,
            numeric=numeric, vision=vision, iterations=it, history=history,
        )
        history.append(
            f"iter {it}: numeric={'PASS' if numeric.passed else 'FAIL'} "
            f"vision={vision.score:.1f}/10 — {vision.summary[:100]}"
        )

        # keep the best candidate: numeric pass first, then vision score
        if best is None or (
            (cand.numeric.passed, cand.vision.score)
            > (best.numeric.passed, best.vision.score)
        ):
            best = cand

        if cand.passed:
            _log(f"Iteration {it}: PASSED both audits.")
            break

        # build correction feedback for the next round (numeric first — it dominates)
        feedback = list(numeric.failures) + list(numeric.correction_hints)
        feedback += vision.corrections[:4]
        _log(f"Iteration {it}: rejected — regenerating with {len(feedback)} corrections.")

    if best is not None:
        best.history = history
    return best or GeometryResult(route=route, history=history)
