"""
AI CAD Geometry Agent.

Turns a PhysicalObject (captured by the shape interview) into a REAL, watertight
CAD solid using an LLM that writes CadQuery code. The solid is exported as STEP
(for FEA simulation) and STL (for an in-app preview).

Why this exists: text-to-3D tools like Meshy produce decorative surface shells
(non-watertight, thousands of disconnected pieces) that cannot be FEA-meshed.
CadQuery builds true B-rep solids on the OpenCASCADE kernel — the same geometry
engine professional CAD tools use — so the generated STEP imports and meshes
cleanly in SimScale.

Flow:
    code = build_cad_code(physical_object)          # Gemini writes CadQuery code
    step_path, stl_path, code = generate_step(obj)  # execute → STEP + STL preview

If the generated code fails to execute, callers should fall back to the
parametric primitive in services.simulation.physical_object_to_stl().
"""

import os
import re
import math
from typing import Optional, Tuple

from agent.llm import call_llm
from agent.config import AGENT_LLM_MODEL, AGENT_LLM_MODEL_FAST
from agent.models import PhysicalObject
from agent.prompts import CADQUERY_CODE_PROMPT, CADQUERY_FIX_PROMPT, render_prompt

# CAD code generation needs strong spatial/code reasoning — the fast model places
# parts in the wrong spots and produces shapes unrelated to the description. Use the
# strong model (Sonnet) here; it's a one-shot code task where quality matters most.
CAD_MODEL = AGENT_LLM_MODEL


def build_cad_code(physical_object: PhysicalObject) -> str:
    """Ask the LLM to write CadQuery code for the object. Returns raw code (fences stripped)."""
    dims = physical_object.dimensions or {}
    dims_str = ", ".join(f"{k}: {v}" for k, v in dims.items()) or "unspecified"

    features = []
    if physical_object.boundary_conditions:
        features.append("BCs: " + "; ".join(physical_object.boundary_conditions))
    for load in (physical_object.applied_loads or []):
        features.append(f"load: {load}")
    features_str = " | ".join(features) or "none specified"

    prompt = render_prompt(
        CADQUERY_CODE_PROMPT,
        name=physical_object.name,
        shape_type=physical_object.shape_type,
        shape_description=physical_object.shape_description,
        dimensions=dims_str,
        features=features_str,
    )

    raw = call_llm(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1600,
        model=CAD_MODEL,
    )
    return _strip_code_fences(raw or "")


def fix_cad_code(code: str, error: str) -> str:
    """Ask the LLM to repair CadQuery code that failed to execute."""
    prompt = render_prompt(CADQUERY_FIX_PROMPT, error=str(error)[:600], code=code)
    raw = call_llm(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1600,
        model=CAD_MODEL,
    )
    return _strip_code_fences(raw or "")


def _strip_code_fences(text: str) -> str:
    """Remove ```python / ``` fences and any stray prose around the code block."""
    fence = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _execute_cad_code(code: str):
    """
    Execute LLM-generated CadQuery code in a restricted namespace and return the
    resulting object. Raises on any failure (caller decides on fallback).

    Security note: this executes generated code locally. The prompt forbids imports
    and file/IO calls, and we run it with a minimal builtins set. This is acceptable
    for a local, single-user research tool driven by the user's own API key.
    """
    import cadquery as cq

    # Allow the LLM's natural `import cadquery as cq` / `import math` lines, but
    # whitelist only safe modules.
    _allowed = {"cadquery", "math", "numpy"}

    def _safe_import(name, *args, **kwargs):
        if name.split(".")[0] not in _allowed:
            raise ImportError(f"import of '{name}' is not allowed")
        return __import__(name, *args, **kwargs)

    safe_builtins = {
        "range": range, "len": len, "min": min, "max": max, "abs": abs,
        "round": round, "float": float, "int": int, "enumerate": enumerate,
        "zip": zip, "list": list, "tuple": tuple, "sum": sum, "pow": pow,
        "True": True, "False": False, "None": None, "__import__": _safe_import,
    }
    namespace = {"__builtins__": safe_builtins, "cq": cq, "math": math}

    exec(code, namespace)  # noqa: S102 — intentional, sandboxed namespace

    if "result" not in namespace:
        raise ValueError("Generated code did not define a `result` variable")
    return namespace["result"]


def _to_shape(result):
    """Normalize a CadQuery result (Workplane or Shape) and validate it's a real solid."""
    import cadquery as cq

    shape = result
    if isinstance(result, cq.Workplane):
        shape = result.val()

    # Validate it is a solid with positive volume
    try:
        vol = shape.Volume()
    except Exception as e:
        raise ValueError(f"Generated geometry is not a valid solid: {e}")
    if vol <= 0:
        raise ValueError("Generated geometry has non-positive volume")
    return result, vol


def _export(result, physical_object, dest_dir):
    """Export a CadQuery result to STEP + STL; return (step_path, stl_path)."""
    import cadquery as cq
    os.makedirs(dest_dir, exist_ok=True)
    safe_name = "".join(
        c for c in (physical_object.name or "object") if c.isalnum() or c in "-_"
    )[:40] or "object"
    step_path = os.path.join(dest_dir, f"{safe_name}.step")
    stl_path = os.path.join(dest_dir, f"{safe_name}_preview.stl")
    cq.exporters.export(result, step_path)
    cq.exporters.export(result, stl_path)
    return step_path, stl_path


def _fallback_primitive_code(physical_object: PhysicalObject) -> str:
    """
    Guaranteed-valid CadQuery code for a simple primitive built from the dimensions
    (millimeters). Used when LLM generation can't produce a valid solid.
    """
    dims = physical_object.dimensions or {}

    def _mm(*keys, default):
        for k in keys:
            for cand in (k, k + "_m", k.replace("_m", "")):
                if cand in dims and dims[cand]:
                    return float(dims[cand]) * 1000.0
        return default

    shape = (physical_object.shape_type or "box").lower()
    if shape in ("cylinder", "rod", "shaft", "pipe", "tube"):
        r = _mm("radius", default=_mm("diameter", default=100.0) / 2.0)
        h = _mm("height", "length", "span", default=500.0)
        return f"result = cq.Workplane('XY').cylinder({h}, {r})"
    if shape in ("sphere", "ball"):
        r = _mm("radius", default=_mm("diameter", default=200.0) / 2.0)
        return f"result = cq.Workplane('XY').sphere({r})"
    L = _mm("length", "span", default=200.0)
    W = _mm("width", "depth", "chord", default=100.0)
    H = _mm("height", "thickness", "wall_thickness", default=50.0)
    return f"result = cq.Workplane('XY').box({L}, {W}, {H})"


def generate_step(
    physical_object: PhysicalObject,
    dest_dir: str = "sim_geometry",
    code: Optional[str] = None,
    max_attempts: int = 3,
) -> Tuple[str, str, str, bool]:
    """
    Generate a CAD solid for the object and export STEP (for FEA) + STL (for preview).

    Self-repair loop: if the generated CadQuery code fails to execute (e.g. a fillet
    radius too large, a bad selector), the error is fed back to the LLM to fix its own
    code, up to `max_attempts` times.

    Fallback: in auto mode (code=None), if all attempts fail, a guaranteed-valid
    parametric primitive (box/cylinder/sphere from the dimensions) is built instead,
    so the pipeline always yields a usable solid. If `code` is supplied explicitly
    (user edited it), failures raise so the user sees the error.

    Returns:
        (step_path, stl_path, code, used_fallback)
    """
    explicit = code is not None
    if code is None:
        code = build_cad_code(physical_object)

    last_error = None
    for attempt in range(max_attempts):
        try:
            result, _vol = _to_shape(_execute_cad_code(code))
            step_path, stl_path = _export(result, physical_object, dest_dir)
            return step_path, stl_path, code, False
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                code = fix_cad_code(code, str(e))

    if explicit:
        raise RuntimeError(f"CAD code failed after {max_attempts} attempts: {last_error}")

    # Auto mode → guaranteed primitive fallback
    fallback_code = _fallback_primitive_code(physical_object)
    result, _vol = _to_shape(_execute_cad_code(fallback_code))
    step_path, stl_path = _export(result, physical_object, dest_dir)
    return step_path, stl_path, fallback_code, True
