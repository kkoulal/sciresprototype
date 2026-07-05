"""
Scientific Object Specification — the source of truth of the geometry pipeline.

Principle: the object is not just geometry. It is geometry + material + physics +
validation targets. The LLM never generates CAD directly from conversation; it goes
    Conversation → ScientificObjectSpec → (audit) → Geometry IR → geometry.

This module provides:
    build_spec(interview_messages, physical_object)  → ScientificObjectSpec
    audit_spec(spec)                                 → AuditReport (consistency check)
"""

import json
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field

from agent.llm import call_llm
from agent.config import AGENT_LLM_MODEL
from agent.models import PhysicalObject


# ── Spec models ────────────────────────────────────────────────────────────────

class ComponentSpec(BaseModel):
    """One logical part of the object (a lens ring, a temple arm, a branch...)."""
    id: str
    description: str = ""
    # geometric role hints for the router / generators
    kind: str = "solid"          # solid | shell | tube | branch | cavity
    dimensions_mm: Dict[str, float] = Field(default_factory=dict)
    position_hint: str = ""      # natural-language placement, e.g. "left of center"


class GeometrySpec(BaseModel):
    shape_class: str = "parametric_cad"   # parametric_cad | procedural_organic
    description: str = ""
    overall_dimensions_mm: Dict[str, float] = Field(default_factory=dict)
    components: List[ComponentSpec] = Field(default_factory=list)
    symmetry: str = "none"                # none | mirror_x | mirror_y | radial


class ValidationTargets(BaseModel):
    """Numeric targets the produced geometry MUST satisfy (within tolerance)."""
    bounding_box_mm: Optional[List[float]] = None   # [x, y, z]
    volume_mm3: Optional[float] = None
    tolerance_percent: float = 10.0
    must_be_watertight: bool = True
    must_be_single_body: bool = True


class ScientificObjectSpec(BaseModel):
    name: str = "object"
    purpose: str = ""                     # what study this object is for
    geometry: GeometrySpec = Field(default_factory=GeometrySpec)
    material: Dict[str, Any] = Field(default_factory=dict)
    physics: Dict[str, Any] = Field(default_factory=dict)
    validation: ValidationTargets = Field(default_factory=ValidationTargets)


class AuditIssue(BaseModel):
    severity: str = "warning"             # error | warning
    message: str = ""


class AuditReport(BaseModel):
    status: str = "ok"                    # ok | warnings | errors
    issues: List[AuditIssue] = Field(default_factory=list)


# ── Spec builder ───────────────────────────────────────────────────────────────

_SPEC_PROMPT = """You are a simulation engineer. Distill the interview below into a
Scientific Object Specification JSON. Think about what the object IS geometrically:
decompose it into named components with sizes and positions.

Rules:
- All dimensions in MILLIMETERS (convert from meters where needed: m × 1000).
- shape_class: "procedural_organic" ONLY for biological/organic freeform shapes
  (artery, bone, tumor, coral, root...). Everything manufactured — even curved
  (glasses, handles, blades, brackets) — is "parametric_cad".
- components: 2-8 logical parts, each with its key dimensions_mm and a short
  position_hint (e.g. "centered", "left end", "between the two rings").
- validation.bounding_box_mm: the expected overall [x,y,z] extents in mm. Only be
  precise about extents the user actually stated. For tubular/organic objects the
  cross-section extents equal the largest outer diameter (a 100mm artery of 8mm
  diameter with branches spreading ~30mm has bbox ≈ [30, 10, 100], NOT [30, 20, 100]).
- validation.tolerance_percent: 8-10 for manufactured parts; 20-30 for organic shapes
  (their exact extents are estimates, not requirements).
- Fill unknowns with sensible engineering estimates; never leave dimensions empty.

Respond ONLY with a JSON object with keys:
  name, purpose,
  geometry: {shape_class, description, overall_dimensions_mm, components:[{id,description,kind,dimensions_mm,position_hint}], symmetry},
  material: {name, E_pa, nu, rho_kg_m3},
  physics:  {analysis_type, loads},
  validation: {bounding_box_mm:[x,y,z], tolerance_percent, must_be_watertight, must_be_single_body}

Interview + object draft:
"""


def build_spec(
    interview_messages: List[Dict[str, str]],
    physical_object: Optional[PhysicalObject] = None,
) -> ScientificObjectSpec:
    """Distill the interview (+ draft) into a validated-shape ScientificObjectSpec."""
    convo = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in (interview_messages or [])
        if m.get("role") in ("user", "assistant") and m.get("content") != "start"
    )
    if physical_object is not None:
        convo += "\n\nObject draft JSON:\n" + physical_object.model_dump_json()

    data = call_llm(
        [{"role": "user", "content": _SPEC_PROMPT + convo}],
        temperature=0.2,
        max_tokens=3500,           # a full spec JSON can be long — avoid truncation
        model=AGENT_LLM_MODEL,     # strong model — this is the source of truth
        extract_json=True,
    )
    if not isinstance(data, dict) or not data:
        raise RuntimeError("Spec builder returned no parseable JSON")
    return ScientificObjectSpec(**data)


# ── Requirements auditor ───────────────────────────────────────────────────────

def audit_spec(spec: ScientificObjectSpec) -> AuditReport:
    """
    Deterministic consistency checks + one LLM sanity pass.
    Errors block generation; warnings are shown but don't block.
    """
    issues: List[AuditIssue] = []

    # deterministic checks
    bbox = spec.validation.bounding_box_mm
    if not bbox or len(bbox) != 3 or any((not isinstance(v, (int, float)) or v <= 0) for v in bbox):
        issues.append(AuditIssue(severity="error",
                                 message="validation.bounding_box_mm missing or non-positive"))
    else:
        if max(bbox) > 50_000:
            issues.append(AuditIssue(severity="warning",
                                     message=f"Very large object ({max(bbox)/1000:.1f} m) — check units"))
        if min(bbox) < 0.5:
            issues.append(AuditIssue(severity="warning",
                                     message="Thinnest overall dimension < 0.5 mm — may not mesh"))
    if not spec.geometry.components:
        issues.append(AuditIssue(severity="warning",
                                 message="No components decomposed — geometry will be one blob"))
    for comp in spec.geometry.components:
        for k, v in comp.dimensions_mm.items():
            if not isinstance(v, (int, float)) or v <= 0:
                issues.append(AuditIssue(severity="error",
                                         message=f"Component '{comp.id}': dimension {k}={v} invalid"))

    status = "ok"
    if any(i.severity == "warning" for i in issues):
        status = "warnings"
    if any(i.severity == "error" for i in issues):
        status = "errors"
    return AuditReport(status=status, issues=issues)
