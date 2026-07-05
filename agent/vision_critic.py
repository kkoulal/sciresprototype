"""
Visual Critic — the generator can finally SEE what it built.

Renders the candidate geometry from several angles (VTK offscreen — no window
needed), sends the renders + the object description to a vision model, and gets a
structured verdict: similarity score, missing/incorrect parts, and specific
corrections. The router feeds those corrections back into regeneration.

    verdict = critique(stl_path, spec)          # → VisionVerdict
"""

import base64
import json
import os
import tempfile
from typing import List

from pydantic import BaseModel, Field

from agent.llm import call_llm
from agent.config import AGENT_LLM_MODEL
from agent.spec import ScientificObjectSpec


class VisionVerdict(BaseModel):
    score: float = 0.0                 # 0–10 similarity to the described object
    passed: bool = False               # score >= threshold
    issues: List[str] = Field(default_factory=list)
    corrections: List[str] = Field(default_factory=list)   # actionable fixes
    summary: str = ""


# ── Offscreen multi-view renderer ──────────────────────────────────────────────

def render_views(stl_path: str, out_dir: str = None, size: int = 448) -> List[str]:
    """Render 4 views (front / side / top / isometric) of an STL. Returns PNG paths."""
    import vtk

    out_dir = out_dir or tempfile.mkdtemp(prefix="scires_render_")
    os.makedirs(out_dir, exist_ok=True)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.75, 0.78, 0.82)
    actor.GetProperty().SetSpecular(0.3)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.AddActor(actor)
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(size, size)
    rw.AddRenderer(renderer)

    # camera orientations: (azimuth°, elevation°, label)
    views = [(0, 0, "front"), (90, 0, "side"), (0, 89, "top"), (45, 30, "isometric")]
    paths: List[str] = []
    for az, el, label in views:
        cam = vtk.vtkCamera()
        renderer.SetActiveCamera(cam)
        renderer.ResetCamera()
        cam.Azimuth(az)
        cam.Elevation(el)
        cam.OrthogonalizeViewUp()
        renderer.ResetCameraClippingRange()
        rw.Render()

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(rw)
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        p = os.path.join(out_dir, f"view_{label}.png")
        writer.SetFileName(p)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        paths.append(p)
    return paths


# ── Vision critique ────────────────────────────────────────────────────────────

_CRITIC_PROMPT = """You are a CAD reviewer. The renders below show a generated 3D solid
(front, side, top, isometric views on white background). Judge how well it matches the
target object described here:

TARGET OBJECT: {name}
DESCRIPTION: {description}
EXPECTED COMPONENTS: {components}

Score similarity 0-10 (10 = clearly the described object with right proportions;
5 = recognizable but wrong proportions/missing parts; 0 = unrelated shape).
Ignore surface finish, faceting, and color — judge SHAPE and STRUCTURE only.

Respond ONLY with compact JSON (max 3 issues, max 3 corrections, keep each under
25 words — prioritize the most important geometric problems):
{{"score": <0-10>, "issues": ["..."], "corrections": ["specific geometric fix"], "summary": "one sentence"}}
"""


def critique(
    stl_path: str,
    spec: ScientificObjectSpec,
    pass_threshold: float = 6.5,
) -> VisionVerdict:
    """Render + ask the vision model to judge the geometry against the spec."""
    png_paths = render_views(stl_path)

    content = [{
        "type": "text",
        "text": _CRITIC_PROMPT.format(
            name=spec.name,
            description=spec.geometry.description,
            components=", ".join(
                f"{c.id} ({c.position_hint})" for c in spec.geometry.components
            ) or "not specified",
        ),
    }]
    for p in png_paths:
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    data = call_llm(
        [{"role": "user", "content": content}],
        temperature=0.1,
        max_tokens=1200,             # verdict JSON must never truncate
        model=AGENT_LLM_MODEL,       # vision-capable strong model
        extract_json=True,
    )
    if not isinstance(data, dict) or "score" not in data:
        # vision unavailable → neutral pass so the numeric auditor still governs
        return VisionVerdict(score=5.0, passed=True,
                             summary="Vision critique unavailable — numeric audit only")

    score = float(data.get("score", 0))
    return VisionVerdict(
        score=score,
        passed=score >= pass_threshold,
        issues=[str(x) for x in data.get("issues", [])],
        corrections=[str(x) for x in data.get("corrections", [])],
        summary=str(data.get("summary", "")),
    )
