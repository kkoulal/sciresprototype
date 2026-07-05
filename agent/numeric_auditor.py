"""
Numerical Geometry Auditor — exact measured properties vs. the spec's targets.

The vision critic judges *looks*; this auditor judges *numbers*. For a scientific
platform the numbers win: a geometry that looks right but is 30% too small is FAIL.

    report = audit_geometry(mesh_or_path, spec)
    report.passed / report.failures / report.measurements
"""

import os
from typing import List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from agent.spec import ScientificObjectSpec


class GeometryMeasurements(BaseModel):
    bounding_box_mm: List[float] = Field(default_factory=list)
    volume_mm3: float = 0.0
    surface_area_mm2: float = 0.0
    watertight: bool = False
    body_count: int = 0
    face_count: int = 0
    center_of_mass_mm: List[float] = Field(default_factory=list)


class NumericAuditReport(BaseModel):
    passed: bool = False
    measurements: GeometryMeasurements = Field(default_factory=GeometryMeasurements)
    failures: List[str] = Field(default_factory=list)     # blocking
    warnings: List[str] = Field(default_factory=list)     # non-blocking
    correction_hints: List[str] = Field(default_factory=list)  # fed back to the generator


def _load_mesh(mesh_or_path):
    import trimesh
    if isinstance(mesh_or_path, trimesh.Trimesh):
        return mesh_or_path
    return trimesh.load(str(mesh_or_path), force="mesh")


def measure(mesh_or_path) -> GeometryMeasurements:
    """Measure a mesh (units assumed mm)."""
    m = _load_mesh(mesh_or_path)
    extents = (m.bounds[1] - m.bounds[0]).tolist()
    return GeometryMeasurements(
        bounding_box_mm=[round(float(x), 3) for x in extents],
        volume_mm3=round(abs(float(m.volume)), 3),
        surface_area_mm2=round(float(m.area), 3),
        watertight=bool(m.is_watertight),
        body_count=int(m.body_count),
        face_count=int(len(m.faces)),
        center_of_mass_mm=[round(float(x), 3) for x in m.center_mass.tolist()]
        if m.is_watertight else [],
    )


def audit_geometry(
    mesh_or_path,
    spec: ScientificObjectSpec,
    mesh_scale_to_mm: float = 1.0,
) -> NumericAuditReport:
    """
    Compare measured geometry against spec.validation targets.

    mesh_scale_to_mm: multiply mesh units by this to get mm (STEP preview STLs from
    CadQuery are already mm → 1.0; trimesh primitives in meters → 1000).
    """
    m = _load_mesh(mesh_or_path)
    if mesh_scale_to_mm != 1.0:
        m = m.copy()
        m.apply_scale(mesh_scale_to_mm)

    meas = measure(m)
    failures: List[str] = []
    warnings: List[str] = []
    hints: List[str] = []

    v = spec.validation
    tol = max(float(v.tolerance_percent), 1.0) / 100.0

    # topology gates
    if v.must_be_watertight and not meas.watertight:
        failures.append("Geometry is not watertight (open surface — cannot volume-mesh)")
        hints.append("Ensure all parts overlap and the SDF/solid is closed")
    if v.must_be_single_body and meas.body_count != 1:
        failures.append(f"Geometry has {meas.body_count} disconnected bodies (must be 1)")
        hints.append("Increase overlaps / smooth-union blending so all parts connect")

    # bounding box vs target — order-insensitive (renderer axes may be permuted)
    if v.bounding_box_mm and len(v.bounding_box_mm) == 3 and meas.bounding_box_mm:
        target = sorted(float(x) for x in v.bounding_box_mm)
        actual = sorted(meas.bounding_box_mm)
        for t, a, label in zip(target, actual, ("min", "mid", "max")):
            if t <= 0:
                continue
            rel = abs(a - t) / t
            abs_dev = abs(a - t)
            if rel > tol:
                msg = (f"Bounding-box {label} extent {a:.1f} mm vs target {t:.1f} mm "
                       f"({rel*100:.0f}% off, tol {v.tolerance_percent:.0f}%)")
                # blocking only when BOTH relative (2× tol) AND absolute (>1.5 mm)
                # deviations are meaningful — a 1 mm difference on a thin wall is
                # engineering noise, not a wrong object
                if rel > 2 * tol and abs_dev > 1.5:
                    failures.append(msg)
                    hints.append(f"Scale/resize so the {label} extent ≈ {t:.1f} mm")
                else:
                    warnings.append(msg)

    # volume vs target
    if v.volume_mm3 and v.volume_mm3 > 0:
        rel = abs(meas.volume_mm3 - v.volume_mm3) / v.volume_mm3
        if rel > 2 * tol:
            failures.append(f"Volume {meas.volume_mm3:.0f} mm³ vs target "
                            f"{v.volume_mm3:.0f} mm³ ({rel*100:.0f}% off)")
        elif rel > tol:
            warnings.append(f"Volume off by {rel*100:.0f}%")

    # degenerate scale check
    if meas.bounding_box_mm and max(meas.bounding_box_mm) < 0.1:
        failures.append("Geometry is degenerate (max extent < 0.1 mm) — likely unit error")
        hints.append("Check meters→millimeters conversion (multiply by 1000)")

    return NumericAuditReport(
        passed=not failures,
        measurements=meas,
        failures=failures,
        warnings=warnings,
        correction_hints=hints,
    )
