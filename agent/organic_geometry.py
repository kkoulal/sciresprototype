"""
Procedural organic geometry engine (Geometry IR → implicit SDF → watertight mesh).

The organic route of the Geometry Router. The LLM does NOT write code here — it
emits a structured Geometry IR (JSON): primitives (capsule chains along centerlines,
spheres, ellipsoids, tori) combined with smooth booleans. This module evaluates the
IR deterministically on a signed-distance grid and extracts a guaranteed-watertight
surface with marching cubes.

Use for: arteries, bronchial trees, bones, tumors, roots, corals, tubular biology.

IR schema (what the LLM emits):
{
  "units": "mm",
  "primitives": [
    {"id":"main","type":"capsule_chain",
     "points":[[x,y,z],...], "radii":[r0,r1,...]},
    {"id":"bulge","type":"ellipsoid","center":[x,y,z],"radii":[rx,ry,rz]},
    {"id":"head","type":"sphere","center":[x,y,z],"radius":r},
    {"id":"ring","type":"torus","center":[x,y,z],"axis":[0,0,1],
     "major_radius":R,"minor_radius":r}
  ],
  "operations": [
    {"type":"smooth_union","targets":["main","bulge"],"k":3.0},
    {"type":"subtract","targets":["result","cavity"]},
    {"type":"shell","thickness":1.2}          # hollow wall (e.g. artery wall)
  ]
}
Operations are applied in order; "result" refers to the running combination.
"""

import numpy as np
from typing import Dict, Any, Tuple


# ── SDF primitives (vectorized over an (N,3) point array) ─────────────────────

def _sdf_sphere(p: np.ndarray, center, radius) -> np.ndarray:
    return np.linalg.norm(p - np.asarray(center, dtype=np.float32), axis=1) - float(radius)


def _sdf_ellipsoid(p: np.ndarray, center, radii) -> np.ndarray:
    # standard approximate ellipsoid SDF (good enough for iso-extraction)
    q = (p - np.asarray(center, dtype=np.float32)) / np.asarray(radii, dtype=np.float32)
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / np.asarray(radii, dtype=np.float32), axis=1)
    k1 = np.where(k1 < 1e-9, 1e-9, k1)
    return k0 * (k0 - 1.0) / k1


def _sdf_capsule_segment(p: np.ndarray, a, b, ra, rb) -> np.ndarray:
    """Rounded cone between a (radius ra) and b (radius rb) — linear radius interp."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab)) or 1e-9
    t = np.clip(np.einsum("ij,j->i", p - a, ab) / denom, 0.0, 1.0)
    closest = a + t[:, None] * ab
    r = ra + (rb - ra) * t
    return np.linalg.norm(p - closest, axis=1) - r


def _sdf_capsule_chain(p: np.ndarray, points, radii) -> np.ndarray:
    """Union of rounded-cone segments along a polyline with per-point radii."""
    pts = [np.asarray(q, dtype=np.float32) for q in points]
    rs = list(radii)
    if len(rs) == 1:
        rs = rs * len(pts)
    d = None
    for i in range(len(pts) - 1):
        seg = _sdf_capsule_segment(p, pts[i], pts[i + 1], float(rs[i]), float(rs[i + 1]))
        d = seg if d is None else np.minimum(d, seg)
    if d is None:  # single point → sphere
        d = _sdf_sphere(p, pts[0], float(rs[0]))
    return d


def _sdf_torus(p: np.ndarray, center, axis, major_radius, minor_radius) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    ax = np.asarray(axis, dtype=np.float32)
    ax = ax / (np.linalg.norm(ax) or 1.0)
    q = p - c
    h = np.einsum("ij,j->i", q, ax)               # height along axis
    radial = np.linalg.norm(q - h[:, None] * ax, axis=1)
    return np.sqrt((radial - float(major_radius)) ** 2 + h ** 2) - float(minor_radius)


def _sdf_box(p: np.ndarray, center, size) -> np.ndarray:
    q = np.abs(p - np.asarray(center, dtype=np.float32)) - np.asarray(size, dtype=np.float32) / 2.0
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


_PRIMITIVES = {
    "sphere":        lambda p, s: _sdf_sphere(p, s["center"], s["radius"]),
    "ellipsoid":     lambda p, s: _sdf_ellipsoid(p, s["center"], s["radii"]),
    "capsule_chain": lambda p, s: _sdf_capsule_chain(p, s["points"], s["radii"]),
    "torus":         lambda p, s: _sdf_torus(p, s["center"], s.get("axis", [0, 0, 1]),
                                             s["major_radius"], s["minor_radius"]),
    "box":           lambda p, s: _sdf_box(p, s["center"], s["size"]),
}


# ── Boolean operations ─────────────────────────────────────────────────────────

def _smooth_union(d1, d2, k: float):
    k = max(float(k), 1e-6)
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 + (d1 - d2) * h - k * h * (1.0 - h)


def _union(d1, d2):
    return np.minimum(d1, d2)


def _subtract(d1, d2):
    return np.maximum(d1, -d2)


def _intersect(d1, d2):
    return np.maximum(d1, d2)


# ── IR evaluation ──────────────────────────────────────────────────────────────

def _primitive_bounds(prim: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Conservative AABB of one primitive."""
    t = prim["type"]
    if t == "sphere":
        c = np.asarray(prim["center"], dtype=np.float32); r = float(prim["radius"])
        return c - r, c + r
    if t == "ellipsoid":
        c = np.asarray(prim["center"], dtype=np.float32)
        r = np.asarray(prim["radii"], dtype=np.float32)
        return c - r, c + r
    if t == "capsule_chain":
        pts = np.asarray(prim["points"], dtype=np.float32)
        rmax = float(max(prim["radii"]))
        return pts.min(axis=0) - rmax, pts.max(axis=0) + rmax
    if t == "torus":
        c = np.asarray(prim["center"], dtype=np.float32)
        r = float(prim["major_radius"]) + float(prim["minor_radius"])
        return c - r, c + r
    if t == "box":
        c = np.asarray(prim["center"], dtype=np.float32)
        s = np.asarray(prim["size"], dtype=np.float32) / 2.0
        return c - s, c + s
    raise ValueError(f"Unknown primitive type: {t}")


def evaluate_ir(ir: Dict[str, Any], resolution: int = 128):
    """
    Evaluate a Geometry IR to a watertight trimesh.Trimesh (units: mm).

    resolution: grid cells along the longest axis (128 ≈ good detail/speed balance).
    """
    import trimesh
    from skimage.measure import marching_cubes

    prims = ir.get("primitives", [])
    if not prims:
        raise ValueError("Geometry IR has no primitives")

    # 1. bounds with margin (surface must not touch the grid boundary)
    lo = np.full(3, np.inf, dtype=np.float32)
    hi = np.full(3, -np.inf, dtype=np.float32)
    for prim in prims:
        plo, phi = _primitive_bounds(prim)
        lo = np.minimum(lo, plo)
        hi = np.maximum(hi, phi)
    max_k = max([float(op.get("k", 0)) for op in ir.get("operations", [])] + [0.0])
    shell_t = max([float(op.get("thickness", 0)) for op in ir.get("operations", [])
                   if op.get("type") == "shell"] + [0.0])
    margin = float(max(2.0, max_k * 2.0, shell_t * 2.0))
    lo -= margin
    hi += margin

    # 2. grid (cap total cells to keep memory sane)
    extents = hi - lo
    spacing = float(extents.max()) / float(resolution)
    dims = np.maximum((extents / spacing).astype(int) + 1, 8)
    while int(np.prod(dims)) > 12_000_000:
        spacing *= 1.26
        dims = np.maximum((extents / spacing).astype(int) + 1, 8)

    xs = lo[0] + np.arange(dims[0], dtype=np.float32) * spacing
    ys = lo[1] + np.arange(dims[1], dtype=np.float32) * spacing
    zs = lo[2] + np.arange(dims[2], dtype=np.float32) * spacing
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # 3. evaluate primitives
    fields: Dict[str, np.ndarray] = {}
    for prim in prims:
        fn = _PRIMITIVES.get(prim["type"])
        if fn is None:
            raise ValueError(f"Unknown primitive type: {prim['type']}")
        fields[prim["id"]] = fn(pts, prim).astype(np.float32)

    # 4. apply operations in order ("result" = running combination)
    ids = [p["id"] for p in prims]
    result = fields[ids[0]]
    consumed = {ids[0]}
    for op in ir.get("operations", []):
        typ = op.get("type", "union")
        if typ == "shell":
            result = np.abs(result) - float(op.get("thickness", 1.0)) / 2.0
            continue
        targets = [t for t in op.get("targets", []) if t != "result"]
        for t in targets:
            if t not in fields:
                continue
            d2 = fields[t]
            consumed.add(t)
            if typ == "smooth_union":
                result = _smooth_union(result, d2, op.get("k", 2.0))
            elif typ == "union":
                result = _union(result, d2)
            elif typ == "subtract":
                result = _subtract(result, d2)
            elif typ == "intersect":
                result = _intersect(result, d2)
    # any primitive never referenced by an operation → plain union
    for pid in ids[1:]:
        if pid not in consumed:
            result = _union(result, fields[pid])

    volume = result.reshape(dims)

    # 5. extract surface
    if volume.min() > 0 or volume.max() < 0:
        raise ValueError("SDF has no zero crossing — primitives outside grid or empty")
    verts, faces, _, _ = marching_cubes(volume, level=0.0, spacing=(spacing,) * 3)
    verts += lo  # grid → world (mm)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    # keep the largest component if stray shells appear
    if mesh.body_count > 1:
        mesh = max(mesh.split(only_watertight=False), key=lambda m: abs(m.volume))
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=8)
    if not mesh.is_watertight:
        mesh.fill_holes()
    return mesh
