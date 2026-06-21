"""
Pluggable physics simulation adapter.

Current implementation: SimScale REST API (structural static stress analysis).
Swap ACTIVE_SOLVER in online/config.py to route to a different backend.

Verified SimScale v0 API flow (probed live 2026-06-19):
  1. POST /storage              → {storageId, url}
  2. PUT  {url}                 → upload STL to S3 (no auth header)
  3. POST /projects/{p}/geometryimports
                                → {geometryImportId}  (format must be STL/STEP/IGES, NOT OBJ)
  4. GET  /projects/{p}/geometryimports/{importId}
                                → poll until status == "FINISHED", then read geometryId
  5. POST /projects/{p}/simulations
                                → body: {name, version:"34.0", geometryId, model:{type:"STATIC_ANALYSIS",...}}
  6. POST /projects/{p}/simulations/{s}/runs
                                → {runId}
  7. POST /projects/{p}/simulations/{s}/runs/{r}/start
  8. GET  /projects/{p}/simulations/{s}/runs/{r}
                                → poll until status == "FINISHED"
  9. GET  /projects/{p}/simulations/{s}/runs/{r}/results
"""

import os
import time
import json
import httpx
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agent.config import (
    SIMSCALE_API_KEY, SIMSCALE_BASE_URL, SIMSCALE_TIMEOUT,
    SIMSCALE_POLL_INTERVAL, SIMSCALE_POLL_TIMEOUT,
    ACTIVE_SOLVER,
)
from agent.models import PhysicalObject, SimulationPackage, SimulationJob, SimulationResult


# ── Abstract base ─────────────────────────────────────────────────────────────

class SimulationAdapter(ABC):
    @abstractmethod
    def prepare(
        self,
        obj_path: str,
        physical_object: PhysicalObject,
        sim_package: SimulationPackage,
        cad_path: str = None,
        analysis_type: str = "static",
    ) -> SimulationJob:
        """
        Upload geometry + configure simulation. Returns a job with tracking IDs.
        If cad_path (a STEP/IGES file) is given, the true CAD geometry is simulated;
        otherwise a clean parametric primitive is built from the object's dimensions.
        analysis_type selects the physics ("static" stress, "frequency" modal, ...).
        """

    @abstractmethod
    def start(self, job: SimulationJob) -> SimulationJob:
        """Start the solver run. Returns updated job with status=RUNNING."""

    @abstractmethod
    def poll(self, job: SimulationJob, progress_callback=None) -> SimulationJob:
        """Block until job is FINISHED or FAILED. Returns updated job."""

    @abstractmethod
    def fetch_results(self, job: SimulationJob) -> SimulationResult:
        """Download / parse results after job is FINISHED."""

    def run_full(
        self,
        obj_path: str,
        physical_object: PhysicalObject,
        sim_package: SimulationPackage,
        progress_callback=None,
        cad_path: str = None,
        analysis_type: str = "static",
    ) -> SimulationResult:
        """Convenience: prepare → start → poll → results."""
        job = self.prepare(
            obj_path, physical_object, sim_package,
            cad_path=cad_path, analysis_type=analysis_type,
        )
        job = self.start(job)
        job = self.poll(job, progress_callback=progress_callback)
        return self.fetch_results(job)


# ── Mesh conversion helper ─────────────────────────────────────────────────────

def obj_to_stl(obj_path: str) -> str:
    """
    Convert an OBJ file to STL (required by SimScale).
    Returns path to the generated .stl file next to the source.
    Requires: pip install trimesh
    """
    try:
        import trimesh
    except ImportError:
        raise RuntimeError(
            "trimesh is required for OBJ→STL conversion. "
            "Run: pip install trimesh"
        )

    mesh = trimesh.load(obj_path, force="mesh")
    stl_path = str(Path(obj_path).with_suffix(".stl"))
    mesh.export(stl_path)
    return stl_path


def physical_object_to_stl(physical_object: "PhysicalObject", dest_dir: str = "sim_geometry") -> str:
    """
    Build a CLEAN, watertight parametric primitive from the PhysicalObject's
    shape_type + dimensions, and export it as STL for SimScale FEA.

    Why not the Meshy mesh? Meshy produces decorative surface shells (multiple
    disconnected, non-watertight bodies) that SimScale's solid mesher rejects.
    A parametric primitive built from the captured dimensions is watertight,
    single-body, and dimensionally accurate — so it always meshes and solves.
    Meshy stays as the visual/digital-twin preview.

    Dimensions are read in meters; common key spellings are accepted.
    Returns the path to the generated .stl file.
    """
    import trimesh

    dims = physical_object.dimensions or {}

    def _d(*keys, default):
        for k in keys:
            for cand in (k, k + "_m", k.replace("_m", "")):
                if cand in dims and dims[cand]:
                    return float(dims[cand])
        return default

    shape = (physical_object.shape_type or "box").lower()

    if shape in ("cylinder", "rod", "shaft", "pipe", "tube"):
        radius = _d("radius", default=_d("diameter", default=0.1) / 2.0)
        height = _d("height", "length", "span", default=0.5)
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    elif shape in ("sphere", "ball"):
        radius = _d("radius", default=_d("diameter", default=0.2) / 2.0)
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    else:  # box / airfoil / plate / custom → bounding box from dimensions
        length = _d("length", "span", default=0.2)
        width = _d("width", "depth", "chord", default=0.1)
        height = _d("height", "thickness", "wall_thickness", default=0.05)
        mesh = trimesh.creation.box(extents=[length, width, height])

    # Guarantee a clean solid
    mesh.process()
    if not mesh.is_watertight:
        mesh.fill_holes()

    os.makedirs(dest_dir, exist_ok=True)
    safe_name = "".join(c for c in (physical_object.name or "object") if c.isalnum() or c in "-_")[:40] or "object"
    stl_path = os.path.join(dest_dir, f"{safe_name}.stl")
    mesh.export(stl_path)
    return stl_path


# ── SimScale adapter ───────────────────────────────────────────────────────────

class SimScaleAdapter(SimulationAdapter):
    """
    Structural static linear stress analysis via SimScale REST API v0.
    Requires SIMSCALE_API_KEY in .env / environment.
    """

    _SCHEMA_VERSION = "34.0"  # latest confirmed 2026-06-19

    def __init__(self):
        self._base = SIMSCALE_BASE_URL   # https://api.simscale.com/v0
        self._h = {
            "X-API-KEY": SIMSCALE_API_KEY,
            "Content-Type": "application/json",
        }
        self._timeout = SIMSCALE_TIMEOUT

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    def _get(self, path: str) -> dict:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.get(f"{self._base}{path}", headers=self._h)
            r.raise_for_status()
            return r.json()

    def _post(self, path: str, body: dict) -> dict:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.post(f"{self._base}{path}", json=body, headers=self._h)
            if not r.is_success:
                raise RuntimeError(
                    f"POST {path} → {r.status_code}\n"
                    f"Request body: {json.dumps(body, indent=2)[:1500]}\n"
                    f"Response: {r.text[:1000]}"
                )
            return r.json()

    def _post_empty(self, path: str, params: dict = None) -> None:
        """POST with an empty body (e.g. start run / start mesh)."""
        with httpx.Client(timeout=self._timeout) as c:
            r = c.post(f"{self._base}{path}", headers=self._h, params=params or {})
            r.raise_for_status()

    def _put(self, path: str, body: dict) -> None:
        """PUT a full resource (e.g. update simulation spec to attach a meshId)."""
        with httpx.Client(timeout=self._timeout) as c:
            r = c.put(f"{self._base}{path}", json=body, headers=self._h)
            if not r.is_success:
                raise RuntimeError(
                    f"PUT {path} → {r.status_code}\nResponse: {r.text[:1000]}"
                )

    # ── Step 1b: mesh generation ──────────────────────────────────────────────

    def _mesh_geometry(self, project_id: str, geometry_id: str, simulation_id: str) -> str:
        """
        Create + start + poll an automatic solid mesh operation.
        Returns meshId required when creating a simulation run.

        Confirmed working API:
          POST /projects/{p}/meshoperations  → {meshOperationId}
          POST /projects/{p}/meshoperations/{id}/start?simulationId={sid}  → 202
          GET  /projects/{p}/meshoperations/{id}  → poll until status==FINISHED, read meshId
        """
        op = self._post(
            f"/projects/{project_id}/meshoperations",
            {
                "name": "auto_mesh",
                "version": "10.0",
                "geometryId": geometry_id,
                "model": {
                    "type": "SIMMETRIX_MESHING_SOLID",
                    "sizing": {"type": "AUTOMATIC_V9", "fineness": 3},
                    "numOfProcessors": 4,
                    "maxMeshingRunTime": {"value": 600, "unit": "s"},
                },
            },
        )
        mesh_op_id: str = op["meshOperationId"]

        self._post_empty(
            f"/projects/{project_id}/meshoperations/{mesh_op_id}/start",
            params={"simulationId": simulation_id},
        )

        deadline = time.time() + 600
        while time.time() < deadline:
            data = self._get(f"/projects/{project_id}/meshoperations/{mesh_op_id}")
            status = data.get("status", "")
            if status == "FINISHED":
                mesh_id = data.get("meshId")
                if not mesh_id:
                    raise RuntimeError("Mesh finished but meshId is missing in response")
                return mesh_id
            if status in ("FAILED", "CANCELED"):
                raise RuntimeError(f"Mesh operation {status}")
            time.sleep(10)

        raise TimeoutError("Mesh operation did not finish within 600 seconds")

    # ── Step 1: geometry upload ───────────────────────────────────────────────

    def _import_geometry(self, project_id: str, file_path: str) -> tuple:
        """
        Upload + import geometry. Returns (geometryId, region_entities, face_entities).
        region_entities: volume names (class "region", e.g. ["B1_TE5"]) — material assignment.
        face_entities:   surface names (class "face", e.g. ["B1_TE2", ...]) — BC assignment.

        facetSplit=True splits a faceted box into its 6 sides so fixed support and
        load can be assigned to different faces (a real cantilever, not a fully
        constrained block). sewing=True stitches the STL into a single watertight
        solid the Simmetrix mesher can volume-mesh.
        """
        # 1a. storage slot
        storage = self._post("/storage", {})
        storage_id: str = storage["storageId"]
        upload_url: str = storage["url"]

        # 1b. S3 upload
        with open(file_path, "rb") as f:
            content = f.read()
        with httpx.Client(timeout=120) as c:
            r = c.put(upload_url, content=content)
            r.raise_for_status()

        # 1c. POST /geometryimports — options field is required (not nullable)
        ext = Path(file_path).suffix.upper().lstrip(".")
        fmt_map = {"STL": "STL", "STEP": "STEP", "STP": "STEP", "IGES": "IGES", "IGS": "IGES"}
        fmt = fmt_map.get(ext, "STL")

        # STL is a faceted triangle soup → facetSplit recovers individual faces.
        # STEP/IGES are BREP CAD with real faces already → no facetSplit needed.
        # Units: our trimesh primitives are authored in meters; CadQuery/CAD STEP
        # files are authored in millimeters (the CAD convention).
        if fmt == "STL":
            options = {"facetSplit": True, "sewing": True, "improve": True, "optimizeForLBMSolver": False}
            input_unit = "m"
        else:
            options = {"facetSplit": False, "sewing": True, "improve": True, "optimizeForLBMSolver": False}
            input_unit = "mm"

        imp = self._post(
            f"/projects/{project_id}/geometryimports",
            {
                "name": Path(file_path).stem[:40],
                "location": {"storageId": storage_id},
                "format": fmt,
                "inputUnit": input_unit,
                "options": options,
            },
        )
        import_id: str = imp["geometryImportId"]

        # 1d. Poll until FINISHED
        deadline = time.time() + 120
        geometry_id = None
        while time.time() < deadline:
            g = self._get(f"/projects/{project_id}/geometryimports/{import_id}")
            status = g.get("status", "")
            if status == "FINISHED":
                geometry_id = g["geometryId"]
                break
            if status in ("FAILED", "CANCELED"):
                reason = g.get("failureReason", {})
                raise RuntimeError(f"Geometry import {status}: {reason}")
            time.sleep(5)

        if not geometry_id:
            raise TimeoutError("Geometry import did not finish within 120 seconds")

        # 1e. Fetch entity names for topological references.
        #     class "region" = the solid volume (material MUST target a Volume entity);
        #     class "face"   = the surfaces (boundary conditions target Faces).
        # For STEP→Parasolid imports the region (volume) mapping can populate a few
        # seconds after the faces, so poll until at least one region appears.
        region_entities: list = []
        face_entities: list = []
        deadline = time.time() + 60
        while time.time() < deadline:
            region_entities, face_entities = [], []
            try:
                mappings = self._get(
                    f"/projects/{project_id}/geometries/{geometry_id}/mappings?limit=500"
                )
                for entry in mappings.get("_embedded", []):
                    cls = entry.get("class", "")
                    name = entry.get("name", "")
                    if not name:
                        continue
                    if cls == "region":
                        region_entities.append(name)
                    elif cls == "face":
                        face_entities.append(name)
            except Exception:
                pass
            # Material needs a volume; wait for it. (Faces alone aren't enough.)
            if region_entities:
                break
            time.sleep(4)

        return geometry_id, region_entities, face_entities

    # ── Step 2: simulation spec ───────────────────────────────────────────────

    def _build_sim_spec(
        self,
        name: str,
        geometry_id: str,
        physical_object: PhysicalObject,
        region_entities: list = None,
        face_entities: list = None,
        analysis_type: str = "static",
    ) -> dict:
        """
        Build the simulation creation body for the chosen analysis type.

        Supported analysis_type (validated against the live API):
          - "static"    → STATIC_ANALYSIS: stress & displacement under a load.
          - "frequency" → FREQUENCY_ANALYSIS: natural (modal) frequencies, no load.

        Entity assignment:
          - material      → a Volume (class "region"); faces are rejected by the API.
          - fixed support → the first face.
          - pressure load → a different face (static only → real cantilever).
        """
        region_entities = region_entities or []
        face_entities = face_entities or []
        analysis_type = (analysis_type or "static").lower()

        props = physical_object.material_properties
        E   = float(props.get("E",   210e9))
        nu  = float(props.get("nu",  0.3))
        rho = float(props.get("rho", 7850.0))

        def _C(val):
            """Wrap a scalar in the CONSTANT DimensionalFunction envelope."""
            return {"type": "CONSTANT", "value": val}

        material = {
            "name": physical_object.material or "Steel",
            "materialBehavior": {
                "type": "LINEAR_ELASTIC",
                "directionalDependency": {
                    "type": "ISOTROPIC",
                    "youngsModulus": {"value": _C(E), "unit": "Pa"},
                    "poissonsRatio": _C(nu),          # dimensionless — no unit wrapper
                },
            },
            "density": {"value": _C(rho), "unit": "kg/m^3"},
            # Material must reference a Volume (region) entity.
            "topologicalReference": {"entities": region_entities, "sets": []},
        }

        fixed_entities = face_entities[:1] if face_entities else []
        pressure_entities = face_entities[1:2] if len(face_entities) > 1 else face_entities[:1]

        # Both analyses anchor the part with a fixed support.
        bcs = [
            {
                "type": "FIXED_SUPPORT",
                "name": "fixed_support",
                "topologicalReference": {"entities": fixed_entities, "sets": []},
            }
        ]

        common = {
            "elementTechnology": {
                "elementTechnology3D": {"definitionMethod": {"type": "AUTOMATIC"}}
            },
            "model": {},  # required by API; empty for linear
        }

        if analysis_type == "frequency":
            # Modal analysis: no external load — compute natural frequencies.
            model = {
                "type": "FREQUENCY_ANALYSIS",
                **common,
                "numerics": {
                    "solver": {
                        "type": "MUMPS",
                        "advancedMumpsSettings": {
                            "precisionSingularityDetection": 8, "stopIfSingular": True,
                        },
                    },
                    "solveModel": {},
                    "calculateFrequency": {},
                    "eigenMode": {},
                    "eigenSolver": {"type": "QZ"},
                },
                "simulationControl": {
                    "eigenfrequencyScope": {"type": "CENTER", "numberOfModes": 6},
                    "maxRunTime": {"value": 3600, "unit": "s"},
                    "processors": {"value": 1},
                },
                "resultControl": {},
                "materials": [material],
                "boundaryConditions": bcs,
            }
        else:
            # Static stress analysis: needs a load. Parse one from applied_loads
            # (flexible), else default to 1 bar so the analysis is valid.
            pressure_val = 0.0
            for load in (physical_object.applied_loads or []):
                ltype = str(load.get("type", "")).lower()
                if "pressure" in ltype or "force" in ltype or "load" in ltype or not ltype:
                    for key in ("value", "magnitude", "pressure", "pa"):
                        if load.get(key) not in (None, ""):
                            try:
                                pressure_val = float(load[key]); break
                            except (TypeError, ValueError):
                                continue
                if pressure_val:
                    break
            if not pressure_val:
                pressure_val = 1.0e5

            if pressure_entities:
                bcs.append({
                    "type": "PRESSURE",
                    "name": "applied_pressure",
                    "pressure": {"value": _C(pressure_val), "unit": "Pa"},
                    "topologicalReference": {"entities": pressure_entities, "sets": []},
                })

            model = {
                "type": "STATIC_ANALYSIS",
                **common,
                "numerics": {
                    "solver": {
                        "type": "MUMPS",
                        "advancedMumpsSettings": {
                            "precisionSingularityDetection": 8, "stopIfSingular": True,
                        },
                    }
                },
                "simulationControl": {
                    "maxRunTime": {"value": 3600, "unit": "s"},
                    "processors": {"value": 1},
                },
                "resultControl": {
                    "solutionFields": [
                        {"type": "DISPLACEMENT", "name": "Displacement"},
                        {"type": "STRESS", "name": "Stress",
                         "stressType": {"type": "SIGNED_VON_MISES"}},
                    ],
                },
                "materials": [material],
                "boundaryConditions": bcs,
            }

        return {
            "name": name[:60],
            "version": self._SCHEMA_VERSION,
            "geometryId": geometry_id,
            "model": model,
        }

    # ── Adapter interface ─────────────────────────────────────────────────────

    def prepare(
        self,
        obj_path: str,
        physical_object: PhysicalObject,
        sim_package: SimulationPackage,
        cad_path: str = None,
        analysis_type: str = "static",
    ) -> SimulationJob:
        job = SimulationJob(solver="simscale", analysis_type=(analysis_type or "static").lower())

        # Create project — include sim package title in description for traceability
        pkg_title = getattr(sim_package, "model_brief", "")[:80].split("\n")[0].strip(" #")
        proj = self._post("/projects", {
            "name": f"scires_{physical_object.name[:35]}",
            "description": f"Auto-generated via sciresVprototype. {pkg_title}".strip(),
        })
        job.project_id = proj["projectId"]

        # Geometry source:
        #   - If a real CAD file (STEP/IGES) is provided, simulate the TRUE geometry.
        #   - Otherwise build a clean watertight parametric primitive from the
        #     captured dimensions. The Meshy mesh (obj_path) is decorative only and
        #     is NOT FEA-meshable; it stays as the visual preview in the UI.
        if cad_path and Path(cad_path).suffix.lower() in (".step", ".stp", ".iges", ".igs"):
            mesh_path = cad_path
        else:
            mesh_path = physical_object_to_stl(physical_object)

        # Upload geometry + fetch topological entity names for BC/load assignment.
        geometry_id, region_entities, face_entities = self._import_geometry(
            job.project_id, mesh_path
        )

        # Create simulation spec (material → region/volume, BCs → faces)
        sim_spec = self._build_sim_spec(
            name=f"{analysis_type}_{physical_object.name[:28]}",
            geometry_id=geometry_id,
            physical_object=physical_object,
            region_entities=region_entities,
            face_entities=face_entities,
            analysis_type=analysis_type,
        )
        sim = self._post(f"/projects/{job.project_id}/simulations", sim_spec)
        job.simulation_id = sim["simulationId"]

        # Generate mesh — required before a run can be created
        job.mesh_id = self._mesh_geometry(job.project_id, geometry_id, job.simulation_id)

        # Attach the mesh to the simulation spec (meshId is a SimulationSpec field,
        # NOT a run field). GET current spec → set meshId → PUT it back.
        spec = self._get(f"/projects/{job.project_id}/simulations/{job.simulation_id}")
        spec["meshId"] = job.mesh_id
        self._put(f"/projects/{job.project_id}/simulations/{job.simulation_id}", spec)

        job.status = "PREPARED"
        return job

    def start(self, job: SimulationJob) -> SimulationJob:
        # meshId is already attached to the simulation; the run body takes only a name.
        run = self._post(
            f"/projects/{job.project_id}/simulations/{job.simulation_id}/runs",
            {"name": "run_1"},
        )
        job.run_id = run["runId"]

        self._post_empty(
            f"/projects/{job.project_id}/simulations/{job.simulation_id}/runs/{job.run_id}/start"
        )
        job.status = "RUNNING"
        return job

    def poll(self, job: SimulationJob, progress_callback=None) -> SimulationJob:
        """
        Poll run until status is FINISHED or FAILED.
        SimScale Status enum: READY | QUEUED | RUNNING | FINISHED | CANCELED | FAILED
        """
        deadline = time.time() + SIMSCALE_POLL_TIMEOUT
        transient_errors = 0
        while time.time() < deadline:
            # The run resource can momentarily 404/5xx while SimScale provisions
            # the solver. Tolerate a handful of transient failures before giving up.
            try:
                data = self._get(
                    f"/projects/{job.project_id}/simulations/{job.simulation_id}/runs/{job.run_id}"
                )
                transient_errors = 0
            except Exception as e:
                transient_errors += 1
                if transient_errors >= 5:
                    job.status = "FAILED"
                    job.error_message = f"Polling failed repeatedly: {e}"
                    return job
                time.sleep(SIMSCALE_POLL_INTERVAL)
                continue

            status = data.get("status", "RUNNING")
            if progress_callback:
                progress_callback(status)

            if status == "FINISHED":
                job.status = "FINISHED"
                return job
            if status in ("FAILED", "CANCELED"):
                job.status = "FAILED"
                job.error_message = str(data.get("error", status))
                return job

            time.sleep(SIMSCALE_POLL_INTERVAL)

        job.status = "FAILED"
        job.error_message = "Poll timeout exceeded"
        return job

    def fetch_results(self, job: SimulationJob) -> SimulationResult:
        if job.status != "FINISHED":
            return SimulationResult(
                job_id=job.job_id or "",
                solver=job.solver,
                analysis_type=job.analysis_type,
                summary=f"Job did not finish: {job.error_message}",
            )

        try:
            data = self._get(
                f"/projects/{job.project_id}/simulations/{job.simulation_id}"
                f"/runs/{job.run_id}/results"
            )
        except Exception:
            data = {}
        raw = data if isinstance(data, dict) else {}

        viewer_url = (
            f"https://www.simscale.com/workbench/?pid={job.project_id}"
            f"&mi={job.simulation_id}&rp={job.run_id}"
        )

        if job.analysis_type == "frequency":
            summary = (
                "Modal (frequency) analysis finished. The natural frequencies and mode "
                "shapes are in the SimScale viewer — open it to see how the part flexes "
                "at each resonant frequency."
            )
        else:
            summary = (
                "Static stress analysis finished. Open the SimScale viewer to inspect "
                "the von Mises stress and displacement fields (read the peak values from "
                "the colour legend)."
            )

        return SimulationResult(
            job_id=job.job_id or "",
            solver="simscale",
            analysis_type=job.analysis_type,
            project_id=job.project_id,
            simulation_id=job.simulation_id,
            run_id=job.run_id,
            result_viewer_url=viewer_url,
            summary=summary,
            raw_results=raw,
        )

    # ── Numeric extraction (on-demand) ─────────────────────────────────────────

    def extract_numbers(
        self,
        project_id: str,
        simulation_id: str,
        run_id: str,
        material_properties: dict = None,
    ) -> dict:
        """
        Best-effort extraction of scalar results from a FINISHED run.

        Flow (validated live):
          1. GET .../results                       → resultId
          2. POST /projects/{p}/export {resultId, format: VTK}  → exportId
          3. GET  /projects/{p}/export/{exportId}  → poll until status == "DONE", read S3 url
          4. download + unzip → region*.vtu        → parse with vtk
        Returns a dict of any values found:
          {max_von_mises_stress_pa, max_displacement_m, min_safety_factor}
        Raises on failure so callers can fall back to viewer-only behaviour.
        """
        # 1. result id
        res = self._get(f"/projects/{project_id}/simulations/{simulation_id}/runs/{run_id}/results")
        embedded = res.get("_embedded", [])
        if not embedded:
            raise RuntimeError("No result fields are available for this run yet.")
        result_id = embedded[0]["resultId"]

        # 2. create VTK export
        exp = self._post(f"/projects/{project_id}/export", {"resultId": result_id, "format": "VTK"})
        export_id = exp["exportId"]

        # 3. poll until DONE
        deadline = time.time() + 300
        download_url = None
        while time.time() < deadline:
            e = self._get(f"/projects/{project_id}/export/{export_id}")
            status = e.get("status", "")
            if status == "DONE":
                download_url = e.get("url")
                break
            if status in ("FAILED", "ERROR", "CANCELED"):
                raise RuntimeError(f"Result export {status}: {e.get('errorCode')}")
            time.sleep(6)
        if not download_url:
            raise TimeoutError("Result export did not finish in time")

        # 4. download + parse
        import io as _io, zipfile, tempfile
        with httpx.Client(timeout=180) as c:
            r = c.get(download_url, follow_redirects=True)  # presigned S3 — no auth header
            r.raise_for_status()
        z = zipfile.ZipFile(_io.BytesIO(r.content))
        tmpd = tempfile.mkdtemp()
        z.extractall(tmpd)

        vtu_path = None
        for name in z.namelist():
            if name.lower().endswith(".vtu"):
                vtu_path = os.path.join(tmpd, name)
                break
        if not vtu_path or not os.path.exists(vtu_path):
            raise RuntimeError("No .vtu result field found in export archive")

        return self._parse_vtu(vtu_path, material_properties or {})

    @staticmethod
    def _parse_vtu(vtu_path: str, material_properties: dict) -> dict:
        """Parse a VTU result file → max von Mises stress, max displacement, safety factor."""
        import numpy as np
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(vtu_path)
        reader.Update()
        grid = reader.GetOutput()
        pd = grid.GetPointData()

        out = {}
        for i in range(pd.GetNumberOfArrays()):
            name = (pd.GetArrayName(i) or "").lower()
            arr = vtk_to_numpy(pd.GetArray(i))
            if "von mises" in name or "vonmises" in name or name == "stress":
                out["max_von_mises_stress_pa"] = float(np.abs(arr).max())
            elif "displacement" in name:
                mag = np.linalg.norm(arr, axis=1) if arr.ndim > 1 else np.abs(arr)
                out["max_displacement_m"] = float(mag.max())

        # Safety factor if a yield strength is known (accept several key spellings)
        yield_pa = None
        for k in ("yield_strength", "yield_strength_pa", "sigma_y", "yield"):
            v = material_properties.get(k)
            if v:
                try:
                    yield_pa = float(v); break
                except (TypeError, ValueError):
                    pass
        peak = out.get("max_von_mises_stress_pa")
        if yield_pa and peak and peak > 0:
            out["min_safety_factor"] = yield_pa / peak

        return out


# ── Factory ────────────────────────────────────────────────────────────────────

def get_adapter(solver: str = None) -> SimulationAdapter:
    """Return the adapter for the configured or requested solver."""
    s = (solver or ACTIVE_SOLVER).lower()
    if s == "simscale":
        return SimScaleAdapter()
    raise NotImplementedError(
        f"Solver '{s}' not yet implemented. "
        "Available: simscale. Add omniverse/ansys/fenics adapters in services/simulation.py."
    )
