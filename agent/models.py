from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

class PolicyScenario(BaseModel):
    scenario_title: str = Field(default="Unknown Scenario")
    intervention: str = Field(default="")
    target_population: str = Field(default="")
    outcome_of_interest: List[str] = Field(default_factory=list)
    time_horizon: str = Field(default="")
    simulation_environment: str = Field(default="generic")
    domain_hints: List[str] = Field(default_factory=list)
    geographic_scope: str = Field(default="")

class SubQuery(BaseModel):
    id: str
    query: str
    rationale: str
    target_component: str  # e.g., "agent_behaviour_rule", "calibration_target", "effect_size"

class ExtractedParameter(BaseModel):
    paper_doi: str = Field(default="N/A")
    parameter_type: str = Field(default="unknown")
    parameter_name: str = Field(default="unknown")
    value: Union[float, str] = Field(default="")
    unit: str = Field(default="")
    confidence_interval: Optional[List[float]] = None
    population: str = Field(default="")
    conditions: str = Field(default="")
    simulation_component: str = Field(default="")
    reliability: str = Field(default="unknown")
    study_design: str = Field(default="unknown")
    n: Optional[int] = None

class SimulationPackage(BaseModel):
    parameters: Dict[str, Any] = Field(default_factory=dict)
    agent_rules: str = Field(default="")
    calibration_targets: Dict[str, Any] = Field(default_factory=dict)
    model_brief: str = Field(default="")
    uncertainty_report: str = Field(default="")


# ── Phase-2: 3D Object + Simulation models ────────────────────────────────────

class PhysicalObject(BaseModel):
    name: str = Field(default="")
    shape_type: str = Field(default="custom")   # box | cylinder | airfoil | custom
    shape_description: str = Field(default="")  # natural language, fed to Meshy
    dimensions: Dict[str, float] = Field(default_factory=dict)   # {"length_m": 2.5, ...}
    material: str = Field(default="")
    material_properties: Dict[str, Any] = Field(default_factory=dict)  # E, nu, rho, ...
    boundary_conditions: List[str] = Field(default_factory=list)
    applied_loads: List[Dict[str, Any]] = Field(default_factory=list)
    environment: Dict[str, Any] = Field(default_factory=dict)    # temp, pressure, medium
    visual_style: str = Field(default="realistic")               # Meshy art style hint


class MeshyResult(BaseModel):
    task_id: str = Field(default="")
    status: str = Field(default="PENDING")       # PENDING | IN_PROGRESS | SUCCEEDED | FAILED
    progress: int = Field(default=0)             # 0-100
    model_urls: Dict[str, str] = Field(default_factory=dict)  # glb | obj | fbx -> url
    thumbnail_url: Optional[str] = None
    local_glb_path: Optional[str] = None
    local_obj_path: Optional[str] = None
    error_message: Optional[str] = None


class SimulationJob(BaseModel):
    job_id: str = Field(default="")
    solver: str = Field(default="simscale")
    analysis_type: str = Field(default="static")  # static | frequency
    status: str = Field(default="PENDING")       # PENDING | RUNNING | FINISHED | FAILED
    project_id: Optional[str] = None
    simulation_id: Optional[str] = None
    mesh_id: Optional[str] = None
    run_id: Optional[str] = None
    results_url: Optional[str] = None
    error_message: Optional[str] = None


class SimulationResult(BaseModel):
    job_id: str = Field(default="")
    solver: str = Field(default="simscale")
    analysis_type: str = Field(default="static")  # static | frequency
    # Locators so numeric extraction can be triggered later (on-demand button)
    project_id: Optional[str] = None
    simulation_id: Optional[str] = None
    run_id: Optional[str] = None
    max_von_mises_stress_pa: Optional[float] = None
    max_displacement_m: Optional[float] = None
    min_safety_factor: Optional[float] = None
    natural_frequencies_hz: list = Field(default_factory=list)  # modal results
    result_viewer_url: Optional[str] = None      # SimScale online viewer link
    numbers_extracted: bool = Field(default=False)
    summary: str = Field(default="")             # plain-language summary
    interpretation: str = Field(default="")      # LLM engineering interpretation
    raw_results: Dict[str, Any] = Field(default_factory=dict)
