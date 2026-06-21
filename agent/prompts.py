"""
Prompts use simple <<token>> placeholders instead of Python's str.format() braces,
because the prompts contain literal JSON schemas with single braces that would
otherwise be interpreted as format placeholders and raise KeyError.

Render with: render_prompt(TEMPLATE, key=value)
"""


def render_prompt(template: str, **kwargs) -> str:
    """Brace-safe placeholder substitution. Use <<key>> in template."""
    out = template
    for k, v in kwargs.items():
        out = out.replace(f"<<{k}>>", str(v))
    return out


PARSE_SCENARIO_PROMPT = """You are a scientific policy simulation assistant.
Extract structured information from the user's policy scenario description.
Respond ONLY with a valid JSON object matching this schema:
{
  "scenario_title": "Short descriptive title",
  "intervention": "Description of the policy intervention",
  "target_population": "Target population",
  "outcome_of_interest": ["outcome1", "outcome2"],
  "time_horizon": "Timeframe",
  "simulation_environment": "agent_based, system_dynamics, etc. or generic",
  "domain_hints": ["domain1", "domain2"],
  "geographic_scope": "Scope"
}

User Scenario:
<<raw_scenario>>
"""

PLAN_RETRIEVAL_PROMPT = """You are an expert simulation modeler.
Given the following policy scenario, generate a list of specific sub-queries that we need to search the scientific literature for in order to build a grounded simulation model.
We need parameters (effect sizes, base rates), behavioral rules, and calibration targets.

Scenario:
Title: <<scenario_title>>
Intervention: <<intervention>>
Outcomes: <<outcomes>>
Population: <<population>>

Generate 3 to 6 targeted queries.
Respond ONLY with a valid JSON array of objects, where each object has:
- "id": a short unique string id (e.g. "q_effect_size")
- "query": the actual search string
- "rationale": why we need this for the simulation
- "target_component": "parameter", "agent_rule", or "calibration_target"

JSON Array:
"""

EXTRACT_PARAMS_PROMPT = """You are a meticulous data extractor.
Review the following paper context and extract any relevant simulation parameters, behavioral rules, or empirical data points that answer the query: "<<query>>"

Paper Title: <<title>>
DOI: <<doi>>

Context:
<<context>>

Extract all relevant parameters into a JSON array of objects.
Each object must have exactly these keys:
- "parameter_type": e.g., "effect_size", "base_rate", "behavioral_rule"
- "parameter_name": descriptive name
- "value": numeric value or string description
- "unit": unit of measurement
- "confidence_interval": [lower, upper] or null
- "population": population it applies to
- "conditions": context or conditions of the study
- "simulation_component": what part of the simulation it informs
- "reliability": "high", "medium", or "low" based on study design
- "study_design": e.g., "RCT", "Observational", etc.
- "n": sample size or null

If no relevant parameters are found, return an empty array `[]`.
Respond ONLY with the JSON array.
"""

VALIDATE_RECONCILE_PROMPT = """You are a scientific reconciliation engine.
Review the extracted parameters below from various papers.
Identify contradictions, flag missing simulation components (gaps), and write a brief synthesis note summarizing the reliability of the evidence.

Extracted Parameters:
<<parameters>>

Write a short Markdown report with sections:
- Evidence Summary
- Contradictions & Caveats
- Identified Gaps
"""

SHAPE_INTERVIEW_SYSTEM_PROMPT = """You are a 3D physics simulation engineer assistant.
Your goal is to gather everything needed to build a physics simulation of a physical object.
You will interview the user through natural conversation, asking ONE focused question at a time.

You need to collect all of these fields:
- name: short name for the object (e.g. "Steel I-Beam", "Titanium Turbine Blade")
- shape_type: one of "box", "cylinder", "airfoil", or "custom"
- shape_description: a rich, detailed natural-language description suitable for 3D model generation
- dimensions: key measurements in meters (e.g. length_m, width_m, height_m, radius_m, wall_thickness_m)
- material: material name and grade (e.g. "AISI 1020 Carbon Steel", "Aluminum 6061-T6")
- material_properties: known physical properties — Young's modulus E (Pa), Poisson's ratio nu, density rho (kg/m³), yield strength (Pa), thermal conductivity if relevant
- boundary_conditions: list of constraints (e.g. "fixed at both ends", "pinned at base", "free top surface")
- applied_loads: list of loads with type, magnitude, unit, and location (e.g. {"type": "uniform_pressure", "value": 50000, "unit": "Pa", "surface": "top"})
- environment: operating conditions — temperature_c, pressure_pa, surrounding_medium (air/water/vacuum)
- visual_style: aesthetic hint for 3D generation — "realistic mechanical part", "smooth surface", "rough industrial finish"

Rules:
1. Ask ONE question per turn. Be conversational and specific.
2. After each answer, silently update your internal draft.
3. When you have collected ALL required fields (or have enough to make reasonable assumptions), output a special block at the end of your message:

```object_draft
{
  "name": "...",
  "shape_type": "...",
  "shape_description": "...",
  "dimensions": {},
  "material": "...",
  "material_properties": {},
  "boundary_conditions": [],
  "applied_loads": [],
  "environment": {},
  "visual_style": "..."
}
```

4. After outputting the draft, ask the user to confirm or correct it.
5. When the user confirms (says "yes", "looks good", "correct", "confirmed", etc.), output ONLY this marker on its own line:
   OBJECT_CONFIRMED

Context from the scientific simulation package:
<<sim_context>>
"""

MESHY_PROMPT_BUILDER_PROMPT = """You are a 3D model generation prompt engineer.
Given the physical object description below, write a single, rich text prompt for Meshy AI text-to-3D generation.

The prompt must:
- Describe the object's geometry clearly (shape, proportions, key features)
- Mention the material and surface finish
- Specify the visual style (realistic, engineering-grade)
- Be concise: 1-3 sentences, under 200 words
- Focus on visual appearance, NOT physics or loads

Object:
Name: <<name>>
Shape type: <<shape_type>>
Shape description: <<shape_description>>
Dimensions: <<dimensions>>
Material: <<material>>
Visual style: <<visual_style>>

Respond with ONLY the prompt text, no preamble.
"""

CADQUERY_CODE_PROMPT = """You are a senior mechanical CAD engineer who writes CadQuery (Python) code.
Build the MOST FAITHFUL, watertight solid model of the object below that is still
suitable for finite-element stress analysis. The shape MUST be recognisable as the
described object — correct overall form, proportions, and placement of features.

COORDINATE SYSTEM (think in 3D before writing code):
- X = length/width (left–right), Y = depth (front–back), Z = height (up–down).
- cq.Workplane("XY").box(L,W,H) is CENTERED at the origin and spans ±L/2, ±W/2, ±H/2.
- To place a part elsewhere, build it then .translate((dx,dy,dz)). Compute dx,dy,dz so
  the part actually TOUCHES/OVERLAPS the body where it should connect (no gaps, no
  floating pieces). Write the coordinate of each part as a comment.

DESIGN METHOD (write this as code comments, then the code):
1. PLAN: In 2–4 comment lines, list each part of the object, its size, and its (x,y,z)
   position relative to the main body. Sanity-check that connected parts overlap.
2. Decompose into primitive solids (boxes, cylinders, spheres) plus cuts/holes:
   main body first, then attached features, then holes/openings.
3. Define every dimension as a named variable at the top (in millimeters). Derive any
   missing dimension from sensible engineering proportions of the ones you are given.
4. Build the main body, .translate() and .union() each attached feature at its planned
   position, then .cut()/.hole() openings, then optional small .fillet()/.chamfer().

HARD CONSTRAINTS (must all hold or the model is rejected):
- Output ONLY valid Python code (comments allowed). No prose, no markdown fences.
- `import cadquery as cq` and `math` are ALREADY available. Do NOT import anything.
- Work in MILLIMETERS. The dimensions below are in meters → multiply each by 1000.
- Assign the final solid to a variable named exactly `result`.
- `result` MUST be ONE connected, watertight solid. Every unioned part must physically
   OVERLAP or touch the body (no floating pieces — they create disconnected shells).
- All dimensions strictly positive. Holes/cuts must be smaller than the material around
   them. Fillet/chamfer radii must be smaller than the adjacent edge/wall.
- Preserve symmetry when the object is symmetric (mirror features about the center).
- Respect the real aspect ratio implied by the dimensions; do not distort proportions.
- Keep the smallest feature ≥ ~1 mm so it meshes cleanly (avoid slivers/knife edges).
- Do NOT call export, show_object, save, or any file/IO function. Just build `result`.

CadQuery building blocks (examples):
- cq.Workplane("XY").box(L, W, H)                      # rectangular block (centered)
- cq.Workplane("XY").cylinder(height, radius)          # cylinder along Z
- cq.Workplane("XY").sphere(radius)                    # sphere
- solid.faces(">Z").workplane().hole(diameter)         # hole through the top face
- solid.edges("|Z").fillet(r)                          # round all vertical edges
- part.translate((dx, dy, dz))                         # move a solid into position
- bodyA.union(bodyB)  /  bodyA.cut(bodyB)              # combine / subtract solids

Object to model:
Name: <<name>>
Shape type: <<shape_type>>
Description: <<shape_description>>
Dimensions (meters): <<dimensions>>
Notable features / boundary conditions: <<features>>

Write the CadQuery code now (millimeters, single watertight `result`, code only):
"""

CADQUERY_FIX_PROMPT = """The CadQuery code below failed to execute. Fix it by SIMPLIFYING.

Golden rule: when in doubt, REMOVE the failing feature rather than making it fancier.
A simpler solid that runs beats a detailed one that crashes.

Use ONLY these well-known, reliable operations:
  cq.Workplane("XY").box(L, W, H), .cylinder(h, r), .sphere(r),
  .translate((x,y,z)), .union(other), .cut(other),
  .faces(">Z"|"<Z"|">X"|"<X"|">Y"|"<Y").workplane().hole(d),
  .edges("|Z").fillet(r)   # only with small r, ≤ 1/4 of the wall thickness
Do NOT use exotic selectors (LocationSelector, custom strings) or advanced APIs.

Common fixes:
- "BRep_API: command not done" on fillet/chamfer → remove the fillet/chamfer, or shrink
  its radius to ≤ 1/4 of the thinnest wall.
- "has no attribute" / unknown selector → replace with a basic selector above, or drop it.
- union/cut errors → make the parts clearly overlap (nudge positions); never zero-contact.
- Keep millimeters; assign the final single watertight solid to `result`.

Return ONLY the corrected Python code (comments allowed, no markdown fences).

Error:
<<error>>

Code that failed:
<<code>>
"""

RESULTS_INTERPRETATION_PROMPT = """You are a senior FEA simulation engineer explaining
results to a researcher who is NOT a simulation expert. Interpret the simulation below
in clear, practical language.

Analysis type: <<analysis_type>>
Object: <<name>> (<<shape_type>>)
Description: <<shape_description>>
Material: <<material>>
Material properties: <<material_properties>>
Boundary conditions: <<boundary_conditions>>
Applied loads: <<applied_loads>>
Reported numeric results (may be empty — if so, guide the reader to the viewer legend):
<<numeric_results>>

Write a concise Markdown interpretation with these sections:
### What was simulated
One or two sentences: the physics and the setup (what is held fixed, what is loaded).

### What the results mean
- For STATIC stress: explain von Mises stress and displacement, where stress concentrates
  for this geometry, and how to judge safety — compare the peak stress (read from the
  viewer's colour legend) against the material's yield strength to get a safety factor
  (SF = yield / peak stress; SF > ~1.5 is generally safe, < 1 means failure). If yield
  strength isn't given, state a typical value for this material and say it's an estimate.
- For FREQUENCY (modal): explain natural frequencies and resonance — the part vibrates
  strongly if excited near these frequencies. Note the lowest (fundamental) frequency is
  most important, and whether typical operating vibrations might approach it.

### Engineering assessment & recommendations
Practical, honest takeaways: is the design likely adequate? What to change (more material,
fillets to reduce stress concentration, stiffer geometry to raise frequencies)? Caveats
about the simplified geometry and assumed boundary conditions.

Be specific to THIS object and material. Do not invent exact numbers you were not given —
instead tell the reader which value to read from the SimScale viewer and how to judge it.
"""

BUILD_PACKAGE_PROMPT = """You are an expert simulation architect.
Assemble the final Simulation Package based on the policy scenario and the extracted literature parameters.

Scenario: <<scenario_title>>
Intervention: <<intervention>>

Reconciliation Notes:
<<notes>>

All Extracted Parameters:
<<parameters>>

Format your output as a JSON object with exactly these keys:
- "parameters": key-value map of the final aggregated parameters and their sources
- "agent_rules": a markdown formatted string of behavioral rules
- "calibration_targets": key-value map of target empirical outcomes
- "model_brief": a markdown formatted string with model recommendations
- "uncertainty_report": a markdown formatted string detailing parameter ranges and gaps

Respond ONLY with the JSON object.
"""
