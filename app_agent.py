import streamlit as st
import json
from agent.graph import create_agent_graph
from agent.object_agent import step as interview_step, build_meshy_prompt
from agent.models import PhysicalObject, MeshyResult, SimulationResult

st.set_page_config(page_title="Digital Twin Brain", layout="wide", initial_sidebar_state="expanded")

st.title("🧠 Digital Twin Brain")
st.markdown("Conversational interface grounded in your 3,000 papers. Ask questions, explore the corpus, or generate simulation packages.")

# ── Session state init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_memory" not in st.session_state:
    st.session_state.session_memory = {
        "topics": [],
        "domains": [],
        "simulations": [],
        "notes": "",
    }
# Phase-2 state
if "active_sim_package" not in st.session_state:
    st.session_state.active_sim_package = None
if "active_scenario" not in st.session_state:
    st.session_state.active_scenario = None
if "interview_messages" not in st.session_state:
    st.session_state.interview_messages = []
if "current_draft" not in st.session_state:
    st.session_state.current_draft = None
if "interview_done" not in st.session_state:
    st.session_state.interview_done = False
if "meshy_result" not in st.session_state:
    st.session_state.meshy_result = None
if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = None
if "meshy_prompt" not in st.session_state:
    st.session_state.meshy_prompt = ""
# AI CAD geometry state
if "cad_code" not in st.session_state:
    st.session_state.cad_code = ""
if "cad_step_path" not in st.session_state:
    st.session_state.cad_step_path = None
if "cad_stl_path" not in st.session_state:
    st.session_state.cad_stl_path = None
if "cad_error" not in st.session_state:
    st.session_state.cad_error = ""
if "cad_fallback" not in st.session_state:
    st.session_state.cad_fallback = False


# ── Memory helpers ─────────────────────────────────────────────────────────────

def _extract_topic_keywords(text: str, max_words: int = 6) -> str:
    stop = {"the", "a", "an", "is", "are", "what", "how", "why", "can", "do",
            "does", "tell", "me", "about", "give", "show", "find", "list"}
    words = [w.strip("?,.'\"") for w in text.split() if w.lower().strip("?,.'\"") not in stop]
    return " ".join(words[:max_words]).lower()


def _update_memory_qa(query: str, answer: str) -> None:
    mem = st.session_state.session_memory
    topic = _extract_topic_keywords(query)
    if topic and topic not in mem["topics"]:
        mem["topics"].append(topic)
    mem["topics"] = mem["topics"][-10:]


def _update_memory_simulate(scenario) -> None:
    if not scenario:
        return
    mem = st.session_state.session_memory
    title = scenario.scenario_title
    if title and title not in mem["simulations"]:
        mem["simulations"].append(title)
    mem["simulations"] = mem["simulations"][-5:]


def _update_memory_domains(answer: str) -> None:
    known_domains = [
        "Machine Learning", "Computer Vision", "Natural Language Processing",
        "Climate Science", "Economics", "Healthcare", "Epidemiology",
        "Materials Science", "Environmental Science", "Robotics",
        "Sports Analytics", "Bioinformatics", "Energy", "Finance",
    ]
    mem = st.session_state.session_memory
    for d in known_domains:
        if d.lower() in answer.lower() and d not in mem["domains"]:
            mem["domains"].append(d)
    mem["domains"] = mem["domains"][-6:]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧩 Session Memory")
    mem = st.session_state.session_memory

    if mem["topics"]:
        st.markdown("**Topics explored:**")
        st.markdown(" ".join([f"`{t}`" for t in mem["topics"]]))
    else:
        st.markdown("*No topics yet.*")

    if mem["domains"]:
        st.markdown("**Domains touched:**")
        st.markdown(" ".join([f"`{d}`" for d in mem["domains"]]))

    if mem["simulations"]:
        st.markdown("**Simulations built:**")
        for s in mem["simulations"]:
            st.markdown(f"- {s}")

    st.divider()
    notes_val = st.text_area(
        "📝 Your notes (visible to the agent)",
        value=mem["notes"],
        height=100,
        placeholder="e.g. 'Focus on European policy scenarios'",
        key="notes_input",
    )
    if notes_val != mem["notes"]:
        st.session_state.session_memory["notes"] = notes_val

    st.divider()
    if st.button("🗑 Clear memory", help="Wipe session memory (keeps chat)"):
        st.session_state.session_memory = {"topics": [], "domains": [], "simulations": [], "notes": ""}
        st.rerun()
    if st.button("🗑 Clear everything"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.session_memory = {"topics": [], "domains": [], "simulations": [], "notes": ""}
        st.session_state.active_sim_package = None
        st.session_state.active_scenario = None
        st.session_state.interview_messages = []
        st.session_state.current_draft = None
        st.session_state.interview_done = False
        st.session_state.meshy_result = None
        st.session_state.simulation_result = None
        st.session_state.meshy_prompt = ""
        st.session_state.cad_code = ""
        st.session_state.cad_step_path = None
        st.session_state.cad_stl_path = None
        st.session_state.cad_error = ""
        st.session_state.cad_fallback = False
        st.rerun()


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _render_md(text: str, fallback: str = "*Not available.*"):
    if not text or not text.strip():
        st.markdown(fallback)
        return
    st.markdown(text.replace("\\n", "\n").strip())


def _package_download(pkg, scenario=None, validation_notes: str = "") -> None:
    export = {
        "scenario": scenario.model_dump() if scenario else {},
        "validation_notes": validation_notes,
        **pkg.model_dump(),
    }
    title = (scenario.scenario_title[:40].replace(" ", "_") if scenario else "simulation")
    st.download_button(
        label="⬇ Download Simulation Package (.json)",
        data=json.dumps(export, indent=2, ensure_ascii=False),
        file_name=f"sim_{title}.json",
        mime="application/json",
    )


def _render_package(pkg, scenario=None, validation_notes: str = "") -> None:
    _package_download(pkg, scenario=scenario, validation_notes=validation_notes)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Parameters", "🤖 Agent Rules", "📋 Model Brief", "⚠️ Uncertainty", "🔩 Raw JSON"
    ])
    with tab1:
        st.json(pkg.parameters)
    with tab2:
        _render_md(pkg.agent_rules, fallback="*No agent rules extracted.*")
    with tab3:
        _render_md(pkg.model_brief, fallback="*No model brief available.*")
    with tab4:
        _render_md(pkg.uncertainty_report, fallback="*No uncertainty report available.*")
    with tab5:
        st.json(pkg.model_dump())


def _render_3d_viewer(stl_path: str, height: int = 420) -> None:
    """
    Interactive realtime 3D viewer for a CAD solid. Converts the STL to GLB and
    embeds Google's <model-viewer> (orbit / zoom / pan, auto-rotate) — no extra
    Python deps, fully interactive in the browser.
    """
    import base64
    import streamlit.components.v1 as components
    try:
        import trimesh
        mesh = trimesh.load(stl_path, force="mesh")
        glb = mesh.export(file_type="glb")
        b64 = base64.b64encode(glb).decode()
    except Exception as e:
        st.warning(f"Could not render 3D preview: {e}")
        return

    html = f"""
    <script type="module"
      src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <model-viewer
        src="data:model/gltf-binary;base64,{b64}"
        camera-controls
        auto-rotate
        rotation-per-second="20deg"
        shadow-intensity="1"
        environment-image="neutral"
        exposure="1.1"
        style="width:100%; height:{height-20}px; background:#1e1e1e; border-radius:8px;">
    </model-viewer>
    <p style="color:#888; font-family:sans-serif; font-size:12px; margin:4px 0 0;">
        Drag to rotate · scroll to zoom · right-drag to pan
    </p>
    """
    components.html(html, height=height)


def _render_draft_card(draft: PhysicalObject) -> None:
    """Compact live preview of the PhysicalObject being built."""
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {draft.name or '—'}")
            st.markdown(f"**Shape:** {draft.shape_type or '—'}")
            st.markdown(f"**Material:** {draft.material or '—'}")
        with col2:
            dims = ", ".join(f"{k}: {v}" for k, v in draft.dimensions.items()) or "—"
            st.markdown(f"**Dimensions:** {dims}")
            bcs = ", ".join(draft.boundary_conditions) or "—"
            st.markdown(f"**BCs:** {bcs}")
        if draft.shape_description:
            st.caption(f"_{draft.shape_description}_")


# ── Phase-2: 3D Object + Simulation section ───────────────────────────────────

def _render_3d_section(sim_package, scenario) -> None:
    """
    Full Phase-2 UI: shape interview → Meshy 3D generation → SimScale run.
    Shown below the simulation package when intent == 'simulate'.
    """
    st.divider()
    st.subheader("🧊 3D Object Simulation")
    st.markdown(
        "Describe the physical object you want to simulate in 3D. "
        "The agent will ask you questions, generate a 3D mesh via Meshy AI, "
        "then run it through SimScale for physics simulation."
    )

    # ── Step 1: Shape interview ──────────────────────────────────────────────
    st.markdown("#### Step 1 — Describe your object")

    # Two-column layout: chat left, live draft card right
    chat_col, draft_col = st.columns([3, 2])

    with draft_col:
        st.markdown("**Live Object Draft**")
        if st.session_state.current_draft:
            _render_draft_card(st.session_state.current_draft)
        else:
            st.info("The draft will appear here as you answer questions.")
        if st.session_state.interview_done:
            st.success("✅ Object confirmed — proceed to Step 2.")

    with chat_col:
        # Render interview history
        for msg in st.session_state.interview_messages:
            if msg.get("role") == "user" and msg.get("content") == "start":
                continue  # hide the hidden kick-off message
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.interview_done:
            st.success("Interview complete. See Step 2 below.")
        else:
            # Kick off interview with opening question if empty
            if not st.session_state.interview_messages:
                with st.spinner("Starting interview..."):
                    opening, draft, done = interview_step(
                        [{"role": "user", "content": "start"}],
                        sim_package,
                    )
                st.session_state.interview_messages = [
                    {"role": "user", "content": "start"},
                    {"role": "assistant", "content": opening},
                ]
                if draft:
                    st.session_state.current_draft = draft
                st.rerun()

            # User input for interview
            user_input = st.chat_input(
                "Describe your object...",
                key="interview_input",
            )
            if user_input:
                st.session_state.interview_messages.append({"role": "user", "content": user_input})
                with st.spinner("Agent is thinking..."):
                    response, draft, done = interview_step(
                        st.session_state.interview_messages,
                        sim_package,
                    )
                st.session_state.interview_messages.append({"role": "assistant", "content": response})
                if draft:
                    st.session_state.current_draft = draft
                if done:
                    st.session_state.interview_done = True
                st.rerun()

    # Self-heal: interview confirmed but no draft captured → synthesize one now
    # (covers sessions confirmed before the draft was parseable).
    if st.session_state.interview_done and not st.session_state.current_draft:
        from agent.object_agent import synthesize_object
        with st.spinner("Finalizing object details..."):
            recovered = synthesize_object(st.session_state.interview_messages, sim_package)
        if recovered:
            st.session_state.current_draft = recovered
            st.rerun()
        else:
            st.warning(
                "Couldn't finalize the object from the conversation. "
                "Please add one more detail in the chat above (e.g. a dimension or material)."
            )

    # ── Step 2: AI CAD solid generation ──────────────────────────────────────
    if st.session_state.interview_done and st.session_state.current_draft:
        with st.expander("Step 2 — Generate CAD Solid (AI)", expanded=st.session_state.cad_step_path is None):
            obj = st.session_state.current_draft
            st.markdown("**Confirmed object:**")
            _render_draft_card(obj)
            st.caption(
                "An AI engineer writes CadQuery code from your description and builds a "
                "real, watertight CAD solid (STEP) — ready for physics simulation. "
                "This is the actual shape that gets simulated."
            )

            if st.session_state.cad_step_path:
                if st.session_state.cad_fallback:
                    st.warning(
                        "The AI couldn't build a valid detailed solid for this description, "
                        f"so a clean parametric **{obj.shape_type}** was used instead. "
                        "Try rewording the shape, or edit the CadQuery code below."
                    )
                else:
                    st.success("CAD solid generated and ready to simulate.")
                # Realtime interactive 3D preview of the generated solid
                if st.session_state.cad_stl_path:
                    _render_3d_viewer(st.session_state.cad_stl_path)
                import os as _os
                c1, c2 = st.columns(2)
                with c1:
                    with open(st.session_state.cad_step_path, "rb") as f:
                        st.download_button(
                            "⬇ Download STEP (CAD)",
                            data=f.read(),
                            file_name=_os.path.basename(st.session_state.cad_step_path),
                            mime="application/step",
                            key="dl_step",
                        )
                with c2:
                    if st.session_state.cad_stl_path and _os.path.exists(st.session_state.cad_stl_path):
                        with open(st.session_state.cad_stl_path, "rb") as f:
                            st.download_button(
                                "⬇ Download STL (preview)",
                                data=f.read(),
                                file_name=_os.path.basename(st.session_state.cad_stl_path),
                                mime="model/stl",
                                key="dl_stl",
                            )
                with st.expander("View / edit the generated CadQuery code"):
                    edited = st.text_area(
                        "CadQuery code (millimeters; final solid assigned to `result`):",
                        value=st.session_state.cad_code,
                        height=260,
                        key="cad_code_area",
                    )
                    if st.button("Rebuild from edited code", key="rebuild_cad"):
                        from agent.geometry_agent import generate_step
                        try:
                            step, stl, code, fb = generate_step(obj, code=edited)
                            st.session_state.cad_step_path = step
                            st.session_state.cad_stl_path = stl
                            st.session_state.cad_code = code
                            st.session_state.cad_fallback = fb
                            st.session_state.cad_error = ""
                            st.rerun()
                        except Exception as e:
                            st.error(f"Rebuild failed: {e}")
                if st.button("Regenerate from description", key="regen_cad"):
                    st.session_state.cad_step_path = None
                    st.session_state.cad_code = ""
                    st.rerun()
            else:
                if st.session_state.cad_error:
                    st.error(st.session_state.cad_error)
                if st.button("Generate CAD Solid", type="primary", key="run_cad"):
                    from agent.geometry_agent import generate_step
                    with st.spinner("AI is designing the CAD solid (with self-repair)..."):
                        try:
                            step, stl, code, fb = generate_step(obj)
                            st.session_state.cad_step_path = step
                            st.session_state.cad_stl_path = stl
                            st.session_state.cad_code = code
                            st.session_state.cad_fallback = fb
                            st.session_state.cad_error = ""
                            st.rerun()
                        except Exception as e:
                            st.session_state.cad_error = f"CAD generation failed: {e}"
                            st.rerun()

        # ── Optional: Meshy realistic visual render (not used for physics) ──
        with st.expander("Optional — Realistic visual render (Meshy)", expanded=False):
            obj = st.session_state.current_draft
            st.caption(
                "Meshy makes a pretty textured render for presentations. It is NOT used "
                "for the simulation (decorative meshes can't be FEA-meshed)."
            )
            if not st.session_state.meshy_prompt:
                st.session_state.meshy_prompt = build_meshy_prompt(obj)
            st.session_state.meshy_prompt = st.text_area(
                "Meshy prompt:", value=st.session_state.meshy_prompt, height=70, key="meshy_prompt_area"
            )
            if st.session_state.meshy_result and st.session_state.meshy_result.status == "SUCCEEDED":
                if st.session_state.meshy_result.thumbnail_url:
                    st.image(st.session_state.meshy_result.thumbnail_url, width=300, caption="Meshy render")
            else:
                if st.button("Generate visual render", key="run_meshy"):
                    from services import meshy as meshy_svc
                    progress_bar = st.progress(0, text="Submitting to Meshy AI...")
                    try:
                        task_id = meshy_svc.create_task(
                            prompt=st.session_state.meshy_prompt, art_style="realistic", topology="quad",
                        )

                        def _on_progress(pct, status):
                            progress_bar.progress(pct, text=f"Meshy: {status} ({pct}%)")

                        result = meshy_svc.poll_task(task_id, progress_callback=_on_progress)
                        if result.status == "SUCCEEDED":
                            result = meshy_svc.download_model(result, dest_dir="meshy_models")
                        st.session_state.meshy_result = result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Meshy error: {e}")

    # ── Step 3: SimScale simulation ───────────────────────────────────────────
    # Step 3 unlocks once the AI CAD solid is ready (the geometry we simulate).
    cad_ready = bool(st.session_state.cad_step_path)
    if cad_ready:
        with st.expander("Step 3 — Run Physics Simulation", expanded=st.session_state.simulation_result is None):
            obj = st.session_state.current_draft

            st.markdown("**Solver:** SimScale")

            # ── Analysis type selection ──────────────────────────────────────
            ANALYSIS_TYPES = {
                "Static stress (loads → stress & displacement)": "static",
                "Modal / natural frequencies (vibration & resonance)": "frequency",
            }
            choice = st.selectbox(
                "Analysis type",
                list(ANALYSIS_TYPES.keys()),
                key="analysis_choice",
                help="Static: how the part deforms and where stress concentrates under a load. "
                     "Modal: the part's natural vibration frequencies (resonance risk).",
            )
            analysis_type = ANALYSIS_TYPES[choice]

            # ── Geometry source: the AI-generated STEP, or override with your own CAD ──
            cad_file = st.file_uploader(
                "Optional: upload your own CAD file (STEP / IGES) to override the AI geometry",
                type=["step", "stp", "iges", "igs"],
                key="cad_upload",
                help="By default the AI-generated CAD solid from Step 2 is simulated. "
                     "Upload your own STEP/IGES here to use that instead.",
            )
            if cad_file is not None:
                st.success(f"Will simulate your uploaded CAD: **{cad_file.name}**")
            else:
                st.info("Will simulate the **AI-generated CAD solid** from Step 2.")

            if st.session_state.simulation_result:
                sim_res: SimulationResult = st.session_state.simulation_result
                st.success("Simulation complete!")
                if sim_res.result_viewer_url:
                    st.markdown(f"### [🔎 Open the 3D results in the SimScale Viewer]({sim_res.result_viewer_url})")
                st.caption(sim_res.summary)

                # ── Numeric results (on-demand extraction) ────────────────────
                has_numbers = sim_res.numbers_extracted
                if has_numbers:
                    cols = st.columns(3)
                    if sim_res.max_von_mises_stress_pa is not None:
                        mpa = sim_res.max_von_mises_stress_pa / 1e6
                        cols[0].metric("Max Von Mises Stress", f"{mpa:.3g} MPa")
                    if sim_res.max_displacement_m is not None:
                        cols[1].metric("Max Displacement", f"{sim_res.max_displacement_m*1000:.3g} mm")
                    if sim_res.min_safety_factor is not None:
                        cols[2].metric("Safety Factor", f"{sim_res.min_safety_factor:.2f}")
                    if sim_res.natural_frequencies_hz:
                        cols[2].metric("Fundamental Frequency", f"{sim_res.natural_frequencies_hz[0]:.1f} Hz")
                else:
                    st.markdown(
                        "Exact peak values aren't pulled in yet. Click below to fetch the real "
                        "numbers from SimScale (exports + parses the result — ~1–3 min)."
                    )
                    if st.button("📊 Extract numbers", key="extract_btn"):
                        from services.simulation import get_adapter
                        adapter = get_adapter("simscale")
                        try:
                            with st.spinner("Exporting + parsing results from SimScale..."):
                                nums = adapter.extract_numbers(
                                    sim_res.project_id, sim_res.simulation_id, sim_res.run_id,
                                    material_properties=obj.material_properties,
                                )
                            sim_res.max_von_mises_stress_pa = nums.get("max_von_mises_stress_pa")
                            sim_res.max_displacement_m = nums.get("max_displacement_m")
                            sim_res.min_safety_factor = nums.get("min_safety_factor")
                            sim_res.numbers_extracted = True
                            sim_res.interpretation = ""  # re-interpret with real numbers
                            st.session_state.simulation_result = sim_res
                            st.rerun()
                        except Exception as e:
                            st.warning(
                                f"Couldn't extract numbers ({e}). The full results are still "
                                "available in the SimScale viewer above."
                            )

                # ── AI interpretation ────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 🧠 Engineering Interpretation")
                if sim_res.interpretation:
                    st.markdown(sim_res.interpretation)
                else:
                    if st.button("Interpret results", type="primary", key="interpret_btn"):
                        from agent.results_agent import interpret_results
                        with st.spinner("Interpreting results..."):
                            interp = interpret_results(obj, sim_res, sim_res.analysis_type)
                        sim_res.interpretation = interp
                        st.session_state.simulation_result = sim_res
                        st.rerun()

                if st.button("Run another analysis", key="rerun_sim"):
                    st.session_state.simulation_result = None
                    st.rerun()
            else:
                if st.button("Run Simulation", type="primary", key="run_simscale"):
                    from services.simulation import get_adapter
                    import os, tempfile
                    adapter = get_adapter("simscale")
                    status_text = st.empty()

                    # Use an uploaded CAD if provided, else the AI-generated STEP.
                    if cad_file is not None:
                        suffix = os.path.splitext(cad_file.name)[1] or ".step"
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        tmp.write(cad_file.getvalue())
                        tmp.close()
                        cad_path = tmp.name
                    else:
                        cad_path = st.session_state.cad_step_path

                    try:
                        def _on_sim_progress(status):
                            status_text.markdown(f"*Solver status: `{status}`*")

                        with st.spinner(f"Running {analysis_type} analysis on SimScale (a few minutes)..."):
                            sim_res = adapter.run_full(
                                obj_path="",
                                physical_object=obj,
                                sim_package=sim_package,
                                progress_callback=_on_sim_progress,
                                cad_path=cad_path,
                                analysis_type=analysis_type,
                            )
                        st.session_state.simulation_result = sim_res
                        status_text.empty()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Simulation error: {e}")


# ── Show past messages ─────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "package" in msg:
            with st.expander("📦 Simulation Package"):
                _render_package(msg["package"], scenario=msg.get("scenario"))


# ── Chat input ─────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask a question, cluster papers, or build a simulation...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_text = st.empty()
        response_placeholder = st.empty()

        graph = create_agent_graph()
        initial_state = {
            "raw_scenario": prompt,
            "chat_history": list(st.session_state.chat_history),
            "session_memory": dict(st.session_state.session_memory),
            "intent": "",
            "parsed_scenario": None,
            "sub_queries": [],
            "retrieved_docs": {},
            "retrieve_rounds": {},
            "extracted_parameters": [],
            "validation_notes": "",
            "simulation_package": None,
            "final_answer": "",
            "errors": [],
            "ui_placeholder": response_placeholder,
        }

        final_state = None
        for event in graph.stream(initial_state):
            for node_name, state in event.items():
                if node_name not in ["qa", "explore"]:
                    status_text.markdown(f"*(Agent is running: `{node_name.upper()}`...)*")
                final_state = state

        status_text.empty()

        if not final_state:
            st.error("Agent failed to process the request.")
            st.stop()

        intent = final_state.get("intent", "qa")

        if intent in ["qa", "explore"]:
            ans = final_state.get("final_answer", "No answer generated.")
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": ans[:800]})
            st.session_state.chat_history = st.session_state.chat_history[-6:]
            _update_memory_qa(prompt, ans)
            _update_memory_domains(ans)

        elif intent == "simulate":
            pkg = final_state.get("simulation_package")
            scenario = final_state.get("parsed_scenario")
            validation_notes = final_state.get("validation_notes", "")

            if pkg:
                st.success("✅ Simulation Package Generated!")
                _render_package(pkg, scenario=scenario, validation_notes=validation_notes)

                # Store for Phase-2 access; reset the 3D pipeline state
                st.session_state.active_sim_package = pkg
                st.session_state.active_scenario = scenario
                st.session_state.interview_messages = []
                st.session_state.current_draft = None
                st.session_state.interview_done = False
                st.session_state.meshy_result = None
                st.session_state.simulation_result = None
                st.session_state.meshy_prompt = ""
                st.session_state.cad_code = ""
                st.session_state.cad_step_path = None
                st.session_state.cad_stl_path = None
                st.session_state.cad_error = ""
                st.session_state.cad_fallback = False

                _update_memory_simulate(scenario)
                summary = "✅ Generated a targeted simulation package based on the literature."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": summary,
                    "package": pkg,
                    "scenario": scenario,
                })
            else:
                errs = final_state.get("errors", [])
                st.error("Failed to generate simulation package.")
                if errs:
                    st.write(errs)


# ── Phase-2: 3D section (shown when a sim package is active) ──────────────────
if st.session_state.active_sim_package:
    _render_3d_section(
        st.session_state.active_sim_package,
        st.session_state.active_scenario,
    )
