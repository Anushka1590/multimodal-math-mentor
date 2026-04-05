import os
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from agents import run_pipeline
from input_handlers import extract_text_from_image, extract_text_from_audio
from memory import add_memory, retrieve_similar, update_feedback, get_memory_summary

load_dotenv()

if not os.path.exists("data/langchain_faiss_index"):
    from rag_pipeline import build_index
    build_index()

st.set_page_config(page_title="Math Mentor", page_icon="🧮", layout="wide")

st.markdown("""
<style>
.agent-box {
    background-color: #1e1e2e;
    border-left: 4px solid #7c3aed;
    padding: 10px 15px;
    border-radius: 6px;
    margin: 6px 0;
    font-family: monospace;
    font-size: 0.85rem;
    color: #e2e8f0;
}
.source-box {
    background-color: #1a2744;
    border-left: 4px solid #2563eb;
    padding: 8px 12px;
    border-radius: 6px;
    margin: 4px 0;
    font-size: 0.85rem;
    color: #e2e8f0;
}
.memory-box {
    background-color: #1a2e1a;
    border-left: 4px solid #16a34a;
    padding: 8px 12px;
    border-radius: 6px;
    margin: 4px 0;
    font-size: 0.85rem;
    color: #e2e8f0;
}
.hitl-box {
    background-color: #2e1a1a;
    border-left: 4px solid #dc2626;
    padding: 10px 15px;
    border-radius: 6px;
    margin: 6px 0;
    color: #e2e8f0;
}
.confidence-high { color: #16a34a; font-weight: bold; }
.confidence-med  { color: #d97706; font-weight: bold; }
.confidence-low  { color: #dc2626; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

for key, default in {
    "result":              None,
    "memory_id":           None,
    "raw_input":           "",
    "extracted_text":      "",
    "input_confirmed":     False,
    "solving":             False,
    "show_correction_box": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🧮 Math Mentor")
st.caption("JEE-level Math Problem Solver - RAG + Multi-Agent + Memory")
st.divider()

left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.subheader("📥 Input")

    prev_mode  = st.session_state.get("input_mode", None)
    input_mode = st.radio("Select input mode:", ["✏️ Text", "🖼️ Image", "🎙️ Audio"], horizontal=True)

    if prev_mode and prev_mode != input_mode:
        st.session_state.result          = None
        st.session_state.memory_id       = None
        st.session_state.raw_input       = ""
        st.session_state.extracted_text  = ""
        st.session_state.input_confirmed = False

    st.session_state.input_mode = input_mode

    if input_mode == "✏️ Text":
        text_input = st.text_area(
            "Type your math problem:",
            placeholder="e.g. If alpha and beta are roots of x^2 - 5x + 6 = 0, find alpha^2 + beta^2",
            height=120
        )
        if text_input:
            if text_input != st.session_state.get("last_input", ""):
                st.session_state.result    = None
                st.session_state.memory_id = None
            st.session_state.last_input      = text_input
            st.session_state.raw_input       = text_input
            st.session_state.extracted_text  = text_input
            st.session_state.input_confirmed = True

    elif input_mode == "🖼️ Image":
        uploaded_image = st.file_uploader("Upload image of math problem (JPG/PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            st.image(Image.open(uploaded_image), caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Extract Text from Image"):
                with st.spinner("Extracting math problem from image..."):
                    ocr_result = extract_text_from_image(uploaded_image.getvalue())

                if ocr_result["success"]:
                    st.session_state.extracted_text = ocr_result["extracted_text"]
                    if ocr_result["confidence"] < 0.8:
                        st.warning(f"⚠️ Low confidence ({ocr_result['confidence']:.0%}) — please review the extracted text.")
                    else:
                        st.success(f"✅ Extracted with {ocr_result['confidence']:.0%} confidence")
                else:
                    st.error(f"❌ Extraction failed: {ocr_result['error']}")

            if st.session_state.extracted_text:
                st.session_state.extracted_text = st.text_area(
                    "📝 Extracted text (edit if needed):",
                    value=st.session_state.extracted_text,
                    height=100
                )
                if st.button("✅ Confirm Extracted Text"):
                    st.session_state.raw_input       = st.session_state.extracted_text
                    st.session_state.input_confirmed = True
                    st.success("Text confirmed!")

    elif input_mode == "🎙️ Audio":
        uploaded_audio = st.file_uploader("Upload audio file (WAV/MP3/M4A):", type=["wav", "mp3", "m4a", "ogg"])
        if uploaded_audio:
            st.audio(uploaded_audio)

            if st.button("🎙️ Transcribe Audio"):
                with st.spinner("Transcribing audio..."):
                    asr_result = extract_text_from_audio(
                        uploaded_audio.getvalue(),
                        uploaded_audio.name.split(".")[-1]
                    )

                if asr_result["success"]:
                    st.session_state.extracted_text = asr_result["transcript"]
                    if asr_result["confidence"] < 0.8:
                        st.warning("⚠️ Low confidence — please review the transcript.")
                    else:
                        st.success("✅ Transcribed successfully")
                else:
                    st.error(f"❌ Transcription failed: {asr_result['error']}")

            if st.session_state.extracted_text:
                st.session_state.extracted_text = st.text_area(
                    "📝 Transcript (edit if needed):",
                    value=st.session_state.extracted_text,
                    height=100
                )
                if st.button("✅ Confirm Transcript"):
                    st.session_state.raw_input       = st.session_state.extracted_text
                    st.session_state.input_confirmed = True
                    st.success("Transcript confirmed!")

    st.divider()

    if st.session_state.input_confirmed and st.session_state.raw_input:
        similar     = retrieve_similar(st.session_state.raw_input, top_k=2)
        successful  = similar.get("successful", [])
        corrections = similar.get("corrections", [])

        if successful or corrections:
            st.subheader("🧠 Similar Problems from Memory")
            for s in successful:
                st.markdown(f"""<div class='memory-box'>
                    <b>#{s['id']} [{s['topic'].upper()}]</b> {s['problem_text'][:80]}...<br>
                    <small>✅ Correct | Confidence: {s['confidence']:.0%}</small>
                </div>""", unsafe_allow_html=True)
            for s in corrections:
                st.markdown(f"""<div class='hitl-box'>
                    <b>#{s['id']} [{s['topic'].upper()}]</b> {s['problem_text'][:80]}...<br>
                    <small>❌ Was wrong — Correction: {s.get('user_correction', 'N/A')[:60]}</small>
                </div>""", unsafe_allow_html=True)

    st.divider()
    solve_clicked = st.button(
        "🚀 Solve Problem",
        disabled=not st.session_state.input_confirmed,
        use_container_width=True,
        type="primary"
    )

    if solve_clicked and st.session_state.raw_input:
        with st.spinner("Running agents... this may take 10-20 seconds"):
            result     = run_pipeline(st.session_state.raw_input)
            input_type = (
                "image" if input_mode == "🖼️ Image"
                else "audio" if input_mode == "🎙️ Audio"
                else "text"
            )
            mem_id = add_memory(
                problem_text   = st.session_state.raw_input,
                parsed_problem = result["parsed_problem"],
                solution       = result["solution_data"]["solution"],
                explanation    = result["explanation"],
                verification   = result["verification"],
                input_type     = input_type
            )
            st.session_state.result    = result
            st.session_state.memory_id = mem_id
        st.rerun()

with right_col:
    if st.session_state.result is None:
        st.info("👈 Enter a problem and click **Solve Problem** to see results here.")
    else:
        result       = st.session_state.result
        parsed       = result["parsed_problem"]
        routing      = result["routing_info"]
        solution     = result["solution_data"]
        verification = result["verification"]
        explanation  = result["explanation"]

        memory_used = result.get("memory_used", [])
        if memory_used:
            st.success(f"🧠 Memory reused {len(memory_used)} verified past solution(s) to guide this answer")
            with st.expander("📝 Past solutions used", expanded=False):
                for m in memory_used:
                    st.markdown(f"""<div class='memory-box'>
                        <b>#{m['id']} [{m['topic'].upper()}]</b> {m['problem_text'][:80]}...<br>
                        <small>✅ Correct | Similarity score: {m['similarity_score']:.3f}</small>
                    </div>""", unsafe_allow_html=True)

        with st.expander("🔍 Agent Trace", expanded=False):
            for name, detail in [
                ("1 Parser Agent",        f"Topic: {parsed['topic']} | Needs clarification: {parsed['needs_clarification']}"),
                ("2 Intent Router Agent", f"Subtopic: {routing['subtopic']} | Difficulty: {routing['difficulty']}"),
                ("3 Solver Agent",        f"Sources: {', '.join(solution['sources_used'])}"),
                ("4 Verifier Agent",      f"Correct: {verification['is_correct']} | Confidence: {verification['confidence']:.0%}"),
                ("5 Explainer Agent",     "Generated student-friendly explanation"),
            ]:
                st.markdown(f"<div class='agent-box'>✅ <b>Agent {name}</b><br>{detail}</div>", unsafe_allow_html=True)

        with st.expander("📚 Retrieved Knowledge Base Sources", expanded=False):
            if solution["context_docs"]:
                for doc in solution["context_docs"]:
                    st.markdown(
                        f"<div class='source-box'><b>[{doc['topic'].upper()}] {doc['title']}</b></div>",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("Answer served from memory — no retrieval needed.")

        conf  = verification["confidence"]
        level = "high" if conf >= 0.85 else "med" if conf >= 0.7 else "low"
        label = "High ✅" if conf >= 0.85 else "Medium ⚠️" if conf >= 0.7 else "Low ❌"
        st.markdown(
            f"**Confidence:** <span class='confidence-{level}'>{conf:.0%} — {label}</span>",
            unsafe_allow_html=True
        )

        if verification.get("needs_human_review") or parsed.get("needs_clarification") or conf < 0.8:
            st.markdown("""<div class='hitl-box'>
                ⚠️ <b>Human Review Requested</b><br>
                The system is not fully confident. Please review the solution carefully.
            </div>""", unsafe_allow_html=True)

            if parsed.get("needs_clarification"):
                st.warning(f"Clarification needed: {parsed.get('clarification_reason', '')}")
            if verification.get("issues_found"):
                st.warning("Issues found: " + ", ".join(verification["issues_found"]))

        with st.expander("🧩 Parsed Problem", expanded=False):
            st.json(parsed)

        st.subheader("📖 Solution & Explanation")
        st.markdown(explanation)

        st.divider()
        st.subheader("💬 Was this solution correct?")
        fb_col1, fb_col2 = st.columns(2)

        with fb_col1:
            if st.button("✅ Correct", use_container_width=True):
                if st.session_state.memory_id:
                    update_feedback(st.session_state.memory_id, "correct")
                st.success("Marked as correct ✅")

        with fb_col2:
            if st.button("❌ Incorrect", use_container_width=True):
                st.session_state.show_correction_box = True

        if st.session_state.get("show_correction_box", False):
            st.warning("Please provide the correct answer so the system can learn from this mistake.")
            correction = st.text_input(
                "What is the correct answer?",
                placeholder="e.g. x = 1 and x = -2, so 2 real solutions",
                key="correction_input"
            )
            if st.button("💾 Save Correction", type="primary"):
                if correction.strip():
                    if st.session_state.memory_id:
                        update_feedback(
                            st.session_state.memory_id,
                            "incorrect",
                            user_correction=correction.strip()
                        )
                    st.error("Marked as incorrect ❌ — correction saved.")
                    st.session_state.show_correction_box = False
                else:
                    st.warning("Please type the correct answer before saving.")

with st.sidebar:
    st.header("🧠 Memory Summary")
    summary = get_memory_summary()
    st.metric("Total Problems Solved", summary["total"])
    st.metric("Marked Correct",        summary["correct"])
    st.metric("Marked Incorrect",      summary["incorrect"])
    st.metric("Pending Feedback",      summary["pending"])

    if summary.get("topics"):
        st.caption("Topics: " + ", ".join(summary["topics"]))

    st.divider()
    st.caption("Math Mentor v1.0")
    st.caption("Powered by Groq + LLaMA 3.3")