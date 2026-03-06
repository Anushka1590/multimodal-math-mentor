import os
import json
import streamlit as st
from PIL import Image
import io
from dotenv import load_dotenv

from agents import run_pipeline, parser_agent
from input_handlers import extract_text_from_image, extract_text_from_audio
from memory import add_memory, retrieve_similar, update_feedback, get_memory_summary

# Build FAISS index on startup if it doesn't exist
import os
if not os.path.exists("data/faiss_index.bin"):
    from rag_pipeline import build_index
    build_index()

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Math Mentor",
    page_icon="🧮",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
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


# ── Session State Init ────────────────────────────────────────────────────────
for key, default in {
    "result":         None,
    "memory_id":      None,
    "raw_input":      "",
    "extracted_text": "",
    "input_confirmed": False,
    "solving":        False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧮 Math Mentor")
st.caption("JEE-level Math Problem Solver - RAG + Multi-Agent + Memory")
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.2, 1], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Input + Controls
# ════════════════════════════════════════════════════════════════════════════
with left_col:
    st.subheader("📥 Input")

    input_mode = st.radio(
        "Select input mode:",
        ["✏️ Text", "🖼️ Image", "🎙️ Audio"],
        horizontal=True
    )

    # ── TEXT INPUT ────────────────────────────────────────────────────────────
    if input_mode == "✏️ Text":
        text_input = st.text_area(
            "Type your math problem:",
            placeholder="e.g. If alpha and beta are roots of x^2 - 5x + 6 = 0, find alpha^2 + beta^2",
            height=120
        )
        if text_input:
            st.session_state.raw_input      = text_input
            st.session_state.extracted_text = text_input
            st.session_state.input_confirmed = True

    # ── IMAGE INPUT ───────────────────────────────────────────────────────────
    elif input_mode == "🖼️ Image":
        uploaded_image = st.file_uploader(
            "Upload image of math problem (JPG/PNG):",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Extract Text from Image"):
                with st.spinner("Extracting math problem from image..."):
                    image_bytes = uploaded_image.getvalue()
                    ocr_result  = extract_text_from_image(image_bytes)

                if ocr_result["success"]:
                    st.session_state.extracted_text = ocr_result["extracted_text"]
                    confidence = ocr_result["confidence"]
                    if confidence < 0.8:
                        st.warning(f"⚠️ Low confidence ({confidence:.0%}) — please review and edit the extracted text.")
                    else:
                        st.success(f"✅ Extracted with {confidence:.0%} confidence")
                else:
                    st.error(f"❌ Extraction failed: {ocr_result['error']}")

            # Show editable extracted text
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

    # ── AUDIO INPUT ───────────────────────────────────────────────────────────
    elif input_mode == "🎙️ Audio":
        uploaded_audio = st.file_uploader(
            "Upload audio file of your math question (WAV/MP3/M4A):",
            type=["wav", "mp3", "m4a", "ogg"]
        )
        if uploaded_audio:
            st.audio(uploaded_audio)

            if st.button("🎙️ Transcribe Audio"):
                with st.spinner("Transcribing audio..."):
                    audio_bytes = uploaded_audio.getvalue()
                    ext         = uploaded_audio.name.split(".")[-1]
                    asr_result  = extract_text_from_audio(audio_bytes, ext)

                if asr_result["success"]:
                    st.session_state.extracted_text = asr_result["transcript"]
                    confidence = asr_result["confidence"]
                    if confidence < 0.8:
                        st.warning(f"⚠️ Low confidence — please review transcript.")
                    else:
                        st.success(f"✅ Transcribed successfully")
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

    # ── SIMILAR PROBLEMS FROM MEMORY ──────────────────────────────────────────
    if st.session_state.input_confirmed and st.session_state.raw_input:
        similar = retrieve_similar(st.session_state.raw_input, top_k=2)
        if similar:
            st.subheader("🧠 Similar Problems from Memory")
            for s in similar:
                st.markdown(f"""<div class='memory-box'>
                    <b>#{s['id']} [{s['topic'].upper()}]</b> {s['problem_text'][:80]}...<br>
                    <small>Feedback: {s['feedback']} | Confidence: {s['confidence']:.0%}</small>
                </div>""", unsafe_allow_html=True)

    # ── SOLVE BUTTON ──────────────────────────────────────────────────────────
    st.divider()
    solve_clicked = st.button(
        "🚀 Solve Problem",
        disabled=not st.session_state.input_confirmed,
        use_container_width=True,
        type="primary"
    )

    if solve_clicked and st.session_state.raw_input:
        with st.spinner("Running agents... this may take 10-20 seconds"):
            result = run_pipeline(st.session_state.raw_input)
            st.session_state.result = result

            # Auto-save to memory
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
            st.session_state.memory_id = mem_id
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Results
# ════════════════════════════════════════════════════════════════════════════
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

        # ── AGENT TRACE ───────────────────────────────────────────────────────
        with st.expander("🔍 Agent Trace", expanded=False):
            agents_trace = [
                ("1 Parser Agent",        f"Topic: {parsed['topic']} | Needs clarification: {parsed['needs_clarification']}"),
                ("2 Intent Router Agent", f"Subtopic: {routing['subtopic']} | Difficulty: {routing['difficulty']}"),
                ("3 Solver Agent",        f"Sources used: {', '.join(solution['sources_used'])}"),
                ("4 Verifier Agent",      f"Correct: {verification['is_correct']} | Confidence: {verification['confidence']:.0%}"),
                ("5 Explainer Agent",     "Generated student-friendly explanation"),
            ]
            for name, detail in agents_trace:
                st.markdown(
                    f"<div class='agent-box'>✅ <b>Agent {name}</b><br>{detail}</div>",
                    unsafe_allow_html=True
                )

        # ── RETRIEVED CONTEXT ─────────────────────────────────────────────────
        with st.expander("📚 Retrieved Knowledge Base Sources", expanded=False):
            for doc in solution["context_docs"]:
                st.markdown(
                    f"<div class='source-box'><b>[{doc['topic'].upper()}] {doc['title']}</b>"
                    f"<br><small>Relevance score: {doc['score']:.2f}</small></div>",
                    unsafe_allow_html=True
                )

        # ── CONFIDENCE INDICATOR ──────────────────────────────────────────────
        conf  = verification["confidence"]
        level = "high" if conf >= 0.85 else "med" if conf >= 0.7 else "low"
        label = "High ✅" if conf >= 0.85 else "Medium ⚠️" if conf >= 0.7 else "Low ❌"
        st.markdown(
            f"**Confidence:** <span class='confidence-{level}'>{conf:.0%} — {label}</span>",
            unsafe_allow_html=True
        )

        # ── HITL TRIGGER ──────────────────────────────────────────────────────
        if (
            verification.get("needs_human_review")
            or parsed.get("needs_clarification")
            or conf < 0.8
        ):
            st.markdown("""<div class='hitl-box'>
                ⚠️ <b>Human Review Requested</b><br>
                The system is not fully confident. Please review the solution below carefully.
            </div>""", unsafe_allow_html=True)

            if parsed.get("needs_clarification"):
                st.warning(f"Clarification needed: {parsed.get('clarification_reason', '')}")

            if verification.get("issues_found"):
                st.warning("Issues found: " + ", ".join(verification["issues_found"]))

        # ── PARSED PROBLEM ────────────────────────────────────────────────────
        with st.expander("🧩 Parsed Problem", expanded=False):
            st.json(parsed)

        # ── EXPLANATION ───────────────────────────────────────────────────────
        st.subheader("📖 Solution & Explanation")
        st.markdown(explanation)

        # ── FEEDBACK ─────────────────────────────────────────────────────────
        st.divider()
        st.subheader("💬 Was this solution correct?")
        fb_col1, fb_col2 = st.columns(2)

        with fb_col1:
            if st.button("✅ Correct", use_container_width=True):
                if st.session_state.memory_id:
                    update_feedback(st.session_state.memory_id, "correct")
                st.success("Thanks! Marked as correct ✅")

        with fb_col2:
            if st.button("❌ Incorrect", use_container_width=True):
                if st.session_state.memory_id:
                    update_feedback(st.session_state.memory_id, "incorrect")
                st.error("Sorry! Marked as incorrect ❌")

        comment = st.text_input("Optional comment:")
        if comment:
            st.caption(f"Comment noted: {comment}")


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Memory Summary
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🧠 Memory Summary")
    summary = get_memory_summary()
    st.metric("Total Problems Solved", summary["total"])
    st.metric("Marked Correct",        summary["correct"])
    st.metric("Marked Incorrect",      summary["incorrect"])
    st.metric("Pending Feedback",      summary["pending"])

    if summary.get("topics"):
        st.caption("Topics covered: " + ", ".join(summary["topics"]))

    st.divider()
    st.caption("Math Mentor v1.0")
    st.caption("Powered by Groq + LLaMA 3.3")