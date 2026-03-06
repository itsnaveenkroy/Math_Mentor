"""
Math Mentor - Streamlit frontend.
Handles input (text/image/audio), runs the agent pipeline,
and displays results with full transparency.
"""

import sys
import os
import json

# make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from agents.orchestrator import run_pipeline
from input_handlers.text_handler import process_text_input
from input_handlers.image_handler import extract_text_from_image
from input_handlers.audio_handler import transcribe_audio
from memory.memory_store import store_feedback, get_topic_stats, store_correction


# -- page config --
st.set_page_config(
    page_title="Math Mentor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("Math Mentor")
    st.caption("Multimodal math problem solver with step-by-step explanations")

    # -- sidebar --
    with st.sidebar:
        st.header("Input Options")
        input_mode = st.radio(
            "How do you want to input your problem?",
            ["Type it", "Upload image", "Upload audio"],
            index=0,
        )

        st.divider()
        st.header("About")
        st.write(
            "This tool uses a multi-agent pipeline to parse, route, solve, "
            "verify, and explain math problems. It pulls from a curated "
            "knowledge base and remembers your past problems."
        )

        # show topic stats if we have any
        try:
            stats = get_topic_stats()
            if stats:
                st.divider()
                st.header("Your History")
                for topic, s in stats.items():
                    avg = s["avg_confidence"]
                    count = s["count"]
                    st.write(f"**{topic}**: {count} problems, avg confidence {avg:.0%}")
        except Exception:
            pass

    # -- main area: input --
    raw_text = ""
    input_source = "text"
    needs_review = False
    ocr_message = None

    if input_mode == "Type it":
        raw_text = st.text_area(
            "Enter your math problem:",
            height=120,
            placeholder="e.g. Solve x^2 - 5x + 6 = 0",
        )
        input_source = "text"

    elif input_mode == "Upload image":
        uploaded = st.file_uploader(
            "Upload an image of the math problem",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
        )
        if uploaded:
            st.image(uploaded, caption="Uploaded image", width=400)
            with st.spinner("Running OCR..."):
                ocr_result = extract_text_from_image(uploaded.getvalue())

            raw_text = ocr_result.get("text", "")
            confidence = ocr_result.get("confidence", 0)
            input_source = "image"

            if ocr_result.get("raw_text") and ocr_result["raw_text"] != raw_text:
                with st.expander("OCR details"):
                    st.write(f"**Raw OCR output:** {ocr_result['raw_text']}")
                    st.write(f"**After math cleanup:** {raw_text}")
                    st.write(f"**Confidence:** {confidence:.0%}")

            if ocr_result.get("message"):
                ocr_message = ocr_result["message"]

            if ocr_result.get("needs_hitl"):
                needs_review = True

    elif input_mode == "Upload audio":
        audio_method = st.radio(
            "How do you want to provide audio?",
            ["Upload a file", "Record with microphone"],
            index=0,
            key="audio_method",
        )

        audio_bytes = None
        audio_filename = "audio.wav"

        if audio_method == "Upload a file":
            uploaded = st.file_uploader(
                "Upload an audio file",
                type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
            )
            if uploaded:
                st.audio(uploaded)
                audio_bytes = uploaded.getvalue()
                audio_filename = uploaded.name

        else:
            recorded = st.audio_input("Record your math question:")
            if recorded:
                audio_bytes = recorded.getvalue()
                # use actual name/type if available; fall back to .wav
                audio_filename = getattr(recorded, 'name', None) or "recording.wav"
                mime = getattr(recorded, 'type', '')
                st.caption(f"Audio: {len(audio_bytes)} bytes, name=`{audio_filename}`, mime=`{mime}`, "
                           f"magic=`{audio_bytes[:12].hex() if audio_bytes else ''}`")

        if audio_bytes:
            with st.spinner("Transcribing audio..."):
                audio_result = transcribe_audio(audio_bytes, filename=audio_filename)

            raw_text = audio_result.get("text", "")
            input_source = "audio"

            if audio_result.get("raw_text"):
                with st.expander("Transcription details"):
                    st.write(f"**Raw transcript:** {audio_result['raw_text']}")
                    st.write(f"**After cleanup:** {raw_text}")

            if audio_result.get("message"):
                st.info(audio_result["message"])

            if audio_result.get("needs_hitl"):
                needs_review = True

    # -- human-in-the-loop: let user edit extracted text --
    original_extracted = raw_text  # save original for correction learning
    if raw_text and (needs_review or input_source != "text"):
        if ocr_message:
            st.warning(ocr_message)
        st.write("**Extracted text** (edit if needed):")
        raw_text = st.text_area("Review and fix:", value=raw_text, height=100, key="hitl_edit")

        # learn from user corrections (OCR/audio self-learning)
        if raw_text != original_extracted and raw_text.strip():
            try:
                store_correction(original_extracted, raw_text, source=input_source)
            except Exception:
                pass

    # -- solve button --
    if st.button("Solve", type="primary", disabled=not raw_text.strip()):
        _run_and_display(raw_text, input_source)


def _run_and_display(raw_text, input_source):
    """Run the pipeline and show results."""

    with st.spinner("Working on it..."):
        result = run_pipeline(raw_text, input_source=input_source)

    if result.get("error"):
        st.error(f"Something went wrong: {result['error']}")

    # -- HITL: show clarification warning if parser flagged ambiguity --
    if result.get("clarification_reason"):
        st.warning(f"⚠️ Possible ambiguity: {result['clarification_reason']}. "
                   "Please review the answer carefully or edit your input above and re-solve.")

    # -- answer section --
    st.header("Answer")

    confidence = result.get("confidence", 0)
    if confidence >= 0.8:
        conf_label = "High confidence"
    elif confidence >= 0.5:
        conf_label = "Medium confidence"
    else:
        conf_label = "Low confidence - review recommended"

    col1, col2 = st.columns([3, 1])
    with col1:
        answer = result.get("answer", "No answer generated")
        st.markdown(f"**{answer}**")
    with col2:
        st.metric("Confidence", f"{confidence:.0%}")
        st.caption(conf_label)

    # HITL: offer re-solve when confidence is low or verifier flagged issues
    if result.get("needs_hitl") and confidence < 0.6:
        st.warning("Low confidence — the system is not sure about this answer. "
                   "You can edit your input above and click Solve again, or request a re-check.")
        if st.button("🔄 Request Re-check", key="recheck_btn"):
            with st.spinner("Re-checking..."):
                result = run_pipeline(raw_text, input_source=input_source)
            st.rerun()

    if result.get("method"):
        st.write(f"**Method used:** {result['method']}")

    # -- explanation --
    if result.get("explanation"):
        st.header("Explanation")
        st.markdown(result["explanation"])

    # -- steps --
    steps = result.get("steps", [])
    if steps:
        with st.expander(f"Solution steps ({len(steps)} steps)"):
            for s in steps:
                step_num = s.get("step", "?")
                desc = s.get("description", "")
                work = s.get("work", "")
                st.markdown(f"**Step {step_num}:** {desc}")
                if work:
                    st.code(work)

    # -- verification --
    issues = result.get("verification_issues", [])
    if issues:
        with st.expander("Verification notes"):
            for issue in issues:
                st.write(f"- {issue}")

    # -- concepts and tips --
    concepts = result.get("key_concepts", [])
    mistakes = result.get("common_mistakes", [])
    related = result.get("related_topics", [])

    if concepts or mistakes or related:
        with st.expander("Concepts and tips"):
            if concepts:
                st.write("**Key concepts:**")
                for c in concepts:
                    st.write(f"- {c}")
            if mistakes:
                st.write("**Watch out for:**")
                for m in mistakes:
                    st.write(f"- {m}")
            if related:
                st.write("**Related topics to review:**")
                for r in related:
                    st.write(f"- {r}")

    # -- similar past problems --
    similar = result.get("similar_problems", [])
    if similar:
        with st.expander("Similar problems you've asked before"):
            for sp in similar:
                st.write(f"- (similarity: {sp['similarity']:.0%}) {sp['text'][:150]}...")

    # -- knowledge base context --
    context = result.get("context_used", [])
    if context:
        with st.expander("Knowledge base context used"):
            for c in context:
                st.markdown(f"**{c.get('source', '')}** (relevance: {c.get('similarity', 0):.0%})")
                st.caption(c.get("text", "")[:300])
                st.divider()

    # -- agent trace (for transparency) --
    trace = result.get("agent_trace", [])
    if trace:
        with st.expander("Agent pipeline trace"):
            for t in trace:
                agent = t.get("agent", "?")
                elapsed = t.get("time", 0)
                st.markdown(f"**{agent}** ({elapsed}s)")
                out = t.get("output", {})
                if isinstance(out, dict):
                    st.json(out)
                else:
                    st.write(str(out)[:500])

    timing = result.get("timing", {})
    if timing.get("total"):
        st.caption(f"Total time: {timing['total']}s")

    # -- feedback section --
    st.divider()
    st.subheader("Was this helpful?")
    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)

    with feedback_col1:
        if st.button("Yes, correct"):
            store_feedback(raw_text, "helpful")
            st.success("Thanks for the feedback.")

    with feedback_col2:
        if st.button("Wrong answer"):
            store_feedback(raw_text, "wrong")
            st.warning("Noted. We'll try to improve.")

    with feedback_col3:
        if st.button("Confusing explanation"):
            store_feedback(raw_text, "confusing")
            st.info("Got it. We'll work on clarity.")

    comment = st.text_input("Any other feedback?", key="feedback_comment")
    if comment and st.button("Submit comment"):
        store_feedback(raw_text, "comment", comment=comment)
        st.success("Comment saved.")


if __name__ == "__main__":
    main()
