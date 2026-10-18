"""AceNotes - clean handwritten notes and turn textbook photos into quizzes.
Powered by Gemma 4 (open-weight model) through the Gemini API.
"""
import io
import json
import os
import re

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageOps

st.set_page_config(page_title="AceNotes", page_icon="🎓", layout="centered")

# Model names from the official Gemma on Gemini API docs.
MODELS = {
    "Gemma 4 31B (best quality)": "gemma-4-31b-it",
    "Gemma 4 26B (faster)": "gemma-4-26b-a4b-it",
}


# ----------------------------------------------------------------------------
# Settings (sidebar)
# ----------------------------------------------------------------------------
def default_key():
    env = os.getenv("GEMINI_API_KEY", "")
    if env:
        return env
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Gemini API key",
        value=default_key(),
        type="password",
        help="Get a free key at aistudio.google.com",
    )
    model_label = st.selectbox("Gemma model", list(MODELS))
    MODEL = MODELS[model_label]
    language = st.radio("Output language", ["English", "Bangla"])
    st.divider()
    st.subheader("⭐ AceNotes Pro (coming soon)")
    st.markdown(
        "- Unlimited pages\n- Flashcards\n- PDF export\n- Exam study plan"
    )
    st.caption("Open source. Built with Gemma 4.")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def prepare_image(uploaded, max_side=1600):
    """Fix rotation, shrink big phone photos, and convert to JPEG bytes."""
    img = Image.open(io.BytesIO(uploaded.getvalue()))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def ask_gemma(prompt, image_bytes=None):
    """Send text (and optional images) to Gemma. Returns the reply text."""
    if not api_key:
        st.error("Please paste your Gemini API key in the sidebar first.")
        st.stop()
    client = genai.Client(api_key=api_key)
    parts = [
        types.Part.from_bytes(data=b, mime_type="image/jpeg")
        for b in (image_bytes or [])
    ]
    parts.append(types.Part.from_text(text=prompt))

    last_error = None
    for _ in range(2):  # one retry for temporary errors
        try:
            response = client.models.generate_content(model=MODEL, contents=parts)
            if response.text:
                return response.text.strip()
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise RuntimeError(f"Gemma did not answer. Details: {last_error}")


def parse_quiz(text):
    """Turn Gemma's JSON reply into a clean Python dict."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start : end + 1])

    questions = []
    for q in data["questions"]:
        options = [str(o).strip() for o in q["options"]][:4]
        raw = q.get("answer_index", 0)
        if isinstance(raw, str) and raw.strip().upper() in list("ABCD"):
            idx = "ABCD".index(raw.strip().upper())
        else:
            idx = int(raw)
        if len(options) >= 2 and 0 <= idx < len(options):
            questions.append(
                {
                    "question": str(q["question"]).strip(),
                    "options": options,
                    "answer_index": idx,
                    "explanation": str(q.get("explanation", "")).strip(),
                }
            )
    if not questions:
        raise ValueError("No valid questions found.")
    return {
        "summary": str(data.get("summary", "")).strip(),
        "key_terms": data.get("key_terms", []) or [],
        "questions": questions,
    }


def make_quiz(source_instruction, n, difficulty, image_bytes=None, source_text=""):
    """Ask Gemma for a summary + quiz as JSON. Retries once if JSON is broken."""
    prompt = f"""{source_instruction}

Create study material in {language}. Use simple words and short sentences.
Make exactly {n} multiple-choice questions. Difficulty: {difficulty}.
Use only information from the material. Do not invent facts.

Reply with ONLY valid JSON. No markdown. No code fences. Use this exact shape:
{{
  "summary": "A short summary in 4 to 6 simple sentences.",
  "key_terms": [{{"term": "word", "meaning": "simple meaning"}}],
  "questions": [
    {{
      "question": "Question text?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer_index": 0,
      "explanation": "One short sentence on why this is correct."
    }}
  ]
}}
Rules: "options" must have exactly 4 items. "answer_index" is a number from 0 to 3.
Give 3 to 5 key_terms.
{source_text}"""
    last_error = None
    for attempt in range(2):
        reply = ask_gemma(prompt, image_bytes)
        try:
            return parse_quiz(reply)
        except Exception as e:  # noqa: BLE001
            last_error = e
            prompt += "\n\nIMPORTANT: Your last reply was not valid JSON. Reply with JSON only."
    raise RuntimeError(f"Could not read the quiz format. Please try again. ({last_error})")


def render_quiz(quiz, key):
    st.subheader("🧠 Quiz")
    with st.form(f"form_{key}"):
        answers = []
        for i, q in enumerate(quiz["questions"]):
            st.markdown(f"**Q{i + 1}. {q['question']}**")
            answers.append(
                st.radio(
                    "Choose one",
                    q["options"],
                    index=None,
                    key=f"{key}_q{i}",
                    label_visibility="collapsed",
                )
            )
        submitted = st.form_submit_button("✅ Check my answers")

    if submitted:
        score = 0
        for i, (q, picked) in enumerate(zip(quiz["questions"], answers)):
            correct = q["options"][q["answer_index"]]
            if picked == correct:
                score += 1
                st.success(f"Q{i + 1}: Correct. {q['explanation']}")
            elif picked is None:
                st.warning(f"Q{i + 1}: Not answered. Correct answer: {correct}. {q['explanation']}")
            else:
                st.error(f"Q{i + 1}: Not quite. Correct answer: {correct}. {q['explanation']}")
        st.metric("Your score", f"{score} / {len(quiz['questions'])}")


def quiz_to_markdown(quiz):
    lines = ["# Summary", quiz["summary"], ""]
    if quiz["key_terms"]:
        lines.append("# Key terms")
        for t in quiz["key_terms"]:
            lines.append(f"- **{t.get('term', '')}**: {t.get('meaning', '')}")
        lines.append("")
    lines.append("# Quiz")
    for i, q in enumerate(quiz["questions"], 1):
        lines.append(f"{i}. {q['question']}")
        for j, o in enumerate(q["options"]):
            lines.append(f"   - {'ABCD'[j]}. {o}")
        lines.append(f"   - Answer: {'ABCD'[q['answer_index']]}. {q['explanation']}")
    return "\n".join(lines)


def show_previews(files):
    if files:
        st.image(files, width=150)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("🎓 AceNotes")
st.write("Study faster. Clean your messy notes. Turn textbook pages into quizzes.")

tab_notes, tab_quiz = st.tabs(["📝 Clean My Notes", "📖 Textbook → Quiz"])

# ----------------------------- TAB 1: Notes cleaner -------------------------
with tab_notes:
    st.write("Upload photos of your handwritten notes. Get clean, organized notes.")

    style_options = {
        "Headings and bullet points": "Use headings and short bullet points.",
        "Short summary": "Write a short summary in a few clear paragraphs.",
        "Study sheet with key terms": "Use headings, bullet points, and a Key Terms list at the end.",
    }
    style = st.selectbox("Note style", list(style_options))
    note_files = st.file_uploader(
        "Upload your notes (you can add many pages)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="note_files",
    )
    show_previews(note_files)

    if st.button("✨ Clean my notes", type="primary", disabled=not note_files):
        prompt = f"""You are an expert note-taking assistant.
The attached images are photos of handwritten notes. Read every page carefully, in order.
Rewrite them as clean, well-organized study notes.

Rules:
- Write the notes in {language}.
- Style: {style_options[style]}
- Keep the original meaning. Do not invent facts that are not in the notes.
- Fix spelling and grammar. Use short, simple sentences.
- Group related ideas under clear headings.
- If a word is unreadable, write [unclear] instead of guessing.
- If the images do not contain notes or text, reply only: I could not find any notes in this image.
- Output Markdown only. No introduction and no closing remarks."""
        with st.spinner("Reading your handwriting..."):
            try:
                imgs = [prepare_image(f) for f in note_files]
                st.session_state["clean_notes"] = ask_gemma(prompt, imgs)
                st.session_state.pop("notes_quiz", None)
            except Exception as e:  # noqa: BLE001
                st.error(f"Something went wrong: {e}")

    if "clean_notes" in st.session_state:
        st.subheader("✨ Your clean notes")
        st.markdown(st.session_state["clean_notes"])
        st.download_button(
            "⬇️ Download notes (.md)",
            st.session_state["clean_notes"],
            file_name="clean_notes.md",
        )

        st.divider()
        st.write("Want to test yourself on these notes?")
        n_notes = st.slider("Number of questions", 3, 10, 5, key="n_notes")
        if st.button("🧠 Make a quiz from my notes"):
            with st.spinner("Writing your quiz..."):
                try:
                    st.session_state["notes_quiz"] = make_quiz(
                        "Below are a student's study notes.",
                        n_notes,
                        "medium",
                        source_text="\nNOTES:\n" + st.session_state["clean_notes"],
                    )
                    st.session_state["notes_quiz_id"] = st.session_state.get("notes_quiz_id", 0) + 1
                except Exception as e:  # noqa: BLE001
                    st.error(f"Something went wrong: {e}")

    if "notes_quiz" in st.session_state:
        render_quiz(st.session_state["notes_quiz"], f"nq{st.session_state['notes_quiz_id']}")

# ----------------------------- TAB 2: Textbook quiz -------------------------
with tab_quiz:
    st.write("Upload a photo of a textbook page. Get a summary and a quiz.")

    book_files = st.file_uploader(
        "Upload textbook page photo(s)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="book_files",
    )
    show_previews(book_files)
    col1, col2 = st.columns(2)
    n_book = col1.slider("Number of questions", 3, 10, 5, key="n_book")
    difficulty = col2.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)

    if st.button("📖 Make summary and quiz", type="primary", disabled=not book_files):
        with st.spinner("Reading the page and writing your quiz..."):
            try:
                imgs = [prepare_image(f) for f in book_files]
                st.session_state["book_quiz"] = make_quiz(
                    "The attached images are photos of textbook pages. Read them carefully.",
                    n_book,
                    difficulty,
                    image_bytes=imgs,
                )
                st.session_state["book_quiz_id"] = st.session_state.get("book_quiz_id", 0) + 1
            except Exception as e:  # noqa: BLE001
                st.error(f"Something went wrong: {e}")

    if "book_quiz" in st.session_state:
        bq = st.session_state["book_quiz"]
        st.subheader("📄 Summary")
        st.write(bq["summary"])
        if bq["key_terms"]:
            with st.expander("🔑 Key terms"):
                for t in bq["key_terms"]:
                    st.markdown(f"**{t.get('term', '')}**: {t.get('meaning', '')}")
        render_quiz(bq, f"bq{st.session_state['book_quiz_id']}")
        st.download_button(
            "⬇️ Download summary and quiz (.md)",
            quiz_to_markdown(bq),
            file_name="textbook_quiz.md",
        )
