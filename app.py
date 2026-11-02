"""
AI Meeting Minutes Generator
----------------------------
Upload meeting notes (TXT / PDF / DOCX) or recordings (MP3 / WAV / M4A),
or paste notes manually. The app transcribes audio with Groq Whisper and
uses a Groq LLM to create structured meeting minutes.

Run with:  streamlit run app.py
"""

import json
import os
import re
from datetime import date
from io import BytesIO

import pandas as pd
import streamlit as st
from docx import Document
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    Groq,
    NotFoundError,
    RateLimitError,
)
from pypdf import PdfReader

# =============================================================================
# CONFIGURATION
# =============================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#  WHERE TO ADD YOUR GROQ API KEY  (never paste a real key into this file!)
#
#  Option 1 (recommended): create the file  .streamlit/secrets.toml  with:
#       GROQ_API_KEY = "your-groq-api-key-here"
#
#  Option 2: set an environment variable before running the app:
#       Windows (PowerShell):  $env:GROQ_API_KEY="your-groq-api-key-here"
#       Mac / Linux:           export GROQ_API_KEY="your-groq-api-key-here"
#
#  Option 3: paste the key into the sidebar box while the app is running.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Groq changes its model list often, so the app asks Groq which models your key can
# use and lets you choose in the sidebar. These lists are only the PREFERRED order
# (the first one that is available gets selected by default). Edit them if you like.
PREFERRED_LLMS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]
PREFERRED_STT = ["whisper-large-v3-turbo", "whisper-large-v3"]
MAX_AUDIO_MB = 25                          # Groq audio upload limit (free tier)
MAX_TEXT_CHARS = 30000                     # Keeps the request within model limits
NOT_SPECIFIED = "Not specified"
STATUS_OPTIONS = [NOT_SPECIFIED, "Pending", "In Progress", "Completed"]

# =============================================================================
# PAGE SETUP & STYLE
# =============================================================================
st.set_page_config(page_title="AI Meeting Minutes Generator", page_icon="📝", layout="wide")

st.markdown(
    """
    <style>
        .main-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0;}
        .subtitle {font-size: 1.05rem; color: #6b7280; margin-bottom: 1.2rem;}
        .section-card {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_api_key(sidebar_key: str) -> str:
    """Look for the Groq API key in: Streamlit secrets -> env variable -> sidebar."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass  # No secrets file exists - that's fine, try the next option
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    return sidebar_key.strip()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_available_models(key: str) -> list:
    """Ask Groq which model IDs this API key can use."""
    return sorted(m.id for m in Groq(api_key=key).models.list().data)


def split_models(model_ids: list):
    """Separate chat (LLM) models from speech-to-text models."""
    skip = ("whisper", "tts", "guard", "orpheus", "playai", "embed", "safeguard", "compound")
    chat = [m for m in model_ids if not any(w in m.lower() for w in skip)]
    stt = [m for m in model_ids if "whisper" in m.lower()]
    return chat, stt


def default_index(options: list, preferred: list) -> int:
    """Index of the first preferred model that is available (else 0)."""
    for name in preferred:
        if name in options:
            return options.index(name)
    return 0


def read_text_file(uploaded_file) -> str:
    """Read a .txt file, trying common encodings."""
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the text file.")


def read_pdf_file(uploaded_file) -> str:
    """Extract text from every page of a PDF."""
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    if reader.is_encrypted:
        raise ValueError("The PDF is password-protected.")
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def read_docx_file(uploaded_file) -> str:
    """Extract paragraphs and table cells from a Word document."""
    doc = Document(BytesIO(uploaded_file.getvalue()))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_notes_text(uploaded_file) -> str:
    """Pick the right reader based on the file extension."""
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        text = read_text_file(uploaded_file)
    elif name.endswith(".pdf"):
        text = read_pdf_file(uploaded_file)
    elif name.endswith(".docx"):
        text = read_docx_file(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {uploaded_file.name}")
    if not text.strip():
        raise ValueError(
            f"'{uploaded_file.name}' contains no readable text "
            "(scanned PDFs without text are not supported)."
        )
    return text.strip()


def transcribe_audio(client: Groq, audio_file, stt_model: str) -> str:
    """Convert an audio recording to text using Groq Whisper."""
    data = audio_file.getvalue()
    if len(data) == 0:
        raise ValueError(f"'{audio_file.name}' is empty.")
    if len(data) > MAX_AUDIO_MB * 1024 * 1024:
        raise ValueError(
            f"'{audio_file.name}' is larger than {MAX_AUDIO_MB} MB. "
            "Please compress or split the recording."
        )
    result = client.audio.transcriptions.create(
        file=(audio_file.name, data),
        model=stt_model,
        response_format="json",
    )
    text = getattr(result, "text", None) or (result if isinstance(result, str) else "")
    if not text.strip():
        raise ValueError("No speech was detected in the recording.")
    return text.strip()


SYSTEM_PROMPT = """You are an expert meeting secretary. You read meeting notes or a
transcript and produce accurate, structured meeting minutes.

STRICT RULES:
- Use ONLY information that appears in the provided text. Never invent facts,
  decisions, people, tasks or deadlines.
- Only list a decision if the text clearly shows it was decided or agreed.
  If no decisions were made, return an empty list.
- Only list an action item if a task was actually mentioned.
- If an assigned person, deadline, responsible person or status is not
  mentioned, use exactly the string "Not specified".
- For status use one of: "Not specified", "Pending", "In Progress", "Completed".
- Write in the same language as the meeting content.

Respond with ONLY valid JSON in exactly this structure:
{
  "meeting_title": "short title, or 'Not specified'",
  "summary": "clear, concise summary in 3-6 sentences",
  "main_topics": ["topic 1", "topic 2"],
  "key_points": ["important point 1", "important point 2"],
  "decisions": [
    {"decision": "...", "related_topic": "...", "responsible_person": "..."}
  ],
  "action_items": [
    {"task": "...", "assigned_to": "...", "deadline": "...", "status": "..."}
  ]
}"""


def parse_json_response(raw: str) -> dict:
    """Safely turn the LLM reply into a Python dictionary."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def clean_value(value) -> str:
    """Return a trimmed string, or 'Not specified' if the value is empty."""
    text = str(value).strip() if value is not None else ""
    if text.lower() in ("", "none", "null", "n/a", "na", "unknown", "not mentioned"):
        return NOT_SPECIFIED
    return text


def normalize_minutes(data: dict) -> dict:
    """Make sure the LLM output has every field we need, in the right format."""
    decisions = []
    for d in data.get("decisions", []) or []:
        if isinstance(d, dict) and str(d.get("decision", "")).strip():
            decisions.append(
                {
                    "Decision": str(d["decision"]).strip(),
                    "Related Topic": clean_value(d.get("related_topic")),
                    "Responsible Person": clean_value(d.get("responsible_person")),
                }
            )

    actions = []
    for a in data.get("action_items", []) or []:
        if isinstance(a, dict) and str(a.get("task", "")).strip():
            status = clean_value(a.get("status"))
            matched = next((s for s in STATUS_OPTIONS if s.lower() == status.lower()), NOT_SPECIFIED)
            actions.append(
                {
                    "Task": str(a["task"]).strip(),
                    "Assigned To": clean_value(a.get("assigned_to")),
                    "Deadline": clean_value(a.get("deadline")),
                    "Status": matched,
                }
            )

    return {
        "title": clean_value(data.get("meeting_title")),
        "summary": str(data.get("summary", "")).strip(),
        "topics": [str(t).strip() for t in data.get("main_topics", []) or [] if str(t).strip()],
        "key_points": [str(k).strip() for k in data.get("key_points", []) or [] if str(k).strip()],
        "decisions": pd.DataFrame(decisions, columns=["Decision", "Related Topic", "Responsible Person"]),
        "actions": pd.DataFrame(actions, columns=["Task", "Assigned To", "Deadline", "Status"]),
    }


def generate_minutes(client: Groq, meeting_text: str, llm_model: str) -> dict:
    """Send the meeting text to the Groq LLM and return structured minutes."""
    request = dict(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"MEETING CONTENT:\n\n{meeting_text}"},
        ],
        temperature=0.1,  # low = more factual, less creative
    )
    try:
        response = client.chat.completions.create(**request, response_format={"type": "json_object"})
    except BadRequestError:
        # Some models don't support JSON mode - retry without it (we parse the text ourselves)
        response = client.chat.completions.create(**request)
    return normalize_minutes(parse_json_response(response.choices[0].message.content))


def clean_table(df: pd.DataFrame, required_col: str) -> pd.DataFrame:
    """Drop empty rows and fill blank cells with 'Not specified'."""
    df = df.copy().fillna("")
    df = df[df[required_col].astype(str).str.strip() != ""]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace("", NOT_SPECIFIED)
    return df.reset_index(drop=True)


def build_txt(title, summary, topics, key_points, decisions_df, actions_df) -> str:
    """Create the full meeting minutes as plain text."""
    line = "=" * 60
    out = [line, "MEETING MINUTES", line, f"Title: {title}", f"Generated on: {date.today():%B %d, %Y}", ""]
    out += ["SUMMARY", "-" * 60, summary or NOT_SPECIFIED, ""]
    out += ["MAIN TOPICS", "-" * 60] + ([f"- {t}" for t in topics] or [NOT_SPECIFIED]) + [""]
    out += ["KEY DISCUSSION POINTS", "-" * 60] + ([f"- {k}" for k in key_points] or [NOT_SPECIFIED]) + [""]

    out += ["DECISIONS", "-" * 60]
    if decisions_df.empty:
        out.append("No decisions were recorded.")
    for i, row in decisions_df.iterrows():
        out.append(f"{i + 1}. {row['Decision']}")
        out.append(f"   Related Topic: {row['Related Topic']}")
        out.append(f"   Responsible Person: {row['Responsible Person']}")
    out.append("")

    out += ["ACTION ITEMS", "-" * 60]
    if actions_df.empty:
        out.append("No action items were recorded.")
    for i, row in actions_df.iterrows():
        out.append(f"{i + 1}. {row['Task']}")
        out.append(f"   Assigned To: {row['Assigned To']}")
        out.append(f"   Deadline: {row['Deadline']}")
        out.append(f"   Status: {row['Status']}")
    out.append("")
    return "\n".join(out)


# =============================================================================
# SESSION STATE (keeps results on screen while the user edits tables)
# =============================================================================
for key, default in {"minutes": None, "transcript": "", "source_text": ""}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    sidebar_key = st.text_input(
        "Groq API Key (optional)",
        type="password",
        placeholder="Paste key here if not using secrets / env",
        help="The app first checks Streamlit secrets and the GROQ_API_KEY environment variable.",
    )
    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Upload notes / recordings or paste text\n"
        "2. Click *Generate Meeting Minutes*\n"
        "3. Review and edit the tables\n"
        "4. Download TXT / CSV files"
    )

api_key = get_api_key(sidebar_key)

# Ask Groq which models are available for this key (cached for 10 minutes)
available_chat, available_stt = [], []
if api_key:
    try:
        available_chat, available_stt = split_models(fetch_available_models(api_key))
    except AuthenticationError:
        st.sidebar.error("Invalid Groq API key.")
    except Exception as e:
        st.sidebar.warning(f"Could not load the model list ({e}). Using defaults.")

llm_options = available_chat or PREFERRED_LLMS
stt_options = available_stt or PREFERRED_STT

with st.sidebar:
    llm_model = st.selectbox("LLM model", llm_options, index=default_index(llm_options, PREFERRED_LLMS),
                             help="If you get a 'model not found' error, pick another model here.")
    stt_model = st.selectbox("Speech-to-text model", stt_options, index=default_index(stt_options, PREFERRED_STT))

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="main-title">📝 AI Meeting Minutes Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Turn meeting notes or recordings into clear summaries, '
    "decisions and action items - powered by Groq AI.</div>",
    unsafe_allow_html=True,
)

if not api_key:
    st.warning(
        "🔑 No Groq API key found. Add `GROQ_API_KEY` to `.streamlit/secrets.toml`, "
        "set it as an environment variable, or paste it in the sidebar."
    )

# =============================================================================
# 1. INPUT SECTION
# =============================================================================
st.header("1️⃣ Provide Meeting Content")
col_notes, col_audio = st.columns(2)

with col_notes:
    notes_files = st.file_uploader(
        "📄 Upload meeting notes (TXT, PDF, DOCX)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )
with col_audio:
    audio_files = st.file_uploader(
        "🎙️ Upload meeting recordings (MP3, WAV, M4A)",
        type=["mp3", "wav", "m4a"],
        accept_multiple_files=True,
    )

manual_notes = st.text_area(
    "✍️ Or paste your meeting notes here",
    height=180,
    placeholder="Paste meeting notes or a transcript here...",
)

generate_clicked = st.button("🚀 Generate Meeting Minutes", type="primary", use_container_width=True)

# =============================================================================
# PROCESSING
# =============================================================================
if generate_clicked:
    if not api_key:
        st.error("❌ Missing Groq API key. Please add it as described above and try again.")
    elif not (notes_files or audio_files or manual_notes.strip()):
        st.error("❌ Please upload a file, upload a recording, or paste some meeting notes.")
    else:
        try:
            client = Groq(api_key=api_key)
            text_parts, transcript_parts, problems = [], [], []

            with st.spinner("Reading your content..."):
                # Notes files
                for f in notes_files or []:
                    try:
                        text_parts.append(f"[Notes from {f.name}]\n{extract_notes_text(f)}")
                    except Exception as e:
                        problems.append(f"📄 {f.name}: {e}")

                # Pasted notes
                if manual_notes.strip():
                    text_parts.append(f"[Pasted notes]\n{manual_notes.strip()}")

            # Audio files
            for f in audio_files or []:
                with st.spinner(f"Transcribing {f.name} (this may take a moment)..."):
                    try:
                        t = transcribe_audio(client, f, stt_model)
                        transcript_parts.append(f"[Transcript of {f.name}]\n{t}")
                        text_parts.append(f"[Transcript of {f.name}]\n{t}")
                    except AuthenticationError:
                        raise
                    except Exception as e:
                        problems.append(f"🎙️ {f.name}: transcription failed - {e}")

            for p in problems:
                st.warning(p)

            if not text_parts:
                st.error("❌ No usable content could be read. Please check your files and try again.")
            else:
                combined = "\n\n".join(text_parts)
                if len(combined) > MAX_TEXT_CHARS:
                    st.warning(
                        f"⚠️ The content is very long. Only the first {MAX_TEXT_CHARS:,} "
                        "characters were analysed."
                    )
                    combined = combined[:MAX_TEXT_CHARS]

                with st.spinner("AI is writing the meeting minutes..."):
                    minutes = generate_minutes(client, combined, llm_model)

                st.session_state.minutes = minutes
                st.session_state.transcript = "\n\n".join(transcript_parts)
                st.session_state.source_text = combined
                st.success("✅ Meeting minutes generated!")

        except AuthenticationError:
            st.error("❌ Invalid Groq API key. Please check your key and try again.")
        except NotFoundError:
            st.error(
                f"❌ The model '{llm_model}' is not available for your account. "
                "Choose a different model in the sidebar and try again."
            )
        except RateLimitError:
            st.error("⏳ Groq rate limit reached. Please wait a minute and try again.")
        except APIConnectionError:
            st.error("🌐 Could not connect to Groq. Please check your internet connection.")
        except APIStatusError as e:
            st.error(f"❌ Groq API error ({e.status_code}): {e.message}")
        except json.JSONDecodeError:
            st.error("❌ The AI returned an unreadable response. Please click Generate again.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# =============================================================================
# RESULTS
# =============================================================================
minutes = st.session_state.minutes

if minutes:
    st.divider()

    # ---- Transcript ----
    if st.session_state.transcript:
        st.header("🗣️ Transcript")
        with st.expander("Show audio transcript", expanded=False):
            st.text_area("Transcript", st.session_state.transcript, height=260, label_visibility="collapsed")

    # ---- Summary ----
    st.header("2️⃣ Meeting Summary")
    title = st.text_input("Meeting title", value=minutes["title"])
    summary = st.text_area("Summary (editable)", value=minutes["summary"], height=150)

    col_topics, col_points = st.columns(2)
    with col_topics:
        st.subheader("📌 Main Topics")
        if minutes["topics"]:
            for t in minutes["topics"]:
                st.markdown(f"- {t}")
        else:
            st.info(NOT_SPECIFIED)
    with col_points:
        st.subheader("⭐ Key Discussion Points")
        if minutes["key_points"]:
            for k in minutes["key_points"]:
                st.markdown(f"- {k}")
        else:
            st.info(NOT_SPECIFIED)

    # ---- Decisions ----
    st.header("3️⃣ Decisions")
    if minutes["decisions"].empty:
        st.info("No decisions were mentioned in this meeting. You can add rows below if needed.")
    edited_decisions = st.data_editor(
        minutes["decisions"],
        num_rows="dynamic",
        use_container_width=True,
        key="decisions_editor",
        column_config={
            "Decision": st.column_config.TextColumn("Decision", width="large"),
            "Related Topic": st.column_config.TextColumn("Related Topic"),
            "Responsible Person": st.column_config.TextColumn("Responsible Person"),
        },
    )

    # ---- Action items ----
    st.header("4️⃣ Action Items")
    if minutes["actions"].empty:
        st.info("No action items were mentioned in this meeting. You can add rows below if needed.")
    edited_actions = st.data_editor(
        minutes["actions"],
        num_rows="dynamic",
        use_container_width=True,
        key="actions_editor",
        column_config={
            "Task": st.column_config.TextColumn("Task", width="large"),
            "Assigned To": st.column_config.TextColumn("Assigned To"),
            "Deadline": st.column_config.TextColumn("Deadline"),
            "Status": st.column_config.SelectboxColumn("Status", options=STATUS_OPTIONS),
        },
    )

    # ---- Downloads ----
    st.header("5️⃣ Download")
    final_decisions = clean_table(edited_decisions, "Decision")
    final_actions = clean_table(edited_actions, "Task")
    txt_content = build_txt(title, summary, minutes["topics"], minutes["key_points"],
                            final_decisions, final_actions)

    d1, d2, d3 = st.columns(3)
    d1.download_button("📄 Meeting Minutes (TXT)", txt_content.encode("utf-8"),
                       file_name="meeting_minutes.txt", mime="text/plain", use_container_width=True)
    d2.download_button("✅ Action Items (CSV)", final_actions.to_csv(index=False).encode("utf-8"),
                       file_name="action_items.csv", mime="text/csv", use_container_width=True)
    d3.download_button("📋 Decisions (CSV)", final_decisions.to_csv(index=False).encode("utf-8"),
                       file_name="decisions.csv", mime="text/csv", use_container_width=True)

    with st.expander("Preview full meeting minutes"):
        st.text(txt_content)

st.caption("Always review AI-generated minutes for accuracy before sharing.")
