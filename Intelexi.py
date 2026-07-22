import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import wikipedia
import numpy as np
import io
import tempfile
import os
from faster_whisper import WhisperModel
from groq import Groq

# -------------------------------
client = Groq(api_key=st.secrets.get("GROQ_API_KEY", ""))
model_name = "llama-3.1-8b-instant"
print("Loaded client successfully!")


# -------------------------------
def clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())

def load_whisper_model():
    # CHANGED: use session_state to ensure single load and re-use
    if "whisper_model" not in st.session_state:
        st.session_state["whisper_model"] = WhisperModel( "tiny",              
            device="cpu",          # no GPU needed
            compute_type="int8",   # prevents meta-tensor error
            cpu_threads=2,         # safe for cloud
            download_root="models" # cached download
        )

    return st.session_state["whisper_model"]

def load_pdf_text(files):
    full_text = ""
    failed_files = []

    for f in files:
        try:
            with pdfplumber.open(f) as pdf:
                file_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        file_text += page_text + " "
            if file_text.strip() == "":
                failed_files.append(f.name)
            else:
                full_text += file_text

        except Exception:
            failed_files.append(getattr(f, "name", "unknown"))

    if failed_files:
        st.warning(f"⚠️ Could not extract text from: {', '.join(failed_files)}")

    return clean_text(full_text)

def split_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(model, chunks):
    emb = model.encode(chunks, convert_to_tensor=False)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb

def get_top_k_chunks(query, model, chunks, chunk_embeddings, k=3):
    query_emb = model.encode([query])[0]
    query_emb = query_emb / np.linalg.norm(query_emb)
    # protect shapes
    if chunk_embeddings is None or len(chunk_embeddings) == 0:
        return []
    scores = np.dot(chunk_embeddings, query_emb)

    boost = np.array([3 if "preamble" in c.lower() else 1 for c in chunks])
    scores = scores * boost

    top_idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_idx]

def ask_model(client, model_name, prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


st.set_page_config(page_title="Intelexi.ai", page_icon="🔍", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(135deg, #0f1724 0%, #122033 50%, #173146 100%);
        color: #e6eef8;
        font-family: 'Segoe UI', Roboto, system-ui, -apple-system, 'Helvetica Neue', Arial;
    }
    .user-bubble, .bot-bubble {
        padding: 12px 16px;
        border-radius: 12px;
        margin: 10px 0;
        max-width: 75%;
        font-size: 16px;
        line-height: 1.5;
        white-space: pre-wrap;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }
    .user-bubble {
        background-color: rgba(40, 120, 90, 0.14);
        border: 1px solid rgba(40,160,80,0.18);
        margin-left: auto;
    }
    .bot-bubble {
        background-color: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        margin-right: auto;
    }

 
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(2,6,23,0.6);
        transition: transform .15s ease, box-shadow .15s ease;
        cursor: pointer;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .card:hover { transform: translateY(-6px); box-shadow: 0 16px 40px rgba(2,6,23,0.7); }
    .card h2 { margin: 6px 0 4px 0; font-size: 20px; }
    .card p { margin: 0; opacity: 0.8; font-size: 14px; }


    .stButton>button {
        background: linear-gradient(90deg,#0ea5a9,#7c3aed); /* gradient filled */
        color: white;
        padding: 12px 18px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 8px 24px rgba(124,58,237,0.25);
        font-weight: 600;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 14px 32px rgba(124,58,237,0.32); }

    /* small tweak for inputs */
    .stTextInput>div>div>input { border-radius: 8px; padding: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("""
<div style="display: flex; align-items: center; gap: 20px;">
    <h1 style="margin: 0; font-size: 40px;">🤖👾INTELEXI</h1>
    <span style="font-size: 15px; color: #ccc;">( Ask. Understand. Intelexi )</span>
</div>
""", unsafe_allow_html=True)

st.markdown("#### 🤖 Hi Buddy! How can I assist you?")


if "chat" not in st.session_state:
    st.session_state["chat"] = []
if "mode" not in st.session_state:
    st.session_state["mode"] = "home"  # home / voice / text
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "chunk_embeddings" not in st.session_state:
    st.session_state["chunk_embeddings"] = np.array([])
if "embedder" not in st.session_state:
    st.session_state["embedder"] = None
if "last_transcription" not in st.session_state:
    st.session_state["last_transcription"] = ""
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""


with st.sidebar:
    st.header("⚙️ Settings")
    st.text_input("🤖 Model Name (GROQ Models)", value=model_name, key="sidebar_modelname")
    if st.button("🗑 Clear Chat & Reset"):
        #CHANGED: clear relevant keys but keep whisper model loaded
        keep = {}
        if "whisper_model" in st.session_state:
            keep["whisper_model"] = st.session_state["whisper_model"]
        st.session_state.clear()
        st.session_state.update(keep)
        st.session_state["mode"] = "home"
        st.rerun()


def process_uploaded_files(files, force_reprocess=False):
    """
    Loads PDFs, splits, and embeds if needed.
    Stores results in session_state to avoid double work.
    """

    if not files:
        return

    # check if new files or force_reprocess
    uploaded_names = [getattr(f, "name", None) for f in files]
    prev_names = [getattr(f, "name", None) for f in st.session_state.get("uploaded_files", [])]

    if force_reprocess or uploaded_names != prev_names or st.session_state["embedder"] is None:
        st.session_state["uploaded_files"] = files
        with st.spinner("📄 Reading documents..."):
            full_text = load_pdf_text(files)
        if full_text.strip() == "":
            st.error("⚠️ Could not read any text from the PDF.")
            st.session_state["chunks"] = []
            st.session_state["chunk_embeddings"] = np.array([])
            return
        chunks = split_text(full_text)
        st.session_state["chunks"] = chunks
        if st.session_state["embedder"] is None:
            st.session_state["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
        with st.spinner("🧠 Processing document..."):
            st.session_state["chunk_embeddings"] = embed_chunks(st.session_state["embedder"], chunks)


if st.session_state["mode"] == "home":
    st.markdown("### Choose Input Mode")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        # Card for Voice Mode
        if st.button("🎤 Voice Mode — Speak to ask"):
            st.session_state["mode"] = "voice"
            st.rerun()
        st.markdown(
            """
            <div class="card" aria-hidden="true">
                <h2>🎤 Voice Mode</h2>
                <p>Ask questions by speaking. Upload PDFs in the voice screen to search your documents by voice.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        # Card for Text Mode
        if st.button("⌨️ Text Mode — Type your question"):
            st.session_state["mode"] = "text"
            st.rerun()
        st.markdown(
            """
            <div class="card" aria-hidden="true">
                <h2>⌨️ Text Mode</h2>
                <p>Type your question. Upload PDFs in the text screen to search your documents by typed queries.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:

        if st.button("💬 View Chat"):
            st.session_state["mode"] = "chat"
            st.rerun()
        st.markdown(
            f"""
            <div class="card" aria-hidden="true">
                <h2>📂 {len(st.session_state['chat'])//2} Conversations</h2>
                <p>View the chat history or return here anytime.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------------------
# CHAT VIEW (simple back link)
# -------------------------------
elif st.session_state["mode"] == "chat":
    st.subheader("💬 Chat History")
    if st.button("⬅️ Back to Home"):
        st.session_state["mode"] = "home"
        st.rerun()
    st.markdown("---")
    for role, message in st.session_state["chat"]:
        if role == "user":
            st.markdown(f"<div class='user-bubble'><b>🧑 You:</b><br>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'><b>🤖 AI:</b><br>{message}</div>", unsafe_allow_html=True)
    st.markdown("<div id='chat-end'></div>", unsafe_allow_html=True)

# -------------------------------
# VOICE MODE Screen
# -------------------------------
elif st.session_state["mode"] == "voice":
    # Voice Mode UI unchanged
    st.subheader("🎤 Voice Question Mode")
    col_left, col_right = st.columns([3,1])
    with col_right:
        if st.button("⬅️ Back to Home"):
            st.session_state["mode"] = "home"
            st.rerun()

    st.markdown("**Upload PDFs (optional) — these will be searched first when you ask by voice**")
    uploaded_files_voice = st.file_uploader("Upload PDF(s) for Voice Mode", type=["pdf"], accept_multiple_files=True, key="uploader_voice")
    if uploaded_files_voice:
        process_uploaded_files(uploaded_files_voice)

    whisper_model = load_whisper_model()

    audio_file = st.audio_input("Click to record your voice question")

    if audio_file is not None:
        audio_bytes = audio_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        segments, info = whisper_model.transcribe(
            temp_audio_path,
            beam_size=5,  # allowed
            temperature=0.0,  # allowed
            initial_prompt=(
                "Transcribe exactly as spoken. "
                "The user will give programming commands such as: "
                "'write a python code', 'generate java program', "
                "'build algorithm', 'create a function'. "
                "Do NOT convert words like 'write' to numbers like '5th'. "
                "Do NOT auto-correct, guess, or interpret. "
                "Strictly transcribe as text commands."
            )
        )

        texts = [seg.text if hasattr(seg, "text") else seg[2] for seg in segments]
        transcribed_text = " ".join(texts).strip()

        st.success(f"Transcribed: **{transcribed_text}**")

        if transcribed_text and transcribed_text != st.session_state.get("last_transcription", ""):
            st.session_state["last_transcription"] = transcribed_text
            question = transcribed_text


            if st.session_state.get("chunks") and len(st.session_state["chunks"]) > 0:
                top_chunks = get_top_k_chunks(
                    question,
                    st.session_state["embedder"],
                    st.session_state["chunks"],
                    st.session_state["chunk_embeddings"],
                    k=5
                )
                context = "\n\n".join(top_chunks)

                prompt = f"""
You are an AI assistant. Answer the question ONLY using the provided document context.
Provide a detailed answer in 3–5 sentences by explaining each term clearly.
If the answer is not found, respond: "Information not in document."

CONTEXT:
{context}

QUESTION:
{question}
"""
                answer = ask_model(client, model_name, prompt)

                # If document does NOT contain the answer → fallback to Wikipedia or Model
                if "Information not in document" in answer:
                    try:
                        wiki_summary = wikipedia.summary(question, sentences=4)
                        answer = f"It’s not mentioned in the document, but here’s what I found:\n\n{wiki_summary}"
                    except:
                        # Wikipedia failed → model gives its own answer (fix for coding questions)
                        answer = ask_model(client, model_name, question)


            else:
                try:
                    wiki_summary = wikipedia.summary(question, sentences=4)
                    prompt = f"""
Use this Wikipedia information if helpful — otherwise provide the correct answer.

{wiki_summary}

Question: {question}
"""
                    answer = ask_model(client, model_name, prompt)

                except:

                    prompt = question
                    answer = ask_model(client, model_name, prompt)

            st.session_state["chat"].append(("user", question))
            st.session_state["chat"].append(("assistant", answer))

    # Show chat preview area
    st.markdown("---")
    st.markdown("### Recent conversation")
    for role, message in st.session_state["chat"][-6:]:
        if role == "user":
            st.markdown(f"<div class='user-bubble'><b>🧑 You:</b><br>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'><b>🤖 AI:</b><br>{message}</div>", unsafe_allow_html=True)
    st.markdown("<div id='chat-end'></div>", unsafe_allow_html=True)


# -------------------------------
# TEXT MODE Screen
# -------------------------------
elif st.session_state["mode"] == "text":

    st.subheader("⌨️ Text Question Mode")
    col_left, col_right = st.columns([3,1])
    with col_right:
        if st.button("⬅️ Back to Home"):
            st.session_state["mode"] = "home"
            st.rerun()

    st.markdown("**Upload PDFs (optional) — these will be searched first for typed queries**")
    uploaded_files_text = st.file_uploader("Upload PDF(s) for Text Mode", type=["pdf"], accept_multiple_files=True, key="uploader_text")
    if uploaded_files_text:
        process_uploaded_files(uploaded_files_text)


    if st.session_state.get("uploaded_files"):
        process_uploaded_files(st.session_state["uploaded_files"])

    st.markdown("### 💬 Ask your question")
    query = st.text_input("Type your question here:", key="text_input_mode")

    if query and query.strip() and query.strip() != st.session_state.get("last_query", ""):
        st.session_state["last_query"] = query.strip()
        question = query.strip()


        if st.session_state.get("chunks") and len(st.session_state["chunks"]) > 0:
            top_chunks = get_top_k_chunks(
                question,
                st.session_state["embedder"],
                st.session_state["chunks"],
                st.session_state["chunk_embeddings"],
                k=5
            )
            context = "\n\n".join(top_chunks)

            prompt = f"""
You are an AI assistant. Answer the question using ONLY the provided document context.
Give a detailed explanation in 3–5 sentences.
Expand the meaning of concepts from the document.
If the answer is not found, respond with: "Information not in document."

CONTEXT:
{context}

QUESTION:
{question}
"""
            answer = ask_model(client, model_name, prompt)

            # Document did NOT have answer → Wikipedia fallback
            if "Information not in document" in answer:
                try:
                    wiki_summary = wikipedia.summary(question, sentences=4)
                    answer = f"It’s not mentioned in the document, but here’s what Wikipedia says:\n\n{wiki_summary}"
                except:
                    # Wikipedia failed → general response
                    answer = ask_model(client, model_name, question)


        else:
            try:
                wiki_summary = wikipedia.summary(question, sentences=4)
                prompt = f"""
Use this Wikipedia information if helpful — otherwise provide the correct answer.

{wiki_summary}

Question: {question}
"""
                answer = ask_model(client, model_name, prompt)

            except:

                prompt = question
                answer = ask_model(client, model_name, prompt)

        # store in chat
        st.session_state["chat"].append(("user", question))
        st.session_state["chat"].append(("assistant", answer))

        # scroll
        st.markdown("""<script>window.location.href = "#chat-end";</script>""", unsafe_allow_html=True)

    # Suggested prompts
    with st.expander("✨ Suggested Prompts"):
        st.markdown("""
            - **📖 Summarize the document in 5 points**
            - **🔍 list the key concepts in the pdf**
            - **📌 Extract all key points and important terms from this pdf**
            - **📚 What does this pdf say about "<topic>"?**
        """)

    # Show chat
    st.markdown("---")
    st.markdown("### Recent conversation")
    for role, message in st.session_state["chat"][-6:]:
        if role == "user":
            st.markdown(f"<div class='user-bubble'><b>🧑 You:</b><br>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'><b>🤖 AI:</b><br>{message}</div>", unsafe_allow_html=True)
    st.markdown("<div id='chat-end'></div>", unsafe_allow_html=True)


else:
    st.session_state["mode"] = "home"
    st.rerun()
