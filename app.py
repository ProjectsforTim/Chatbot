import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import textwrap
import base64

# --- PAGE SETUP ---
st.set_page_config(page_title="Tim's DroneShield Chatbot", page_icon="🛡️")

# ---------- STYLING ----------
st.markdown("""
    <style>
        .stApp {
            background: radial-gradient(circle at top center, rgba(255,122,0,0.9) 0%, #0D0D0D 90%);
            color: white;
            text-align: center;
        }

        .main-title {
            background: linear-gradient(90deg, rgba(255,122,0,1) 0%, rgba(255,102,0,0.9) 100%);
            box-shadow: 0 0 35px rgba(255,122,0,0.6);
            color: white;
            font-weight: 700;
            font-size: 30px;
            border-radius: 12px;
            display: inline-block;
            padding: 12px 40px;
            margin-top: 10px;
            transition: all 0.4s ease-in-out;
        }
        .title-grey { color: #bfbfbf; font-weight: 600; }
        .title-black { color: #333333; font-weight: 500; }

        .main-title:hover {
            box-shadow: 0 0 50px rgba(255,150,50,0.9);
            transform: scale(1.03);
        }

        .caption {
            color: #e0e0e0;
            font-size: 15px;
            margin-bottom: 25px;
        }

        .answer-card {
            background: rgba(0, 0, 0, 0.85);
            color: white;
            border-radius: 10px;
            padding: 22px;
            text-align: left;
            box-shadow: inset 0px 0px 15px rgba(255,122,0,0.25), 0 0 20px rgba(0,0,0,0.6);
        }

        .footer {
            font-size: 13px;
            color: #d9d9d9;
            opacity: 0.8;
            margin-top: 40px;
            border-top: 1px solid rgba(255,255,255,0.1);
            padding-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
logo_path = "droneshield_logo.png"
if os.path.exists(logo_path):
    encoded_logo = base64.b64encode(open(logo_path, "rb").read()).decode()
    st.markdown(f"""
        <div style='display:flex;justify-content:center;align-items:center;flex-direction:column;margin-bottom:-10px;'>
            <img src="data:image/png;base64,{encoded_logo}" width="140"/>
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Logo not found — please upload 'droneshield_logo.png'.")

st.markdown("<div class='main-title'>Tim's <span class='title-grey'>Drone</span><span class='title-black'>Shield</span> Chatbot</div>", unsafe_allow_html=True)
st.markdown("<p class='caption'>Built for clarity and precision! Answers grounded in verified content from droneshield.com.</p>", unsafe_allow_html=True)

# ---------- LOAD CONTEXT ----------
try:
    with open("droneshield_parsed_data.json.txt", "r", encoding="utf-8") as f:
        context_data = json.load(f)

    # If the JSON file contains a single string (stringified JSON), parse it again
    if isinstance(context_data, str):
        context_data = json.loads(context_data)

    # Ensure it’s a list
    if not isinstance(context_data, list):
        context_data = [context_data]

    documents = [item.get("content", "") for item in context_data if isinstance(item, dict)]
    metas = [{"url": item.get("url", ""), "title": item.get("title", "")} for item in context_data if isinstance(item, dict)]
except Exception as e:
    st.error(f"⚠️ Error loading context file: {e}")
    documents, metas = [], []

# ---------- EMBEDDING MODEL ----------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve(query, k=4):
    if not documents:
        return []
    query_emb = model.encode(query)
    doc_embs = model.encode(documents)
    scores = np.dot(doc_embs, query_emb)
    top_k = np.argsort(scores)[-k:][::-1]
    return [(documents[i], metas[i]) for i in top_k]

# ---------- OPENAI SETUP ----------
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("⚠️ Missing OpenAI API key. Please set it in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=openai_key)

SYSTEM_PROMPT = """You are the DroneShield AI assistant.
Use provided context where available.
If info not found, answer accurately from verified knowledge and note it was external.
Always end with 'Sources:' followed by bullet-point URLs.
"""

def answer(query):
    hits = retrieve(query)
    context = "\n\n".join([h[0] for h in hits]) if hits else "No relevant company data found."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"}
            ],
            temperature=0.3,
            max_tokens=500,
        )
        text = resp.choices[0].message.content
    except Exception as e:
        text = f"⚠️ GPT-4 error: {e}\n\nExtractive fallback:\n\n" + textwrap.fill(context[:1000], 110)

    sources, seen = [], set()
    for h in hits:
        url = h[1].get("url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    return text, sources

# ---------- UI ----------
query = st.text_input("Ask a question about DroneShield:")

if query:
    with st.spinner("Thinking..."):
        resp_text, srcs = answer(query)
    st.markdown(f"<div class='answer-card'>{resp_text}</div>", unsafe_allow_html=True)
    if srcs:
        st.subheader("Sources:")
        for s in srcs:
            st.write(f"- [{s}]({s})")

# ---------- FOOTER ----------
st.markdown("<div class='footer'>⚠️ This chatbot is an independent personal project inspired by DroneShield. Not affiliated with or endorsed by DroneShield Ltd. GPT-4 assists if website data is insufficient.</div>", unsafe_allow_html=True)
