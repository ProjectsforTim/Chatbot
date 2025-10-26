import streamlit as st
import json
import os
import textwrap
import numpy as np
import base64
from openai import OpenAI
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Tim's DroneShield Chatbot", page_icon="🛡️")

# ---------- STYLING ----------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top center, rgba(255,122,0,0.85) 0%, #0D0D0D 85%);
    color: white;
    text-align: center;
}
.logo-wrapper img {
    width: 130px;
    filter: drop-shadow(0 0 25px rgba(255,122,0,0.7));
    transition: all 0.6s ease-in-out;
}
.logo-wrapper img:hover {
    transform: scale(1.05);
    filter: drop-shadow(0 0 45px rgba(255,180,80,1));
}
.main-title {
    background: linear-gradient(90deg, rgba(255,122,0,1) 0%, rgba(255,102,0,0.9) 100%);
    box-shadow: 0 0 30px rgba(255,122,0,0.5);
    color: white;
    font-weight: 700;
    font-size: 30px;
    border-radius: 12px;
    display: inline-block;
    padding: 12px 40px;
    margin-top: 10px;
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

# ---------- LOGO ----------
if os.path.exists("droneshield_logo.png"):
    encoded_logo = base64.b64encode(open("droneshield_logo.png", "rb").read()).decode()
    st.markdown(f"<div class='logo-wrapper'><img src='data:image/png;base64,{encoded_logo}'/></div>", unsafe_allow_html=True)
else:
    st.warning("⚠️ Logo not found — please upload 'droneshield_logo.png'.")

st.markdown("<div class='main-title'>Tim's DroneShield Chatbot</div>", unsafe_allow_html=True)
st.markdown("<p class='caption'>Built for clarity and precision — grounded in verified DroneShield.com content with GPT-4 fallback.</p>", unsafe_allow_html=True)

# ---------- CONTEXT ----------
try:
    with open("droneshield_parsed_data.json.txt", "r", encoding="utf-8") as f:
        context_data = json.load(f)
except Exception as e:
    st.error(f"⚠️ Could not load context: {e}")
    context_data = []

documents = [item["content"] for item in context_data if isinstance(item, dict) and "content" in item]
metas = [{"url": item.get("url", "")} for item in context_data if isinstance(item, dict)]

try:
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
except Exception as e:
    st.warning(f"⚠️ Embedding model issue: {e}")
    model = None

def retrieve(query, k=4):
    if not model or not documents:
        return []
    query_emb = model.encode([query])[0]
    doc_embs = model.encode(documents)
    scores = np.dot(doc_embs, query_emb)
    top_k_idx = np.argsort(scores)[-k:][::-1]
    return [(documents[i], metas[i]) for i in top_k_idx]

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key) if openai_key else None

query = st.text_input("Ask a question about DroneShield:")

if query:
    with st.spinner("Thinking..."):
        hits = retrieve(query)
        context = "\n\n".join([h[0] for h in hits])
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are the DroneShield assistant."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ]
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"⚠️ GPT error: {e}\n\nExtractive fallback:\n{textwrap.fill(context[:500], 100)}"

    st.markdown(f"<div class='answer-card'>{answer}</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>⚠️ Independent project inspired by DroneShield. Not affiliated with or endorsed by DroneShield Ltd.</div>", unsafe_allow_html=True)
