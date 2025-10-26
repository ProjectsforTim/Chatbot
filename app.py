import streamlit as st
import os, json, base64, textwrap, chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------- PAGE CONFIG ----------
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
        .logo-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            margin-bottom: -10px;
        }
        .logo-wrapper img {
            width: 140px;
            transition: all 0.5s ease-in-out;
            filter: drop-shadow(0 0 25px rgba(255,122,0,0.7));
        }
        @keyframes glowPulse {
            0% { filter: drop-shadow(0 0 20px rgba(255,122,0,0.6)); transform: scale(1.00); }
            50% { filter: drop-shadow(0 0 45px rgba(255,180,80,1)); transform: scale(1.05); }
            100% { filter: drop-shadow(0 0 20px rgba(255,122,0,0.6)); transform: scale(1.00); }
        }
        .logo-wrapper img:hover {
            animation: glowPulse 2.5s ease-in-out infinite;
        }
        .logo-reflection {
            width: 120px;
            height: 18px;
            border-radius: 50%;
            background: radial-gradient(ellipse at center, rgba(255,122,0,0.35) 0%, rgba(255,122,0,0) 70%);
            filter: blur(8px);
            margin-top: -5px;
            opacity: 0.7;
            transition: all 0.5s ease;
        }
        .logo-wrapper:hover .logo-reflection {
            opacity: 0.45;
            transform: translateY(3px) scale(1.1);
        }
        .footer {
            font-size: 13px;
            color: #d9d9d9;
            opacity: 0.8;
            margin-top: 40px;
            border-top: 1px solid rgba(255,255,255,0.1);
            padding-top: 10px;
        }
        .error-box {
            background: rgba(255, 0, 0, 0.1);
            border: 1px solid rgba(255,0,0,0.4);
            color: #ffaaaa;
            padding: 10px;
            border-radius: 6px;
            margin: 10px auto;
            width: 70%;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- LOGO ----------
logo_path = "droneshield_logo.png"
if os.path.exists(logo_path):
    encoded_logo = base64.b64encode(open(logo_path, "rb").read()).decode()
    st.markdown(f"""
        <div class="logo-wrapper">
            <img src="data:image/png;base64,{encoded_logo}" />
            <div class="logo-reflection"></div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<div class='error-box'>⚠️ Logo not found — please upload 'droneshield_logo.png'.</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-top:-10px'></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='main-title'>Tim's <span class='title-grey'>Drone</span><span class='title-black'>Shield</span> Chatbot</div>",
    unsafe_allow_html=True
)
st.markdown("<p class='caption'>Built for clarity and precision — answers grounded in verified DroneShield content.</p>", unsafe_allow_html=True)

# ---------- LOAD CONTEXT ----------
try:
    with open("droneshield_parsed_data.json.txt", "r", encoding="utf-8") as f:
        context_data = json.load(f)
except Exception as e:
    st.markdown(f"<div class='error-box'>⚠️ Failed to load context data: {e}</div>", unsafe_allow_html=True)
    context_data = []

# Normalize context
if isinstance(context_data, list) and all(isinstance(i, dict) for i in context_data):
    documents = [item.get("content", "") for item in context_data]
    metas = [{"url": item.get("url", ""), "title": item.get("title", "")} for item in context_data]
else:
    documents = [str(item) for item in context_data]
    metas = [{"url": "", "title": ""} for _ in context_data]

# ---------- EMBEDDING MODEL ----------
try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
except Exception as e:
    st.markdown(f"<div class='error-box'>⚠️ Error loading embedding model: {e}</div>", unsafe_allow_html=True)
    st.stop()

def retrieve(query, k=4):
    query_emb = model.encode(query)
    doc_embs = model.encode(documents)
    scores = (query_emb @ doc_embs.T)
    top_k = scores.argsort()[-k:][::-1]
    return [(documents[i], metas[i]) for i in top_k]

# ---------- OPENAI CLIENT ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.markdown("<div class='error-box'>⚠️ Missing OpenAI API key. Please set it as an environment variable named 'OPENAI_API_KEY'.</div>", unsafe_allow_html=True)
else:
    openai_client = OpenAI(api_key=api_key)

    def answer(query):
        hits = retrieve(query)
        context = "\n\n".join([h[0] for h in hits]) if hits else "No relevant company data found."
        SYSTEM_PROMPT = """You are the DroneShield AI assistant.
        Use provided context where available.
        If info not found, answer accurately from verified knowledge and note it was external.
        Always end with 'Sources:' followed by bullet-point URLs.
        """
        try:
            resp = openai_client.chat.completions.create(
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
st.markdown("<div class='footer'>⚠️ This chatbot is an independent personal project inspired by DroneShield. Not affiliated with or endorsed by DroneShield Ltd. GPT-4 assists when company data is insufficient.</div>", unsafe_allow_html=True)
