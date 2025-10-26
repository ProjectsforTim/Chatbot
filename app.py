import streamlit as st
import os, json, base64, textwrap, numpy as np, hashlib
from openai import OpenAI

st.set_page_config(page_title="Tim's DroneShield Chatbot", page_icon="🛡️")

# ---------- CUSTOM STYLING ----------
st.markdown("""
    <style>
        .stApp {
            background: radial-gradient(circle at top center, rgba(255,122,0,0.9) 0%, #0D0D0D 90%);
            color: white; text-align: center;
            animation: fadeIn 1s ease-in;
        }

        /* --- Title Styling with Glow + Gradient --- */
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
            animation: titlePulse 4s ease-in-out infinite;
        }
        .title-grey { color: #bfbfbf; font-weight: 600; }
        .title-black { color: #333333; font-weight: 500; }

        @keyframes titlePulse {
            0% { box-shadow: 0 0 25px rgba(255,122,0,0.4); transform: scale(1.00); }
            50% { box-shadow: 0 0 55px rgba(255,180,80,1); transform: scale(1.03); }
            100% { box-shadow: 0 0 25px rgba(255,122,0,0.4); transform: scale(1.00); }
        }

        /* --- Hover-Animated Logo --- */
        .logo-wrapper {
            display: flex; justify-content: center; align-items: center;
            flex-direction: column; margin-bottom: -10px;
        }
        .logo-wrapper img {
            width: 140px;
            transition: all 0.5s ease-in-out;
            filter: drop-shadow(0 0 25px rgba(255,122,0,0.5));
        }
        .logo-wrapper:hover img {
            transform: translateY(-8px) scale(1.08);
            filter: drop-shadow(0 0 45px rgba(255,180,80,1));
        }

        /* --- Logo Reflection --- */
        .logo-reflection {
            width: 120px; height: 18px; border-radius: 50%;
            background: radial-gradient(ellipse at center, rgba(255,122,0,0.35) 0%, rgba(255,122,0,0) 70%);
            filter: blur(8px); margin-top: -5px; opacity: 0.7;
            transition: all 0.5s ease;
        }
        .logo-wrapper:hover .logo-reflection {
            opacity: 0.45; transform: translateY(3px) scale(1.1);
        }

        /* --- Chat Response Card --- */
        .answer-card {
            background: rgba(0, 0, 0, 0.85); color: white;
            border-radius: 10px; padding: 22px; text-align: left;
            box-shadow: inset 0px 0px 15px rgba(255,122,0,0.25), 0 0 20px rgba(0,0,0,0.6);
        }

        /* --- Footer Disclaimer --- */
        .footer {
            font-size: 13px; color: #d9d9d9; opacity: 0.8;
            margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.1);
            padding-top: 10px;
        }

        /* --- Subtle fade-in animation for content --- */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
if os.path.exists("droneshield_logo.png"):
    encoded_logo = base64.b64encode(open("droneshield_logo.png", "rb").read()).decode()
    st.markdown(f"""
        <div class="logo-wrapper">
            <img src="data:image/png;base64,{encoded_logo}" />
            <div class="logo-reflection"></div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Logo not found — please upload 'droneshield_logo.png'.")

st.markdown("<div class='main-title'>Tim's <span class='title-grey'>Drone</span><span class='title-black'>Shield</span> Chatbot</div>", unsafe_allow_html=True)

# ---------- LOAD CONTEXT ----------
try:
    with open("droneshield_parsed_data.json.txt", "r", encoding="utf-8") as f:
        context_data = json.load(f)
except Exception as e:
    st.error(f"⚠️ Could not load context: {e}")
    context_data = []

documents, metas = [], []
for item in context_data:
    if isinstance(item, dict):
        documents.append(item.get("content", ""))
        metas.append({"url": item.get("url", ""), "title": item.get("title", "")})
    else:
        documents.append(str(item))
        metas.append({"url": "", "title": ""})

# ---------- LIGHT EMBEDDING (no torch) ----------
def cheap_embed(text):
    hash_val = hashlib.sha256(text.encode("utf-8")).digest()
    return np.array([b / 255.0 for b in hash_val[:64]])

def retrieve(query, k=4):
    q_emb = cheap_embed(query)
    doc_embs = [cheap_embed(d) for d in documents]
    sims = [np.dot(q_emb, d) / (np.linalg.norm(q_emb) * np.linalg.norm(d)) for d in doc_embs]
    top_k = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    return [(documents[i], metas[i]) for i in top_k]

# ---------- OPENAI ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ Missing OpenAI API key. Please set it as 'OPENAI_API_KEY'.")
else:
    client = OpenAI(api_key=api_key)

    def answer(query):
        hits = retrieve(query)
        context = "\n\n".join([h[0] for h in hits]) if hits else "No company data found."
        prompt = """You are the DroneShield AI assistant.
        Use the context if relevant, otherwise answer from verified public info and note that it was external.
        Always end with 'Sources:' and bullet-point URLs."""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            text = resp.choices[0].message.content
        except Exception as e:
            text = f"⚠️ GPT error: {e}\n\nExtractive fallback:\n{textwrap.fill(context[:1000], 110)}"

        srcs, seen = [], set()
        for h in hits:
            url = h[1].get("url", "")
            if url and url not in seen:
                seen.add(url)
                srcs.append(url)
        return text, srcs

    query = st.text_input("Ask a question about DroneShield:")
    if query:
        with st.spinner("Thinking..."):
            resp_text, srcs = answer(query)
        st.markdown(f"<div class='answer-card'>{resp_text}</div>", unsafe_allow_html=True)
        if srcs:
            st.subheader("Sources:")
            for s in srcs:
                st.write(f"- [{s}]({s})")

st.markdown("<div class='footer'>⚠️ Independent personal project inspired by DroneShield. Not affiliated with or endorsed by DroneShield Ltd.</div>", unsafe_allow_html=True)
