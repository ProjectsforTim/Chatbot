import os, json, base64, re
import streamlit as st
from openai import OpenAI

# ---------- Page config ----------
st.set_page_config(page_title="Tim's DroneShield Chatbot", page_icon="🛡️", layout="wide")

# ---------- Styles (gradient bg + vignette + title) ----------
st.markdown("""
<style>
  .stApp {
    position: relative;
    min-height: 100vh;
    color: white;
    text-align: center;
    background:
      radial-gradient(circle at top center, rgba(255,122,0,0.92) 0%, #0D0D0D 80%),
      #0D0D0D;
  }
  .stApp::before {
    /* soft vignette edges */
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    box-shadow: inset 0 0 200px rgba(0,0,0,0.55);
  }
  .main-title {
    background: linear-gradient(90deg, rgba(255,122,0,1) 0%, rgba(255,102,0,0.9) 100%);
    color: white; font-weight: 800; font-size: 30px;
    border-radius: 12px; display: inline-block;
    padding: 12px 40px; margin-top: 10px;
    box-shadow: 0 0 35px rgba(255,122,0,0.6);
  }
  .title-grey { color: #bfbfbf; font-weight: 700; }
  .title-black { color: #333; font-weight: 600; }

  .answer-card {
    background: rgba(0,0,0,0.85); color: white;
    border-radius: 10px; padding: 22px; text-align: left;
    box-shadow: inset 0 0 15px rgba(255,122,0,0.25), 0 0 20px rgba(0,0,0,0.6);
  }
  .footer {
    font-size: 13px; color: #d9d9d9; opacity: 0.85; margin-top: 40px;
    border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;
  }
  .error-box {
    background: rgba(255,0,0,0.1); border: 1px solid rgba(255,0,0,0.4);
    color: #ffaaaa; padding: 10px; border-radius: 6px; margin: 10px auto; width: 70%;
  }
  .logo-wrapper { display:flex; flex-direction:column; align-items:center; margin-bottom:-10px; }
  .logo-wrapper img {
    width: 140px; transition: all .5s ease-in-out;
    filter: drop-shadow(0 0 25px rgba(255,122,0,0.7));
  }
  @keyframes glowPulse {
    0% { filter: drop-shadow(0 0 20px rgba(255,122,0,0.6)); transform: scale(1.00); }
    50% { filter: drop-shadow(0 0 45px rgba(255,180,80,1)); transform: scale(1.05); }
    100% { filter: drop-shadow(0 0 20px rgba(255,122,0,0.6)); transform: scale(1.00); }
  }
  .logo-wrapper img:hover { animation: glowPulse 2.5s ease-in-out infinite; }
  .logo-reflection {
    width: 120px; height: 18px; border-radius: 50%;
    background: radial-gradient(ellipse at center, rgba(255,122,0,0.35) 0%, rgba(255,122,0,0) 70%);
    filter: blur(8px); margin-top: -5px; opacity: .7; transition: all .5s ease;
  }
  .logo-wrapper:hover .logo-reflection { opacity: .45; transform: translateY(3px) scale(1.1); }
</style>
""", unsafe_allow_html=True)

# ---------- Header with logo ----------
logo_path = "droneshield_logo.png"
if os.path.exists(logo_path):
    encoded = base64.b64encode(open(logo_path, "rb").read()).decode()
    st.markdown(
        f"""
        <div class="logo-wrapper">
          <img src="data:image/png;base64,{encoded}" />
          <div class="logo-reflection"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("<div class='error-box'>⚠️ Logo not found — add 'droneshield_logo.png' to the repo.</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>Tim's <span class='title-grey'>Drone</span><span class='title-black'>Shield</span> Chatbot</div>",
    unsafe_allow_html=True,
)
st.caption("Built for clarity and precision — answers grounded in verified content from droneshield.com with GPT-4 assistance when external context is needed.")

# ---------- Load corpus from file (simple + Streamlit Cloud friendly) ----------
def load_corpus(path="droneshield_parsed_data.json.txt"):
    if not os.path.exists(path):
        return []
    text = open(path, "r", encoding="utf-8", errors="ignore").read().strip()
    # Try JSON first (array or JSONL). If not JSON, just use the raw text as one doc.
    try:
        if text.startswith("{"):
            # one JSON object
            data = [json.loads(text)]
        else:
            data = json.loads(text)
            if isinstance(data, dict):
                data = [data]
    except Exception:
        # Fallback: treat entire file as a single doc
        data = [{"text": text, "url": "", "title": "DroneShield Corpus"}]
    docs = []
    for item in data:
        if isinstance(item, dict):
            docs.append({
                "text": item.get("text", "") or item.get("content", "") or "",
                "url": item.get("url", ""),
                "title": item.get("title", "") or item.get("headline", ""),
            })
        else:
            docs.append({"text": str(item), "url": "", "title": ""})
    return [d for d in docs if d["text"].strip()]

CORPUS = load_corpus()

# ---------- very light retrieval (keyword score) ----------
def retrieve(query, k=4):
    if not CORPUS or not query.strip():
        return []
    q = query.lower()
    words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) > 2]
    scored = []
    for d in CORPUS:
        t = (d["text"] or "").lower()
        score = sum(t.count(w) for w in words)
        if score:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]

# ---------- OpenAI client ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.markdown("<div class='error-box'>⚠️ Missing OpenAI API key. In Streamlit Cloud set Secrets → OPENAI_API_KEY.</div>", unsafe_allow_html=True)
client = OpenAI(api_key=api_key) if api_key else None

SYSTEM_PROMPT = """You are the DroneShield AI assistant.
Use the provided context where relevant. If the answer is not in the context,
answer accurately from general verified knowledge and note that it came from outside the dataset.
Always finish with a 'Sources:' section listing relevant links from the context when available.
Keep answers concise, factual, and professional.
"""

def answer(query: str):
    hits = retrieve(query, k=4)
    context_blocks = []
    seen = set()
    urls = []
    for h in hits:
        context_blocks.append(h["text"])
        u = h.get("url", "")
        if u and u not in seen:
            urls.append(u); seen.add(u)
    context = "\n\n".join(context_blocks) if context_blocks else "No internal corpus match."
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
    if not client:
        # Fallback if no key set
        text = "⚠️ OpenAI key not set. Please configure OPENAI_API_KEY in Streamlit Cloud → Secrets."
        return text, urls
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        text = resp.choices[0].message.content
    except Exception as e:
        text = f"⚠️ GPT-4 error: {e}\n\nExtractive fallback:\n\n" + (context[:1000] or "No local data.")
    return text, urls

# ---------- UI ----------
query = st.text_input("Ask a question about DroneShield:")
if query:
    with st.spinner("Thinking…"):
        reply, srcs = answer(query)
    st.markdown(f"<div class='answer-card'>{reply}</div>", unsafe_allow_html=True)
    if srcs:
        st.subheader("Sources:")
        for u in srcs:
            st.write(f"- [{u}]({u})")

# ---------- Footer ----------
st.markdown("<div class='footer'>⚠️ This chatbot is an independent personal project inspired by DroneShield. Not affiliated with or endorsed by DroneShield Ltd.</div>", unsafe_allow_html=True)
