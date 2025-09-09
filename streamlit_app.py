import re
import json
from datetime import datetime
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    st.warning("OpenAI SDK not found. Make sure requirements.txt installs 'openai>=1.40.0'.")
    OpenAI = None

st.set_page_config(page_title="Consultant ↔ Customer Mediator", layout="wide")

# -----------------------------
# Auth (PIN Gate)
# -----------------------------
def require_pin():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if "auth_tries" not in st.session_state:
        st.session_state.auth_tries = 0

    configured_pin = st.secrets.get("APP_PIN")
    if not configured_pin:
        st.stop()
    if st.session_state.auth_ok:
        return True

    st.markdown("### 🔐 Enter PIN to access the app")
    with st.form("pin_form", clear_on_submit=False):
        pin = st.text_input("PIN", type="password", help="Contact the owner if you don't have the PIN.")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        if pin == configured_pin:
            st.session_state.auth_ok = True
            st.session_state.auth_tries = 0
            st.success("Unlocked.")
            return True
        else:
            st.session_state.auth_tries += 1
            st.error("Incorrect PIN.")
    st.stop()

def logout():
    st.session_state.auth_ok = False
    st.session_state.auth_tries = 0
    st.rerun()

require_pin()

# -----------------------------
# System Instructions (Single-Focus Version)
# -----------------------------
CONSULTANT_SYSTEM = """You are an AI Business Consultant and Subject Matter Expert (SME).

Your goal is to propose ONE unique online business idea that is:
- Powered primarily by AI (85%+ AI-driven implementation)
- Fully automated (100% AI-run operations)
- Profitable with minimal deployment cost (using existing APIs, SaaS tools, or open-source frameworks)

Behavior:
- Do not list multiple ideas.
- Present only one concise idea.
- Your idea must emphasize effectiveness and profitability.
- Clearly highlight why it has minimal deployment cost.
- Keep your response under 5 concise sentences.
- Be practical, ROI-focused, and persuasive.
- Wait for the Customer to respond with either acceptance or challenges.
- Do not propose another idea unless the first is rejected.
"""

CUSTOMER_SYSTEM = """You are a skeptical Customer seeking a profitable, low-cost, AI-powered online business idea.

Behavior:
- Listen to only ONE idea at a time.
- Challenge the Consultant ONLY on its profitability and low deployment cost.
- Ask for clarification if any part of the idea seems vague, costly, or unconvincing.
- Be concise (2–3 sentence responses).
- Accept the idea only if it is clearly profitable AND has minimal setup cost.
- If convinced, say: "I accept this idea."
- If not convinced, say: "I reject this idea because..."
- Do not ask for another idea unless the first is rejected.
"""

ACCEPT_PATTERNS = [
    r"\bI accept this idea\b",
    r"\bI am convinced\b",
    r"\bThis is (feasible|profitable)\b",
    r"\bI agree to proceed\b"
]
REJECT_PATTERNS = [
    r"\bI reject this idea\b",
    r"\bThis won't work\b",
    r"\bNot acceptable\b",
    r"\bI am not convinced\b"
]

def is_match(text, patterns):
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def call_chat(client, model, system, user, temperature, top_p):
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return r.choices[0].message.content.strip()

# -----------------------------
# UI
# -----------------------------
st.title("🤝 Two-Assistant Mediator: Consultant ↔ Customer")
st.caption("One AI-first business idea only — must be profitable and low-cost to launch.")

with st.sidebar:
    st.header("⚙️ Settings")
    model_consultant = st.text_input("Consultant model", value="gpt-4o")
    model_customer = st.text_input("Customer model", value="gpt-4o")
    temp_consultant = st.slider("Consultant Temperature", 0.0, 1.0, 0.7, 0.05)
    temp_customer = st.slider("Customer Temperature", 0.0, 1.0, 0.45, 0.05)
    top_p_consultant = st.slider("Consultant Top-p", 0.1, 1.0, 1.0, 0.05)
    top_p_customer = st.slider("Customer Top-p", 0.1, 1.0, 1.0, 0.05)
    max_turns = st.number_input("Max dialogue turns", min_value=1, max_value=20, value=6)
    st.markdown("---")
    if st.button("🔓 Log out"):
        logout()
    start_btn = st.button("▶️ Start Dialogue", type="primary")
    clear_btn = st.button("🔄 Clear Transcript")

if clear_btn:
    st.session_state.pop("transcript", None)
    st.rerun()

if "transcript" not in st.session_state:
    st.session_state.transcript = []

transcript = st.session_state.transcript

if start_btn:
    if OpenAI is None or "OPENAI_API_KEY" not in st.secrets:
        st.error("Missing dependencies or API key.")
        st.stop()

    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
    accepted, rejected = False, False
    turn = 0
    placeholder = st.empty()

    # First idea from Consultant
    consultant_prompt = "Propose a unique AI-first online business idea that is profitable and has minimal deployment cost."
    consultant_reply = call_chat(client, model_consultant, CONSULTANT_SYSTEM, consultant_prompt, temp_consultant, top_p_consultant)
    transcript.append({"role": "consultant", "content": consultant_reply})

    while turn < max_turns and not (accepted or rejected):
        turn += 1

        customer_prompt = f"The Consultant proposed this idea:\n\n{consultant_reply}\n\nEvaluate ONLY its profitability and deployment cost. Respond in 2–3 sentences."
        customer_reply = call_chat(client, model_customer, CUSTOMER_SYSTEM, customer_prompt, temp_customer, top_p_customer)
        transcript.append({"role": "customer", "content": customer_reply})

        accepted = is_match(customer_reply, ACCEPT_PATTERNS)
        rejected = is_match(customer_reply, REJECT_PATTERNS)

        if not accepted and not rejected:
            consultant_prompt = f"The Customer replied:\n\n{customer_reply}\n\nRefine the SAME idea to better prove its profitability and minimal deployment cost. Keep your reply short."
            consultant_reply = call_chat(client, model_consultant, CONSULTANT_SYSTEM, consultant_prompt, temp_consultant, top_p_consultant)
            transcript.append({"role": "consultant", "content": consultant_reply})

        with placeholder.container():
            for t in transcript:
                speaker = "Consultant" if t['role'] == 'consultant' else "Customer"
                st.markdown(f"**{speaker}:**\n\n{t['content']}\n\n---")

    if accepted:
        st.success("Customer accepted the idea!")
    elif rejected:
        st.error("Customer rejected the idea.")
    else:
        st.warning("Max turns reached.")

for t in transcript:
    speaker = "Consultant" if t["role"] == "consultant" else "Customer"
    st.markdown(f"**{speaker}:**\n\n{t['content']}\n\n---")

if transcript:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md = ["# Two-Assistant Dialogue Transcript\n"]
    for t in transcript:
        speaker = "Consultant" if t["role"] == "consultant" else "Customer"
        md.append(f"## {speaker}\n\n{t['content']}\n\n---\n")
    md_text = "\n".join(md)
    json_text = json.dumps(transcript, ensure_ascii=False, indent=2)
    st.download_button("⬇️ Download transcript (.md)", data=md_text, file_name=f"dialogue_{ts}.md")
    st.download_button("⬇️ Download transcript (.json)", data=json_text, file_name=f"dialogue_{ts}.json")
