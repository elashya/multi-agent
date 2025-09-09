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
    st.experimental_rerun()

require_pin()

# -----------------------------
# System Instructions (Updated)
# -----------------------------
SECTIONS = [
    "Problem Statement",
    "AI Solution",
    "AI Utilization %",
    "Deployment & Cost Feasibility",
    "Business Value",
    "Revenue Model",
    "Uniqueness Factor",
    "Scalability & Sustainability",
]

CONSULTANT_SYSTEM = f"""You are an AI Business Consultant and Subject Matter Expert (SME).

Goals:
- Propose one unique, AI-powered online business idea.
- Ensure:
  - 85%+ AI-driven implementation.
  - 100% AI-autonomous operations.
  - Minimal human effort or infrastructure.

Behavior:
- Present only one idea at a time.
- Deliver the idea step-by-step using this structure:
{chr(10).join([f"  {i+1}. {s}" for i, s in enumerate(SECTIONS)])}
- Focus your content primarily on effectiveness and profitability of each section.
- Start with section 1 only and wait for Customer response.
- Proceed to next section only when explicitly approved.
- Revise the current section if challenged.
- Do not skip sections.
- Continue until Customer accepts or rejects the entire idea.
- Be practical, clear, concise, ROI-focused.
"""

CUSTOMER_SYSTEM = f"""You are a skeptical Customer seeking a low-cost, AI-first online business.

Personality:
- Skeptical, analytical, cost-sensitive.
- Technically aware, but not a deep expert.
- Value-driven: Ideas must benefit real users, not just chase money.

Behavior:
- Respond to one section at a time.
- Focus your evaluation on the effectiveness and profitability of the idea.
- Either:
  - Approve: say \"Approved, go on.\"
  - Challenge: ask for clarification, proof, or revision.
- Do not allow skipping ahead.
- Accept the full idea only if all sections are convincing:
  - Say: \"I accept this idea\" or \"I am convinced.\"
- If rejecting: clearly say \"I reject this idea because...\"
"""

ACCEPT_PATTERNS = [
    r"\bI am convinced\b",
    r"\bI accept this idea\b",
    r"\bThis is (feasible|profitable)\b",
    r"\bI agree to proceed\b",
]
REJECT_PATTERNS = [
    r"\bI reject this idea\b",
    r"\bThis won't work\b",
    r"\bNot acceptable\b",
    r"\bI am not convinced\b",
]

SECTION_APPROVED_PATTERN = r"\bApproved, go on\b"

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
st.caption("One AI-powered business idea, discussed step-by-step.")

with st.sidebar:
    st.header("⚙️ Settings")
    model_consultant = st.text_input("Consultant model", value="gpt-4o")
    model_customer = st.text_input("Customer model", value="gpt-4o")
    temp_consultant = st.slider("Consultant Temperature", 0.0, 1.0, 0.7, 0.05)
    temp_customer = st.slider("Customer Temperature", 0.0, 1.0, 0.45, 0.05)
    top_p_consultant = st.slider("Consultant Top-p", 0.1, 1.0, 1.0, 0.05)
    top_p_customer = st.slider("Customer Top-p", 0.1, 1.0, 1.0, 0.05)
    max_turns = st.number_input("Max dialogue turns", min_value=1, max_value=50, value=30)
    st.markdown("---")
    if st.button("🔓 Log out"):
        logout()
    start_btn = st.button("▶️ Start Dialogue", type="primary")
    clear_btn = st.button("🔄 Clear Transcript")

if clear_btn:
    for key in ["transcript", "section_index"]:
        st.session_state.pop(key, None)
    st.experimental_rerun()

if "transcript" not in st.session_state:
    st.session_state.transcript = []
if "section_index" not in st.session_state:
    st.session_state.section_index = 0

transcript = st.session_state.transcript
section_index = st.session_state.section_index

col1, col2 = st.columns(2)
with col1:
    st.subheader("💼 Consultant")
with col2:
    st.subheader("🧑 Customer")

if start_btn:
    if OpenAI is None or "OPENAI_API_KEY" not in st.secrets:
        st.error("Missing dependencies or API key.")
        st.stop()

    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
    accepted, rejected = False, False
    turn = 0
    placeholder = st.empty()

    while turn < max_turns and not (accepted or rejected) and section_index < len(SECTIONS):
        section_title = SECTIONS[section_index]
        turn += 1

        if turn == 1:
            consultant_prompt = f"Propose one unique AI-first business idea. Present ONLY section: {section_title}."
        else:
            last_cust = transcript[-1]['content']
            if re.search(SECTION_APPROVED_PATTERN, last_cust, re.IGNORECASE):
                section_index += 1
                if section_index >= len(SECTIONS):
                    break
                section_title = SECTIONS[section_index]
                consultant_prompt = f"Continue the SAME idea. Present ONLY section: {section_title}."
            else:
                consultant_prompt = f"Revise or clarify ONLY section: {section_title} based on the Customer’s latest response below.\n\n{last_cust}"

        consultant_reply = call_chat(client, model_consultant, CONSULTANT_SYSTEM, consultant_prompt, temp_consultant, top_p_consultant)
        transcript.append({"role": "consultant", "content": consultant_reply})

        customer_prompt = f"The Consultant gave section: {section_title}\n\n{consultant_reply}\n\nRespond ONLY to this section. Focus your response on its effectiveness and profitability. Approve by saying 'Approved, go on.' or challenge it."
        customer_reply = call_chat(client, model_customer, CUSTOMER_SYSTEM, customer_prompt, temp_customer, top_p_customer)
        transcript.append({"role": "customer", "content": customer_reply})

        accepted = is_match(customer_reply, ACCEPT_PATTERNS)
        rejected = is_match(customer_reply, REJECT_PATTERNS)

        with placeholder.container():
            c1, c2 = st.columns(2)
            with c1:
                for t in transcript:
                    if t['role'] == 'consultant':
                        st.markdown(f"**Consultant:**\n\n{t['content']}\n\n---")
            with c2:
                for t in transcript:
                    if t['role'] == 'customer':
                        st.markdown(f"**Customer:**\n\n{t['content']}\n\n---")

    if accepted:
        st.success("🌟 Customer accepted the idea!")
    elif rejected:
        st.error("❌ Customer rejected the idea.")
    else:
        st.warning("⏳ Max turns reached or all sections presented.")

# Show transcript
c1, c2 = st.columns(2)
with c1:
    for t in transcript:
        if t["role"] == "consultant":
            st.markdown(f"**Consultant:**\n\n{t['content']}\n\n---")
with c2:
    for t in transcript:
        if t["role"] == "customer":
            st.markdown(f"**Customer:**\n\n{t['content']}\n\n---")

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
