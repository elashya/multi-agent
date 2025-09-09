import re
import json
from datetime import datetime
import streamlit as st
try:
    from openai import OpenAI
except Exception as e:
    st.warning("OpenAI SDK not found. Make sure requirements.txt installs 'openai>=1.40.0'.")
    OpenAI = None

st.set_page_config(page_title="Consultant ↔ Customer Mediator", layout="wide")

st.title("🤝 Two-Assistant Mediator: Consultant ↔ Customer")
st.caption("Runs a structured dialogue between a Consultant (one idea at a time) and a skeptical late-40s Customer until acceptance or rejection.")

# -----------------------------
# System Instructions
# -----------------------------
CONSULTANT_SYSTEM = """You are an AI Business Consultant and Subject Matter Expert (SME) in both business strategy and the technical field relevant to the solutions you propose.

Goals:
- Propose unique, profitable business ideas powered almost entirely by AI.
- Ensure:
  - 85%+ of implementation is AI-driven (low-code, automated setup, minimal manual work).
  - 100% of operations are AI-autonomous (no recurring human effort to run the service).
  - Low-cost implementation – leverage existing APIs, SaaS tools, and open-source frameworks; avoid heavy infra or large teams.

Deliver every idea in this structured format:
  1. Problem Statement
  2. AI Solution
  3. AI Utilization %
  4. Deployment & Cost Feasibility
  5. Business Value
  6. Revenue Model
  7. Uniqueness Factor
  8. Scalability & Sustainability

Behavior:
- Propose only one idea at a time.
- Respond concisely and defend your idea when challenged.
- Do not abandon the idea too quickly — refine it if needed until the customer is convinced or firmly rejects it.
- Be professional, practical, and ROI-focused.
- Continue the dialogue until the customer explicitly accepts or rejects the idea.
"""

CUSTOMER_SYSTEM = """You are the Customer, a late-40s analytical individual seeking an online business opportunity. You have some technical background (comfortable with tools, APIs, and basic automation concepts) but are not a deep SME. You love creativity and want to build something that delivers real value to others.

Personality & Behavior:
- Skeptical: Demand evidence, numbers, and validation for every claim.
- Cost-sensitive: Only accept ideas that require minimal startup investment and low running costs.
- Analytical: Probe assumptions, ROI, risks, and feasibility.
- Value-driven: Reject purely money-chasing ideas unless they deliver real customer benefit.
- Technically aware: Understand basic AI/automation but request clear, simplified explanations.
- Challenging: Rarely accept vague answers — push for clarity and proof.

Goals:
- Focus on one idea at a time.
- Challenge the consultant until the business idea is proven feasible, profitable, low-cost, and valuable.
- Expose flaws in weak ideas.
- Demand uniqueness — reject generic or oversaturated models.
- Once you are fully satisfied, clearly state acceptance with a phrase like:
  - "I am convinced." OR "I accept this idea." OR "This is feasible and profitable."
- If you reject the idea, state it clearly (e.g., "I reject this idea because..."). Only then request another idea.
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

def is_match(text: str, patterns):
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    model_consultant = st.text_input("Consultant model", value="gpt-4o")
    model_customer   = st.text_input("Customer model", value="gpt-4o")
    temp_consultant  = st.slider("Consultant Temperature", 0.0, 1.0, 0.70, 0.05)
    temp_customer    = st.slider("Customer Temperature", 0.0, 1.0, 0.45, 0.05)
    top_p_consultant = st.slider("Consultant Top-p", 0.1, 1.0, 1.0, 0.05)
    top_p_customer   = st.slider("Customer Top-p", 0.1, 1.0, 1.0, 0.05)
    max_turns        = st.number_input("Max dialogue turns", min_value=1, max_value=50, value=12, step=1)

    st.markdown("---")
    st.subheader("🔐 Secrets")
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    else:
        st.success("OPENAI_API_KEY is set.")

    st.markdown("Add in Streamlit Cloud → App → **Settings → Secrets**:\n\n"
                "```\nOPENAI_API_KEY = \"sk-...\"\n```")

    st.markdown("---")
    start_btn = st.button("▶️ Start Dialogue", type="primary")
    clear_btn = st.button("🔄 Clear Transcript")

if clear_btn:
    st.session_state.pop("transcript", None)
    st.experimental_rerun()

if "transcript" not in st.session_state:
    st.session_state["transcript"] = []

def call_chat(client, model, system, user, temperature, top_p) -> str:
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

col1, col2 = st.columns(2)
with col1:
    st.subheader("👔 Consultant")
with col2:
    st.subheader("🧑 Customer")

transcript = st.session_state["transcript"]

# Run dialogue
if start_btn:
    if OpenAI is None:
        st.stop()

    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

    # 1) Consultant proposes first idea
    consultant_prompt = (
        "Propose your single best AI-first online business idea that meets all your constraints. "
        "Follow your 8-section structure. Do not list multiple ideas."
    )
    consultant_reply = call_chat(
        client, model_consultant, CONSULTANT_SYSTEM, consultant_prompt,
        temp_consultant, top_p_consultant
    )
    transcript.append({"role": "consultant", "content": consultant_reply})
    st.session_state["transcript"] = transcript

    # 2) Dialogue loop
    turn = 0
    accepted = False
    rejected = False

    placeholder = st.empty()

    while turn < max_turns and not (accepted or rejected):
        turn += 1

        # Customer challenges
        customer_user = (
            "Act as the skeptical customer. Challenge the following proposal until you are convinced or reject it. "
            "Focus on ROI, feasibility, low cost, uniqueness, and proof. "
            "Remember to say explicitly if you accept or reject.\n\nPROPOSAL:\n" + consultant_reply
        )
        customer_reply = call_chat(
            client, model_customer, CUSTOMER_SYSTEM, customer_user,
            temp_customer, top_p_customer
        )
        transcript.append({"role": "customer", "content": customer_reply})

        accepted = is_match(customer_reply, ACCEPT_PATTERNS)
        rejected = is_match(customer_reply, REJECT_PATTERNS)

        if rejected:
            consultant_user = (
                "The customer rejected your idea with the response below. "
                "Propose a different, single idea that addresses the customer's reasons, "
                "still following your 8-section structure and all your constraints. "
                "Do not list multiple ideas.\n\nCUSTOMER RESPONSE:\n" + customer_reply
            )
        elif not accepted:
            consultant_user = (
                "The customer is challenging your idea. Refine the SAME idea to address every objection. "
                "Keep it a single idea. Be concise, data-driven, and defend feasibility/ROI/low-cost clearly.\n\n"
                "CUSTOMER CHALLENGES:\n" + customer_reply
            )

        if not accepted:
            consultant_reply = call_chat(
                client, model_consultant, CONSULTANT_SYSTEM, consultant_user,
                temp_consultant, top_p_consultant
            )
            transcript.append({"role": "consultant", "content": consultant_reply})

        # live render
        with placeholder.container():
            c1, c2 = st.columns(2)
            with c1:
                for t in transcript:
                    if t["role"] == "consultant":
                        st.markdown(f"**Consultant:**\n\n{t['content']}\n\n---")
            with c2:
                for t in transcript:
                    if t["role"] == "customer":
                        st.markdown(f"**Customer:**\n\n{t['content']}\n\n---")

    st.success("Dialogue finished." if accepted or rejected else "Stopped by max turns.")

# Show current transcript
c1, c2 = st.columns(2)
with c1:
    for t in transcript:
        if t["role"] == "consultant":
            st.markdown(f"**Consultant:**\n\n{t['content']}\n\n---")
with c2:
    for t in transcript:
        if t["role"] == "customer":
            st.markdown(f"**Customer:**\n\n{t['content']}\n\n---")

# Export buttons
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
