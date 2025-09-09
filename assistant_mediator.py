"""
Two-Assistant Mediator (Controller)
-----------------------------------
Orchestrates a dialogue between:
  - Consultant assistant (proposes & defends one idea at a time)
  - Customer assistant (skeptical late-40s customer with some technical background)

Requirements:
  pip install --upgrade openai

Usage:
  1) Set environment var:  export OPENAI_API_KEY="sk-..."
  2) python assistant_mediator.py
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, Any, List
from openai import OpenAI

# -----------------------------
# Configurable parameters
# -----------------------------
MODEL_CONSULTANT = os.getenv("MODEL_CONSULTANT", "gpt-4o")
MODEL_CUSTOMER   = os.getenv("MODEL_CUSTOMER", "gpt-4o")

TEMP_CONSULTANT  = float(os.getenv("TEMP_CONSULTANT", "0.70"))
TEMP_CUSTOMER    = float(os.getenv("TEMP_CUSTOMER", "0.45"))

TOP_P_CONSULTANT = float(os.getenv("TOP_P_CONSULTANT", "1.0"))
TOP_P_CUSTOMER   = float(os.getenv("TOP_P_CUSTOMER", "1.0"))

MAX_TURNS        = int(os.getenv("MAX_TURNS", "12"))   # max back-and-forth pairs
LOG_DIR          = os.getenv("LOG_DIR", ".")

# Acceptance / rejection phrases
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

def is_match(text: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

# -----------------------------
# System instructions
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

# -----------------------------
# OpenAI client
# -----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_chat(model: str, system: str, user: str, temperature: float, top_p: float) -> str:
    """Call Chat Completions and return text content."""
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()

def main() -> None:
    transcript: List[Dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Consultant starts with a single best idea
    consultant_prompt = (
        "Propose your single best AI-first online business idea that meets all your constraints. "
        "Follow your 8-section structure. Do not list multiple ideas."
    )
    consultant_reply = call_chat(
        model=MODEL_CONSULTANT,
        system=CONSULTANT_SYSTEM,
        user=consultant_prompt,
        temperature=TEMP_CONSULTANT,
        top_p=TOP_P_CONSULTANT,
    )
    print("\n👔 Consultant:\n", consultant_reply, "\n")
    transcript.append({"role": "consultant", "content": consultant_reply})

    # 2) Dialogue loop
    turn = 0
    while turn < MAX_TURNS:
        turn += 1

        # Customer challenges
        customer_user = (
            "Act as the skeptical customer. Challenge the following proposal until you are convinced or reject it. "
            "Focus on ROI, feasibility, low cost, uniqueness, and proof. "
            "Remember to say explicitly if you accept or reject.\n\nPROPOSAL:\n" + consultant_reply
        )
        customer_reply = call_chat(
            model=MODEL_CUSTOMER,
            system=CUSTOMER_SYSTEM,
            user=customer_user,
            temperature=TEMP_CUSTOMER,
            top_p=TOP_P_CUSTOMER,
        )
        print("🧑 Customer:\n", customer_reply, "\n")
        transcript.append({"role": "customer", "content": customer_reply})

        # Check acceptance/rejection
        if is_match(customer_reply, ACCEPT_PATTERNS):
            print("✅ Customer accepted the idea.")
            break
        if is_match(customer_reply, REJECT_PATTERNS):
            # Ask consultant to propose a *new* single idea addressing the stated reasons.
            consultant_user = (
                "The customer rejected your idea with the response below. "
                "Propose a different, single idea that addresses the customer's reasons, "
                "still following your 8-section structure and all your constraints. "
                "Do not list multiple ideas.\n\nCUSTOMER RESPONSE:\n" + customer_reply
            )
        else:
            # Customer is challenging; ask consultant to refine the same idea
            consultant_user = (
                "The customer is challenging your idea. Refine the SAME idea to address every objection. "
                "Keep it a single idea. Be concise, data-driven, and defend feasibility/ROI/low-cost clearly.\n\n"
                "CUSTOMER CHALLENGES:\n" + customer_reply
            )

        consultant_reply = call_chat(
            model=MODEL_CONSULTANT,
            system=CONSULTANT_SYSTEM,
            user=consultant_user,
            temperature=TEMP_CONSULTANT,
            top_p=TOP_P_CONSULTANT,
        )
        print("👔 Consultant (refinement):\n", consultant_reply, "\n")
        transcript.append({"role": "consultant", "content": consultant_reply})

    # Save transcript
    os.makedirs(LOG_DIR, exist_ok=True)
    base = os.path.join(LOG_DIR, f"two_assistants_dialog_{timestamp}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("# Two-Assistant Dialogue Transcript\n\n")
        for turn in transcript:
            speaker = "Consultant" if turn["role"] == "consultant" else "Customer"
            f.write(f"## {speaker}\n\n{turn['content']}\n\n---\n\n")

    print(f"Transcript saved to: {base}.json and {base}.md")

if __name__ == "__main__":
    main()
