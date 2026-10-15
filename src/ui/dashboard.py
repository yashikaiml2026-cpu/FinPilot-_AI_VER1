from __future__ import annotations

import json
import re

import streamlit as st

from src.agent.agent import run_finpilot_agent
from src.data.pipeline import ingest_raw_datasets


QUICK_ACTIONS = [
    ("Calculate EMI", "Calculate EMI"),
    ("Apply for Personal Loan", "I want to apply for a personal loan"),
    ("Apply for Home Loan", "I want to apply for a home loan"),
    ("Check Eligibility", "Check my loan eligibility"),
    ("Show Account Details", "Show my account details"),
    ("Track Application", "Track my latest loan application"),
    ("Recall My Preferences", "What do you remember about my name and preferences?"),
]


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
            :root { --ink:#070d13; --panel:#0d1720; --line:#243541; --cyan:#61e7f5; --mint:#a8f5d0; --amber:#ffd58a; --white:#f1f7fa; --muted:#adbdc6; }
            .stApp { background: radial-gradient(ellipse at 50% -20%, #203a46 0%, transparent 45%), linear-gradient(160deg,#071018,#0a131b 48%,#10191d); color:var(--white); font-family:'DM Sans',sans-serif; }
            [data-testid="stHeader"] { background:transparent; }
            .block-container { max-width:1320px; padding-top:2rem; padding-bottom:3rem; }
            h1,h2,h3,p,label,div { letter-spacing:0 !important; }
            h1,h2,h3 { color:var(--white); }
            p, label, [data-testid="stMarkdownContainer"] { color:var(--muted); }
            .command-head { display:flex; align-items:center; gap:1.4rem; border-bottom:1px solid var(--line); padding:0 0 1.5rem; margin-bottom:1.4rem; }
            .eyebrow { color:var(--cyan); text-transform:uppercase; font:500 .75rem 'IBM Plex Mono',monospace; }
            .command-title { color:var(--white); font-size:1.8rem; font-weight:600; margin:.18rem 0; }
            .command-sub { color:var(--muted); font-size:.92rem; }
            .core-wrap { width:104px; height:104px; position:relative; display:grid; place-items:center; flex:none; }
            .core { width:52px; height:52px; border-radius:50%; background:radial-gradient(circle at 35% 32%,#e9ffff 0,#8cf3f3 17%,#2eb9cc 48%,#12505f 78%); box-shadow:0 0 18px #50e7f5,0 0 44px #25b9d588,inset 0 0 14px #d9ffff; animation:core-breathe 2.8s ease-in-out infinite; }
            .core-ring { position:absolute; inset:8px; border:1px solid #63e9f588; border-radius:50%; animation:ring-turn 9s linear infinite; }
            .core-ring:after { content:''; position:absolute; width:5px;height:5px; border-radius:50%; background:var(--mint); top:8px; left:14px; box-shadow:0 0 10px var(--mint); }
            .core-ring.two { inset:0; border-color:#ffd58a55; animation-duration:14s; animation-direction:reverse; }
            @keyframes core-breathe { 0%,100% { transform:scale(.94); filter:brightness(.9); } 50% { transform:scale(1.06); filter:brightness(1.25); } }
            @keyframes ring-turn { to { transform:rotate(360deg); } }
            .state-pill { margin-left:auto; color:var(--mint); border:1px solid #61e7f555; padding:.45rem .75rem; border-radius:999px; font:.72rem 'IBM Plex Mono',monospace; }
            .section-rule { border-top:1px solid var(--line); margin:1rem 0; }
            .stButton > button { border:1px solid #36505b; border-radius:7px; color:var(--white); background:linear-gradient(135deg,#13232c,#0c171e); min-height:2.65rem; transition:border-color .2s,transform .2s,box-shadow .2s; }
            .stButton > button:hover { border-color:var(--cyan); color:var(--white); box-shadow:0 0 18px #49dcec33; transform:translateY(-1px); }
            .stButton > button[kind="primary"] { background:linear-gradient(100deg,#1b7480,#1c5c68); border-color:#63e7ef; }
            [data-testid="stChatMessage"] { border:1px solid #273b46; background:linear-gradient(135deg,#101c25dd,#0b141bdd); border-radius:10px; }
            [data-testid="stChatInput"] textarea { color:var(--white) !important; }
            [data-testid="stExpander"] { border:1px solid var(--line); border-radius:8px; background:#0a141b99; }
            [data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stSelectbox"] div[data-baseweb="select"] { background:#0c1820; color:var(--white); border-color:#36505b; }
            .trace-row { border-left:2px solid #395360; padding:.42rem .9rem; margin-left:.35rem; font:.82rem 'IBM Plex Mono',monospace; color:#d8e7eb; }
            .trace-row.done { border-color:var(--mint); }
            .trace-row.wait { border-color:var(--amber); color:var(--amber); }
            .micro { color:#9fb2bc; font:.73rem 'IBM Plex Mono',monospace; }
            @media(max-width:700px) { .command-head { gap:.7rem; } .core-wrap { width:76px;height:76px; } .command-title { font-size:1.4rem; } .state-pill { display:none; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _initialize_session() -> None:
    defaults = {
        "agent_history": [],
        "agent_messages": [],
        "pending_application": None,
        "identified_customer_id": None,
        "last_intent": None,
        "active_workflow": None,
        "follow_up_fields": [],
        "execution_trace": [],
        "last_tools": [],
        "known_profile": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _dispatch_prompt(prompt: str) -> None:
    previous_customer = st.session_state.identified_customer_id
    pending_offer = st.session_state.pending_application
    affirmative = bool(re.fullmatch(
        r"\s*(?:yes|yeah|yep|confirm(?: and submit)?|i confirm(?: and submit)?|approve|i approve|submit(?: it|(?: the)? loan| this offer| this application)?|go ahead|proceed)[.! ]*\s*",
        prompt,
        re.IGNORECASE,
    ))
    negative = bool(re.fullmatch(r"\s*(?:no|no thanks|decline|reject|cancel)(?: it| this offer| this application)?[.! ]*\s*", prompt, re.IGNORECASE))
    confirm_pending_offer = bool(pending_offer and affirmative)
    decline_pending_offer = bool(pending_offer and negative)
    with st.status("FinPilot is thinking…", expanded=False) as progress:
        try:
            result = run_finpilot_agent(
                prompt,
                identified_customer_id=previous_customer,
                messages=st.session_state.agent_messages,
                previous_intent=st.session_state.active_workflow,
                active_workflow=st.session_state.active_workflow,
                pending_application=st.session_state.pending_application,
                customer_confirmed=confirm_pending_offer,
                customer_declined=decline_pending_offer,
                approved_application=pending_offer if confirm_pending_offer or decline_pending_offer else None,
            )
            progress.update(label="FinPilot response ready", state="complete", expanded=False)
        except Exception as exc:
            progress.update(label="FinPilot could not complete the request", state="error")
            st.error(f"The agent could not complete this request: {exc}")
            return

    if result["identity_changed"] and previous_customer:
        st.session_state.agent_history = []
        st.session_state.agent_messages = []
    st.session_state.agent_messages = result["messages"]
    st.session_state.identified_customer_id = result.get("identified_customer_id")
    st.session_state.pending_application = result.get("pending_application")
    st.session_state.last_intent = result.get("detected_intent")
    st.session_state.active_workflow = result.get("active_workflow")
    st.session_state.follow_up_fields = result.get("follow_up_fields", [])
    st.session_state.execution_trace = result.get("execution_trace", [])
    st.session_state.last_tools = [action.tool for action, _ in result.get("intermediate_steps", [])]
    for action, observation in result.get("intermediate_steps", []):
        if action.tool == "account_information_tool":
            try:
                account_context = json.loads(observation)
                st.session_state.known_profile = account_context.get("account_profile", {})
            except (TypeError, json.JSONDecodeError):
                pass
    st.session_state.agent_history.extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": result["output"]},
    ])


def _render_guided_form() -> None:
    fields = st.session_state.follow_up_fields
    financial_fields = {"loan_amount", "requested loan amount", "tenure_years", "loan tenure", "monthly salary or income"}
    if not fields or not set(fields).intersection(financial_fields):
        return
    is_emi = st.session_state.last_intent == "emi_calculation"
    profile = st.session_state.known_profile
    st.markdown("#### Add the missing loan details")
    with st.form("guided_loan_details"):
        monthly_income = st.number_input(
            "Monthly salary / income",
            min_value=0.0,
            value=float(profile.get("monthly_income") or 0),
            step=1000.0,
        ) if "monthly salary or income" in fields else None
        loan_amount = st.number_input("Loan amount", min_value=0.0, step=10000.0) if "loan_amount" in fields or "requested loan amount" in fields else None
        tenure_years = st.number_input("Tenure (years)", min_value=1, max_value=40, step=1) if "tenure_years" in fields or "loan tenure" in fields else None
        monthly_obligations = st.number_input(
            "Existing monthly obligations",
            min_value=0.0,
            value=float(profile.get("monthly_obligations") or 0),
            step=500.0,
        ) if not is_emi else None
        credit_score = st.number_input(
            "Credit score", min_value=300, max_value=900,
            value=int(profile.get("credit_score") or 700), step=1,
        ) if not is_emi else None
        submit = st.form_submit_button("Continue assessment", type="primary")
    if submit:
        if is_emi:
            _dispatch_prompt(f"Calculate EMI for ₹{loan_amount:,.0f} for {int(tenure_years)} years.")
            st.rerun()
        details = []
        if monthly_income is not None:
            details.append(f"monthly salary is ₹{monthly_income:,.0f}")
        if loan_amount is not None:
            details.append(f"loan amount is ₹{loan_amount:,.0f}")
        if tenure_years is not None:
            details.append(f"tenure is {int(tenure_years)} years")
        if monthly_obligations is not None:
            details.append(f"monthly obligations are ₹{monthly_obligations:,.0f}")
        if credit_score is not None:
            details.append(f"credit score is {int(credit_score)}")
        _dispatch_prompt("For my loan request, " + ", ".join(details) + ".")
        st.rerun()


def _render_identity_form() -> None:
    if st.session_state.identified_customer_id:
        customer_id = st.session_state.identified_customer_id
        st.markdown(f"<div class='micro'>IDENTIFIED SESSION · {customer_id}</div>", unsafe_allow_html=True)
        if st.button("End customer session", key="end_customer_session"):
            st.session_state.identified_customer_id = None
            st.session_state.agent_messages = []
            st.session_state.agent_history = []
            st.session_state.pending_application = None
            st.session_state.known_profile = {}
            st.session_state.last_intent = None
            st.session_state.active_workflow = None
            st.session_state.follow_up_fields = []
            st.session_state.execution_trace = []
            st.rerun()
        return

    with st.expander("Identify your account before retrieval", expanded=True):
        with st.form("customer_identification"):
            identifier_type = st.selectbox("Identification method", ["Customer ID", "Account Number", "Registered Mobile Number"])
            identifier_value = st.text_input("Enter identifier", placeholder="Use the identifier registered with your bank")
            submit = st.form_submit_button("Identify customer", type="primary")
        if submit:
            if not identifier_value.strip():
                st.warning("Enter a Customer ID, account number, or registered mobile number.")
            else:
                prefix = {
                    "Customer ID": "My Customer ID is",
                    "Account Number": "My account number is",
                    "Registered Mobile Number": "My registered mobile number is",
                }[identifier_type]
                _dispatch_prompt(f"{prefix} {identifier_value.strip()}.")
                st.rerun()


def _render_registration_form() -> None:
    with st.expander("New customer registration"):
        with st.form("new_customer_registration"):
            name = st.text_input("Full name")
            mobile = st.text_input("Registered mobile number")
            income = st.number_input("Monthly income (optional)", min_value=0.0, step=1000.0)
            employment = st.selectbox("Employment type", ["Not provided", "Salaried", "Self-employed", "Professional"])
            submit = st.form_submit_button("Create customer profile", type="primary")
        if submit:
            if not name.strip() or len("".join(character for character in mobile if character.isdigit())) < 10:
                st.warning("A full name and valid mobile number are required.")
            else:
                income_detail = f" Monthly income is ₹{income:,.0f}." if income else ""
                employment_detail = f" Employment type is {employment}." if employment != "Not provided" else ""
                _dispatch_prompt(
                    f"Register me as a new customer. My name is {name.strip()}. "
                    f"My registered mobile number is {mobile.strip()}.{income_detail}{employment_detail}"
                )
                st.rerun()


def _render_quick_actions() -> None:
    st.markdown("#### Quick actions")
    for offset in (0, 4):
        columns = st.columns(4)
        for col, (label, prompt) in zip(columns, QUICK_ACTIONS[offset:offset + 4]):
            with col:
                if st.button(label, key=f"quick_{label}", use_container_width=True):
                    _dispatch_prompt(prompt)
                    st.rerun()


def _render_pending_offer() -> None:
    pending = st.session_state.pending_application
    if not pending:
        return
    st.warning(
        f"Draft {pending['application_id']} · Tracking {pending['tracking_number']} · "
        f"₹{pending['loan_amount']:,.0f} · {pending['tenure_years']} years · "
        f"EMI ₹{pending['emi']['monthly_emi']:,.2f}/month · "
        f"{pending['eligibility']['risk_category']} risk. This draft has not been submitted."
    )
    col_confirm, col_decline = st.columns(2)
    with col_confirm:
        if st.button("Confirm and submit this offer", type="primary", key="confirm_offer"):
            result = run_finpilot_agent(
                "I explicitly confirm submission of the exact loan offer shown above.",
                identified_customer_id=st.session_state.identified_customer_id,
                messages=st.session_state.agent_messages,
                customer_confirmed=True,
                approved_application=pending,
                previous_intent="loan_application",
                active_workflow="loan_application",
                pending_application=pending,
            )
            st.session_state.agent_messages = result["messages"]
            st.session_state.pending_application = result.get("pending_application")
            st.session_state.execution_trace = result.get("execution_trace", [])
            st.session_state.last_tools = [action.tool for action, _ in result.get("intermediate_steps", [])]
            st.session_state.agent_history.append({"role": "assistant", "content": result["output"]})
            st.rerun()
    with col_decline:
        if st.button("Decline draft", key="decline_offer"):
            result = run_finpilot_agent(
                "I decline this loan offer. Do not submit it.",
                identified_customer_id=st.session_state.identified_customer_id,
                messages=st.session_state.agent_messages,
                customer_declined=True,
                approved_application=pending,
                previous_intent="loan_application",
                active_workflow="loan_application",
                pending_application=pending,
            )
            st.session_state.agent_messages = result["messages"]
            st.session_state.pending_application = result.get("pending_application")
            st.session_state.execution_trace = result.get("execution_trace", [])
            st.session_state.last_tools = [action.tool for action, _ in result.get("intermediate_steps", [])]
            st.session_state.agent_history.append({"role": "assistant", "content": result["output"]})
            st.rerun()


def render_dashboard() -> None:
    apply_theme()
    ingest_raw_datasets()
    _initialize_session()

    st.markdown(
        """
        <div class="command-head">
            <div class="core-wrap"><div class="core-ring"></div><div class="core-ring two"></div><div class="core"></div></div>
            <div><div class="eyebrow">FINPILOT · BANKING INTELLIGENCE</div><div class="command-title">Command Center</div><div class="command-sub">Your banking requests, handled by one reasoning agent.</div></div>
            <div class="state-pill">● AGENT ONLINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Talk to FinPilot")
    _render_identity_form()
    _render_registration_form()
    _render_quick_actions()

    for entry in st.session_state.agent_history:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

    _render_guided_form()
    prompt = st.chat_input("Ask FinPilot about EMI, eligibility, accounts, applications, or your saved preferences")
    if prompt:
        _dispatch_prompt(prompt)
        st.rerun()

    _render_pending_offer()
