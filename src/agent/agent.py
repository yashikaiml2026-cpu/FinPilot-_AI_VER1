from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, Literal

from langchain.agents import AgentState, create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.agent.tools import (
    calculate_emi,
    evaluate_eligibility,
    get_customer_profile,
    query_policy_knowledge_base,
    submit_loan_application,
)
from src.database.repositories import AuditRepository, CustomerRepository, UserMemoryRepository
from src.database.repositories import LoanApplicationRepository
from src.agent.intents import Intent, IntentName, ROUTES, classify_intent
from src.llm.client import build_llm_client
from src.utils.logging import log_event


@tool
def emi_calculator_tool(loan_amount: float, annual_rate: float, tenure_years: int) -> str:
    """Calculate monthly EMI, total interest and repayment schedule for a user loan request."""
    result = calculate_emi(loan_amount=loan_amount, annual_rate=annual_rate, tenure_years=tenure_years)
    return json.dumps(result, default=str)


@tool
def eligibility_checker_tool(
    monthly_income: float,
    monthly_obligations: float,
    credit_score: int,
    loan_amount: float,
    existing_loans: int = 0,
    annual_rate: float = 9.5,
    tenure_years: int = 10,
) -> str:
    """Evaluate affordability and risk for the requested loan."""
    try:
        policy_context = query_policy_knowledge_base(
            "Loan eligibility: FOIR, monthly repayment burden, credit profile, existing obligations, affordability, and when repayment stress requires escalation or manual review.",
            k=8,
        )
        result = evaluate_eligibility(
            monthly_income=monthly_income,
            monthly_obligations=monthly_obligations,
            credit_score=credit_score,
            loan_amount=loan_amount,
            existing_loans=existing_loans,
            annual_rate=annual_rate,
            tenure_years=tenure_years,
            policy_context=policy_context,
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({
            "status": "insufficient_eligibility_data",
            "message": str(exc),
            "requires_follow_up": True,
            "next_step": "Ask for the missing monthly income, obligations, credit score, or loan amount before continuing the assessment.",
        }, default=str)


@tool
def account_information_tool(customer_id: Annotated[str | None, InjectedState("customer_id")]) -> str:
    """Explicitly retrieve masked account profile plus stored profile facts, preferences, interactions, and applications."""
    if not customer_id:
        return json.dumps({
            "status": "identification_required",
            "message": "No customer record was loaded.",
            "next_step": "Ask for a Customer ID, account number, or registered mobile number before retrying account lookup.",
        })
    try:
        record = get_customer_profile(customer_id)
        context = UserMemoryRepository.get_context(customer_id)
        return json.dumps({"account_profile": record, "persistent_context": context}, default=str)
    except Exception as exc:
        return json.dumps({
            "status": "missing_customer",
            "message": str(exc),
            "requires_follow_up": True,
            "next_step": "Ask the customer for a valid customer identifier and any missing profile details.",
        }, default=str)


@tool
def user_memory_query_tool(
    question: str,
    customer_id: Annotated[str | None, InjectedState("customer_id")],
) -> str:
    """Retrieve saved profile facts and preferences for memory-recall questions without loading account data."""
    if not customer_id:
        return json.dumps({"status": "identification_required", "message": "Identify the customer before memory recall."})
    context = UserMemoryRepository.get_context(customer_id)
    return json.dumps({
        "profile_facts": context["profile_facts"],
        "preferences": context["preferences"],
    }, default=str)


@tool
def application_information_tool(
    customer_id: Annotated[str | None, InjectedState("customer_id")],
    application_id: str | None = None,
) -> str:
    """Retrieve applications belonging only to the identified customer, optionally by application ID."""
    if not customer_id:
        return json.dumps({"status": "identification_required", "message": "Identify the customer before tracking applications."})
    applications = LoanApplicationRepository.get_application(customer_id, application_id)
    return json.dumps({"status": "found" if applications else "not_found", "applications": applications}, default=str)


@tool
def customer_registration_tool(
    full_name: Annotated[str | None, InjectedState("registration_name")],
    registered_mobile_number: Annotated[str | None, InjectedState("registration_mobile")],
    monthly_income: float = 0,
    employment_type: str = "",
    account_type: str = "Savings",
) -> str:
    """Register a new customer, generate a customer ID and account number, and persist the profile."""
    try:
        customer = CustomerRepository.register_customer({
            "full_name": full_name,
            "phone": registered_mobile_number,
            "monthly_income": monthly_income,
            "employment_type": employment_type,
            "account_type": account_type,
        })
        return json.dumps({
            "status": "registered",
            "customer_id": customer["customer_id"],
            "account_number": customer["account_number"],
            "full_name": customer["full_name"],
        })
    except (ValueError, OSError) as exc:
        return json.dumps({"status": "registration_failed", "message": str(exc)})


@tool
def remember_user_information_tool(
    memory_type: Literal["profile", "preference"],
    key: str,
    value: str,
    customer_id: Annotated[str | None, InjectedState("customer_id")],
) -> str:
    """Persist a customer-volunteered non-sensitive profile fact or preference for future account lookups."""
    if not customer_id:
        return json.dumps({"status": "identification_required", "message": "Identify the customer before saving user memory."})
    try:
        UserMemoryRepository.save_memory(customer_id, memory_type, key, value)
        return json.dumps({"status": "stored", "memory_type": memory_type, "key": key})
    except (ValueError, OSError) as exc:
        return json.dumps({"status": "not_stored", "message": str(exc)})


@tool
def policy_retriever_tool(question: str, k: int = 4) -> str:
    """Retrieve relevant RBI and banking policy context for eligibility and compliance decisions."""
    results = query_policy_knowledge_base(question, k=k)
    return json.dumps(results, default=str)


@tool
def request_customer_confirmation_tool(
    customer_id: Annotated[str | None, InjectedState("customer_id")],
    loan_amount: float,
    annual_rate: float,
    tenure_years: int,
    monthly_income: float,
    monthly_obligations: float,
    credit_score: int,
    existing_loans: int = 0,
) -> str:
    """Calculate and present the final offer for explicit customer confirmation; does not submit it."""
    if not customer_id:
        return json.dumps({
            "status": "identification_required",
            "message": "A customer must be identified before preparing an application offer.",
            "next_step": "Ask for a Customer ID, account number, or registered mobile number.",
        })
    try:
        emi = calculate_emi(loan_amount, annual_rate, tenure_years)
        eligibility = evaluate_eligibility(
            monthly_income=monthly_income,
            monthly_obligations=monthly_obligations,
            credit_score=credit_score,
            loan_amount=loan_amount,
            existing_loans=existing_loans,
            annual_rate=annual_rate,
            tenure_years=tenure_years,
            policy_context=query_policy_knowledge_base(
                "Loan eligibility FOIR affordability and escalation policy", k=8
            ),
        )
        application_id = f"APP-{uuid.uuid4().hex[:8].upper()}"
        tracking_number = f"TRK-{uuid.uuid4().hex[:8].upper()}"
        submitted_at = datetime.now(timezone.utc).isoformat()
        draft = LoanApplicationRepository.create_draft({
            "application_id": application_id,
            "customer_id": customer_id,
            "loan_amount": float(loan_amount),
            "interest_rate": float(annual_rate),
            "tenure_months": int(tenure_years) * 12,
            "approval_probability": float(eligibility.get("approval_probability", 0)),
            "tracking_number": tracking_number,
            "reason_codes": eligibility.get("reason_codes", []),
            "report": "Customer-authorized loan application offer",
            "submitted_at": submitted_at,
        })
        offer = {
            "customer_id": customer_id,
            "application_id": draft["application_id"],
            "tracking_number": draft["tracking_number"],
            "submitted_at": draft["submitted_at"],
            "loan_amount": float(loan_amount),
            "annual_rate": float(annual_rate),
            "tenure_years": int(tenure_years),
            "monthly_income": float(monthly_income),
            "monthly_obligations": float(monthly_obligations),
            "credit_score": int(credit_score),
            "existing_loans": int(existing_loans),
            "emi": emi,
            "eligibility": eligibility,
            "report": "Customer-authorized loan application offer",
        }
        return json.dumps({"status": "awaiting_confirmation", "offer": offer}, default=str)
    except Exception as exc:
        return json.dumps({
            "status": "offer_failed",
            "message": str(exc),
            "requires_follow_up": True,
            "next_step": "Collect or correct the required loan details before asking for confirmation.",
        }, default=str)


class FinPilotState(AgentState):
    customer_id: str | None
    active_workflow: str | None
    registration_mobile: str | None
    registration_name: str | None
    customer_confirmed: bool
    customer_declined: bool
    approved_application: dict[str, Any]


@tool
def loan_application_processor_tool(
    customer_confirmed: Annotated[bool, InjectedState("customer_confirmed")],
    approved_application: Annotated[dict[str, Any], InjectedState("approved_application")],
    customer_id: Annotated[str | None, InjectedState("customer_id")],
) -> str:
    """Submit only the exact loan offer approved by the customer in the confirmation control."""
    if not customer_id:
        return json.dumps({"status": "blocked", "message": "Customer identification is required before application submission."})
    if not customer_confirmed or not approved_application:
        return json.dumps({"status": "blocked", "message": "Explicit customer confirmation is required before submission."})
    if approved_application.get("customer_id") != customer_id:
        return json.dumps({"status": "blocked", "message": "The approved offer does not match the active customer."})
    try:
        payload = submit_loan_application(
            customer_id=customer_id,
            loan_amount=approved_application["loan_amount"],
            annual_rate=approved_application["annual_rate"],
            tenure_years=approved_application["tenure_years"],
            eligibility=approved_application["eligibility"],
            report=approved_application["report"],
            status="SUBMITTED",
            application_id=approved_application["application_id"],
            tracking_number=approved_application["tracking_number"],
            submitted_at=approved_application["submitted_at"],
        )
        return json.dumps(payload, default=str)
    except Exception as exc:
        return json.dumps({"status": "submission_failed", "message": str(exc)}, default=str)


@tool
def decline_customer_offer_tool(
    customer_id: Annotated[str | None, InjectedState("customer_id")],
    approved_application: Annotated[dict[str, Any], InjectedState("approved_application")],
) -> str:
    """Mark the identified customer's unsubmitted application draft as rejected after they decline the offer."""
    if not customer_id or approved_application.get("customer_id") != customer_id:
        return json.dumps({"status": "blocked", "message": "The draft does not match the identified customer."})
    try:
        application = LoanApplicationRepository.update_status(
            customer_id,
            approved_application["application_id"],
            "REJECTED",
        )
        return json.dumps({"status": "REJECTED", "application_id": application["application_id"]})
    except Exception as exc:
        return json.dumps({"status": "decline_failed", "message": str(exc)})


TOOLS = [
    emi_calculator_tool,
    eligibility_checker_tool,
    account_information_tool,
    user_memory_query_tool,
    application_information_tool,
    customer_registration_tool,
    remember_user_information_tool,
    policy_retriever_tool,
    request_customer_confirmation_tool,
    loan_application_processor_tool,
    decline_customer_offer_tool,
]


def build_finpilot_agent(
    customer_confirmed: bool = False,
    customer_declined: bool = False,
    customer_identified: bool = False,
    intent_name: IntentName = "general",
    active_workflow: IntentName | None = None,
    pending_application: dict[str, Any] | None = None,
) -> Any:
    llm = build_llm_client()
    confirmation_instruction = (
        "The customer declined the pending offer. You MUST call decline_customer_offer_tool and confirm that no submission occurred."
        if customer_declined
        else
        "The customer has now explicitly confirmed the previously displayed offer. "
        "You MUST call loan_application_processor_tool to submit that exact approved offer before replying."
        if customer_confirmed
        else "Do not submit an application. First present the calculated offer and request explicit customer authorization."
    )
    identification_instruction = (
        "The application has verified an explicitly supplied customer identifier. You may now use customer-specific tools."
        if customer_identified
        else "No customer is identified. Ask for a Customer ID, account number, or registered mobile number before any account lookup, memory access, or loan application offer. Do not assume an identity."
    )
    pending_offer_instruction = (
        f"A pending application offer already exists: {pending_application.get('application_id')} "
        f"(tracking {pending_application.get('tracking_number')}). Do not create a second draft; "
        "answer follow-ups from the existing offer and wait for explicit submit/decline."
        if pending_application else "There is no pending application offer."
    )
    system_prompt = """You are FinPilot AI, an autonomous banking support and loan evaluation agent.

Detected intent: {detected_intent}
Active workflow: {active_workflow}
Pending offer: {pending_offer_instruction}

Your responsibilities:
1. Understand the customer intent and requested loan requirements.
2. Decide when customer information is missing and ask targeted questions.
3. Identification state: {identification_instruction}
4. Use account_information_tool for balances, transactions, and customer profiles. Use application_information_tool for past, recent, or status questions about loan applications; summarize its results and never ask the customer to call a tool.
5. When the customer volunteers reusable, non-sensitive profile information or a preference, persist it using remember_user_information_tool only after identification. Never store credentials, account/card numbers, authentication codes, or payment secrets. Interactions and applications are already stored by the system. For past, recent, or status questions about loan applications, call application_information_tool directly and summarize its results; never ask the customer to call a tool.
6. For Customer Registration, collect the full name and registered mobile number, call customer_registration_tool, and clearly return the generated customer ID.
7. Select only tools relevant to the current request and reason over their outputs before deciding what to do next.
8. Preserve conversation continuity. Interpret short replies using prior user messages and tool results. For “why” or “explain”, answer about the current workflow; for “continue”, resume it; for “track it”, retrieve the existing application. Never restart a completed step or create a duplicate application.
9. Retrieved policy passages are internal evidence, not response text. Synthesize their applicable constraints in your own words, and use the policy-aware eligibility result; never quote or return retrieved passages verbatim.
10. Never claim an application is submitted unless the submission tool returns status SUBMITTED.
11. After assessment, follow this instruction: {confirmation_instruction}
12. Provide a concise report covering EMI, eligibility, and application status, grounded in tool results.
13. Ask for salary, requested amount, and tenure when any is absent. Never invent defaults.
14. User memory is stored in SQLite and used only as future context; conversations never train or update model weights.
"""
    system_prompt = system_prompt.format(
        confirmation_instruction=confirmation_instruction,
        identification_instruction=identification_instruction,
        detected_intent=ROUTES[intent_name].label,
        active_workflow=ROUTES[active_workflow].label if active_workflow in ROUTES else "None",
        pending_offer_instruction=pending_offer_instruction,
    )
    if customer_confirmed:
        allowed_names = {"loan_application_processor_tool"}
    elif customer_declined:
        allowed_names = {"decline_customer_offer_tool"}
    else:
        allowed_names = set(ROUTES[intent_name].tool_names)
    if customer_identified:
        allowed_names.update({
            name for name in allowed_names
            if name in {
                "account_information_tool",
                "application_information_tool",
                "remember_user_information_tool",
                "request_customer_confirmation_tool",
                "loan_application_processor_tool",
                "decline_customer_offer_tool",
            }
        })
    else:
        allowed_names.discard("account_information_tool")
        allowed_names.discard("application_information_tool")
        allowed_names.discard("remember_user_information_tool")
        allowed_names.discard("request_customer_confirmation_tool")
        allowed_names.discard("loan_application_processor_tool")
    available_tools = [tool for tool in TOOLS if tool.name in allowed_names]
    return create_agent(llm, available_tools, system_prompt=system_prompt, state_schema=FinPilotState)


def _record_conversation(customer_id: str, role: str, message: str) -> None:
    if not customer_id:
        return
    try:
        with __import__("src.database.db", fromlist=["get_connection"]).get_connection() as conn:
            conn.execute(
                "INSERT INTO conversation_logs (customer_id, role, message) VALUES (?, ?, ?)",
                (customer_id, role, message),
            )
            conn.commit()
    except Exception:
        pass


def _missing_loan_fields(text: str, intent_name: IntentName = "loan_application") -> list[str]:
    if intent_name not in {"loan_application", "eligibility_query"}:
        return []
    missing = []
    amount = r"\d[\d,]*(?:\.\d+)?\s*(?:k|thousand|lakh|lakhs|crore)?"
    if not re.search(rf"\b(salary|monthly income|income)\b[^\d]{{0,24}}{amount}", text, re.IGNORECASE):
        missing.append("monthly salary or income")
    labeled_amount = re.search(rf"\b(?:loan amount|amount|loan of|borrow(?:ing)?(?: of)?)\b[^\d₹]{{0,24}}{amount}", text, re.IGNORECASE)
    currency_amount = re.search(rf"(?:₹|\b(?:rs\.?|inr)\s*)\s*{amount}", text, re.IGNORECASE)
    amount_before_loan = re.search(rf"{amount}\s*(?:personal|home|auto|car|education|business)?\s*loan\b", text, re.IGNORECASE)
    if not (labeled_amount or currency_amount or amount_before_loan):
        missing.append("requested loan amount")
    if not re.search(r"\b(?:(?:for|over|across)\s+\d+\s*(?:years?|yrs?|months?)|tenure\s*(?:is|of)?\s*\d+\s*(?:years?|yrs?|months?)?)\b", text, re.IGNORECASE):
        missing.append("loan tenure")
    return missing


def _extract_emi_details(text: str) -> dict[str, float | int] | None:
    amount_match = re.search(
        r"(?:₹|\b(?:rs\.?|inr)\s*)\s*(\d[\d,]*(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|crore|cr|k|thousand)?|\b(?:amount|loan)\s*(?:of|is)?\s*(\d[\d,]*(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|crore|cr|k|thousand)?",
        text,
        re.IGNORECASE,
    )
    duration_match = re.search(r"\b(?:for|over|across)\s+(\d+)\s*(years?|yrs?)\b|\btenure\s*(?:of|is)?\s*(\d+)\s*(years?|yrs?)\b", text, re.IGNORECASE)
    if not amount_match or not duration_match:
        return None
    amount_text = amount_match.group(1) or amount_match.group(3)
    unit = (amount_match.group(2) or amount_match.group(4) or "").lower()
    amount = float(amount_text.replace(",", ""))
    if unit in {"lakh", "lakhs", "lac", "lacs"}:
        amount *= 100_000
    elif unit in {"crore", "cr"}:
        amount *= 10_000_000
    elif unit in {"k", "thousand"}:
        amount *= 1_000
    years = int(duration_match.group(1) or duration_match.group(3))
    rate_match = re.search(r"\b(?:at|rate|interest rate)\s*(\d+(?:\.\d+)?)\s*%?", text, re.IGNORECASE)
    return {
        "loan_amount": amount,
        "tenure_years": years,
        "annual_rate": float(rate_match.group(1)) if rate_match else 9.5,
    }


def _previous_user_message(messages: list[Any]) -> str:
    for item in reversed(messages):
        if isinstance(item, HumanMessage):
            return str(item.content)
    return ""


def _extract_memory_facts(text: str) -> list[tuple[str, str, str]]:
    facts: dict[tuple[str, str], str] = {}
    for match in re.finditer(r"\bmy\s+([a-z][a-z _-]{0,40}?)\s+is\s+([^.!?\n]+)", text, re.IGNORECASE):
        key = re.sub(r"\s+", " ", match.group(1)).strip().lower()
        value = match.group(2).strip(" ,;:")
        if re.search(r"customer id|account|mobile|phone|number|redacted customer identifier", key + " " + value, re.IGNORECASE):
            continue
        memory_type = "preference" if key.startswith(("preferred ", "preference ")) else "profile"
        facts[(memory_type, key)] = value
    for match in re.finditer(r"\b(preferred\s+(?:loan\s+)?(?:type|language))\s+is\s+([^.!?\n]+)", text, re.IGNORECASE):
        facts[("preference", re.sub(r"\s+", " ", match.group(1)).strip().lower())] = match.group(2).strip(" ,;:")
    for match in re.finditer(r"\bi\s+prefer\s+([^.!?\n]+)", text, re.IGNORECASE):
        facts[("preference", "general preference")] = match.group(1).strip(" ,;:")
    return [(memory_type, key, value) for (memory_type, key), value in facts.items()]


def _summary_response(
    output: str,
    steps: list[tuple[Any, Any]],
    awaiting_confirmation: bool,
    intent_name: IntentName | None = None,
) -> str:
    if intent_name not in {None, "loan_application", "eligibility_query"}:
        return output.strip()
    if intent_name is not None and not steps:
        return output.strip()
    tool_results: dict[str, list[dict[str, Any]]] = {}
    for action, observation in steps:
        try:
            parsed = json.loads(observation)
        except (TypeError, json.JSONDecodeError):
            continue
        tool_results.setdefault(action.tool, []).append(parsed)

    offers = tool_results.get("request_customer_confirmation_tool", [])
    offer = next((item.get("offer") for item in reversed(offers) if item.get("offer")), {})
    emi_results = tool_results.get("emi_calculator_tool", [])
    eligibility_results = tool_results.get("eligibility_checker_tool", [])
    applications = tool_results.get("loan_application_processor_tool", [])
    emi = offer.get("emi") or next((item for item in reversed(emi_results) if "monthly_emi" in item), {})
    eligibility = offer.get("eligibility") or next((item for item in reversed(eligibility_results) if "approval_probability" in item), {})
    submitted = next((item for item in reversed(applications) if item.get("status") == "SUBMITTED"), None)

    if submitted:
        application_status = f"Submitted ({submitted.get('application_id', 'application ID unavailable')})."
    elif awaiting_confirmation:
        application_status = "Awaiting explicit customer confirmation; not submitted."
    elif applications:
        application_status = "Not submitted; submission was blocked or failed."
    else:
        application_status = "Not submitted."

    emi_text = f"₹{emi['monthly_emi']:,.2f} per month" if emi.get("monthly_emi") is not None else "Not calculated"
    if eligibility.get("risk_category"):
        eligibility_text = f"{eligibility['risk_category']} risk; approval score {eligibility.get('approval_probability', 'unavailable')}% (assessment, not a guarantee)"
        if eligibility.get("policy_decision"):
            eligibility_text += f"; policy outcome: {eligibility['policy_decision']}"
    else:
        eligibility_text = "Not assessed"
    summary = (
        "\n\n**Assessment Summary**\n"
        f"- EMI: {emi_text}\n"
        f"- Eligibility result: {eligibility_text}\n"
        f"- Application status: {application_status}"
    )
    return f"{output.strip()}{summary}" if output.strip() else summary.strip()


def _build_execution_trace(
    intent: Intent,
    steps: list[tuple[Any, Any]],
    awaiting_confirmation: bool,
    output: str,
) -> list[dict[str, str]]:
    trace = [{"stage": "Intent detected", "status": "complete", "detail": intent.label}]
    if steps:
        selected = ", ".join(dict.fromkeys(action.tool for action, _ in steps))
        trace.append({"stage": "Tools selected", "status": "complete", "detail": selected})
        trace.extend(
            {"stage": "Tool executed", "status": "complete", "detail": action.tool}
            for action, _ in steps
        )
        trace.append({
            "stage": "Reasoning",
            "status": "complete",
            "detail": f"Agent received {len(steps)} current-turn tool observation(s)",
        })
        policy_outcomes = []
        for action, observation in steps:
            if action.tool == "eligibility_checker_tool":
                try:
                    outcome = json.loads(observation).get("policy_decision")
                    if outcome:
                        policy_outcomes.append(outcome)
                except (TypeError, json.JSONDecodeError):
                    pass
        if policy_outcomes:
            trace.append({"stage": "Decision", "status": "complete", "detail": f"Policy result: {policy_outcomes[-1]}"})
    if awaiting_confirmation:
        trace.append({"stage": "Decision", "status": "complete", "detail": "Offer prepared from current tool results"})
        trace.append({"stage": "Customer confirmation", "status": "waiting", "detail": "Offer prepared; not submitted"})
    elif any(action.tool == "loan_application_processor_tool" and _tool_result_status(result) == "SUBMITTED" for action, result in steps):
        trace.append({"stage": "Decision", "status": "complete", "detail": "Customer confirmed the exact pending offer"})
        trace.append({"stage": "Application submitted", "status": "complete", "detail": "Submission tool confirmed persistence"})
    elif any(action.tool == "decline_customer_offer_tool" and _tool_result_status(result) == "REJECTED" for action, result in steps):
        trace.append({"stage": "Decision", "status": "complete", "detail": "Customer declined the pending offer"})
        trace.append({"stage": "Offer declined", "status": "complete", "detail": "Draft marked rejected; no submission"})
    if output:
        trace.append({"stage": "Response ready", "status": "complete", "detail": "Tool results synthesized for the customer"})
    return trace


def _extract_customer_identifier(message: str) -> tuple[str, str] | None:
    patterns = (
        ("customer_id", r"\b(?:customer\s*id|customer\s*identifier|id)\s*(?:is|:|#)?\s*(CUST[- ]?\d+)\b"),
        ("customer_id", r"\b(CUST[- ]?\d+)\b"),
        ("account_number", r"\b(?:account(?:\s*(?:number|no\.?))?|a/c)\s*(?:is|:|#)?\s*([\d\s-]{8,24})"),
        ("mobile_number", r"\b(?:registered\s+)?(?:mobile|phone)(?:\s*(?:number|no\.?))?\s*(?:is|:|#)?\s*(\+?[\d\s()-]{10,20})"),
    )
    for identifier_type, pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return identifier_type, match.group(1).strip()

    standalone = message.strip()
    if re.fullmatch(r"CUST[- ]?\d+", standalone, re.IGNORECASE):
        return "customer_id", standalone
    if re.fullmatch(r"[\d\s()+-]{8,24}", standalone):
        digits = re.sub(r"\D", "", standalone)
        if len(digits) == 10:
            return "mobile_number", standalone
        if 11 <= len(digits) <= 18:
            return "account_number", standalone
    return None


def _requires_account_identification(text: str) -> bool:
    return bool(re.search(
        r"\b(account|balance|profile|transaction history|application history|my applications|saved preferences?)\b",
        text,
        re.IGNORECASE,
    ))


def run_finpilot_agent(
    message: str,
    identified_customer_id: str | None = None,
    messages: list[Any] | None = None,
    customer_confirmed: bool = False,
    customer_declined: bool = False,
    approved_application: dict[str, Any] | None = None,
    previous_intent: IntentName | None = None,
    progress_callback: Callable[[dict[str, str]], None] | None = None,
    active_workflow: IntentName | None = None,
    pending_application: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conversation = list(messages or [])
    workflow_context = active_workflow or previous_intent
    identifier = _extract_customer_identifier(message)
    preliminary_intent = classify_intent(message, workflow_context)
    if identified_customer_id is None and any(
        classify_intent(str(item.content)).name == "customer_registration"
        for item in conversation
        if isinstance(item, HumanMessage)
    ):
        preliminary_intent = ROUTES["customer_registration"]
    registration_mobile = None
    registration_name = None
    if preliminary_intent.name == "customer_registration":
        registration_text = "\n".join(
            [str(item.content) for item in conversation if isinstance(item, HumanMessage)] + [message]
        )
        if identifier and identifier[0] == "mobile_number":
            registration_mobile = identifier[1]
        else:
            mobile_match = re.search(
                r"\b(?:registered\s+)?(?:phone|mobile|number)(?:\s+number)?\s*(?:is|:)?\s*(\+?[\d ()-]{10,20})",
                registration_text,
                re.IGNORECASE,
            )
            if mobile_match:
                registration_mobile = mobile_match.group(1).strip()
                identifier = ("mobile_number", registration_mobile)
        name_match = re.search(
            r"\b(?:my\s+)?name\s*(?:is|:)\s*([A-Za-z][A-Za-z '-]{1,60})",
            registration_text,
            re.IGNORECASE,
        )
        if name_match:
            registration_name = re.split(
                r"\s+and\s+(?:phone|mobile|number)\b",
                name_match.group(1),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip().rstrip(" ,.;:")
    customer_id = identified_customer_id
    identity_error = False
    identity_changed = False
    if identifier and registration_mobile is None:
        resolved_customer_id = CustomerRepository.identify_customer(*identifier)
        customer_id = resolved_customer_id
        identity_error = resolved_customer_id is None
        identity_changed = identified_customer_id is not None and resolved_customer_id != identified_customer_id
        if identity_changed:
            conversation = []

    redact_identifier = identifier
    safe_message = re.sub(re.escape(redact_identifier[1]), "[REDACTED CUSTOMER IDENTIFIER]", message, count=1, flags=re.IGNORECASE) if redact_identifier else message
    identifier_only = False
    if identifier:
        remainder = re.sub(re.escape(identifier[1]), " ", message, count=1, flags=re.IGNORECASE)
        remainder = re.sub(r"\b(?:my|registered|customer|account|mobile|phone|number|no|id|is|the)\b", " ", remainder, flags=re.IGNORECASE)
        identifier_only = not re.search(r"[A-Za-z0-9]", remainder)
    previous_user_text = _previous_user_message(conversation) if identifier_only else ""
    routing_text = f"{previous_user_text} {safe_message}".strip() if previous_user_text else safe_message
    detected_intent: Intent = (
        ROUTES["loan_application"]
        if customer_confirmed or customer_declined
        else preliminary_intent
        if preliminary_intent.name == "customer_registration"
        else classify_intent(routing_text, workflow_context)
    )
    continuation = detected_intent.name == workflow_context and bool(
        re.fullmatch(r"\s*(?:continue|proceed|go ahead|keep going|next)\b[.! ]*\s*", safe_message, re.IGNORECASE)
    )
    if continuation:
        prior_user_text = _previous_user_message(conversation)
        if prior_user_text:
            routing_text = f"{prior_user_text} {safe_message}".strip()
    conversation.append(HumanMessage(content=safe_message))
    if customer_id:
        _record_conversation(customer_id, "user", safe_message)
    missing = [] if customer_confirmed or customer_declined else _missing_loan_fields(routing_text, detected_intent.name)
    follow_up_fields = list(missing)

    if identity_error:
        output = "I could not match that identifier. Please check it and provide your Customer ID, account number, or registered mobile number."
        steps: list[tuple[Any, Any]] = []
        awaiting_confirmation = False
    elif not customer_id and detected_intent.name in {
        "memory_query", "memory_update", "account_query", "application_query", "loan_application",
    }:
        output = "Before I access or save customer-specific information, please provide your Customer ID, account number, or registered mobile number."
        steps = []
        awaiting_confirmation = False
        follow_up_fields = ["customer_identifier"]
    elif detected_intent.name == "customer_registration" and not registration_mobile:
        output = "To register you, please provide your full name and registered mobile number."
        steps = []
        awaiting_confirmation = False
        follow_up_fields = ["full_name", "registered_mobile_number"]
    elif detected_intent.name == "customer_registration" and not registration_name:
        output = "I have your mobile number. Please provide your full name to complete registration."
        steps = []
        awaiting_confirmation = False
        follow_up_fields = ["full_name"]
    elif detected_intent.name == "customer_registration":
        observation = customer_registration_tool.func(
            registration_name,
            registration_mobile,
            0,
            "",
            "Savings",
        )
        registration = json.loads(observation)
        action = type("Action", (), {
            "tool": "customer_registration_tool",
            "tool_input": {"full_name": registration_name},
        })()
        steps = [(action, observation)]
        if registration.get("status") == "registered":
            customer_id = registration["customer_id"]
            identity_changed = True
            _record_conversation(customer_id, "user", safe_message)
            output = f"Welcome, {registration_name}. Your customer profile is ready. Your Customer ID is **{customer_id}**."
        else:
            output = registration.get("message", "I couldn't complete registration. Please check the name and mobile number.")
        awaiting_confirmation = False
    elif customer_confirmed:
        observation = loan_application_processor_tool.func(
            customer_confirmed=True,
            approved_application=approved_application or {},
            customer_id=customer_id,
        )
        submission = json.loads(observation)
        action = type("Action", (), {
            "tool": "loan_application_processor_tool",
            "tool_input": {"application_id": (approved_application or {}).get("application_id")},
        })()
        steps = [(action, observation)]
        if submission.get("status") == "SUBMITTED":
            output = (
                f"Your loan application **{submission['application_id']}** has been submitted. "
                f"Tracking number: **{submission['tracking_number']}**."
            )
            approved_application = None
        else:
            output = f"I couldn't submit the pending offer: {submission.get('message', submission.get('status', 'unknown error'))}."
        awaiting_confirmation = False
    elif customer_declined:
        observation = decline_customer_offer_tool.func(
            customer_id=customer_id,
            approved_application=approved_application or {},
        )
        decline_result = json.loads(observation)
        action = type("Action", (), {
            "tool": "decline_customer_offer_tool",
            "tool_input": {"application_id": (approved_application or {}).get("application_id")},
        })()
        steps = [(action, observation)]
        output = "The pending application draft has been declined. No loan application was submitted." if decline_result.get("status") == "REJECTED" else f"I couldn't decline the pending draft: {decline_result.get('message', 'unknown error')}."
        approved_application = None
        awaiting_confirmation = False
    elif pending_application and continuation:
        output = (
            f"Your application offer **{pending_application['application_id']}** is still ready for review. "
            "No new application was created. Reply ‘yes’ to submit it or ‘no’ to decline it."
        )
        steps = []
        awaiting_confirmation = True
        approved_application = pending_application
    elif missing:
        output = "To assess the loan request, please provide your " + ", ".join(missing) + "."
        conversation.append(AIMessage(content=output))
        steps = []
        awaiting_confirmation = False
    elif detected_intent.name == "emi_calculation":
        emi_inputs = _extract_emi_details(routing_text)
        if not emi_inputs:
            output = "To calculate EMI, please provide the loan amount and tenure, for example: ₹500000 for 3 years."
            steps = []
            follow_up_fields = ["loan_amount", "tenure_years"]
        else:
            if progress_callback:
                progress_callback({"stage": "Tool selected", "detail": "emi_calculator_tool"})
            observation = emi_calculator_tool.invoke(emi_inputs)
            emi_result = json.loads(observation)
            action = type("Action", (), {"tool": "emi_calculator_tool", "tool_input": emi_inputs})()
            steps = [(action, observation)]
            if progress_callback:
                progress_callback({"stage": "Tool completed", "detail": "emi_calculator_tool"})
            output = (
                f"For ₹{emi_result['loan_amount']:,.0f} over {emi_result['tenure_years']} years "
                f"at {emi_result['annual_rate']:.2f}% p.a., the estimated EMI is "
                f"₹{emi_result['monthly_emi']:,.2f} per month. Total interest: "
                f"₹{emi_result['total_interest']:,.2f}."
            )
        awaiting_confirmation = False
    elif detected_intent.name == "application_query" and customer_id:
        application_match = re.search(r"\bAPP-[A-Z0-9-]+\b", routing_text, re.IGNORECASE)
        application_id = application_match.group(0).upper() if application_match else None
        observation = application_information_tool.func(customer_id, application_id)
        application_result = json.loads(observation)
        action = type("Action", (), {
            "tool": "application_information_tool",
            "tool_input": {"application_id": application_id} if application_id else {},
        })()
        steps = [(action, observation)]
        applications = application_result.get("applications", [])
        if applications:
            summaries = [
                f"Application **{item['application_id']}**: status **{item['status']}**, "
                f"tracking number **{item['tracking_number']}**, loan amount ₹{item['loan_amount']:,.0f}."
                for item in applications
            ]
            output = "\n".join(summaries)
        else:
            output = "I couldn't find any loan applications for your identified customer profile."
        awaiting_confirmation = False
    elif detected_intent.name == "memory_update" and customer_id:
        memory_facts = _extract_memory_facts(routing_text)
        if memory_facts:
            steps = []
            for memory_type, key, value in memory_facts:
                if progress_callback:
                    progress_callback({"stage": "Tool selected", "detail": "remember_user_information_tool"})
                observation = remember_user_information_tool.func(memory_type, key, value, customer_id)
                action = type("Action", (), {
                    "tool": "remember_user_information_tool",
                    "tool_input": {"memory_type": memory_type, "key": key, "value": value},
                })()
                steps.append((action, observation))
                if progress_callback:
                    progress_callback({"stage": "Tool completed", "detail": "remember_user_information_tool"})
            output = "I've saved " + " and ".join(key for _, key, _ in memory_facts) + " to your profile for future conversations."
            awaiting_confirmation = False
        else:
            agent = build_finpilot_agent(customer_identified=True, intent_name=detected_intent.name)
            result = agent.invoke({
                "messages": conversation,
                "customer_id": customer_id,
                "registration_mobile": registration_mobile,
                "customer_confirmed": False,
                "approved_application": {},
            })
            conversation = result.get("messages", conversation)
            output = str(conversation[-1].content) if conversation and isinstance(conversation[-1], AIMessage) else "I couldn't identify a specific profile fact to save."
            steps = []
            awaiting_confirmation = False
    else:
        agent = build_finpilot_agent(
            customer_confirmed=customer_confirmed,
            customer_declined=customer_declined,
            customer_identified=bool(customer_id),
            intent_name=detected_intent.name,
            active_workflow=workflow_context,
            pending_application=pending_application,
        )
        call_start = len(conversation)
        agent_input = {
            "messages": conversation,
            "customer_id": customer_id,
            "active_workflow": workflow_context,
            "registration_mobile": registration_mobile,
            "registration_name": registration_name,
            "customer_confirmed": customer_confirmed,
            "customer_declined": customer_declined,
            "approved_application": approved_application or {},
        }
        if progress_callback:
            progress_callback({"stage": "Agent started", "detail": detected_intent.label})
        result: dict[str, Any] = {"messages": list(conversation)}
        for update in agent.stream(agent_input, stream_mode="updates"):
            for node_name, node_update in update.items():
                new_messages = node_update.get("messages", []) if isinstance(node_update, dict) else []
                if new_messages:
                    result["messages"].extend(new_messages)
                if isinstance(node_update, dict):
                    result.update({key: value for key, value in node_update.items() if key != "messages"})
                if progress_callback:
                    tool_names = []
                    completed_tools = []
                    for emitted_message in new_messages:
                        tool_names.extend(
                            call.get("name", "unknown_tool")
                            for call in getattr(emitted_message, "tool_calls", [])
                        )
                        if isinstance(emitted_message, ToolMessage):
                            completed_tools.append(emitted_message.name)
                    if completed_tools:
                        progress_callback({"stage": "Tool completed", "detail": ", ".join(completed_tools)})
                    elif tool_names:
                        progress_callback({"stage": "Tool selected", "detail": ", ".join(tool_names)})
                    elif node_name == "model":
                        progress_callback({"stage": "Reasoning", "detail": "Agent received tool results or prepared a response"})
        conversation = result.get("messages", conversation)
        output = ""
        for msg in reversed(conversation):
            if isinstance(msg, AIMessage) and getattr(msg, "content", ""):
                content = msg.content
                output = "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and "text" in part) if isinstance(content, list) else str(content)
                break

        tool_messages = {msg.tool_call_id: msg for msg in conversation if isinstance(msg, ToolMessage)}
        steps = []
        for msg in conversation[call_start:]:
            if isinstance(msg, AIMessage):
                for tool_call in getattr(msg, "tool_calls", []):
                    tool_message = tool_messages.get(tool_call.get("id"))
                    if tool_message:
                        action = type("Action", (), {"tool": tool_call.get("name", "unknown_tool"), "tool_input": tool_call.get("args", {})})()
                        steps.append((action, tool_message.content))

        awaiting_confirmation = False
        for action, observation in steps:
            if action.tool == "request_customer_confirmation_tool":
                try:
                    parsed = json.loads(observation)
                    if parsed.get("status") == "awaiting_confirmation":
                        approved_application = parsed["offer"]
                        awaiting_confirmation = True
                except (TypeError, json.JSONDecodeError, KeyError):
                    pass
            if action.tool == "customer_registration_tool":
                try:
                    registration = json.loads(observation)
                    if registration.get("status") == "registered":
                        customer_id = registration["customer_id"]
                        identity_changed = True
                        _record_conversation(customer_id, "user", safe_message)
                except (TypeError, json.JSONDecodeError, KeyError):
                    pass
        application_submitted = any(
            action.tool == "loan_application_processor_tool"
            and _tool_result_status(observation) == "SUBMITTED"
            for action, observation in steps
        )
        if application_submitted:
            awaiting_confirmation = False
            approved_application = None
        application_declined = any(
            action.tool == "decline_customer_offer_tool"
            and _tool_result_status(observation) == "REJECTED"
            for action, observation in steps
        )
        if application_declined:
            awaiting_confirmation = False
            approved_application = None

    output = _summary_response(output, steps, awaiting_confirmation, detected_intent.name)
    if conversation and isinstance(conversation[-1], AIMessage):
        conversation[-1] = AIMessage(content=output)
    else:
        conversation.append(AIMessage(content=output))
    _record_conversation(customer_id, "assistant", output)
    AuditRepository.log("agent_run", safe_message)
    log_event("agent_run", "Agent processed request", {"customer_id": customer_id, "message": safe_message})
    next_active_workflow = workflow_context if detected_intent.name in {"contextual_followup", "general"} else detected_intent.name
    if pending_application and detected_intent.name == "application_query" and workflow_context == "loan_application":
        next_active_workflow = "loan_application"
    return {
        "output": output,
        "intermediate_steps": steps,
        "customer_id": customer_id,
        "identified_customer_id": customer_id,
        "registration_pending": detected_intent.name == "customer_registration" and not bool(customer_id),
        "identity_changed": identity_changed,
        "detected_intent": detected_intent.name,
        "active_workflow": next_active_workflow,
        "intent_label": detected_intent.label,
        "follow_up_fields": follow_up_fields,
        "execution_trace": _build_execution_trace(detected_intent, steps, awaiting_confirmation, output),
        "messages": conversation,
        "awaiting_confirmation": awaiting_confirmation,
        "pending_application": (
            approved_application if awaiting_confirmation
            else None if customer_confirmed or customer_declined
            else pending_application
        ),
    }


def _tool_result_status(observation: Any) -> str | None:
    try:
        result = json.loads(observation)
    except (TypeError, json.JSONDecodeError):
        return None
    return result.get("status")
