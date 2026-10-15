from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

IntentName = Literal[
    "memory_query",
    "memory_update",
    "emi_calculation",
    "account_query",
    "application_query",
    "eligibility_query",
    "loan_application",
    "customer_registration",
    "contextual_followup",
    "general",
]


@dataclass(frozen=True)
class Intent:
    name: IntentName
    label: str
    tool_names: tuple[str, ...]


ROUTES: dict[IntentName, Intent] = {
    "memory_query": Intent("memory_query", "Memory Recall", ("user_memory_query_tool",)),
    "memory_update": Intent("memory_update", "Save User Memory", ("remember_user_information_tool",)),
    "emi_calculation": Intent("emi_calculation", "EMI Calculation", ("emi_calculator_tool",)),
    "account_query": Intent("account_query", "Account Lookup", ("account_information_tool",)),
    "application_query": Intent("application_query", "Application Tracking", ("application_information_tool",)),
    "eligibility_query": Intent("eligibility_query", "Eligibility Check", ("eligibility_checker_tool", "policy_retriever_tool")),
    "loan_application": Intent(
        "loan_application",
        "Loan Application",
        (
            "account_information_tool",
            "emi_calculator_tool",
            "eligibility_checker_tool",
            "policy_retriever_tool",
            "request_customer_confirmation_tool",
            "loan_application_processor_tool",
            "decline_customer_offer_tool",
        ),
    ),
    "customer_registration": Intent("customer_registration", "Customer Registration", ("customer_registration_tool",)),
    "contextual_followup": Intent("contextual_followup", "Conversation Follow-up", ()),
    "general": Intent("general", "General Conversation", ()),
}


def classify_intent(text: str, previous_intent: IntentName | None = None) -> Intent:
    normalized = re.sub(r"\s+", " ", text).strip().lower()

    if previous_intent and re.fullmatch(r"(?:track|show|check)\s+(?:it|that|the application)", normalized):
        return ROUTES["application_query"]
    if previous_intent and re.fullmatch(r"(?:why|explain|explain it|tell me more|what does that mean)\??", normalized):
        return ROUTES["contextual_followup"]
    if previous_intent and re.fullmatch(r"(?:continue|proceed|go ahead|keep going|yes|okay|ok|next)\b[.! ]*", normalized):
        return ROUTES[previous_intent]

    if re.search(r"\b(register|sign me up|new customer|open (?:a )?(?:bank )?account|create (?:a )?customer profile)\b", normalized):
        return ROUTES["customer_registration"]

    has_registration_name = re.search(r"\b(?:my\s+)?name\s*(?:is|:)\s*[a-z][a-z '-]+", normalized)
    has_registration_mobile = re.search(r"\b(?:phone|mobile|number)(?:\s+number)?\s*(?:is|:)?\s*\+?[\d ()-]{10,}", normalized)
    if has_registration_name and has_registration_mobile:
        return ROUTES["customer_registration"]
    if previous_intent == "customer_registration" and re.search(
        r"\b(?:name\s+is|phone|mobile|registered number)\b", normalized
    ):
        return ROUTES["customer_registration"]

    if re.search(r"\b(my name is|i am called|my preferred(?:\s+[\w-]+){0,3}\s+is\b|i prefer\b|remember that i|please remember)\b", normalized):
        return ROUTES["memory_update"]

    if re.search(r"\b(what('?s| is) my\s+(?:name|preferred|preference|profile fact|employment|city|age|language)|do you remember|remind me|recall|my (?:name|preferred|preference|profile fact))\b", normalized):
        return ROUTES["memory_query"]

    if re.search(
        r"\b(track|status of|where is|check|show|list|past|previous|latest|recent)\b.*\b(applications?|loan applications?)\b"
        r"|\bapplications?\s+(?:id|number|status|history)\b",
        normalized,
    ):
        return ROUTES["application_query"]

    if re.search(r"\b(account|balance|transactions?|account details|account type)\b", normalized):
        return ROUTES["account_query"]

    is_emi_request = re.search(r"\b(emi|monthly installment|monthly payment)\b", normalized)
    is_calculation = re.search(r"\b(calculate|compute|work out|estimate|what would|what is the emi for)\b", normalized)
    is_application = re.search(
        r"\b(apply|application|submit|borrow|loan request|want a loan|need a loan)\b"
        r"|\b(?:want|need|looking for)\s+(?:a\s+)?(?:personal|home|auto|car|education|business)?\s*loan\b",
        normalized,
    )
    if is_emi_request and is_calculation and not is_application:
        return ROUTES["emi_calculation"]

    if re.search(r"\b(eligible|eligibility|qualify|qualification|afford|approval probability)\b", normalized):
        return ROUTES["eligibility_query"]

    if is_application:
        return ROUTES["loan_application"]

    if previous_intent in {"loan_application", "eligibility_query"} and re.search(
        r"\b(salary|income|loan amount|existing emi|obligations|credit score|\d+\s*(?:years?|yrs?))\b",
        normalized,
    ):
        return ROUTES[previous_intent]

    if previous_intent and re.match(r"^(and |what about|how about|also |then |same )", normalized):
        return ROUTES[previous_intent]

    return ROUTES["general"]
