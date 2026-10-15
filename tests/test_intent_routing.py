import pytest

from src.agent.intents import classify_intent
from src.agent import agent as finpilot_agent


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("What is my name and preferred loan type?", "memory_query"),
        ("My name is Rahul. My preferred loan type is Home Loan.", "memory_update"),
        ("Calculate EMI for ₹500000 for 3 years", "emi_calculation"),
        ("Show account details", "account_query"),
        ("Track application APP-123", "application_query"),
        ("past application", "application_query"),
        ("What is my application status?", "application_query"),
        ("Check whether I qualify for a loan", "eligibility_query"),
        ("Apply for a personal loan", "loan_application"),
        ("I want a personal loan of ₹500000 for 3 years", "loan_application"),
        ("I am a new customer, register me", "customer_registration"),
        ("name is Yashik Kalra and phone number is 6280033182", "customer_registration"),
        ("Hello FinPilot", "general"),
    ],
)
def test_intent_routes_by_user_request(message, expected):
    assert classify_intent(message).name == expected


def test_contextual_follow_up_keeps_previous_intent():
    assert classify_intent("What about for 5 years?", previous_intent="emi_calculation").name == "emi_calculation"
    assert classify_intent("continue", previous_intent="loan_application").name == "loan_application"
    assert classify_intent("why", previous_intent="loan_application").name == "contextual_followup"
    assert classify_intent("track it", previous_intent="loan_application").name == "application_query"


def test_memory_recall_takes_precedence_over_loan_keyword():
    intent = classify_intent("What is my preferred loan type?")
    assert intent.name == "memory_query"
    assert intent.tool_names == ("user_memory_query_tool",)


def test_emi_request_routes_only_to_emi_tool():
    intent = classify_intent("Calculate EMI for ₹500000 for 3 years")
    assert intent.name == "emi_calculation"
    assert intent.tool_names == ("emi_calculator_tool",)


def test_agent_tool_allowlist_matches_memory_and_emi_intents(monkeypatch):
    captured = {}
    monkeypatch.setattr(finpilot_agent, "build_llm_client", lambda: object())

    def fake_create_agent(llm, tools, **kwargs):
        captured["names"] = [tool.name for tool in tools]
        return object()

    monkeypatch.setattr(finpilot_agent, "create_agent", fake_create_agent)
    finpilot_agent.build_finpilot_agent(customer_identified=True, intent_name="memory_query")
    assert captured["names"] == ["user_memory_query_tool"]

    finpilot_agent.build_finpilot_agent(intent_name="emi_calculation")
    assert captured["names"] == ["emi_calculator_tool"]

    finpilot_agent.build_finpilot_agent(customer_identified=True, customer_confirmed=True, intent_name="loan_application")
    assert captured["names"] == ["loan_application_processor_tool"]

    finpilot_agent.build_finpilot_agent(customer_identified=True, customer_declined=True, intent_name="loan_application")
    assert captured["names"] == ["decline_customer_offer_tool"]
