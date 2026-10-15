import json

import pandas as pd
from langchain_core.tools import BaseTool
import pytest

from src.agent import agent as finpilot_agent
from src.agent import tools as banking_tools
from src.agent.tools import evaluate_eligibility
from src.data import pipeline
from src.database import db
from src.database.repositories import CustomerRepository, LoanApplicationRepository, UserMemoryRepository
from src.knowledge.rag import PolicyKnowledgeBase
from src.agent.intents import classify_intent


def test_single_agent_registers_required_banking_tools():
    tools = {tool.name: tool for tool in finpilot_agent.TOOLS}
    assert all(isinstance(tool, BaseTool) for tool in tools.values())
    assert {
        "emi_calculator_tool",
        "eligibility_checker_tool",
        "account_information_tool",
        "loan_application_processor_tool",
    } <= tools.keys()
    assert "request_customer_confirmation_tool" in tools


def test_missing_salary_amount_and_tenure_are_requested(monkeypatch):
    monkeypatch.setattr(finpilot_agent, "_record_conversation", lambda *args: None)
    monkeypatch.setattr(finpilot_agent.AuditRepository, "log", lambda *args: None)
    monkeypatch.setattr(finpilot_agent, "log_event", lambda *args: None)
    monkeypatch.setattr(
        finpilot_agent,
        "build_finpilot_agent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("LLM must not run with missing required fields")),
    )

    result = finpilot_agent.run_finpilot_agent(
        "I want to apply for a personal loan",
        identified_customer_id="CUST-TEST",
    )

    assert "monthly salary or income" in result["output"]
    assert "requested loan amount" in result["output"]
    assert "loan tenure" in result["output"]
    assert result["intermediate_steps"] == []


@pytest.mark.parametrize(
    ("request_text", "missing_field"),
    [
        ("I need a loan of ₹500000 for 3 years", "monthly salary or income"),
        ("My monthly salary is ₹60000; I need a loan for 3 years", "requested loan amount"),
        ("My monthly income is ₹60000; I need a loan of ₹500000", "loan tenure"),
    ],
)
def test_each_required_loan_field_is_checked(request_text, missing_field):
    assert missing_field in finpilot_agent._missing_loan_fields(request_text)


def test_eligibility_parser_accepts_currency_and_over_duration_phrasing():
    assert finpilot_agent._missing_loan_fields(
        "Check eligibility for a ₹500000 personal loan over 3 years. "
        "My salary is ₹60000 monthly.",
        "eligibility_query",
    ) == []


def test_final_summary_reports_emi_eligibility_and_application_status():
    action = type("Action", (), {"tool": "emi_calculator_tool"})()
    eligibility_action = type("Action", (), {"tool": "eligibility_checker_tool"})()
    offer_action = type("Action", (), {"tool": "request_customer_confirmation_tool"})()
    offer = {
        "status": "awaiting_confirmation",
        "offer": {
            "emi": {"monthly_emi": 12345.67},
            "eligibility": {"risk_category": "LOW", "approval_probability": 82.5},
        },
    }
    steps = [
        (action, json.dumps({"monthly_emi": 12345.67})),
        (eligibility_action, json.dumps({"risk_category": "LOW", "approval_probability": 82.5})),
        (offer_action, json.dumps(offer)),
    ]

    response = finpilot_agent._summary_response("Offer prepared.", steps, awaiting_confirmation=True)

    assert "EMI: ₹12,345.67 per month" in response
    assert "Eligibility result: LOW risk" in response
    assert "Awaiting explicit customer confirmation; not submitted." in response


def test_final_summary_reports_actual_submission_status():
    action = type("Action", (), {"tool": "loan_application_processor_tool"})()
    response = finpilot_agent._summary_response(
        "Application submitted.",
        [(action, json.dumps({"status": "SUBMITTED", "application_id": "APP-123"}))],
        awaiting_confirmation=False,
    )
    assert "Application status: Submitted (APP-123)." in response


def test_eligibility_uses_requested_loan_tenure():
    short_term = evaluate_eligibility(60000, 5000, 710, 500000, annual_rate=9.5, tenure_years=3)
    long_term = evaluate_eligibility(60000, 5000, 710, 500000, annual_rate=9.5, tenure_years=10)
    assert short_term["debt_to_income"] > long_term["debt_to_income"]


def test_retrieved_policy_changes_eligibility_review_outcome():
    passages = [
        {"content": "FOIR should remain within a prudent threshold for affordability."},
        {"content": "High repayment stress should be escalated for manual review."},
    ]
    result = evaluate_eligibility(
        monthly_income=60000,
        monthly_obligations=30000,
        credit_score=710,
        loan_amount=500000,
        annual_rate=9.5,
        tenure_years=3,
        policy_context=passages,
    )
    assert result["policy_decision"] == "MANUAL_REVIEW"


def test_eligibility_tool_uses_retrieved_policy_context(monkeypatch):
    passages = [
        {"content": "FOIR should remain within a prudent threshold for affordability.", "source": "loan_policies.md"},
        {"content": "High repayment stress should be escalated for manual review.", "source": "risk_assessment.md"},
    ]
    monkeypatch.setattr(finpilot_agent, "query_policy_knowledge_base", lambda *args, **kwargs: passages)

    result = json.loads(finpilot_agent.eligibility_checker_tool.func(
        monthly_income=60000,
        monthly_obligations=30000,
        credit_score=710,
        loan_amount=500000,
        annual_rate=9.5,
        tenure_years=3,
    ))

    assert result["policy_decision"] == "MANUAL_REVIEW"
    assert result["policy_sources"] == ["loan_policies.md", "risk_assessment.md"]


def test_policy_retrieval_ranks_matching_passages(tmp_path):
    knowledge_dir = tmp_path / "policies"
    knowledge_dir.mkdir()
    (knowledge_dir / "loan_policy.md").write_text(
        "FOIR should remain within a prudent affordability threshold.\n\n"
        "High repayment stress should be escalated for manual review.",
        encoding="utf-8",
    )
    knowledge_base = PolicyKnowledgeBase(knowledge_dir=knowledge_dir, persist_dir=tmp_path / "chroma")

    results = knowledge_base.search("FOIR affordability repayment stress escalation manual review", k=2)

    assert len(results) == 2
    assert any("foir" in result["content"].lower() for result in results)
    assert any("manual review" in result["content"].lower() for result in results)


def test_account_lookup_tool_reads_persistent_context_only(monkeypatch):
    monkeypatch.setattr(finpilot_agent, "get_customer_profile", lambda customer_id: {"customer_id": customer_id, "balance": 100})
    monkeypatch.setattr(
        UserMemoryRepository,
        "get_context",
        lambda customer_id: {"preferences": {"language": "English"}, "recent_interactions": [], "recent_applications": []},
    )

    result = json.loads(finpilot_agent.account_information_tool.func("CUST-0001"))

    assert result["account_profile"]["balance"] == 100
    assert result["persistent_context"]["preferences"]["language"] == "English"


def test_user_memory_persists_profile_preferences_interactions_and_applications(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "memory.db")
    db.initialize_database()
    UserMemoryRepository.save_memory("CUST-MEM", "profile", "employment", "Salaried")
    UserMemoryRepository.save_memory("CUST-MEM", "preference", "language", "English")
    with db.get_connection() as conn:
        conn.execute(
            "INSERT INTO conversation_logs (customer_id, role, message) VALUES (?, ?, ?)",
            ("CUST-MEM", "user", "I prefer concise explanations."),
        )
        conn.commit()
    LoanApplicationRepository.save_application({
        "application_id": "APP-MEM",
        "customer_id": "CUST-MEM",
        "loan_amount": 100000,
        "interest_rate": 9.5,
        "tenure_months": 12,
        "status": "SUBMITTED",
        "approval_probability": 80,
        "tracking_number": "TRK-MEM",
        "reason_codes": [],
        "report": "Test application",
    })

    context = UserMemoryRepository.get_context("CUST-MEM")

    assert context["profile_facts"]["employment"] == "Salaried"
    assert context["preferences"]["language"] == "English"
    assert context["recent_interactions"][0]["message"] == "I prefer concise explanations."
    assert context["recent_applications"][0]["application_id"] == "APP-MEM"


@pytest.mark.parametrize(
    ("key", "value"),
    [("password", "example-value"), ("account number", "123456789012")],
)
def test_user_memory_rejects_secrets_and_account_identifiers(key, value):
    with pytest.raises(ValueError):
        UserMemoryRepository.save_memory("CUST-MEM", "profile", key, value)


def test_memory_extraction_skips_redacted_customer_identifiers():
    facts = finpilot_agent._extract_memory_facts(
        "My name is Rahul. My Customer ID is [REDACTED CUSTOMER IDENTIFIER]. "
        "My preferred loan type is Home Loan."
    )
    assert facts == [
        ("profile", "name", "Rahul"),
        ("preference", "preferred loan type", "Home Loan"),
    ]


def test_existing_ingestion_path_imports_customer_rows(monkeypatch, tmp_path):
    customer_file = tmp_path / "customers.csv"
    pd.DataFrame([{"customer_id": "CUST-0001", "full_name": "Customer 1"}]).to_csv(customer_file, index=False)
    imported = []
    monkeypatch.setattr(pipeline, "generate_processed_datasets", lambda: {"customers": str(customer_file)})
    monkeypatch.setattr(CustomerRepository, "upsert_customer", imported.append)

    pipeline.ingest_raw_datasets()

    assert imported == [{"customer_id": "CUST-0001", "full_name": "Customer 1"}]


def test_application_tool_blocks_without_confirmation(monkeypatch):
    submit_calls = []
    monkeypatch.setattr(
        finpilot_agent,
        "submit_loan_application",
        lambda **kwargs: submit_calls.append(kwargs),
    )

    result = json.loads(
        finpilot_agent.loan_application_processor_tool.func(
            customer_confirmed=False,
            approved_application={},
            customer_id="CUST-0001",
        )
    )

    assert result["status"] == "blocked"
    assert submit_calls == []


def test_application_tool_submits_only_injected_approved_offer(monkeypatch):
    expected = {
        "customer_id": "CUST-0001",
        "loan_amount": 500000,
        "annual_rate": 9.5,
        "tenure_years": 3,
        "eligibility": {"approval_probability": 80, "reason_codes": []},
        "report": "Customer-authorized loan application offer",
    }
    monkeypatch.setattr(finpilot_agent, "submit_loan_application", lambda **kwargs: expected)

    result = json.loads(
        finpilot_agent.loan_application_processor_tool.func(
            customer_confirmed=True,
            approved_application=expected,
            customer_id="CUST-0001",
        )
    )

    assert result == expected


@pytest.mark.parametrize(
    ("text", "expected_type"),
    [
        ("My Customer ID is CUST-4242", "customer_id"),
        ("Account number: 7000000000012342", "account_number"),
        ("Registered mobile: +1 (415) 555-2671", "mobile_number"),
        ("4155552671", "mobile_number"),
        ("7000000000012342", "account_number"),
    ],
)
def test_customer_identifier_extraction(text, expected_type):
    result = finpilot_agent._extract_customer_identifier(text)
    assert result is not None
    assert result[0] == expected_type


def test_customer_can_be_identified_by_id_account_or_mobile(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "identify.db")
    db.initialize_database()
    CustomerRepository.upsert_customer({
        "customer_id": "CUST-4242",
        "account_number": "7000000000012342",
        "full_name": "Test Customer",
        "phone": "+1 (415) 555-2671",
    })

    assert CustomerRepository.identify_customer("customer_id", "cust-4242") == "CUST-4242"
    assert CustomerRepository.identify_customer("account_number", "7000000000012342") == "CUST-4242"
    assert CustomerRepository.identify_customer("mobile_number", "4155552671") == "CUST-4242"


def test_new_customer_registration_generates_and_persists_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "onboarding.db")
    db.initialize_database()
    first = CustomerRepository.register_customer({"full_name": "Rahul", "phone": "+91 98765 43210"})
    second = CustomerRepository.register_customer({"full_name": "Maya", "phone": "+91 98765 43211"})

    assert first["customer_id"] == "CUST-000001"
    assert second["customer_id"] == "CUST-000002"
    assert first["account_number"] != second["account_number"]
    assert CustomerRepository.identify_customer("mobile_number", "+91 98765 43210") == first["customer_id"]
    assert CustomerRepository.get_customer(first["customer_id"])["full_name"] == "Rahul"


def test_application_lifecycle_preserves_ids_and_timestamp(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "applications.db")
    db.initialize_database()
    draft = LoanApplicationRepository.create_draft({
        "application_id": "APP-LIFECYCLE",
        "customer_id": "CUST-TEST",
        "loan_amount": 100000,
        "interest_rate": 9.5,
        "tenure_months": 12,
        "approval_probability": 80,
        "tracking_number": "TRK-LIFECYCLE",
        "reason_codes": [],
        "report": "test",
    })
    timestamp = LoanApplicationRepository.get_application("CUST-TEST", draft["application_id"])[0]["submitted_at"]
    for status in ("PENDING", "SUBMITTED", "REVIEW", "APPROVED"):
        application = LoanApplicationRepository.update_status("CUST-TEST", draft["application_id"], status)

    assert application["status"] == "APPROVED"
    assert application["tracking_number"] == "TRK-LIFECYCLE"
    assert application["submitted_at"] == timestamp
    with pytest.raises(ValueError, match="Invalid application status transition"):
        LoanApplicationRepository.update_status("CUST-TEST", draft["application_id"], "REJECTED")


def test_application_lookup_is_customer_scoped(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "application-lookup.db")
    db.initialize_database()
    LoanApplicationRepository.save_application({
        "application_id": "APP-PRIVATE",
        "customer_id": "CUST-ONE",
        "loan_amount": 100000,
        "interest_rate": 9.5,
        "tenure_months": 12,
        "status": "SUBMITTED",
        "approval_probability": 80,
        "tracking_number": "TRK-PRIVATE",
        "reason_codes": [],
    })

    assert LoanApplicationRepository.get_application("CUST-TWO", "APP-PRIVATE") == []


def test_confirmed_application_updates_the_existing_draft(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "confirmed-application.db")
    db.initialize_database()
    CustomerRepository.upsert_customer({"customer_id": "CUST-CONFIRM", "full_name": "Test Customer", "phone": "+1 4155550123"})
    monkeypatch.setattr(finpilot_agent, "query_policy_knowledge_base", lambda *args, **kwargs: [])
    offer_result = json.loads(finpilot_agent.request_customer_confirmation_tool.func(
        customer_id="CUST-CONFIRM",
        loan_amount=500000,
        annual_rate=9.5,
        tenure_years=3,
        monthly_income=60000,
        monthly_obligations=5000,
        credit_score=710,
    ))
    offer = offer_result["offer"]
    draft = LoanApplicationRepository.get_application("CUST-CONFIRM", offer["application_id"])[0]

    submission = json.loads(finpilot_agent.loan_application_processor_tool.func(
        customer_confirmed=True,
        approved_application=offer,
        customer_id="CUST-CONFIRM",
    ))
    persisted = LoanApplicationRepository.get_application("CUST-CONFIRM", offer["application_id"])[0]

    assert draft["status"] == "DRAFT"
    assert submission["status"] == "SUBMITTED"
    assert persisted["status"] == "SUBMITTED"
    assert persisted["tracking_number"] == offer["tracking_number"]
    assert persisted["submitted_at"] == draft["submitted_at"]


def test_declined_offer_marks_the_draft_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "declined-application.db")
    db.initialize_database()
    CustomerRepository.upsert_customer({"customer_id": "CUST-DECLINE", "full_name": "Test Customer", "phone": "+1 4155550124"})
    draft = LoanApplicationRepository.create_draft({
        "application_id": "APP-DECLINED",
        "customer_id": "CUST-DECLINE",
        "loan_amount": 100000,
        "interest_rate": 9.5,
        "tenure_months": 12,
        "approval_probability": 80,
        "tracking_number": "TRK-DECLINED",
        "reason_codes": [],
        "report": "offer",
    })
    result = json.loads(finpilot_agent.decline_customer_offer_tool.func(
        customer_id="CUST-DECLINE",
        approved_application={**draft, "customer_id": "CUST-DECLINE"},
    ))

    assert result["status"] == "REJECTED"
    assert LoanApplicationRepository.get_application("CUST-DECLINE", "APP-DECLINED")[0]["status"] == "REJECTED"


def test_emi_only_route_invokes_only_emi_tool(monkeypatch):
    monkeypatch.setattr(finpilot_agent, "_record_conversation", lambda *args: None)
    monkeypatch.setattr(finpilot_agent.AuditRepository, "log", lambda *args: None)
    monkeypatch.setattr(finpilot_agent, "log_event", lambda *args: None)
    result = finpilot_agent.run_finpilot_agent("Calculate EMI for ₹500000 for 3 years")
    observation = json.loads(result["intermediate_steps"][0][1])

    assert result["detected_intent"] == "emi_calculation"
    assert [action.tool for action, _ in result["intermediate_steps"]] == ["emi_calculator_tool"]
    assert observation["loan_amount"] == 500000
    assert observation["tenure_years"] == 3
    assert "salary" not in result["output"].lower()


def test_execution_trace_contains_only_runtime_tool_events():
    action = type("Action", (), {"tool": "emi_calculator_tool"})()
    trace = finpilot_agent._build_execution_trace(
        classify_intent("Calculate EMI for ₹500000 for 3 years"),
        [(action, json.dumps({"monthly_emi": 16000}))],
        awaiting_confirmation=False,
        output="EMI is calculated.",
    )
    assert [event["stage"] for event in trace] == [
        "Intent detected", "Tools selected", "Tool executed", "Reasoning", "Response ready"
    ]
    assert all(event["detail"] != "Eligibility Check" for event in trace)


def test_profile_lookup_does_not_fabricate_missing_account_number(monkeypatch):
    monkeypatch.setattr(
        banking_tools.CustomerRepository,
        "get_customer",
        lambda _: {"customer_id": "CUST-4242", "account_number": None, "phone": None},
    )
    profile = banking_tools.get_customer_profile("CUST-4242")
    assert profile["account_number"] is None


def test_account_tool_does_not_read_profile_before_identification(monkeypatch):
    monkeypatch.setattr(
        finpilot_agent,
        "get_customer_profile",
        lambda *_: (_ for _ in ()).throw(AssertionError("Profile must not load without identification")),
    )
    result = json.loads(finpilot_agent.account_information_tool.func(None))
    assert result["status"] == "identification_required"


def test_application_offer_requires_customer_identification():
    result = json.loads(finpilot_agent.request_customer_confirmation_tool.func(
        customer_id=None,
        loan_amount=500000,
        annual_rate=9.5,
        tenure_years=3,
        monthly_income=60000,
        monthly_obligations=5000,
        credit_score=710,
    ))
    assert result["status"] == "identification_required"


def test_agent_state_has_no_automatic_customer_default(monkeypatch):
    monkeypatch.setattr(
        finpilot_agent,
        "build_finpilot_agent",
        lambda **_: (_ for _ in ()).throw(AssertionError("Account request must stop before agent/tool execution")),
    )
    monkeypatch.setattr(finpilot_agent, "_record_conversation", lambda *args: None)
    monkeypatch.setattr(finpilot_agent.AuditRepository, "log", lambda *args: None)
    monkeypatch.setattr(finpilot_agent, "log_event", lambda *args: None)

    result = finpilot_agent.run_finpilot_agent("Please retrieve my account balance.")

    assert result["identified_customer_id"] is None
    assert "Customer ID, account number, or registered mobile number" in result["output"]
    assert result["intermediate_steps"] == []
