import pytest

from src.agent.tools import calculate_emi, evaluate_eligibility


def test_emi_calculation_returns_expected_values():
    emi = calculate_emi(loan_amount=1000000, annual_rate=9.5, tenure_years=12)
    assert emi["monthly_emi"] > 0
    assert emi["total_payable"] > emi["loan_amount"]
    assert len(emi["schedule"]) == 12 * 12


def test_eligibility_risk_scoring():
    result = evaluate_eligibility(
        monthly_income=120000,
        monthly_obligations=18000,
        credit_score=710,
        loan_amount=900000,
        existing_loans=1,
    )
    assert result["risk_category"] in {"LOW", "MEDIUM", "HIGH"}
    assert 0 <= result["approval_probability"] <= 100
    assert result["foir"] > 0
