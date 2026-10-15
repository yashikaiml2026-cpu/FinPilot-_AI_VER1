from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from src.database.repositories import CustomerRepository, LoanApplicationRepository
from src.knowledge.rag import query_policy_knowledge_base
from src.utils.security import mask_account_number


def calculate_emi(loan_amount: float, annual_rate: float, tenure_years: int) -> dict[str, Any]:
    principal = float(loan_amount)
    annual = float(annual_rate)
    months = int(tenure_years) * 12
    monthly_rate = (annual / 100) / 12
    if monthly_rate == 0:
        monthly_payment = principal / months if months else 0
    else:
        monthly_payment = principal * monthly_rate * ((1 + monthly_rate) ** months) / (((1 + monthly_rate) ** months) - 1)

    total_payable = monthly_payment * months
    total_interest = total_payable - principal

    balance = principal
    schedule = []
    for month in range(1, months + 1):
        interest = balance * monthly_rate
        principal_paid = monthly_payment - interest
        balance = max(balance - principal_paid, 0.0)
        schedule.append(
            {
                "month": month,
                "emi": round(monthly_payment, 2),
                "principal": round(principal_paid, 2),
                "interest": round(interest, 2),
                "balance": round(balance, 2),
            }
        )

    return {
        "loan_amount": round(principal, 2),
        "annual_rate": annual,
        "tenure_years": tenure_years,
        "monthly_emi": round(monthly_payment, 2),
        "total_interest": round(total_interest, 2),
        "total_payable": round(total_payable, 2),
        "schedule": schedule,
    }


def evaluate_eligibility(
    monthly_income: float,
    monthly_obligations: float,
    credit_score: int,
    loan_amount: float,
    existing_loans: int = 0,
    annual_rate: float = 9.5,
    tenure_years: int = 10,
    policy_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if monthly_income <= 0:
        raise ValueError("Monthly income must be greater than zero.")

    foir = (monthly_obligations / monthly_income) * 100
    emi_estimate = calculate_emi(loan_amount, annual_rate, tenure_years)["monthly_emi"]
    debt_to_income = ((monthly_obligations + emi_estimate) / monthly_income) * 100
    loan_to_income_ratio = (loan_amount / monthly_income) * 100

    base_score = 72
    if credit_score >= 750:
        base_score += 18
    elif credit_score >= 680:
        base_score += 12
    elif credit_score >= 620:
        base_score += 6
    else:
        base_score -= 10

    base_score -= foir * 0.32
    base_score -= existing_loans * 4
    base_score -= loan_to_income_ratio * 0.08

    approval_probability = max(20, min(96, round(base_score, 2)))
    if approval_probability >= 75 and foir <= 45 and credit_score >= 630:
        risk_category = "LOW"
    elif approval_probability >= 55 and foir <= 55:
        risk_category = "MEDIUM"
    else:
        risk_category = "HIGH"

    reason_codes = []
    if credit_score < 650:
        reason_codes.append("Credit score below preferred threshold")
    if foir > 45:
        reason_codes.append("FOIR exceeds preferred affordability threshold")
    if existing_loans > 1:
        reason_codes.append("Multiple active loan obligations")
    if loan_to_income_ratio > 50:
        reason_codes.append("Requested loan amount is elevated relative to income")
    if not reason_codes:
        reason_codes.append("Strong repayment capacity and risk profile")

    policy_passages = [str(item.get("content", "")) for item in (policy_context or [])]
    policy_text = " ".join(policy_passages).lower()
    policy_sources = sorted({str(item.get("source", "unknown")) for item in (policy_context or [])})
    policy_has_affordability_guidance = "foir" in policy_text and any(
        term in policy_text for term in ("prudent", "repayment", "affordability", "debt burden")
    )
    policy_recommends_escalation = any(
        term in policy_text for term in ("escalat", "manual review", "repayment stress", "additional review")
    )
    policy_credit_guidance = "credit profile" in policy_text or "credit score" in policy_text
    policy_review_required = (
        (policy_has_affordability_guidance and policy_recommends_escalation and foir > 45)
        or (policy_credit_guidance and policy_recommends_escalation and credit_score < 630)
    )
    if policy_review_required:
        reason_codes.append("Retrieved policy guidance requires manual review of affordability or credit risk")

    return {
        "foir": round(foir, 2),
        "debt_to_income": round(debt_to_income, 2),
        "risk_category": risk_category,
        "reason_codes": reason_codes,
        "eligibility_score": round(base_score, 2),
        "approval_probability": approval_probability,
        "loan_to_income_ratio": round(loan_to_income_ratio, 2),
        "policy_decision": "MANUAL_REVIEW" if policy_review_required else "CONTINUE_REVIEW" if policy_text else "POLICY_CONTEXT_UNAVAILABLE",
        "policy_sources": policy_sources,
    }


def get_customer_profile(customer_id: str) -> dict[str, Any]:
    customer = CustomerRepository.get_customer(customer_id)
    if customer is None:
        raise ValueError(f"Customer {customer_id} not found in the database.")

    profile = dict(customer)
    account_number = profile.get("account_number")
    profile["account_number"] = mask_account_number(account_number) if account_number else None
    profile["phone"] = profile.get("phone", "").replace(
        profile.get("phone", "")[:3], "***", 1
    ) if profile.get("phone") else "***"
    return profile


def query_policy_knowledge_base(question: str, k: int = 4) -> list[dict[str, Any]]:
    engine = __import__("src.knowledge.rag", fromlist=["query_policy_knowledge_base"]).query_policy_knowledge_base
    return engine(question, k=k)


def submit_loan_application(
    customer_id: str,
    loan_amount: float,
    annual_rate: float,
    tenure_years: int,
    eligibility: dict[str, Any],
    report: str,
    status: str = "PENDING",
    application_id: str | None = None,
    tracking_number: str | None = None,
    submitted_at: str | None = None,
) -> dict[str, Any]:
    application_id = application_id or f"APP-{uuid.uuid4().hex[:8].upper()}"
    tracking_number = tracking_number or f"TRK-{uuid.uuid4().hex[:8].upper()}"
    timestamp = submitted_at or datetime.now(timezone.utc).isoformat()
    payload = {
        "application_id": application_id,
        "customer_id": customer_id,
        "loan_amount": float(loan_amount),
        "interest_rate": float(annual_rate),
        "tenure_months": int(tenure_years) * 12,
        "status": status,
        "approval_probability": float(eligibility.get("approval_probability", 0)),
        "tracking_number": tracking_number,
        "submitted_at": timestamp,
        "reason_codes": eligibility.get("reason_codes", []),
        "report": report,
    }
    if LoanApplicationRepository.get_application(customer_id, application_id):
        current = LoanApplicationRepository.get_application(customer_id, application_id)[0]
        if current["status"] == "DRAFT":
            LoanApplicationRepository.update_status(customer_id, application_id, "PENDING")
        application = LoanApplicationRepository.update_status(customer_id, application_id, status)
        payload["submitted_at"] = application["submitted_at"]
    else:
        LoanApplicationRepository.save_application(payload)
    return payload
