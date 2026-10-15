from __future__ import annotations

import re
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Any

from src.database.db import get_connection


class CustomerRepository:
    @staticmethod
    def identify_customer(identifier_type: str, identifier: str) -> str | None:
        normalized_type = identifier_type.strip().lower()
        value = identifier.strip()
        if normalized_type == "customer_id":
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT customer_id FROM customers WHERE UPPER(customer_id) = UPPER(?)",
                    (value,),
                ).fetchone()
                return row["customer_id"] if row else None

        digits = re.sub(r"\D", "", value)
        if normalized_type == "account_number":
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT customer_id FROM customers WHERE account_number = ?",
                    (digits,),
                ).fetchone()
                return row["customer_id"] if row else None

        if normalized_type == "mobile_number":
            if len(digits) < 10:
                return None
            candidate = digits[-10:]
            with get_connection() as conn:
                rows = conn.execute("SELECT customer_id, phone FROM customers WHERE phone IS NOT NULL").fetchall()
            matches = [
                row["customer_id"] for row in rows
                if re.sub(r"\D", "", row["phone"])[-10:] == candidate
            ]
            return matches[0] if len(matches) == 1 else None

        return None

    @staticmethod
    def register_customer(payload: dict[str, Any]) -> dict[str, Any]:
        full_name = str(payload.get("full_name", "")).strip()
        phone = str(payload.get("phone", "")).strip()
        phone_digits = re.sub(r"\D", "", phone)
        if not full_name or len(phone_digits) < 10:
            raise ValueError("Full name and a valid registered mobile number are required.")

        with get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT customer_id, phone FROM customers WHERE phone IS NOT NULL").fetchall()
            if any(re.sub(r"\D", "", row["phone"])[-10:] == phone_digits[-10:] for row in existing):
                raise ValueError("A customer is already registered with this mobile number.")
            rows = conn.execute(
                "SELECT customer_id FROM customers WHERE customer_id GLOB 'CUST-[0-9]*'"
            ).fetchall()
            last_number = max(
                (int(row["customer_id"].split("-", 1)[1]) for row in rows),
                default=0,
            )
            customer_id = f"CUST-{last_number + 1:06d}"
            account_number = str(secrets.randbelow(9_000_000_000_000_000) + 1_000_000_000_000_000)
            customer = {
                **payload,
                "customer_id": customer_id,
                "account_number": account_number,
                "full_name": full_name,
                "phone": phone,
            }
            CustomerRepository.upsert_customer(customer, connection=conn)
            conn.commit()
        return customer

    @staticmethod
    def get_customer(customer_id: str) -> dict[str, Any] | None:
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
            return dict(row) if row else None

    @staticmethod
    def upsert_customer(payload: dict[str, Any], connection: sqlite3.Connection | None = None) -> None:
        def upsert(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO customers (
                    customer_id, account_number, full_name, age, gender, monthly_income, account_type,
                    credit_score, existing_loans, balance, city, email, phone,
                    employment_type, risk_profile
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    account_number=excluded.account_number,
                    full_name=excluded.full_name,
                    age=excluded.age,
                    gender=excluded.gender,
                    monthly_income=excluded.monthly_income,
                    account_type=excluded.account_type,
                    credit_score=excluded.credit_score,
                    existing_loans=excluded.existing_loans,
                    balance=excluded.balance,
                    city=excluded.city,
                    email=excluded.email,
                    phone=excluded.phone,
                    employment_type=excluded.employment_type,
                    risk_profile=excluded.risk_profile
                """,
                (
                    payload["customer_id"],
                    payload.get("account_number"),
                    payload["full_name"],
                    payload.get("age"),
                    payload.get("gender"),
                    payload.get("monthly_income"),
                    payload.get("account_type"),
                    payload.get("credit_score"),
                    payload.get("existing_loans", 0),
                    payload.get("balance"),
                    payload.get("city"),
                    payload.get("email"),
                    payload.get("phone"),
                    payload.get("employment_type"),
                    payload.get("risk_profile"),
                ),
            )
        if connection is not None:
            upsert(connection)
        else:
            with get_connection() as conn:
                upsert(conn)
                conn.commit()

    @staticmethod
    def get_recent_transactions(customer_id: str, limit: int = 5) -> list[dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM transactions WHERE customer_id = ? ORDER BY txn_date DESC LIMIT ?",
                (customer_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]


class LoanApplicationRepository:
    STATUSES = {"DRAFT", "PENDING", "SUBMITTED", "REVIEW", "APPROVED", "REJECTED"}
    TRANSITIONS = {
        "DRAFT": {"PENDING", "REJECTED"},
        "PENDING": {"SUBMITTED", "REVIEW", "REJECTED"},
        "SUBMITTED": {"REVIEW", "APPROVED", "REJECTED"},
        "REVIEW": {"APPROVED", "REJECTED"},
        "APPROVED": set(),
        "REJECTED": set(),
    }

    @staticmethod
    def save_application(payload: dict[str, Any]) -> None:
        status = str(payload.get("status", "PENDING")).upper()
        if status not in LoanApplicationRepository.STATUSES:
            raise ValueError(f"Unsupported application status: {status}")
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO loan_applications (
                    application_id, customer_id, loan_amount, interest_rate, tenure_months,
                    status, approval_probability, tracking_number, reason_codes, report, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["application_id"],
                    payload["customer_id"],
                    payload["loan_amount"],
                    payload["interest_rate"],
                    payload["tenure_months"],
                    status,
                    payload["approval_probability"],
                    payload["tracking_number"],
                    "; ".join(payload.get("reason_codes", [])),
                    payload.get("report"),
                    payload.get("submitted_at") or datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    @staticmethod
    def create_draft(payload: dict[str, Any]) -> dict[str, Any]:
        draft = {
            **payload,
            "status": "DRAFT",
            "submitted_at": payload.get("submitted_at") or datetime.now(timezone.utc).isoformat(),
        }
        LoanApplicationRepository.save_application(draft)
        return draft

    @staticmethod
    def update_status(customer_id: str, application_id: str, status: str) -> dict[str, Any]:
        next_status = status.strip().upper()
        if next_status not in LoanApplicationRepository.STATUSES:
            raise ValueError(f"Unsupported application status: {next_status}")
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM loan_applications WHERE customer_id = ? AND application_id = ?",
                (customer_id, application_id),
            ).fetchone()
            if row is None:
                raise ValueError("Application not found for the identified customer.")
            current_status = str(row["status"]).upper()
            if next_status != current_status and next_status not in LoanApplicationRepository.TRANSITIONS[current_status]:
                raise ValueError(f"Invalid application status transition: {current_status} -> {next_status}")
            conn.execute(
                "UPDATE loan_applications SET status = ? WHERE customer_id = ? AND application_id = ?",
                (next_status, customer_id, application_id),
            )
            conn.commit()
            updated = conn.execute(
                "SELECT * FROM loan_applications WHERE customer_id = ? AND application_id = ?",
                (customer_id, application_id),
            ).fetchone()
        return dict(updated)

    @staticmethod
    def get_application(customer_id: str, application_id: str | None = None) -> list[dict[str, Any]]:
        if application_id:
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM loan_applications WHERE customer_id = ? AND application_id = ? ORDER BY submitted_at DESC",
                    (customer_id, application_id),
                ).fetchall()
        else:
            return LoanApplicationRepository.list_applications(customer_id)
        return [dict(row) for row in rows]

    @staticmethod
    def list_applications(customer_id: str) -> list[dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM loan_applications WHERE customer_id = ? ORDER BY submitted_at DESC",
                (customer_id,),
            ).fetchall()
            return [dict(row) for row in rows]


class UserMemoryRepository:
    MEMORY_TYPES = {"profile", "preference"}

    @staticmethod
    def save_memory(customer_id: str, memory_type: str, memory_key: str, memory_value: str) -> None:
        normalized_type = memory_type.strip().lower()
        normalized_key = memory_key.strip()[:80]
        normalized_value = memory_value.strip()[:1000]
        if normalized_type not in UserMemoryRepository.MEMORY_TYPES:
            raise ValueError("Memory type must be profile or preference.")
        if not customer_id.strip() or not normalized_key or not normalized_value:
            raise ValueError("Customer, memory key, and value are required.")
        sensitive_terms = re.compile(r"password|passphrase|pin|otp|auth(?:entication)? code|token|secret|cvv|card number|account number", re.IGNORECASE)
        if sensitive_terms.search(normalized_key) or sensitive_terms.search(normalized_value):
            raise ValueError("Credentials, authentication codes, and payment or account numbers cannot be stored.")
        if len(re.sub(r"\D", "", normalized_value)) >= 9:
            raise ValueError("Long numeric identifiers cannot be stored in user memory.")

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_memory (customer_id, memory_type, memory_key, memory_value)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(customer_id, memory_type, memory_key) DO UPDATE SET
                    memory_value=excluded.memory_value,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (customer_id, normalized_type, normalized_key, normalized_value),
            )
            conn.commit()

    @staticmethod
    def get_context(customer_id: str, interaction_limit: int = 6, application_limit: int = 5) -> dict[str, Any]:
        with get_connection() as conn:
            memories = conn.execute(
                """
                SELECT memory_type, memory_key, memory_value, updated_at
                FROM user_memory WHERE customer_id = ? ORDER BY updated_at DESC
                """,
                (customer_id,),
            ).fetchall()
            interactions = conn.execute(
                """
                SELECT role, message, created_at FROM conversation_logs
                WHERE customer_id = ? ORDER BY id DESC LIMIT ?
                """,
                (customer_id, interaction_limit),
            ).fetchall()
            applications = conn.execute(
                """
                SELECT application_id, loan_amount, interest_rate, tenure_months,
                       status, approval_probability, submitted_at
                FROM loan_applications WHERE customer_id = ?
                ORDER BY submitted_at DESC LIMIT ?
                """,
                (customer_id, application_limit),
            ).fetchall()

        return {
            "profile_facts": {
                row["memory_key"]: row["memory_value"]
                for row in memories if row["memory_type"] == "profile"
            },
            "preferences": {
                row["memory_key"]: row["memory_value"]
                for row in memories if row["memory_type"] == "preference"
            },
            "recent_interactions": [
                {"role": row["role"], "message": row["message"][:500], "created_at": row["created_at"]}
                for row in reversed(interactions)
            ],
            "recent_applications": [dict(row) for row in applications],
        }


class AuditRepository:
    @staticmethod
    def log(event_type: str, message: str) -> None:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO audit_logs (event_type, message) VALUES (?, ?)",
                (event_type, message),
            )
            conn.commit()
