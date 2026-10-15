from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config.settings import settings

DATA_DIR = settings.data_dir
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DATASET_SOURCES = [
    "https://www.kaggle.com/datasets/itsmesunil/bank-loan-status-dataset",
    "https://www.kaggle.com/datasets/rohitrox/loan-eligibility-dataset",
    "https://www.kaggle.com/datasets/paololazzari/bank-customers",
    "https://www.kaggle.com/datasets/gauravduttakiit/credit-score-classification",
    "https://www.kaggle.com/datasets/ashishkumarjayswal/bank-loan-approval-data",
]


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def generate_processed_datasets() -> dict[str, str]:
    ensure_directories()
    random.seed(42)

    customers = []
    for idx in range(1, 31):
        income = random.randint(60000, 260000)
        credit = random.randint(550, 860)
        balance = random.randint(40000, 260000)
        customers.append(
            {
                "customer_id": f"CUST-{idx:04d}",
                "account_number": f"700000000000{idx:04d}",
                "full_name": f"Customer {idx}",
                "age": random.randint(26, 62),
                "gender": random.choice(["Male", "Female", "Other"]),
                "monthly_income": income,
                "account_type": random.choice(["Savings", "Current", "Salary"]),
                "credit_score": credit,
                "existing_loans": random.randint(0, 3),
                "balance": balance,
                "city": random.choice(["Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune", "Chennai"]),
                "email": f"customer{idx}@bank.example",
                "phone": f"+91{random.randint(7000000000, 9999999999)}",
                "employment_type": random.choice(["Salaried", "Self-Employed", "Professional"]),
                "risk_profile": "LOW" if credit >= 700 else "MEDIUM" if credit >= 620 else "HIGH",
            }
        )

    dfs_customers = pd.DataFrame(customers)
    customer_file = PROCESSED_DIR / "customers.csv"
    dfs_customers.to_csv(customer_file, index=False)

    transactions: list[dict] = []
    txn_id = 1
    for customer in customers:
        balance = float(customer["balance"])
        for month in range(1, 7):
            for _ in range(6):
                txn_date = (datetime(2024, month, 1) + timedelta(days=random.randint(1, 24))).strftime("%Y-%m-%d")
                tx_type = random.choice(["CREDIT", "DEBIT", "UPI", "EMI", "SALARY", "RENT"])
                amount = random.randint(2500, 50000)
                balance = balance + amount if tx_type in {"SALARY", "CREDIT"} else balance - amount
                transactions.append(
                    {
                        "txn_id": f"TXN-{txn_id:06d}",
                        "customer_id": customer["customer_id"],
                        "txn_date": txn_date,
                        "type": tx_type,
                        "amount": amount,
                        "merchant": random.choice(["Amazon", "HDFC", "Airtel", "Office", "Rent", "Grocer", "Travel"]),
                        "balance_after": round(balance, 2),
                    }
                )
                txn_id += 1

    transactions_df = pd.DataFrame(transactions)
    tx_file = PROCESSED_DIR / "transactions.csv"
    transactions_df.to_csv(tx_file, index=False)

    app_file = PROCESSED_DIR / "loan_applications.csv"
    loan_rows = [
        {
            "application_id": "APP-0001",
            "customer_id": customers[0]["customer_id"],
            "loan_amount": 800000,
            "interest_rate": 9.5,
            "tenure_months": 120,
            "status": "PENDING",
            "approval_probability": 82.4,
            "tracking_number": "TRK-0001",
            "reason_codes": "Strong income profile; stable savings behavior",
        }
    ]
    pd.DataFrame(loan_rows).to_csv(app_file, index=False)

    dataset_doc = DATA_DIR / "dataset_documentation.md"
    dataset_doc.write_text(
        """# FinPilot dataset documentation

## Data sources

The platform is designed to support Kaggle banking datasets such as:

- Bank loan status dataset
- Loan eligibility dataset
- Bank customers profile data
- Credit score classification data
- Bank loan approval data

See the source list in the project documentation and the pipeline configuration for the official links.

## Processed outputs

- customers.csv: anonymized customer profiles with income, score, loans, and city summary.
- transactions.csv: recent transaction history with debit/credit patterns.
- loan_applications.csv: application metadata for processing and tracking.

## Validation rules

- Customer IDs are unique and non-null.
- Monthly income and credit scores are numeric.
- Transactions are grouped by customer and must have a valid transaction date.
- Loan application records are associated with a valid customer profile.
""",
        encoding="utf-8",
    )

    return {
        "customers": str(customer_file),
        "transactions": str(tx_file),
        "loan_applications": str(app_file),
        "documentation": str(dataset_doc),
    }


def ingest_raw_datasets() -> dict[str, str]:
    ensure_directories()
    processed = generate_processed_datasets()
    from src.database.repositories import CustomerRepository

    customers = pd.read_csv(processed["customers"])
    for customer in customers.to_dict(orient="records"):
        CustomerRepository.upsert_customer(customer)
    return processed
