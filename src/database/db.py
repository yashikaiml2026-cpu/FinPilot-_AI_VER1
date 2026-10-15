from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config.settings import settings


DB_PATH = settings.db_path
DATA_DIR = settings.data_dir
DATA_DIR.mkdir(exist_ok=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                account_number TEXT UNIQUE,
                full_name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                monthly_income REAL,
                account_type TEXT,
                credit_score INTEGER,
                existing_loans INTEGER DEFAULT 0,
                balance REAL,
                city TEXT,
                email TEXT,
                phone TEXT,
                employment_type TEXT,
                risk_profile TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        customer_columns = {row[1] for row in conn.execute("PRAGMA table_info(customers)")}
        if "account_number" not in customer_columns:
            conn.execute("ALTER TABLE customers ADD COLUMN account_number TEXT")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_customers_account_number "
            "ON customers(account_number) WHERE account_number IS NOT NULL"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                txn_id TEXT PRIMARY KEY,
                customer_id TEXT,
                txn_date TEXT,
                type TEXT,
                amount REAL,
                merchant TEXT,
                balance_after REAL,
                FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS loan_applications (
                application_id TEXT PRIMARY KEY,
                customer_id TEXT,
                loan_amount REAL,
                interest_rate REAL,
                tenure_months INTEGER,
                status TEXT,
                approval_probability REAL,
                tracking_number TEXT,
                reason_codes TEXT,
                submitted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                report TEXT,
                FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                role TEXT,
                message TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                action_name TEXT,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                message TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memory (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                memory_type TEXT NOT NULL CHECK (memory_type IN ('profile', 'preference')),
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(customer_id, memory_type, memory_key)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_loan_applications_customer_id ON loan_applications(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_customer_id ON conversation_logs(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_memory_customer_id ON user_memory(customer_id)")
        conn.commit()


initialize_database()
