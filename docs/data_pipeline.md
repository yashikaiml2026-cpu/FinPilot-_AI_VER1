# Data Source and Preprocessing Plan

## Source strategy
FinPilot is designed to consume authentic banking and risk datasets from public Kaggle sources, including loan approval, credit scoring, personal loan, and customer profile datasets. The pipeline is configured to support quick ingestion when Kaggle credentials are available, while retaining a realistic local fallback for development.

## Required data domains
- Personal loan data
- Bank customer financial profile data
- Transaction data
- Credit score and risk signals
- Loan eligibility and application history

## Preprocessing steps
- Normalize column names and types
- Validate required fields like customer ID, income, balance, and credit score
- Mask or exclude personally sensitive details from UI output
- Standardize dates and transaction categories
- Reconcile duplicate records and missing values
- Produce processed datasets under data/processed

## Validation
- Unique customer ID checks
- Numeric validation for income and score fields
- Boundary ranges for loan amounts and tenures
- Consistency between customer summarization and transaction records
