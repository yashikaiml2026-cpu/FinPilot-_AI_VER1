# FinPilot AI

AI-Powered Banking Support & Loan Processing Assistant

## Project vision
FinPilot AI is a production-style banking assistant designed around a single intelligent agent that supports customer conversations, validates policy fit, calculates affordability, and prepares a lending application workflow.

## Core design principles
- Realistic banking logic for EMI estimation and eligibility review
- Policy-aware decisioning through a ChromaDB-backed retrieval pipeline
- SQLite persistence for customer, loan, and audit records
- Premium dark-mode Streamlit dashboard with banking-grade analytics
- Modular structure to support future extension with a live LLM provider

## Folder structure
- app.py: Streamlit app entry point
- src/agent: tool layer and LangGraph orchestration
- src/config: environment configuration
- src/data: Kaggle-ready preprocessing and normalization pipeline
- src/database: SQLite schema and repository layer
- src/knowledge: policy retrieval and ChromaDB embeddings
- src/ui: premium dashboard design and analytics rendering
- knowledge_base: policy documents and retrieval sources
- docs: architecture and data design documentation
- data: processed datasets and local SQLite database

## Environment configuration
Create a local .env from the example file and set only the required values:
- MODEL_PROVIDER
- MODEL_NAME
- API_KEY

All other parameters are derived automatically from the local project configuration.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the application
```bash
streamlit run app.py
```

## Dataset strategy
The project is built to support a Kaggle-first data ingestion workflow for personal loan, bank customer, transaction, credit, and lending datasets. A local synthetic but realistic fallback is generated when the Kaggle sources are unavailable, and the pipeline documentation outlines the validation rules and expected schema.

## RAG strategy
The knowledge base includes policy summaries for loan affordability, KYC, compliance, customer service, risk assessment, and eligibility norms. These are chunked and embedded into ChromaDB for grounded retrieval with citations.

## Workflow summary
1. Intent analysis
2. Validation of information
3. Missing data checks
4. Customer lookup
5. EMI calculation
6. Eligibility analysis
7. Policy retrieval
8. Compliance validation
9. Decision engine
10. Customer confirmation
11. Application submission
12. Report generation
13. Audit logging

## Project status
This repository includes the application structure, the SQLite layer, the policy knowledge base, the loan tool logic, and the Streamlit UI scaffolding required for a production-grade banking workflow prototype.
