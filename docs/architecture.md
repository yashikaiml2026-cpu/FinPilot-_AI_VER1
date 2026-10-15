# FinPilot AI Architecture

## 1. Solution overview
FinPilot AI is a premium banking assistant built around a single intelligent agent that understands customer financing requests, validates information, applies a policy-aware decision workflow, and escalates or submits loan applications when appropriate.

## 2. Core principles
- Realistic banking decisioning over demo logic
- One intelligent agent with tool selection and chaining
- Policy retrieval through a RAG pipeline grounded in banking documents
- Secure SQLite persistence and audit-friendly logging
- Premium Streamlit experience with analytics dashboarding

## 3. Layered architecture
### Presentation layer
- Streamlit dashboard with premium dark theme and glassmorphism styling
- Customer profile, risk summary, application timeline, and analytics panels

### Application layer
- LangGraph state machine orchestrating the workflow
- Banking tools for EMI, eligibility, customer lookup, and loan submission
- Decision engine and compliance checks

### Data layer
- SQLite for customer and application records
- ChromaDB for policy retrieval embeddings
- Processed datasets under data/processed

### Platform layer
- Python environment, environment config, and logging
- Security checks for masked customer data and input validation

## 4. LangGraph workflow
1. Intent analysis
2. Information validation
3. Missing data check
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

## 5. RAG design
The knowledge base is composed of curated policy documents under knowledge_base. Each Markdown file is chunked and embedded into a ChromaDB vector store. Customer questions are transformed into vector queries that retrieve relevant policy context and provide citations.

## 6. Data lifecycle
The platform supports a Kaggle-compatible ingestion pipeline with a robust fallback dataset builder so the banking workflow remains operable in local environments where full Kaggle downloads are not available.
