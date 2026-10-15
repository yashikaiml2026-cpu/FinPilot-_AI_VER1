# RAG Reference Manual for Banking Policy Retrieval

## Retrieval objective
The RAG layer retrieves curated banking policy passages that explain lending rules, KYC checks, service standards, and compliance expectations.

## Source categories
1. RBI publications
2. Personal loan eligibility guidance
3. KYC and onboarding rules
4. Banking operations standards
5. Customer service guidance
6. Risk and underwriting frameworks

## Expected use
Each policy answer should cite the relevant policy paragraph, explain its connection to the banking scenario, and highlight any decision implications.

## Example query mapping
- "loan affordability" -> loan_policies.md and eligibility_rules.md
- "KYC verification" -> kyc_guidelines.md
- "risk review" -> risk_assessment.md
- "customer communication" -> customer_service.md
- "compliance" -> banking_compliance.md
