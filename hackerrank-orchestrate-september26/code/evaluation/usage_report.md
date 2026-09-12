# Token Usage Report

## Summary
- **Model Provider**: Google / Rule-Based Engine
- **Model Name**: None (deterministic rule-based engine)
- **Total Requests Processed**: 250
- **Approach**: Deterministic 90-day cash-flow forecaster + rule-based decision engine (AGENTS.md §6.3)
- **LLM Enrichment**: Disabled

## Token Counts
- **Input Tokens**: 0 (deterministic engine)
- **Output Tokens**: 0 (deterministic engine)
- **Average Tokens per Request**: 0

## Estimated Cost
- **Total Estimated Cost**: $0.00
- **Average Cost per Request**: $0.00

## Notes
The core financial decisions (amount_safe_to_pay, affordability_status, recommended_payment_method,
payment_plan, earliest_date_for_full_payment, spending_changes_needed) are produced by a
deterministic Python engine implementing the exact rules in AGENTS.md §6.3, without any LLM calls.
This guarantees reproducible, quota-free execution. The decision_explanation field is template-generated.
