# Buy-or-Wait Financial Agent

This is a deterministic, rule-based financial agent that evaluates user requests against their financial profile and 90-day cash flow forecast. It is designed to run locally, rapidly, and without LLM API quota bottlenecks, fulfilling the HackerRank Orchestrate "Buy or Wait?" challenge requirements.

## Overview

The agent determines the safest way for a user to afford a requested expense by simulating their running balance over a 90-day horizon. It considers:
- Current available balance & minimum balance to keep.
- Upcoming settled, pending, and scheduled cashflows (applying real-time exchange rates).
- User constraints (willingness to use installments, partial payments, wait, or cut flexible expenses).
- Context from messages (e.g., event cancellations or amendments).

For each request, the system computes:
- `amount_safe_to_pay`
- `affordability_status`
- `recommended_payment_method`
- `payment_plan`
- `earliest_date_for_full_payment`
- `spending_changes_needed`
- `decision_explanation`

## Features

- **100% Deterministic Engine**: No flaky LLM decisions for financial math. Runs all 250 requests in ~25 seconds.
- **LLM Enrichment (Optional)**: Can use the Gemini API (e.g. `gemini-3.6-flash`) to generate natural language explanations instead of relying on templates.

## Requirements

- Python >= 3.13
- `uv` package manager (recommended)

## Setup

1. Install dependencies via `uv`:
   ```bash
   uv sync
   ```

2. (Optional) If using LLM enrichment, create a `.env` file in the `code/` directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

Run the agent from within the `code/` directory:

```bash
uv run python main.py
```

By default, the agent reads from `../dataset/` and writes its predictions to `../output.csv` (the repository root, per hackathon specifications), as well as a token usage report to `evaluation/usage_report.md`.

### Optional Flags

- `--test-run N`: Only process the first `N` requests (useful for quick testing).
- `--llm`: Enable LLM-generated decision explanations via the Gemini API.

```bash
uv run python main.py --test-run 10 --llm
```

## Project Structure

- `main.py`: The complete agent implementation, comprising data loading, cashflow simulation, and decision-tree logic.
- `pyproject.toml` / `uv.lock`: Project dependencies.
- `evaluation/usage_report.md`: Generated report on token usage and model behavior.
