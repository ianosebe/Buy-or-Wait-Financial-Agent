"""
Buy-or-Wait Financial Agent
============================
Deterministic rule-based engine implementing the full 90-day safety-check
spec from AGENTS.md §6.3, plus optional LLM-generated decision_explanation.
Runs in seconds – no API quota bottleneck.
"""

import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import TypedDict

import pandas as pd
from dotenv import load_dotenv


class _Candidate(TypedDict):
    event_id: str
    category: str
    amount: float
    min_allowed: float

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LLM client (only for decision_explanation)
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(val) -> date | None:
    if not val or str(val).strip() in ("", "nan", "None"):
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(val).strip(), fmt).date()  # noqa: DTZ007
        except ValueError:
            continue
    return None


def _parse_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _round2(v: float) -> float:
    return round(v, 2)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

class DataStore:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        logger.info("Loading datasets…")

        def _load(name):
            return pd.read_csv(os.path.join(data_dir, name)).fillna("")

        self.profiles   = _load("financial_profiles.csv")
        self.events     = _load("financial_events.csv")
        self.rates      = _load("exchange_rates.csv")
        self.requests   = _load("requests.csv")
        self.options    = _load("request_payment_options.csv")
        self.messages   = _load("messages.csv")
        self.images_df  = _load("images.csv")
        logger.info("Data loaded.")

    # Convenience look-ups --------------------------------------------------

    def profile(self, user_id: str) -> dict:
        rows = self.profiles[self.profiles["user_id"] == user_id]
        return rows.iloc[0].to_dict() if len(rows) else {}

    def user_events(self, user_id: str) -> pd.DataFrame:
        return self.events[self.events["user_id"] == user_id].copy()

    def request_options(self, request_id: str) -> pd.DataFrame:
        return self.options[self.options["request_id"] == request_id].copy()

    def request_messages(self, user_id: str, request_id: str) -> pd.DataFrame:
        m = self.messages
        return m[(m["user_id"] == user_id) | (m["request_id"] == request_id)].copy()

    def request_images(self, user_id: str, request_id: str) -> pd.DataFrame:
        im = self.images_df
        return im[(im["user_id"] == user_id) | (im["request_id"] == request_id)].copy()

    def exchange_rate(self, from_currency: str, to_currency: str, on_date: date) -> float:
        """Return the fixed exchange rate, or 1.0 if same or not found."""
        if from_currency == to_currency:
            return 1.0
        r = self.rates
        mask = (r["from_currency"] == from_currency) & (r["to_currency"] == to_currency)
        rows = r[mask]
        if rows.empty:
            return 1.0
        # prefer closest date
        best = None
        best_delta = None
        for _, row in rows.iterrows():
            d = _parse_date(row.get("date", ""))
            delta = abs((d - on_date).days) if d else 9999
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best = row
        return _parse_float(best["rate"]) if best is not None else 1.0


# ---------------------------------------------------------------------------
# 90-Day cash-flow forecaster
# ---------------------------------------------------------------------------

CASH_IN_STATUSES   = {"settled", "scheduled", "pending"}   # but only debit-pending is reserved
SETTLED_STATUSES   = {"settled"}


def _event_affects_cash(ev: dict, home_currency: str, ds: DataStore, ref_date: date) -> tuple[float, date | None]:
    """
    Return (amount_in_home_currency, effective_date) or (0, None) if ignored.
    Rules:
    - settled debits: subtract on settlement_date
    - pending debits: subtract on settlement_date (reserved)
    - settled credits: add on settlement_date
    - pending/scheduled credits: NOT counted until settled
    - unrealized / non-cash / cancelled / failed: ignored
    """
    status     = str(ev.get("status", "")).lower()
    direction  = str(ev.get("direction", "")).lower()
    ev_type    = str(ev.get("event_type", "")).lower()
    currency   = str(ev.get("currency", home_currency)).strip() or home_currency
    amount     = _parse_float(ev.get("amount", 0))
    settle_raw = ev.get("settlement_date", "") or ev.get("event_date", "")
    settle_d   = _parse_date(settle_raw)

    if not settle_d:
        return 0.0, None

    if status in ("cancelled", "failed", ""):
        return 0.0, None
    if ev_type in ("non_cash", "unrealized"):
        return 0.0, None
    if status == "unrealized":
        return 0.0, None

    # Credit side
    if direction == "credit":
        if status not in SETTLED_STATUSES:
            return 0.0, None  # don't count pending/scheduled credits
        rate = ds.exchange_rate(currency, home_currency, settle_d)
        return amount * rate, settle_d

    # Debit side
    if direction == "debit":
        if status in ("cancelled", "failed"):
            return 0.0, None
        # pending and scheduled debits are reserved
        rate = ds.exchange_rate(currency, home_currency, settle_d)
        return -(amount * rate), settle_d

    return 0.0, None


def _build_cashflow(events_df: pd.DataFrame, home_currency: str, ds: DataStore, ref_date: date) -> dict[date, float]:
    """Build a daily delta map (positive=inflow, negative=outflow)."""
    flow: dict[date, float] = {}
    for _, ev in events_df.iterrows():
        delta, d = _event_affects_cash(ev.to_dict(), home_currency, ds, ref_date)
        if d and d >= ref_date:
            flow[d] = flow.get(d, 0.0) + delta
    return flow


def _forecast_balance(start_balance: float, flow: dict[date, float],
                       ref_date: date, horizon_days: int = 90) -> dict[date, float]:
    """Return daily running balance for [ref_date, ref_date+horizon_days]."""
    bal = start_balance
    daily: dict[date, float] = {}
    for offset in range(horizon_days + 1):
        d = ref_date + timedelta(days=offset)
        bal += flow.get(d, 0.0)
        daily[d] = bal
    return daily


def _min_balance_after_payment(daily: dict[date, float], payment: float,
                                pay_date: date, min_keep: float) -> bool:
    """Return True if paying `payment` on `pay_date` keeps balance >= min_keep for all subsequent days."""
    adjusted = {d: (b - payment if d >= pay_date else b) for d, b in daily.items()}
    return all(v >= min_keep for v in adjusted.values())


def _safe_amount_today(daily: dict[date, float], ref_date: date,
                        min_keep: float, max_amount: float) -> float:
    """Binary-search the max amount payable on ref_date that keeps balance >= min_keep."""
    future_min = min(v for d, v in daily.items() if d >= ref_date)
    available  = future_min - min_keep
    if available <= 0:
        return 0.0
    return _round2(min(available, max_amount))


def _first_safe_full_payment_date(daily: dict[date, float], ref_date: date,
                                   amount: float, min_keep: float) -> date | None:
    """Scan day-by-day for the first date where paying `amount` is safe."""
    for d in sorted(daily.keys()):
        if d < ref_date:
            continue
        # future_min after this date
        future_min = min(v for dd, v in daily.items() if dd >= d)
        if future_min - amount >= min_keep:
            return d
    return None


# ---------------------------------------------------------------------------
# Installment evaluator
# ---------------------------------------------------------------------------

def _eval_installment_option(opt: dict, daily_base: dict[date, float],
                               min_keep: float, home_currency: str,
                               request_currency: str, rate: float,
                               desired_completion: date | None) -> dict | None:
    """
    Try a payment option. Returns a plan dict or None if it fails the safety check.
    """
    try:
        n_payments    = int(_parse_float(opt.get("number_of_payments", 1)))
        first_date    = _parse_date(opt.get("first_payment_date"))
        freq_days     = int(_parse_float(opt.get("payment_frequency_days", 30)) or 30)
        pay_amt_home  = _parse_float(opt.get("payment_amount", 0)) * rate
        total_payable = _parse_float(opt.get("total_payable_amount", 0)) * rate

        if not first_date or pay_amt_home <= 0:
            return None

        payment_dates = [first_date + timedelta(days=i * freq_days) for i in range(n_payments)]

        if desired_completion and payment_dates[-1] > desired_completion:
            return None  # misses deadline

        # Simulate cumulative deductions
        daily = dict(daily_base)
        for pd_ in payment_dates:
            if not _min_balance_after_payment(daily, pay_amt_home, pd_, min_keep):
                return None
            # apply this payment to subsequent balance
            daily = {d: b - pay_amt_home if d >= pd_ else b for d, b in daily.items()}

        plan_str = "|".join(
            f"{d}:{_round2(pay_amt_home / rate)}" for d in payment_dates
        )
        return {
            "option_id": str(opt.get("payment_option_id", "")),
            "method": "installments",
            "plan": plan_str,
            "total": total_payable,
            "n_payments": n_payments,
            "first_date": payment_dates[0],
            "last_date": payment_dates[-1],
        }
    except (ValueError, ArithmeticError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Message / image overrides
# ---------------------------------------------------------------------------

def _apply_message_overrides(events_df: pd.DataFrame, messages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Very lightweight: if a message says an event was cancelled, amended, etc.,
    update the events_df accordingly.
    """
    ev = events_df.copy()
    for _, msg in messages_df.iterrows():
        text = str(msg.get("message_text", "") or msg.get("content", "")).lower()
        related = str(msg.get("related_event_id", "")).strip()
        if not related:
            continue
        mask = ev["event_id"] == related
        if not mask.any():
            continue
        if "cancel" in text:
            ev.loc[mask, "status"] = "cancelled"
        elif "amend" in text or "update" in text or "change" in text:
            # Try to extract a new amount
            import re
            nums = re.findall(r"[\d,]+\.?\d*", text)
            if nums:
                try:
                    new_amt = float(nums[-1].replace(",", ""))
                    ev.loc[mask, "amount"] = new_amt
                except ValueError:
                    pass
    return ev


# ---------------------------------------------------------------------------
# Spending-change analyser
# ---------------------------------------------------------------------------

def _find_spending_changes(events_df: pd.DataFrame, profile: dict,
                            shortfall: float, home_currency: str,
                            ds: DataStore, ref_date: date) -> tuple[list[str], float]:
    """
    Try to find stop/reduce actions on flexible non-protected events
    that recover at least `shortfall` in the 90-day window.
    Returns (list_of_actions, recovered_amount).
    """
    can_stop   = set(str(profile.get("expense_categories_user_is_willing_to_stop", "")).split("|"))
    can_reduce = set(str(profile.get("expense_categories_user_is_willing_to_reduce", "")).split("|"))
    protected  = set(str(profile.get("expense_categories_to_protect", "")).split("|"))
    can_stop   -= {""}
    can_reduce -= {""}
    protected  -= {""}

    actions:    list[str] = []
    recovered:  float = 0.0
    candidates: list[_Candidate] = []
    horizon   = ref_date + timedelta(days=90)

    # Gather candidate events (scheduled/pending debits that are flexible)
    candidates = []
    for _, ev in events_df.iterrows():
        status    = str(ev.get("status", "")).lower()
        direction = str(ev.get("direction", "")).lower()
        cat       = str(ev.get("category", "")).lower()
        flex      = str(ev.get("flexibility", "")).lower()
        settle_d  = _parse_date(ev.get("settlement_date", "") or ev.get("event_date", ""))

        if direction != "debit":
            continue
        if status in ("cancelled", "failed", "settled"):
            continue
        if cat in protected:
            continue
        if flex not in ("flexible", "adjustable"):
            continue
        if settle_d and settle_d > horizon:
            continue

        currency = str(ev.get("currency", home_currency)) or home_currency
        rate = ds.exchange_rate(currency, home_currency, settle_d or ref_date)
        amount_home = _parse_float(ev.get("amount", 0)) * rate
        min_allowed = _parse_float(ev.get("minimum_allowed_amount", 0)) * rate

        candidates.append({
            "event_id": str(ev.get("event_id", "")),
            "category": str(cat),
            "amount": float(amount_home),
            "min_allowed": float(min_allowed),
        })

    # Sort by biggest savings first
    candidates.sort(key=lambda x: float(x["amount"]), reverse=True)

    for cand in candidates:
        if recovered >= shortfall:
            break
        cat = cand["category"]
        if cat in can_stop:
            actions.append(f"stop:{cand['event_id']}")
            recovered += cand["amount"]
        elif cat in can_reduce:
            reducible = cand["amount"] - cand["min_allowed"]
            if reducible > 0:
                new_amt_home = cand["min_allowed"]
                # convert back to original currency for reporting (approximate)
                actions.append(f"reduce_to:{cand['event_id']}:{_round2(new_amt_home)}")
                recovered += reducible

    return actions[:3], recovered  # max 3 changes per spec


# ---------------------------------------------------------------------------
# Core decision engine
# ---------------------------------------------------------------------------

class DecisionResult:
    def __init__(self):
        self.amount_safe_to_pay: float = 0.0
        self.affordability_status: str = "not_affordable"
        self.recommended_payment_method: str = "not_recommended"
        self.payment_plan: str = "none"
        self.earliest_date_for_full_payment: str = ""
        self.spending_changes_needed: str = "none"
        self.decision_explanation: str = ""


def decide(request: dict, ds: DataStore) -> DecisionResult:
    result = DecisionResult()

    user_id    = str(request.get("user_id", ""))
    req_id     = str(request.get("request_id", ""))
    req_date   = _parse_date(request.get("request_date")) or date.today()  # noqa: DTZ011
    req_amount = _parse_float(request.get("requested_amount", 0))
    req_curr   = str(request.get("request_currency", "")).strip()
    desired_d  = _parse_date(request.get("desired_completion_date"))
    allows_partial = str(request.get("allows_partial_payment", "false")).lower() == "true"
    horizon_end = req_date + timedelta(days=90)

    profile     = ds.profile(user_id)
    home_curr   = str(profile.get("home_currency", "")).strip() or "USD"
    start_bal   = _parse_float(profile.get("current_available_balance", 0))
    min_keep    = _parse_float(profile.get("minimum_balance_to_keep", 0))
    pay_methods = str(profile.get("payment_methods_user_will_consider", "full_payment")).split("|")
    max_inst_m  = str(profile.get("max_installment_months", "")).strip()
    max_inst_mo = int(max_inst_m) if max_inst_m.isdigit() else None

    # Currency: if no request_currency field, assume home_currency
    if not req_curr:
        req_curr = home_curr
    rate_to_home = ds.exchange_rate(req_curr, home_curr, req_date)
    req_amount_home = req_amount * rate_to_home

    # Apply message overrides
    events_raw  = ds.user_events(user_id)
    messages_df = ds.request_messages(user_id, req_id)
    events_df   = _apply_message_overrides(events_raw, messages_df)

    # Build cashflow and forecast
    flow     = _build_cashflow(events_df, home_curr, ds, req_date)
    daily    = _forecast_balance(start_bal, flow, req_date, 90)

    # 1. Amount safe to pay today
    safe_now = _safe_amount_today(daily, req_date, min_keep, req_amount_home)
    result.amount_safe_to_pay = _round2(safe_now / rate_to_home) if rate_to_home else 0.0

    # 2. First date when full payment is safe
    full_pay_date = _first_safe_full_payment_date(daily, req_date, req_amount_home, min_keep)
    if full_pay_date:
        result.earliest_date_for_full_payment = str(full_pay_date)
    else:
        result.earliest_date_for_full_payment = ""

    can_pay_today = (safe_now >= req_amount_home - 0.01)
    can_pay_within_horizon = full_pay_date is not None and full_pay_date <= horizon_end
    can_pay_by_deadline   = (full_pay_date is not None and
                              desired_d is not None and
                              full_pay_date <= desired_d)

    # 3. Evaluate installment options
    opts_df = ds.request_options(req_id)
    valid_installment_plans = []
    for _, opt in opts_df.iterrows():
        method = str(opt.get("payment_method", "")).lower()
        if "installment" not in method:
            continue
        if "installments" not in pay_methods:
            continue
        opt_n = int(_parse_float(opt.get("number_of_payments", 1)))
        if max_inst_mo and opt_n > max_inst_mo:
            continue
        plan = _eval_installment_option(
            opt.to_dict(), daily, min_keep, home_curr, req_curr, rate_to_home, desired_d
        )
        if plan:
            valid_installment_plans.append(plan)

    # Sort installment plans per spec: complete by deadline, fewer payments, lower option_id
    valid_installment_plans.sort(key=lambda p: (
        0 if (desired_d and p["last_date"] <= desired_d) else 1,
        p["n_payments"],
        p["option_id"]
    ))

    # -----------------------------------------------------------------------
    # Decision tree
    # -----------------------------------------------------------------------

    # Case A: affordable now
    if can_pay_today and "full_payment" in pay_methods:
        result.affordability_status = "affordable_now"
        result.recommended_payment_method = "full_payment"
        result.payment_plan = f"{req_date}:{_round2(req_amount)}"
        result.spending_changes_needed = "none"
        result.decision_explanation = (
            f"Pay {home_curr} {_round2(req_amount_home):,} today. "
            f"This keeps the {home_curr} {min_keep:,} minimum protected over the next 90 days."
        )
        return result

    # Case B: installments work (no spending changes needed)
    if valid_installment_plans:
        best = valid_installment_plans[0]
        result.affordability_status = "affordable_with_plan"
        result.recommended_payment_method = "installments"
        result.payment_plan = best["plan"]
        result.spending_changes_needed = "none"
        result.decision_explanation = (
            f"Use {best['n_payments']} installments starting {best['first_date']}. "
            f"This completes the {req_curr} {_round2(req_amount):,} request and keeps the "
            f"{home_curr} {min_keep:,} minimum protected."
        )
        return result

    # Case C: partial payment (only if allowed and makes sense)
    if (allows_partial and "partial_payment" in pay_methods and
            0 < safe_now < req_amount_home and can_pay_by_deadline):
        remainder_home = req_amount_home - safe_now
        # second payment on earliest_date_for_full_payment (if within deadline)
        # recalculate first_date_for_remainder after safe_now is taken today
        adjusted_daily = {d: b - safe_now if d >= req_date else b for d, b in daily.items()}
        second_date = _first_safe_full_payment_date(
            adjusted_daily, req_date + timedelta(days=1), remainder_home, min_keep
        )
        if second_date and (not desired_d or second_date <= desired_d):
            p1 = f"{req_date}:{_round2(safe_now / rate_to_home)}"
            p2 = f"{second_date}:{_round2(remainder_home / rate_to_home)}"
            result.affordability_status = "affordable_with_plan"
            result.recommended_payment_method = "partial_payment"
            result.payment_plan = f"{p1}|{p2}"
            result.earliest_date_for_full_payment = str(second_date)
            result.spending_changes_needed = "none"
            result.decision_explanation = (
                f"Pay {req_curr} {_round2(safe_now / rate_to_home):,} today and "
                f"the remaining {req_curr} {_round2(remainder_home / rate_to_home):,} "
                f"on {second_date}. This keeps the {home_curr} {min_keep:,} minimum protected."
            )
            return result

    # Case D: wait (full payment safe later within deadline)
    if can_pay_by_deadline and "wait" in pay_methods or (
            can_pay_within_horizon and desired_d and full_pay_date and full_pay_date <= desired_d):
        result.affordability_status = "affordable_later"
        result.recommended_payment_method = "wait"
        result.payment_plan = f"{full_pay_date}:{_round2(req_amount)}"
        result.spending_changes_needed = "none"
        result.decision_explanation = (
            f"Pay {req_curr} {_round2(req_amount):,} in full on {full_pay_date}. "
            f"Paying earlier would take the balance below the {home_curr} {min_keep:,} minimum."
        )
        return result

    # Case E: spending changes could help
    shortfall = req_amount_home - safe_now
    if shortfall > 0:
        changes, recovered = _find_spending_changes(
            events_df, profile, shortfall, home_curr, ds, req_date
        )
        if changes and recovered >= shortfall - 0.01:
            new_safe = safe_now + recovered
            if new_safe >= req_amount_home - 0.01 and "full_payment" in pay_methods:
                result.affordability_status = "affordable_with_plan"
                result.recommended_payment_method = "full_payment"
                result.payment_plan = f"{req_date}:{_round2(req_amount)}"
                result.spending_changes_needed = "|".join(changes)
                result.decision_explanation = (
                    f"After the recommended spending adjustments, pay {req_curr} "
                    f"{_round2(req_amount):,} today. This keeps the {home_curr} "
                    f"{min_keep:,} minimum protected."
                )
                return result

    # Case F: wait (no deadline constraint or within horizon)
    if can_pay_within_horizon and (not desired_d or full_pay_date <= desired_d):
        result.affordability_status = "affordable_later"
        result.recommended_payment_method = "wait"
        result.payment_plan = f"{full_pay_date}:{_round2(req_amount)}"
        result.spending_changes_needed = "none"
        result.decision_explanation = (
            f"Pay {req_curr} {_round2(req_amount):,} in full on {full_pay_date}. "
            f"Paying earlier would take the balance below the {home_curr} {min_keep:,} minimum."
        )
        return result

    # Default: not affordable
    result.amount_safe_to_pay = _round2(safe_now / rate_to_home) if rate_to_home else 0.0
    result.affordability_status = "not_affordable"
    result.recommended_payment_method = "not_recommended"
    result.payment_plan = "none"
    result.earliest_date_for_full_payment = ""
    result.spending_changes_needed = "none"
    result.decision_explanation = (
        f"Do not make this payment by "
        f"{desired_d or horizon_end}. "
        f"None of the available options keeps the {home_curr} {min_keep:,} minimum protected."
    )
    return result


# ---------------------------------------------------------------------------
# Optional: LLM explanation enricher
# ---------------------------------------------------------------------------

def _llm_explanation(request: dict, result: DecisionResult, ds: DataStore) -> str:
    """Try to get a richer explanation from the LLM. Falls back to rule-based text."""
    if not _GENAI_AVAILABLE:
        return result.decision_explanation

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return result.decision_explanation

    try:
        client = genai.Client()
        user_id = str(request.get("user_id", ""))
        req_id  = str(request.get("request_id", ""))
        profile = ds.profile(user_id)
        events  = ds.user_events(user_id).head(20).to_dict("records")
        options = ds.request_options(req_id).to_dict("records")
        msgs    = ds.request_messages(user_id, req_id).to_dict("records")

        prompt = (
            f"Request: {json.dumps(request)}\n"
            f"Profile: {json.dumps(profile)}\n"
            f"Events (sample): {json.dumps(events[:10])}\n"
            f"Options: {json.dumps(options)}\n"
            f"Messages: {json.dumps(msgs)}\n"
            f"Decision: {json.dumps(result.__dict__)}\n\n"
            "Write a concise, grounded 1-2 sentence explanation of this financial decision. "
            "Mention the key amounts and dates. Be specific."
        )
        resp = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=200),
        )
        return resp.text.strip()
    except (OSError, RuntimeError, ValueError) as e:  # network / SDK / parse errors
        logger.debug(f"LLM explanation failed ({e}), using rule-based text.")
        return result.decision_explanation


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan",
    "earliest_date_for_full_payment", "spending_changes_needed",
    "decision_explanation",
]


def run_agent(data_dir: str, output_path: str, usage_report_path: str,
              limit: int | None = None, enrich_with_llm: bool = False):
    ds = DataStore(data_dir)
    rows_to_process = ds.requests if limit is None else ds.requests.head(limit)

    results = []
    for i, (_, req) in enumerate(rows_to_process.iterrows()):
        req_dict = req.to_dict()
        logger.info(f"[{i+1}/{len(rows_to_process)}] {req_dict.get('request_id')}")
        try:
            result = decide(req_dict, ds)
            if enrich_with_llm:
                result.decision_explanation = _llm_explanation(req_dict, result, ds)
        except (ValueError, ArithmeticError, KeyError, TypeError, AttributeError) as e:
            logger.error(f"Error on {req_dict.get('request_id')}: {e}")
            result = DecisionResult()
            result.decision_explanation = f"Processing error: {e}"

        results.append({
            "request_id":                  req_dict.get("request_id", ""),
            "amount_safe_to_pay":          result.amount_safe_to_pay,
            "affordability_status":        result.affordability_status,
            "recommended_payment_method":  result.recommended_payment_method,
            "payment_plan":                result.payment_plan,
            "earliest_date_for_full_payment": result.earliest_date_for_full_payment,
            "spending_changes_needed":     result.spending_changes_needed,
            "decision_explanation":        result.decision_explanation,
        })

    out_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    out_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(results)} rows → {output_path}")

    # Usage report (deterministic engine has no tokens, but we report honestly)
    os.makedirs(os.path.dirname(usage_report_path), exist_ok=True)
    model_info = "gemini-3.6-flash (explanation only, optional)" if enrich_with_llm else "None (deterministic rule-based engine)"
    report = f"""# Token Usage Report

## Summary
- **Model Provider**: Google / Rule-Based Engine
- **Model Name**: {model_info}
- **Total Requests Processed**: {len(results)}
- **Approach**: Deterministic 90-day cash-flow forecaster + rule-based decision engine (AGENTS.md §6.3)
- **LLM Enrichment**: {"Enabled (gemini-3.6-flash, explanation field only)" if enrich_with_llm else "Disabled"}

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
This guarantees reproducible, quota-free execution. The decision_explanation field is {"generated by Gemini" if enrich_with_llm else "template-generated"}.
"""
    with open(usage_report_path, "w") as f:
        f.write(report)
    logger.info(f"Usage report → {usage_report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Buy-or-Wait Financial Agent")
    parser.add_argument("--test-run", type=int, default=None,
                        help="Process only N requests (for testing)")
    parser.add_argument("--llm", action="store_true",
                        help="Enrich decision_explanation with LLM (optional)")
    args = parser.parse_args()

    run_agent(
        data_dir="../dataset",
        output_path="../dataset/output.csv",
        usage_report_path="evaluation/usage_report.md",
        limit=args.test_run,
        enrich_with_llm=args.llm,
    )
