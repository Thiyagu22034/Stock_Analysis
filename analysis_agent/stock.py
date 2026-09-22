import json
import re
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD

# AutoGen v0.4 (AgentChat)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# =========================
# PAGE
# =========================
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

st.set_page_config(page_title="Stock Analyst Agent", page_icon="📈", layout="wide")
st.title("📈 Stock Analyst Agent (Real Data + Gemini)")
st.caption("Enter a company name (e.g., Apple, Microsoft, Reliance). I’ll find the ticker, fetch data, and Gemini will recommend BUY / HOLD / SELL.")

with st.sidebar:
    st.header("Gemini")
    gemini_api_key = st.text_input(
        "API key",
        type="password",
        placeholder="Paste your Gemini API key",
        help="From Google AI Studio. Used only in this session. It is not written to a file.",
    )
    st.caption(f"Model: {GEMINI_MODEL}")

if not gemini_api_key.strip():
    st.info("Enter your Gemini API key in the sidebar before analyzing.", icon="🔑")


# =========================
# Session State
# =========================
def _init_state():
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.loop)

    if "model_client" not in st.session_state:
        st.session_state.model_client: Optional[OpenAIChatCompletionClient] = None

    if "agent" not in st.session_state:
        st.session_state.agent: Optional[AssistantAgent] = None

    if "team" not in st.session_state:
        st.session_state.team: Optional[RoundRobinGroupChat] = None

    if "latest_json" not in st.session_state:
        st.session_state.latest_json: Optional[Dict[str, Any]] = None

    if "matches" not in st.session_state:
        st.session_state.matches = []

    if "searched_query" not in st.session_state:
        st.session_state.searched_query = ""

    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol = None

    if "analyze_now" not in st.session_state:
        st.session_state.analyze_now = False

_init_state()


# =========================
# Data / Indicator Helpers
# =========================
# Trading-day windows for the price chart. Fetch covers the longest window.
LOOKBACK_BARS = {
    "3mo": 63,
    "6mo": 126,
    "1y": 252,
    "2y": 504,
    "3y": 756,
}


def fetch_stock_data(symbol: str) -> Dict[str, Any]:
    """
    Pull price history (3y daily), fast info, and basic financial ratios if available.
    """
    tk = yf.Ticker(symbol)

    # Price history: 3 years plus a buffer. Yahoo treats `end` as exclusive,
    # so the end date is two days ahead and today's session is included.
    end = datetime.utcnow() + timedelta(days=2)
    start = datetime.utcnow() - timedelta(days=365 * 3 + 45)
    hist = tk.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=True)

    # Fast info (robust vs legacy .info)
    fast = {}
    try:
        fast = tk.fast_info or {}
    except Exception:
        fast = {}

    # Fundamentals (best-effort; yfinance varies by ticker)
    fin = {}
    try:
        # yfinance v0.2+ exposes .get_financials, .get_income_stmt, etc. as DataFrames
        income = tk.income_stmt
        bal = tk.balance_sheet
        cash = tk.cashflow
        fin = {
            "income_stmt_cols": list(income.columns) if isinstance(income, pd.DataFrame) else [],
            "balance_sheet_cols": list(bal.columns) if isinstance(bal, pd.DataFrame) else [],
            "cashflow_cols": list(cash.columns) if isinstance(cash, pd.DataFrame) else [],
        }
    except Exception:
        pass

    hist, quote = _merge_live_quote(hist, fast)
    return {"hist": hist, "fast": fast, "fin": fin, "quote": quote}


def _quote_number(fast, *names):
    if not fast:
        return None
    for name in names:
        value = None
        try:
            value = fast[name]
        except Exception:
            value = getattr(fast, name, None)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0:
            return number
    return None


def _merge_live_quote(hist: pd.DataFrame, fast) -> tuple:
    """
    Use the live quote Groww shows when the daily bar is still the previous close.
    """
    live = _quote_number(fast, "lastPrice", "last_price")
    previous = _quote_number(fast, "previousClose", "previous_close", "regularMarketPreviousClose")
    info = {"today_price": live, "previous_close": previous}
    if hist is None or hist.empty or live is None:
        return hist, info

    today = (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()
    last_day = _bar_date(hist.index[-1])
    last_close = float(hist["Close"].iloc[-1])
    hist = hist.copy()
    if last_day >= today:
        if abs(live - last_close) / last_close > 0.001:
            hist.iloc[-1, hist.columns.get_loc("Close")] = live
            if "High" in hist.columns:
                hist.iloc[-1, hist.columns.get_loc("High")] = max(float(hist["High"].iloc[-1]), live)
            if "Low" in hist.columns:
                hist.iloc[-1, hist.columns.get_loc("Low")] = min(float(hist["Low"].iloc[-1]), live)
        return hist, info

    ts = pd.Timestamp(today)
    if getattr(hist.index, "tz", None) is not None:
        ts = ts.tz_localize(hist.index.tz)
    row = {col: np.nan for col in hist.columns}
    row["Close"] = live
    if "Open" in row:
        row["Open"] = previous if previous else live
    if "High" in row:
        row["High"] = max(live, previous or live)
    if "Low" in row:
        row["Low"] = min(live, previous or live)
    volume = _quote_number(fast, "lastVolume", "last_volume")
    if "Volume" in row and volume:
        row["Volume"] = volume
    extra = pd.DataFrame([row], index=pd.DatetimeIndex([ts]))
    hist = pd.concat([hist, extra])
    return hist, info


def compute_indicators(hist: pd.DataFrame) -> Dict[str, Any]:
    if hist is None or hist.empty:
        return {"error": "No price history"}

    close = hist["Close"].dropna()
    if len(close) < 60:
        return {"error": "Insufficient history (need ~60+ days)"}

    # Simple MAs
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    # RSI
    rsi = RSIIndicator(close, window=14).rsi()

    # MACD
    macd_calc = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd = macd_calc.macd()
    macd_signal = macd_calc.macd_signal()
    macd_diff = macd_calc.macd_diff()

    # 52w high/low
    last_252 = close.tail(252)
    high_52w = float(last_252.max())
    low_52w = float(last_252.min())

    last = float(close.iloc[-1])
    above_50 = last > float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else False
    above_200 = last > float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else False

    # Volume trend
    vol = hist["Volume"].dropna()
    vol_avg_20 = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
    vol_last = float(vol.iloc[-1]) if len(vol) else np.nan
    vol_ratio = (vol_last / vol_avg_20) if vol_avg_20 else np.nan

    return {
        "last_price": last,
        "sma20": float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else None,
        "sma50": float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None,
        "sma200": float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None,
        "rsi14": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None,
        "macd": float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else None,
        "macd_signal": float(macd_signal.iloc[-1]) if not np.isnan(macd_signal.iloc[-1]) else None,
        "macd_hist": float(macd_diff.iloc[-1]) if not np.isnan(macd_diff.iloc[-1]) else None,
        "52w_high": high_52w,
        "52w_low": low_52w,
        "above_sma50": above_50,
        "above_sma200": above_200,
        "volume_last": vol_last,
        "volume_avg20": vol_avg_20,
        "volume_ratio": float(vol_ratio) if not np.isnan(vol_ratio) else None,
    }


def _bar_date(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        # Exchange calendar date, not the UTC date.
        return datetime.strptime(ts.strftime("%Y-%m-%d"), "%Y-%m-%d").date()
    return ts.date()


def _next_weekday(day):
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def _swing_points(close: pd.Series, span: int, kind: str) -> list:
    """Indices of swing highs or lows. Nearby hits are collapsed into one extreme."""
    values = close.to_numpy(dtype=float)
    n = len(values)
    raw = []
    for i in range(span, n - span):
        window = values[i - span:i + span + 1]
        if not np.isfinite(window).all():
            continue
        if kind == "high" and values[i] >= np.max(window):
            raw.append(i)
        elif kind == "low" and values[i] <= np.min(window):
            raw.append(i)
    if not raw:
        return []

    groups = [[raw[0]]]
    for i in raw[1:]:
        if i - groups[-1][-1] <= span:
            groups[-1].append(i)
        else:
            groups.append([i])

    picked = []
    for group in groups:
        if kind == "high":
            picked.append(max(group, key=lambda i: values[i]))
        else:
            picked.append(min(group, key=lambda i: values[i]))
    return picked


def _forward_high_returns(values: np.ndarray, bars_ahead: int) -> np.ndarray:
    """How far price rose above each close over the following bars."""
    bars_ahead = max(1, int(bars_ahead))
    if len(values) <= bars_ahead + 1:
        return np.array([])
    rets = []
    for i in range(len(values) - bars_ahead):
        base = values[i]
        if not np.isfinite(base) or base == 0:
            continue
        future_max = np.nanmax(values[i + 1:i + 1 + bars_ahead])
        if np.isfinite(future_max):
            rets.append(future_max / base - 1.0)
    return np.asarray(rets, dtype=float)


def forecast_outlook(hist: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate the path (up, down, or sideways) and the next swing-high date.
    Direction comes from moving averages, MACD, RSI, and 1-month through 3-year returns.
    The date comes from the spacing of past swing highs, and the price is kept inside
    the range of past rallies of a similar length.
    """
    if indicators.get("error"):
        return {"error": indicators["error"]}

    close = hist["Close"].dropna()
    if len(close) < 40:
        return {"error": "Not enough history to forecast"}

    last = float(close.iloc[-1])
    as_of = _bar_date(close.index[-1])
    weighted = []
    bullish = []
    bearish = []

    def vote(value, weight, up_text, down_text):
        if value is None or not np.isfinite(value) or value == 0:
            return
        choice = 1 if value > 0 else -1
        weighted.append((choice, weight))
        (bullish if choice > 0 else bearish).append(up_text if choice > 0 else down_text)

    sma50 = indicators.get("sma50")
    sma200 = indicators.get("sma200")
    if sma50:
        vote(
            last - float(sma50), 1.5,
            f"Price is above the 50-day average ({float(sma50):,.2f}).",
            f"Price is below the 50-day average ({float(sma50):,.2f}).",
        )
    if sma200:
        vote(
            last - float(sma200), 1.5,
            f"Price is above the 200-day average ({float(sma200):,.2f}).",
            f"Price is below the 200-day average ({float(sma200):,.2f}).",
        )

    sma20 = close.rolling(20).mean()
    if pd.notna(sma20.iloc[-1]) and pd.notna(sma20.iloc[-10]):
        vote(
            float(sma20.iloc[-1] - sma20.iloc[-10]), 1.2,
            "The 20-day average is rising.",
            "The 20-day average is falling.",
        )

    if indicators.get("macd_hist") is not None:
        vote(
            float(indicators["macd_hist"]), 1.2,
            "MACD momentum is positive.",
            "MACD momentum is negative.",
        )

    rsi = indicators.get("rsi14")
    if rsi is not None and 30 < float(rsi) < 70:
        vote(
            float(rsi) - 50, 0.8,
            f"RSI is {float(rsi):.0f}, on the strong side of 50.",
            f"RSI is {float(rsi):.0f}, on the weak side of 50.",
        )

    returns = []
    for label, bars, weight in (
        ("1M", 21, 1.2),
        ("3M", 63, 1.1),
        ("6M", 126, 1.0),
        ("1Y", 252, 1.0),
        ("2Y", 504, 0.7),
        ("3Y", 756, 0.6),
    ):
        if len(close) <= bars:
            continue
        ret = float(close.iloc[-1] / close.iloc[-1 - bars] - 1)
        returns.append({"label": label, "return_pct": round(ret * 100, 1)})
        vote(
            ret, weight,
            f"The {label.lower()} return is {ret * 100:+.1f}%.",
            f"The {label.lower()} return is {ret * 100:+.1f}%.",
        )

    span = 10 if len(close) < 400 else 15
    if len(close) < span * 4:
        span = max(3, len(close) // 8)
    high_idx = _swing_points(close, span, "high")
    low_idx = _swing_points(close, span, "low")
    if len(high_idx) >= 2:
        vote(
            float(close.iloc[high_idx[-1]] - close.iloc[high_idx[-2]]), 1.3,
            "Recent swing highs are stepping up.",
            "Recent swing highs are stepping down.",
        )

    if not weighted:
        return {"error": "Not enough signals to forecast"}

    score = sum(choice * weight for choice, weight in weighted) / sum(weight for _, weight in weighted)
    if score >= 0.15:
        direction = "Upward"
        picked = bullish
    elif score <= -0.15:
        direction = "Downward"
        picked = bearish
    else:
        direction = "Sideways"
        picked = bullish[:2] + bearish[:2]

    gaps = []
    for left, right in zip(high_idx, high_idx[1:]):
        gap_days = (_bar_date(close.index[right]) - _bar_date(close.index[left])).days
        if gap_days > 0:
            gaps.append(gap_days)
    median_gap = int(round(float(np.median(gaps)))) if gaps else 28
    median_gap = max(median_gap, 7)

    anchor = _bar_date(close.index[high_idx[-1]]) if high_idx else as_of
    next_date = anchor + timedelta(days=median_gap)
    for _ in range(36):
        if next_date > as_of:
            break
        next_date += timedelta(days=median_gap)
    next_date = _next_weekday(next_date)

    bars_ahead = max(1, int(round((next_date - as_of).days * 5 / 7)))
    window = min(len(close), max(40, bars_ahead * 3))
    y = close.iloc[-window:].to_numpy(dtype=float)
    slope, intercept = np.polyfit(np.arange(window, dtype=float), y, 1)
    trend_price = float(intercept + slope * (window - 1 + bars_ahead))

    amplitudes = []
    for high_i in high_idx:
        prior_lows = [i for i in low_idx if i < high_i]
        if prior_lows:
            amplitudes.append(float(close.iloc[high_i] - close.iloc[prior_lows[-1]]))
    amplitude = float(np.median(amplitudes)) if amplitudes else abs(last) * 0.02
    model_price = trend_price + 0.35 * max(amplitude, 0.0)

    hist_rets = _forward_high_returns(close.to_numpy(dtype=float), min(bars_ahead, max(1, len(close) // 5)))
    if len(hist_rets):
        typical = float(np.median(hist_rets))
        cap = float(np.quantile(hist_rets, 0.9))
    else:
        typical = 0.02
        cap = 0.08
    cap = max(cap, 0.01)
    model_ret = (model_price / last - 1.0) if last else 0.0
    blended = 0.65 * model_ret + 0.35 * max(typical, 0.0)
    blended = float(min(max(blended, 0.0), cap))
    high_price = last * (1.0 + blended)

    if gaps and np.mean(gaps):
        regularity = 1.0 - min(float(np.std(gaps) / np.mean(gaps)), 1.0)
    else:
        regularity = 0.0
    sample = min(len(gaps) / 5.0, 1.0)
    confidence = int(round(100 * (0.5 * min(abs(score), 1.0) + 0.35 * regularity + 0.15 * sample)))
    confidence = int(np.clip(confidence, 28, 80))

    reasons = picked[:3] + [
        f"Past highs have been about {median_gap} days apart, so the next high is estimated on {next_date.strftime('%d %b %Y')}.",
        f"The estimated price at that high is {high_price:,.2f}, {blended * 100:.1f}% above the latest close of {last:,.2f}.",
    ]

    return {
        "direction": direction,
        "score": round(float(score), 2),
        "confidence": confidence,
        "as_of": as_of.isoformat(),
        "high_date": next_date.isoformat(),
        "high_date_label": next_date.strftime("%d %b %Y"),
        "days_to_high": (next_date - as_of).days,
        "high_price": round(high_price, 2),
        "last_price": round(last, 2),
        "high_vs_last_pct": round(blended * 100, 1),
        "returns": returns,
        "reasons": reasons,
        "median_high_gap_days": median_gap,
    }


def daily_market_read(hist: pd.DataFrame, bars: int) -> pd.DataFrame:
    """
    One row per session. The day read combines that day's move, its place
    versus the 20-day average, and the MACD histogram.
    """
    close = hist["Close"].dropna().astype(float)
    if close.empty:
        return pd.DataFrame()

    volume = hist["Volume"].reindex(close.index).astype(float) if "Volume" in hist.columns else None
    sma20 = close.rolling(20).mean()
    rsi = RSIIndicator(close, window=14).rsi()
    macd_hist = MACD(close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
    change = close.pct_change()
    versus_sma = close / sma20 - 1.0
    if volume is not None:
        vol_vs = volume / volume.rolling(20).mean()
    else:
        vol_vs = pd.Series(np.nan, index=close.index)

    change_vote = pd.Series(
        np.where(change > 0.001, 1.0, np.where(change < -0.001, -1.0, 0.0)),
        index=close.index,
    ).where(change.notna())
    sma_vote = pd.Series(np.where(close > sma20, 1.0, -1.0), index=close.index).where(sma20.notna())
    macd_vote = pd.Series(np.where(macd_hist > 0, 1.0, -1.0), index=close.index).where(macd_hist.notna())
    score = pd.concat([change_vote, sma_vote, macd_vote], axis=1).mean(axis=1, skipna=True)
    day_read = pd.Series(
        np.where(score >= 0.34, "Upward", np.where(score <= -0.34, "Downward", "Sideways")),
        index=close.index,
    ).where(score.notna(), "Sideways")

    frame = pd.DataFrame({
        "Date": [_bar_date(ts) for ts in close.index],
        "Close": close.to_numpy(),
        "Change": change.to_numpy(),
        "RSI": rsi.to_numpy(),
        "MACD": macd_hist.to_numpy(),
        "Vs 20-day average": versus_sma.to_numpy(),
        "Volume vs average": vol_vs.to_numpy(),
        "Day read": day_read.to_numpy(),
    })
    return frame.tail(max(int(bars), 1)).iloc[::-1].reset_index(drop=True)


def projected_sessions(outlook: Dict[str, Any]) -> pd.DataFrame:
    """Trading days from the next session through the expected high."""
    if outlook.get("error") or not outlook.get("high_date"):
        return pd.DataFrame()

    start = datetime.strptime(outlook["as_of"], "%Y-%m-%d").date()
    end = datetime.strptime(outlook["high_date"], "%Y-%m-%d").date()
    days = pd.bdate_range(start + timedelta(days=1), end)
    if len(days) == 0:
        days = pd.DatetimeIndex([pd.Timestamp(end)])

    last = float(outlook["last_price"])
    high = float(outlook["high_price"])
    prices = np.linspace(last, high, len(days) + 1)[1:]
    previous = np.concatenate([[last], prices[:-1]])
    change = np.where(previous != 0, prices / previous - 1.0, 0.0)
    reads = []
    for i, (price, prior) in enumerate(zip(prices, previous)):
        if i == len(prices) - 1:
            reads.append("Expected high")
        elif price > prior:
            reads.append("Upward")
        elif price < prior:
            reads.append("Downward")
        else:
            reads.append("Sideways")

    return pd.DataFrame({
        "Date": [ts.date() for ts in days],
        "Estimated close": np.round(prices, 2),
        "Change": change,
        "Day read": reads,
    })


NEWS_WINDOW_DAYS = 15
PREDICT_SESSIONS = 15

_POSITIVE_WORDS = {
    "upgrade", "upgraded", "surge", "surged", "surges", "rally", "rallied",
    "record", "beat", "beats", "beaten", "bullish", "outperform", "approval",
    "approved", "breakthrough", "buyback", "dividend", "expansion", "partnership",
    "profit", "profits", "growth", "strong", "gain", "gains", "raised", "raises",
}
_NEGATIVE_WORDS = {
    "downgrade", "downgraded", "plunge", "plunged", "plunges", "slump", "slumped",
    "lawsuit", "investigation", "probe", "fraud", "recall", "layoff", "layoffs",
    "bankruptcy", "default", "warning", "miss", "misses", "missed", "underperform",
    "bearish", "decline", "declined", "declines", "crash", "crashed", "halt",
    "halted", "penalty", "selloff", "tumble", "tumbled", "disappoint",
    "disappointed", "loss", "losses", "weak", "weakness", "lowered", "lowers",
}
_POSITIVE_PHRASES = ("price target raised", "beats estimates", "beat estimates", "raises guidance", "share buyback")
_NEGATIVE_PHRASES = ("guidance cut", "price target cut", "job cuts", "profit warning", "sell-off")


def _news_text_score(text: str) -> float:
    lowered = (text or "").lower()
    if not lowered.strip():
        return 0.0
    pos = sum(lowered.count(phrase) for phrase in _POSITIVE_PHRASES)
    neg = sum(lowered.count(phrase) for phrase in _NEGATIVE_PHRASES)
    for word in re.findall(r"[a-z']+", lowered):
        if word in _POSITIVE_WORDS:
            pos += 1
        elif word in _NEGATIVE_WORDS:
            neg += 1
    if pos == neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _tone_label(score: float) -> str:
    if score > 0.05:
        return "Positive"
    if score < -0.05:
        return "Negative"
    return "Neutral"


def _parse_news_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = str(content.get("title") or "").strip()
    if not title:
        return None
    summary = str(content.get("summary") or content.get("description") or "").strip()
    provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
    source = str(provider.get("displayName") or content.get("publisher") or "").strip()
    url = ""
    for key in ("canonicalUrl", "clickThroughUrl"):
        block = content.get(key)
        if isinstance(block, dict) and block.get("url"):
            url = str(block["url"])
            break
    if not url:
        url = str(content.get("link") or "")
    published = content.get("pubDate") or content.get("displayTime")
    if published is None and content.get("providerPublishTime"):
        published = datetime.utcfromtimestamp(int(content["providerPublishTime"])).isoformat() + "Z"
    when = pd.to_datetime(published, utc=True, errors="coerce")
    if pd.isna(when):
        return None
    headline_score = 0.7 * _news_text_score(title) + 0.3 * _news_text_score(summary)
    return {
        "published": when,
        "Date": when.tz_convert("UTC").date(),
        "Source": source or "Yahoo Finance",
        "Headline": title,
        "Tone": _tone_label(headline_score),
        "score": float(headline_score),
        "Link": url,
    }


def fetch_company_news(symbol: str, days: int = NEWS_WINDOW_DAYS) -> list:
    """Headlines for this symbol from the past `days` calendar days."""
    tk = yf.Ticker(symbol)
    raw = []
    try:
        raw.extend(tk.get_news(count=40, tab="all") or [])
    except TypeError:
        raw.extend(tk.news or [])
    except Exception:
        pass
    try:
        raw.extend(yf.Search(symbol, news_count=20).news or [])
    except Exception:
        pass

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    parsed = []
    seen = set()
    for item in raw:
        row = _parse_news_item(item)
        if row is None or row["published"] < cutoff:
            continue
        key = row["Headline"].lower()
        if key in seen:
            continue
        seen.add(key)
        parsed.append(row)
    parsed.sort(key=lambda row: row["published"], reverse=True)
    return parsed


def analyze_recent_news(symbol: str, hist: pd.DataFrame) -> Dict[str, Any]:
    """
    Score the past 15 days of headlines and project the next 15 sessions.
    A downfall is a sharp recent drop, a clearly negative news tone, or a
    forecast that loses ground over those sessions.
    """
    articles = fetch_company_news(symbol)
    scores = [row["score"] for row in articles]
    news_score = float(np.mean(scores)) if scores else 0.0
    positive = sum(1 for row in articles if row["Tone"] == "Positive")
    negative = sum(1 for row in articles if row["Tone"] == "Negative")
    neutral = len(articles) - positive - negative
    if news_score > 0.05:
        tone = "Positive"
    elif news_score < -0.05:
        tone = "Negative"
    elif articles:
        tone = "Mixed"
    else:
        tone = "No headlines"

    close = hist["Close"].dropna().astype(float) if hist is not None else pd.Series(dtype=float)
    ret_1 = ret_5 = ret_15 = drift = typical = 0.0
    as_of = datetime.utcnow().date()
    last_price = None
    if len(close) >= 2:
        as_of = _bar_date(close.index[-1])
        last_price = float(close.iloc[-1])
        changes = close.pct_change().dropna()
        ret_1 = float(changes.iloc[-1])
        ret_5 = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) > 6 else ret_1
        ret_15 = float(close.iloc[-1] / close.iloc[-16] - 1.0) if len(close) > 16 else ret_5
        recent = changes.tail(15)
        drift = float(recent.mean()) if len(recent) else 0.0
        typical = float(recent.std()) if len(recent) else 0.01
        if not np.isfinite(typical) or typical <= 0:
            typical = 0.01

    reasons = []
    if ret_1 <= -0.02:
        reasons.append(f"The last session fell {abs(ret_1) * 100:.1f}%.")
    if ret_5 <= -0.025:
        reasons.append(f"The past 5 sessions are down {abs(ret_5) * 100:.1f}%.")
    if ret_15 <= -0.04:
        reasons.append(f"The past 15 sessions are down {abs(ret_15) * 100:.1f}%.")
    if articles and negative >= max(positive + 2, 3) and news_score <= -0.2:
        reasons.append(
            f"{negative} of {len(articles)} headlines from the past 15 days read negative."
        )

    forecast = pd.DataFrame()
    predicted_return = 0.0
    if last_price is not None:
        days = pd.bdate_range(as_of + timedelta(days=1), periods=PREDICT_SESSIONS)
        price = last_price
        rows = []
        for i, day in enumerate(days):
            fade = 0.82 ** i
            step = 0.55 * drift + 0.45 * news_score * typical * fade
            step = float(np.clip(step, -0.025, 0.025))
            nxt = price * (1.0 + step)
            if step > 0.001:
                read = "Upward"
            elif step < -0.001:
                read = "Downward"
            else:
                read = "Sideways"
            rows.append({
                "Date": day.date(),
                "Estimated close": round(nxt, 2),
                "Change": step,
                "Day read": read,
            })
            price = nxt
        forecast = pd.DataFrame(rows)
        predicted_return = price / last_price - 1.0
        down_days = sum(1 for row in rows if row["Day read"] == "Downward")
        if predicted_return <= -0.02:
            reasons.append(
                f"The next {PREDICT_SESSIONS} sessions are estimated down {abs(predicted_return) * 100:.1f}%."
            )
        elif down_days >= 10 and ret_15 < 0:
            reasons.append(
                f"{down_days} of the next {PREDICT_SESSIONS} sessions are estimated downward."
            )

    news_frame = pd.DataFrame([
        {
            "Date": row["Date"],
            "Source": row["Source"],
            "Tone": row["Tone"],
            "Headline": row["Headline"],
            "Link": row["Link"],
        }
        for row in articles
    ])

    return {
        "articles": [
            {"date": row["Date"].isoformat(), "title": row["Headline"], "tone": row["Tone"]}
            for row in articles[:15]
        ],
        "news_frame": news_frame,
        "forecast": forecast,
        "headline_count": len(articles),
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "news_score": round(news_score, 2),
        "tone": tone,
        "downfall": bool(reasons),
        "reasons": reasons,
        "predicted_return_pct": round(predicted_return * 100, 1),
    }


def lookup_companies(query: str) -> list:
    """
    Resolve a company name (or ticker) to equity matches via Yahoo Finance.
    Each item is {symbol, name, exchange}.
    """
    query = (query or "").strip()
    if not query:
        return []

    found = []
    seen = set()

    def add(symbol, name, exchange, quote_type):
        symbol = str(symbol or "").strip()
        if not symbol or symbol in seen:
            return
        qtype = str(quote_type or "EQUITY").upper()
        if qtype not in {"EQUITY", "ETF"}:
            return
        seen.add(symbol)
        found.append({
            "symbol": symbol,
            "name": str(name or symbol).strip() or symbol,
            "exchange": str(exchange or "").strip(),
            "quote_type": qtype,
        })

    try:
        if hasattr(yf, "Lookup"):
            frame = yf.Lookup(query).get_stock(count=10)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                if "symbol" in frame.columns:
                    rows = frame.to_dict("records")
                    for row in rows:
                        add(row.get("symbol"), row.get("shortName") or row.get("longName"), row.get("exchange") or row.get("exchDisp"), row.get("quoteType"))
                else:
                    for symbol, row in frame.iterrows():
                        add(symbol, row.get("shortName") if hasattr(row, "get") else None, row.get("exchange") if hasattr(row, "get") else None, row.get("quoteType") if hasattr(row, "get") else None)
    except Exception:
        pass

    if not found:
        try:
            search = yf.Search(query, max_results=10)
            for quote in (getattr(search, "quotes", None) or []):
                add(
                    quote.get("symbol"),
                    quote.get("longname") or quote.get("shortname") or quote.get("shortName"),
                    quote.get("exchDisp") or quote.get("exchange"),
                    quote.get("quoteType"),
                )
        except Exception:
            pass

    if not found and query.replace(".", "").replace("-", "").isalnum():
        add(query.upper(), query.upper(), "", "EQUITY")

    q = query.lower()

    def rank(item):
        sym = item["symbol"].lower()
        name = item["name"].lower()
        if sym == q or sym.split(".")[0] == q:
            score = 0
        elif name == q:
            score = 1
        elif name.startswith(q):
            score = 2
        elif q in name:
            score = 3
        else:
            score = 4
        if item.get("quote_type") != "EQUITY":
            score += 10
        return score

    found.sort(key=rank)
    return found


def extract_fundamentals(fast: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull whatever is reliably present in .fast_info (yfinance’s newer API).
    Fields may be missing per ticker.
    """
    def safe(k, default=None):
        v = fast.get(k, default)
        try:
            return float(v)
        except Exception:
            return v

    out = {
        "market_cap": safe("market_cap"),
        "pe_ratio": safe("trailing_pe"),
        "forward_pe": safe("forward_pe"),
        "pb_ratio": safe("price_to_book"),
        "dividend_yield": safe("dividend_yield"),
    }
    return out


# =========================
# Agent (LLM) Setup
# =========================
def build_agent(api_key: str) -> AssistantAgent:
    # AutoGen's OpenAI client routes any model name starting with "gemini-"
    # to Gemini's OpenAI-compatible endpoint when base_url is set.
    key_changed = st.session_state.get("api_key_used") != api_key
    if st.session_state.model_client is None or st.session_state.get("model_name") != GEMINI_MODEL or key_changed:
        st.session_state.model_client = OpenAIChatCompletionClient(
            model=GEMINI_MODEL,
            api_key=api_key,
            base_url=GEMINI_BASE_URL,
            temperature=0.3,  # more deterministic for decisions
            include_name_in_message=False,  # Gemini rejects the OpenAI "name" field
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": True,
                "family": "unknown",
                "structured_output": False,
            },
        )
        st.session_state.model_name = GEMINI_MODEL
        st.session_state.api_key_used = api_key

    system_message = (
        "You are a disciplined equity research assistant. You will be given:\n"
        "1) Technical indicators (RSI, MACD, SMAs, 52w range, volume ratio)\n"
        "2) Fundamental signals (PE, PB, market cap, dividend yield)\n"
        "3) A computed outlook with direction, expected_high_date, and expected_high_price\n"
        "4) Headlines from the past 15 days, each with a tone, plus a downfall flag\n\n"
        "Output strict JSON with fields:\n"
        "{\n"
        '  "action": "BUY" | "HOLD" | "SELL",\n'
        '  "confidence": 0-100,\n'
        '  "technical_summary": "…",\n'
        '  "fundamental_summary": "…",\n'
        '  "risks": ["…", "…"],\n'
        '  "notes": "…"\n'
        "}\n\n"
        "Rules:\n"
        "- Combine both technical + fundamental signals.\n"
        "- Favor BUY if trend is positive (price > SMA50 & SMA200, MACD>=0, RSI 45–65) and valuation not excessive (PE or PB reasonable vs sector).\n"
        "- Favor SELL if trend is negative (price < SMA200, MACD<0, RSI<40) or valuation/risks are severe.\n"
        "- Otherwise HOLD.\n"
        "- Be conservative if data is missing.\n"
        "- If you mention a high date or a direction, use the computed outlook. Do not invent another date.\n"
        "- Weigh the past 15 days of headlines with the technicals.\n"
        "- If downfall is true, name that in risks.\n"
        "- JSON only. No markdown, no extra text."
    )
    agent = AssistantAgent(
        name="stock_agent",
        system_message=system_message,
        model_client=st.session_state.model_client,
    )
    return agent


async def _ask_agent_async(agent: AssistantAgent, payload: Dict[str, Any]) -> str:
    # We use a 1-agent "team" just to reuse the same run loop style if you later add more tools/agents
    team = RoundRobinGroupChat([agent], max_turns=1)
    msg = json.dumps(payload)
    result = await team.run(task=msg)
    # get last message
    if result.messages:
        return result.messages[-1].content
    return ""


def ask_agent(agent: AssistantAgent, payload: Dict[str, Any]) -> str:
    loop = st.session_state.loop
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(_ask_agent_async(agent, payload))


# =========================
# UI
# =========================
col = st.columns([2, 1, 1])
with col[0]:
    company = st.text_input("Company name", value="", placeholder="e.g., Apple, Microsoft, Reliance")

with col[1]:
    lookback = st.selectbox("Price lookback", ["3y", "2y", "1y", "6mo", "3mo"], index=2)

with col[2]:
    run_btn = st.button(
        "Analyze",
        type="primary",
        width="stretch",
        disabled=(not company.strip() or not gemini_api_key.strip()),
    )

st.divider()

if run_btn:
    matches = lookup_companies(company)
    st.session_state.matches = matches
    st.session_state.searched_query = company.strip()
    st.session_state.selected_symbol = matches[0]["symbol"] if matches else None
    st.session_state.analyze_now = bool(matches)
    if not matches:
        st.error(f"No listed company found for “{company.strip()}”. Try the full name or the ticker.")

same_query = company.strip() == st.session_state.searched_query
if same_query and len(st.session_state.matches) > 1:
    labels = []
    label_to_symbol = {}
    for match in st.session_state.matches:
        exchange = f" · {match['exchange']}" if match.get("exchange") else ""
        label = f"{match['name']} ({match['symbol']}){exchange}"
        labels.append(label)
        label_to_symbol[label] = match["symbol"]
    current = next(
        (label for label, symbol in label_to_symbol.items() if symbol == st.session_state.selected_symbol),
        labels[0],
    )
    chosen = st.selectbox("Matching companies", labels, index=labels.index(current))
    chosen_symbol = label_to_symbol[chosen]
    if chosen_symbol != st.session_state.selected_symbol:
        st.session_state.selected_symbol = chosen_symbol
        st.session_state.analyze_now = True

if st.session_state.analyze_now and same_query and st.session_state.selected_symbol:
    match = next(m for m in st.session_state.matches if m["symbol"] == st.session_state.selected_symbol)
    ticker = match["symbol"]
    company_name = match["name"]
    st.session_state.analyze_now = False
    with st.spinner(f"Fetching and analyzing {company_name} ({ticker})…"):
        try:
            data = fetch_stock_data(ticker)
            hist = data["hist"]
            fast = data["fast"]
            quote = data.get("quote") or {}

            if hist is None or hist.empty:
                st.error("No price data found for this symbol.")
            else:
                # Trim lookback for display
                bars = LOOKBACK_BARS.get(lookback, len(hist))
                hist_disp = hist.tail(bars)

                inds = compute_indicators(hist)
                fins = extract_fundamentals(fast)
                outlook = forecast_outlook(hist, inds)
                news_view = analyze_recent_news(ticker, hist)
                payload = {
                    "company_name": company_name,
                    "symbol": ticker,
                    "as_of": datetime.utcnow().isoformat() + "Z",
                    "technical": inds,
                    "fundamental": fins,
                    "outlook": {
                        "direction": outlook.get("direction"),
                        "confidence": outlook.get("confidence"),
                        "expected_high_date": outlook.get("high_date"),
                        "expected_high_price": outlook.get("high_price"),
                    },
                    "news_15d": {
                        "headline_count": news_view["headline_count"],
                        "tone": news_view["tone"],
                        "downfall": news_view["downfall"],
                        "reasons": news_view["reasons"],
                        "headlines": news_view["articles"],
                    },
                }

                if news_view["downfall"]:
                    st.toast(f"Downfall alert for {company_name}")
                    st.error("Downfall alert for " + company_name + " (" + ticker + ").")
                    for reason in news_view["reasons"]:
                        st.write("- " + reason)

                today_price = quote.get("today_price")
                previous_close = quote.get("previous_close")
                if today_price:
                    change = None
                    if previous_close:
                        change = f"{(today_price / previous_close - 1) * 100:+.2f}% from previous close {previous_close:,.2f}"
                    st.metric("Today's price", f"{today_price:,.2f}", delta=change, border=True)
                    if previous_close and abs(today_price - previous_close) >= 0.05:
                        st.caption(
                            f"This matches the live price on Groww. "
                            f"The previous close was {previous_close:,.2f}, which is the last completed session."
                        )

                st.subheader(f"{company_name} ({ticker}) – Market outlook")
                if outlook.get("error"):
                    st.warning(outlook["error"])
                else:
                    headline = (
                        f"{outlook['direction']} — next high estimated "
                        f"{outlook['high_date_label']} ({outlook['days_to_high']} days out)"
                    )
                    if outlook["direction"] == "Upward":
                        st.success(headline)
                    elif outlook["direction"] == "Downward":
                        st.error(headline)
                    else:
                        st.info(headline)

                    with st.container(horizontal=True):
                        st.metric("Direction", outlook["direction"], delta=f"{outlook['score']:+.2f} score", border=True)
                        st.metric(
                            "Expected high date",
                            outlook["high_date_label"],
                            delta=f"{outlook['days_to_high']} days",
                            border=True,
                        )
                        st.metric(
                            "Estimated price at high",
                            f"{outlook['high_price']:,.2f}",
                            delta=f"{outlook['high_vs_last_pct']:+.1f}% vs last close",
                            border=True,
                        )
                        st.metric("Outlook confidence", f"{outlook['confidence']}/100", border=True)

                    if outlook.get("returns"):
                        with st.container(horizontal=True):
                            for item in outlook["returns"]:
                                st.metric(item["label"], f"{item['return_pct']:+.1f}%", border=True)

                    st.markdown("**How this was read**")
                    for reason in outlook["reasons"]:
                        st.write(f"- {reason}")
                    st.caption(
                        "Direction uses moving averages, MACD, RSI, and returns from 1 month through 3 years. "
                        "The high date follows the spacing of past swing highs. "
                        "The price is an estimate kept inside the range of similar past rallies."
                    )

                with st.container(border=True):
                    st.subheader("News, past 15 days")
                    with st.container(horizontal=True):
                        st.metric("Headlines", news_view["headline_count"], border=True)
                        st.metric("Positive", news_view["positive"], border=True)
                        st.metric("Negative", news_view["negative"], border=True)
                        st.metric("News tone", news_view["tone"], border=True)
                    if news_view["news_frame"].empty:
                        st.info("No headlines were returned for this company in the past 15 days.")
                    else:
                        st.dataframe(
                            news_view["news_frame"],
                            column_config={
                                "Date": st.column_config.DateColumn("Date", pinned=True),
                                "Source": st.column_config.TextColumn("Source"),
                                "Tone": st.column_config.TextColumn("Tone"),
                                "Headline": st.column_config.TextColumn("Headline"),
                                "Link": st.column_config.LinkColumn("Link", display_text="Open"),
                            },
                            hide_index=True,
                            height=320,
                        )

                forecast = news_view["forecast"]
                if not forecast.empty:
                    with st.container(border=True):
                        st.subheader("Predicted day-to-day performance")
                        st.caption(
                            f"Next {len(forecast)} trading days, estimated {news_view['predicted_return_pct']:+.1f}% from the latest close. "
                            "Each day blends the past 15 sessions of price with the tone of the past 15 days of headlines. "
                            "The news effect fades on later days."
                        )
                        st.dataframe(
                            forecast,
                            column_config={
                                "Date": st.column_config.DateColumn("Date", pinned=True),
                                "Estimated close": st.column_config.NumberColumn("Estimated close", format="%,.2f"),
                                "Change": st.column_config.NumberColumn("Change", format="percent"),
                                "Day read": st.column_config.TextColumn("Day read"),
                            },
                            hide_index=True,
                            height=380,
                        )

                # Show raw metrics
                st.subheader(f"{company_name} ({ticker}) – Key Metrics")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Technical**")
                    st.json(inds, expanded=False)
                with c2:
                    st.markdown("**Fundamental**")
                    st.json(fins, expanded=False)

                # Chart. The last point is the estimated high when a forecast is available.
                st.markdown("**Price (Adj Close)**")
                chart_close = hist_disp["Close"].copy()
                if not outlook.get("error"):
                    high_ts = pd.Timestamp(outlook["high_date"])
                    if getattr(chart_close.index, "tz", None) is not None:
                        high_ts = high_ts.tz_localize(chart_close.index.tz)
                    chart_close.loc[high_ts] = outlook["high_price"]
                    chart_close = chart_close.sort_index()
                st.line_chart(chart_close)
                if not outlook.get("error"):
                    st.caption(f"The last point is the estimated high on {outlook['high_date_label']}, not a traded price.")

                daily = daily_market_read(hist, LOOKBACK_BARS.get(lookback, len(hist)))
                with st.container(border=True):
                    st.subheader("Day by day")
                    st.caption(
                        f"{len(daily)} sessions in the {lookback} window, newest first. "
                        "Each day read uses that session's move, its place versus the 20-day average, and MACD."
                    )
                    st.dataframe(
                        daily,
                        column_config={
                            "Date": st.column_config.DateColumn("Date", pinned=True),
                            "Close": st.column_config.NumberColumn("Close", format="%,.2f"),
                            "Change": st.column_config.NumberColumn("Change", format="percent"),
                            "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                            "MACD": st.column_config.NumberColumn("MACD", format="%.3f"),
                            "Vs 20-day average": st.column_config.NumberColumn("Vs 20-day average", format="percent"),
                            "Volume vs average": st.column_config.NumberColumn("Volume vs average", format="%.2f"),
                            "Day read": st.column_config.TextColumn("Day read"),
                        },
                        hide_index=True,
                        height=420,
                    )

                upcoming = projected_sessions(outlook)
                if not upcoming.empty:
                    with st.container(border=True):
                        st.subheader("Path to the expected high")
                        st.caption("Estimated closes on each trading day from the next session through the expected high.")
                        st.dataframe(
                            upcoming,
                            column_config={
                                "Date": st.column_config.DateColumn("Date", pinned=True),
                                "Estimated close": st.column_config.NumberColumn("Estimated close", format="%,.2f"),
                                "Change": st.column_config.NumberColumn("Change", format="percent"),
                                "Day read": st.column_config.TextColumn("Day read"),
                            },
                            hide_index=True,
                            height=320,
                        )

                # Call agent for decision
                agent = build_agent(gemini_api_key.strip())
                raw = ask_agent(agent, payload)

                # Try parse
                decision = None
                try:
                    decision = json.loads(raw)
                except Exception:
                    # Sometimes models add stray characters—try a crude fix:
                    try:
                        start = raw.find("{")
                        end = raw.rfind("}")
                        if start != -1 and end != -1:
                            decision = json.loads(raw[start:end+1])
                    except Exception:
                        decision = None

                if not decision or not isinstance(decision, dict) or "action" not in decision:
                    st.error("Agent returned an invalid response. Showing raw output.")
                    st.code(raw)
                else:
                    st.session_state.latest_json = decision
                    a = decision.get("action", "HOLD")
                    conf = decision.get("confidence", 50)
                    st.success(f"**Recommendation: {a}** (confidence: {conf}/100)")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Technical Summary**")
                        st.write(decision.get("technical_summary", ""))
                    with c2:
                        st.markdown("**Fundamental Summary**")
                        st.write(decision.get("fundamental_summary", ""))

                    if decision.get("risks"):
                        st.markdown("**Risks**")
                        st.write("- " + "\n- ".join(decision["risks"]))
                    if decision.get("notes"):
                        st.markdown("**Notes**")
                        st.write(decision["notes"])

        except Exception as e:
            st.error(f"Error: {e}")

# Footer / debug
st.divider()
st.caption("Data and headlines via Yahoo Finance. Outlook, news tone, and day-to-day estimates are for education, not financial advice.")
