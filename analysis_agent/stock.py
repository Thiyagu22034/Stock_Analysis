import json
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

    # Price history: 3 years plus a buffer so the 3y window has a full set of bars.
    end = datetime.utcnow()
    start = end - timedelta(days=365 * 3 + 45)
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

    return {"hist": hist, "fast": fast, "fin": fin}


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
        "2) Fundamental signals (PE, PB, market cap, dividend yield)\n\n"
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
        use_container_width=True,
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

            if hist is None or hist.empty:
                st.error("No price data found for this symbol.")
            else:
                # Trim lookback for display
                bars = LOOKBACK_BARS.get(lookback, len(hist))
                hist_disp = hist.tail(bars)

                inds = compute_indicators(hist)
                fins = extract_fundamentals(fast)
                payload = {
                    "company_name": company_name,
                    "symbol": ticker,
                    "as_of": datetime.utcnow().isoformat() + "Z",
                    "technical": inds,
                    "fundamental": fins,
                }

                # Show raw metrics
                st.subheader(f"{company_name} ({ticker}) – Key Metrics")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Technical**")
                    st.json(inds, expanded=False)
                with c2:
                    st.markdown("**Fundamental**")
                    st.json(fins, expanded=False)

                # Chart
                st.markdown("**Price (Adj Close)**")
                st.line_chart(hist_disp["Close"])

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
st.caption("Data via yfinance. This is educational, not financial advice.")
