#!/usr/bin/env python
# coding: utf-8

"""
app.py — Supply Chain AI Agent (Streamlit UI)
Imports all logic from work_on_this.py
Run: streamlit run app.py
"""

import io
import re
import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage

# ── Import everything from your project module ──────────────────────────────
from work_on_this import SupplyChainState, generate_data, build_graph

# ── Streamlit version compatibility ─────────────────────────────────────────
# st.rerun() was added in Streamlit 1.27. Older versions use experimental_rerun.
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# Patch cache decorators for older Streamlit versions
if not hasattr(st, "cache_resource"):
    st.cache_resource = st.experimental_singleton
if not hasattr(st, "cache_data"):
    st.cache_data = st.experimental_memo

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Supply Chain AI Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
.stApp                       { background: #07090f; color: #c9d1d9; }
.main .block-container       { padding-top: 1.4rem; max-width: 1350px; }

/* ── KPI card ── */
.kpi-card {
    background: linear-gradient(145deg,#0f1623,#141d2e);
    border: 1px solid #1c2a3e; border-radius: 12px;
    padding: 1rem 1.2rem; position: relative; overflow: hidden;
}
.kpi-card::before {
    content:""; position:absolute; top:0; left:0; width:4px; height:100%;
}
.kpi-card.red::before    { background:#ef4444; }
.kpi-card.amber::before  { background:#f59e0b; }
.kpi-card.blue::before   { background:#3b82f6; }
.kpi-card.green::before  { background:#10b981; }
.kpi-card.purple::before { background:#8b5cf6; }
.kpi-label { font-family:'IBM Plex Mono',monospace; font-size:.66rem;
             letter-spacing:.1em; text-transform:uppercase; color:#516070; margin-bottom:.3rem; }
.kpi-value { font-family:'IBM Plex Mono',monospace; font-size:1.85rem;
             font-weight:700; color:#f0f6fc; line-height:1; }
.kpi-sub   { font-size:.72rem; color:#516070; margin-top:.25rem; }

/* ── Chat bubbles ── */
.chat-box {
    background:#0d1320; border:1px solid #1c2a3e; border-radius:14px;
    padding:1.2rem; max-height:480px; overflow-y:auto; margin-bottom:.8rem;
}
.msg-user {
    background:#1a3558; border-left:3px solid #3b82f6;
    border-radius:0 10px 10px 10px; padding:.7rem .95rem;
    margin:.45rem 0 .45rem 3rem; font-size:.9rem; color:#bfdbfe;
}
.msg-ai {
    background:#0f2218; border-left:3px solid #10b981;
    border-radius:10px 10px 10px 0; padding:.7rem .95rem;
    margin:.45rem 3rem .45rem 0; font-size:.9rem; color:#d1fae5;
    white-space:pre-wrap;
}
.msg-tag { font-family:'IBM Plex Mono',monospace; font-size:.62rem;
           letter-spacing:.07em; text-transform:uppercase; font-weight:700; margin-bottom:.2rem; }
.tag-u   { color:#60a5fa; }
.tag-a   { color:#34d399; }

/* ── Section header ── */
.sec-hdr {
    font-family:'IBM Plex Mono',monospace; font-size:.7rem;
    text-transform:uppercase; letter-spacing:.12em; color:#334155;
    border-bottom:1px solid #1c2a3e; padding-bottom:.45rem; margin-bottom:.9rem;
}

/* ── Status badge ── */
.badge { display:inline-block; padding:.18rem .55rem; border-radius:20px;
         font-family:'IBM Plex Mono',monospace; font-size:.65rem; font-weight:700; }
.b-ok  { background:#052e16; color:#34d399; border:1px solid #064e3b; }
.b-run { background:#1e1a03; color:#fbbf24; border:1px solid #78350f; }

/* ── Sidebar ── */
[data-testid="stSidebar"]         { background:#0a0e1a; border-right:1px solid #1c2a3e; }
[data-testid="stSidebar"] h2      { font-family:'IBM Plex Mono',monospace; font-size:.78rem;
                                    text-transform:uppercase; letter-spacing:.1em; color:#3b82f6; }

/* ── Buttons ── */
.stButton > button {
    background:#1d4ed8; color:#fff; border:none; border-radius:8px;
    font-family:'IBM Plex Sans',sans-serif; font-weight:600; width:100%;
    padding:.52rem 1rem; transition:background .18s,transform .1s;
}
.stButton > button:hover  { background:#2563eb; transform:translateY(-1px); }
.stButton > button:active { transform:translateY(0); }

/* run-agent button special */
.run-btn > button {
    background: linear-gradient(135deg,#065f46,#047857) !important;
    font-size:1rem !important; padding:.75rem !important;
    letter-spacing:.04em;
}
.run-btn > button:hover { background: linear-gradient(135deg,#047857,#059669) !important; }

/* ── Text input ── */
.stTextArea textarea, .stTextInput input {
    background:#0a0e1a !important; border:1px solid #1c2a3e !important;
    color:#c9d1d9 !important; border-radius:8px !important;
    font-family:'IBM Plex Sans',sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color:#3b82f6 !important;
    box-shadow:0 0 0 2px rgba(59,130,246,.18) !important;
}

/* ── Result box ── */
.result-box {
    background:#0d1320; border:1px solid #1c2a3e; border-radius:12px;
    padding:1.2rem 1.4rem; white-space:pre-wrap; font-size:.88rem;
    color:#e2f0ff; line-height:1.7; max-height:500px; overflow-y:auto;
}

/* ── Download area ── */
.dl-area {
    background:#0a1a0d; border:1px dashed #1a3a24; border-radius:10px;
    padding:1rem 1.2rem; margin-top:.8rem;
}

hr { border-color:#1c2a3e; }
.stDataFrame  { border:1px solid #1c2a3e !important; border-radius:8px !important; }
.stSpinner > div { border-top-color:#3b82f6 !important; }
.streamlit-expanderHeader { background:#0d1320 !important; color:#516070 !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource(show_spinner="⚙️  Initialising supply chain data and agent graph…")
def load_resources():
    sc = generate_data()
    graph, tools = build_graph(sc)
    return sc, graph, tools


@st.cache_data(show_spinner=False)
def compute_kpis(_sc: SupplyChainState):
    s = _sc
    zero   = int((s.central_inventory["current_stock"] == 0).sum())
    total  = len(s.sku_master)

    danger_df = s.central_inventory.merge(s.sku_master[["sku_id","danger_level"]], on="sku_id")
    danger = int((danger_df["current_stock"] < danger_df["danger_level"]).sum())

    pos = len(s.open_po)

    val_df = s.central_inventory.merge(s.sku_master[["sku_id","unit_cost"]], on="sku_id")
    inv_value = float((val_df["current_stock"] * val_df["unit_cost"]).sum())

    ad = s.demand_history.groupby("sku_id")["daily_demand"].mean().reset_index().rename(columns={"daily_demand":"avg_dd"})
    rr = (s.central_inventory
            .merge(s.sku_master[["sku_id","safety_stock","selling_price"]], on="sku_id")
            .merge(ad, on="sku_id"))
    at_risk = rr[rr["current_stock"] <= rr["safety_stock"]]
    weekly_risk = float((at_risk["avg_dd"] * at_risk["selling_price"] * 7).sum())

    return dict(zero=zero, total=total, danger=danger, pos=pos,
                inv_value=inv_value, weekly_risk=weekly_risk)


# ============================================================
# HELPERS
# ============================================================

def run_agent(query: str, sc_state, graph) -> str:
    """Invoke the LangGraph agent; return the final text response."""
    result = graph.invoke({
        "sc_state": sc_state,
        "messages": [HumanMessage(content=query)],
    })
    last = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)


def response_to_dataframe(text: str) -> pd.DataFrame | None:
    """
    Best-effort parse of a table-like agent response into a DataFrame for CSV download.
    Tries tab-separated, pipe-separated, and whitespace-separated detection.
    Returns None if parsing fails or result is not tabular.
    """
    lines = [l for l in text.strip().splitlines() if l.strip()]

    # ── 1. Pipe-delimited markdown table ──────────────────────────
    pipe_lines = [l for l in lines if "|" in l]
    if len(pipe_lines) >= 2:
        rows = []
        for l in pipe_lines:
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            if all(re.fullmatch(r"[-: ]+", c) for c in cells):
                continue                         # separator row
            rows.append(cells)
        if len(rows) >= 2:
            try:
                df = pd.DataFrame(rows[1:], columns=rows[0])
                if len(df.columns) >= 2:
                    return df
            except Exception:
                pass

    # ── 2. Two-or-more consistent whitespace columns ──────────────
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(text), sep=r"\s{2,}", engine="python")
        if len(df.columns) >= 2 and len(df) >= 1:
            return df
    except Exception:
        pass

    return None


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ============================================================
# SESSION STATE
# ============================================================

for key, default in [
    ("chat_history", []),
    ("last_answer", ""),
    ("last_df", None),
    ("running", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# LOAD
# ============================================================

sc_state, graph, all_tools = load_resources()
kpis = compute_kpis(sc_state)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📦 Supply Chain AI")
    st.markdown("---")

    status_html = (
        '<span class="badge b-run">● Running</span>'
        if st.session_state.running
        else '<span class="badge b-ok">● Agent Online</span>'
    )
    st.markdown(f"**Status** &nbsp; {status_html}", unsafe_allow_html=True)
    st.markdown(f"**{kpis['total']}** SKUs &nbsp;·&nbsp; **10** Stores &nbsp;·&nbsp; **180d** history")
    st.markdown("---")

    st.markdown("## Tools Available")
    for icon, label in [
        ("🔴","Zero Stock"), ("🟠","Reorder Alert"), ("⚠️","Danger Level Breach"),
        ("📉","Stockout Risk"), ("📋","PO Coverage"), ("🛒","Recommended Order Qty"),
        ("🔄","Store Transfer Opp."), ("🏪","Store Stockout Risk"),
        ("💀","Dead Stock"), ("📦","Overstock"), ("💸","Stockout Cost Est."),
        ("🔬","General Analytics"),
    ]:
        st.markdown(f"{icon} &nbsp;{label}", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat & Results"):
        st.session_state.chat_history = []
        st.session_state.last_answer  = ""
        st.session_state.last_df      = None
        _rerun()

    st.markdown("---")
    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:.68rem;color:#334155;">'
        'gpt-4o-mini · LangGraph<br>timeout=45s · retries=3</p>',
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="sec-hdr">Supply Chain Command Centre</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 style="font-family:\'IBM Plex Mono\',monospace;font-size:1.55rem;'
    'color:#f0f6fc;margin:0 0 .25rem 0;">Supply Chain AI Agent</h1>'
    '<p style="color:#516070;font-size:.87rem;margin:0 0 1rem 0;">'
    'Natural-language analytics · LangGraph + GPT-4o-mini · 12 tools</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ============================================================
# KPI STRIP
# ============================================================

st.markdown('<div class="sec-hdr">Live Inventory Snapshot</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""<div class="kpi-card red">
        <div class="kpi-label">Zero-Stock SKUs</div>
        <div class="kpi-value">{kpis['zero']}</div>
        <div class="kpi-sub">Central warehouse</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card amber">
        <div class="kpi-label">Danger Breaches</div>
        <div class="kpi-value">{kpis['danger']}</div>
        <div class="kpi-sub">Emergency reorder</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card blue">
        <div class="kpi-label">Open POs</div>
        <div class="kpi-value">{kpis['pos']}</div>
        <div class="kpi-sub">Pending delivery</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi-card green">
        <div class="kpi-label">Inventory Value</div>
        <div class="kpi-value">${kpis['inv_value']:,.0f}</div>
        <div class="kpi-sub">Central warehouse</div></div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="kpi-card purple">
        <div class="kpi-label">Weekly Rev. at Risk</div>
        <div class="kpi-value">${kpis['weekly_risk']:,.0f}</div>
        <div class="kpi-sub">Stockout exposure</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# AGENT CHAT SECTION
# ============================================================

st.markdown('<div class="sec-hdr">AI Agent</div>', unsafe_allow_html=True)

# ── Quick prompts ──────────────────────────────────────────
QUICK = [
    "How many SKUs have zero stock?",
    "Which SKUs breached danger level?",
    "Give all stocks with less than 10 days cover + recommended order qty",
    "What is the weekly revenue at risk from stockouts?",
    "Top 10 SKUs by margin",
    "Dead stock capital tied up (>180 days)?",
    "Store transfer opportunities to avoid new POs",
    "Overstock SKUs with highest monthly holding cost waste",
]

st.markdown("**Quick Prompts:**")
cols = st.columns(4)
for i, p in enumerate(QUICK):
    with cols[i % 4]:
        if st.button(p, key=f"qp_{i}"):
            st.session_state["pending_q"] = p
            _rerun()

st.markdown("---")

# ── Chat history ───────────────────────────────────────────
if st.session_state.chat_history:
    html = '<div class="chat-box">'
    for m in st.session_state.chat_history:
        txt = m["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        if m["role"] == "user":
            html += f'<div class="msg-user"><div class="msg-tag tag-u">You</div>{txt}</div>'
        else:
            html += f'<div class="msg-ai"><div class="msg-tag tag-a">Agent</div>{txt}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
else:
    st.markdown(
        '<div style="text-align:center;padding:2rem;color:#1c2a3e;">'
        '<span style="font-size:2.5rem">🤖</span><br>'
        '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:.78rem;color:#334155;">'
        'Type a query or pick a quick prompt above</span></div>',
        unsafe_allow_html=True,
    )

# ── Input row ──────────────────────────────────────────────
inp_col, btn_col = st.columns([5, 1])
with inp_col:
    user_query = st.text_input(
        "query",
        placeholder="e.g.  Which Class A SKUs will stock out before replenishment arrives?",
        label_visibility="collapsed",
        key="query_input",
        value=st.session_state.pop("pending_q", ""),
    )
with btn_col:
    # Use markdown wrapper to get the green styling
    st.markdown('<div class="run-btn">', unsafe_allow_html=True)
    run_clicked = st.button("▶ Run Agent", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# AGENT EXECUTION
# ============================================================

if run_clicked and user_query.strip():
    st.session_state.running = True
    query = user_query.strip()
    st.session_state.chat_history.append({"role": "user", "content": query})

    # ── Live status indicator ──────────────────────────────
    status_placeholder = st.empty()
    status_placeholder.markdown(
        '<div style="background:#1e1a03;border:1px solid #78350f;border-radius:8px;'
        'padding:.75rem 1rem;font-family:\'IBM Plex Mono\',monospace;font-size:.82rem;'
        'color:#fbbf24;">⏳ &nbsp;Agent running… calling tools and reasoning over your supply chain data</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("🔍  Agent is thinking — please wait…"):
        try:
            answer = run_agent(query, sc_state, graph)
        except Exception as e:
            answer = f"⚠️  Agent error: {e}"

    status_placeholder.empty()
    st.session_state.running = False
    st.session_state.last_answer = answer
    st.session_state.chat_history.append({"role": "ai", "content": answer})

    # Try to parse a DataFrame for download
    st.session_state.last_df = response_to_dataframe(answer)
    _rerun()


# ============================================================
# LATEST RESULT + CSV DOWNLOAD
# ============================================================

if st.session_state.last_answer:
    st.markdown("---")
    st.markdown('<div class="sec-hdr">Latest Agent Response</div>', unsafe_allow_html=True)

    # Formatted result box
    safe = (st.session_state.last_answer
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    st.markdown(f'<div class="result-box">{safe}</div>', unsafe_allow_html=True)

    # ── Download section ──────────────────────────────────
    st.markdown('<div class="dl-area">', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:.7rem;'
        'text-transform:uppercase;letter-spacing:.1em;color:#1a3a24;">📥 &nbsp; Export</span>',
        unsafe_allow_html=True,
    )

    dl_col1, dl_col2, dl_col3 = st.columns([2, 2, 4])

    # ── Download raw text ────────────────────────────────
    with dl_col1:
        st.download_button(
            label="⬇ Download as TXT",
            data=st.session_state.last_answer.encode("utf-8"),
            file_name="agent_response.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── Download parsed CSV (if table detected) ──────────
    with dl_col2:
        df = st.session_state.last_df
        if df is not None:
            st.download_button(
                label="⬇ Download as CSV",
                data=df_to_csv_bytes(df),
                file_name="agent_response.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            # Try to extract any list/numbers into a minimal CSV
            lines = [l.strip() for l in st.session_state.last_answer.splitlines() if l.strip()]
            fallback_df = pd.DataFrame({"response_line": lines})
            st.download_button(
                label="⬇ Download as CSV",
                data=df_to_csv_bytes(fallback_df),
                file_name="agent_response.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with dl_col3:
        if st.session_state.last_df is not None:
            st.markdown(
                f'<p style="font-size:.78rem;color:#1a3a24;margin:.4rem 0 0 0;">'
                f'✅ Tabular data detected — {len(st.session_state.last_df)} rows, '
                f'{len(st.session_state.last_df.columns)} columns parsed for CSV</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="font-size:.78rem;color:#334155;margin:.4rem 0 0 0;">'
                'Response exported as line-by-line CSV (no table structure detected)</p>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# DATA EXPLORER
# ============================================================

st.markdown("---")
st.markdown('<div class="sec-hdr">Data Explorer</div>', unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.tabs(
    ["SKU Master", "Central Inventory", "Store Inventory", "Open POs", "Inventory Aging"]
)

with t1:
    df = sc_state.sku_master.copy()
    df[["unit_cost","selling_price"]] = df[["unit_cost","selling_price"]].round(2)
    st.dataframe(df, use_container_width=True, height=300)
    st.download_button("⬇ CSV", df_to_csv_bytes(df), "sku_master.csv", "text/csv")

with t2:
    df = sc_state.central_inventory.copy()
    df["last_updated"] = df["last_updated"].dt.strftime("%Y-%m-%d")
    st.dataframe(df, use_container_width=True, height=300)
    st.download_button("⬇ CSV", df_to_csv_bytes(df), "central_inventory.csv", "text/csv")

with t3:
    df = sc_state.store_inventory.copy()
    st.dataframe(df, use_container_width=True, height=300)
    st.download_button("⬇ CSV", df_to_csv_bytes(df), "store_inventory.csv", "text/csv")

with t4:
    df = sc_state.open_po.copy()
    df["expected_delivery_date"] = df["expected_delivery_date"].dt.strftime("%Y-%m-%d")
    st.dataframe(df, use_container_width=True, height=300)
    st.download_button("⬇ CSV", df_to_csv_bytes(df), "open_po.csv", "text/csv")

with t5:
    df = sc_state.inventory_aging.copy()
    st.dataframe(df, use_container_width=True, height=300)
    st.download_button("⬇ CSV", df_to_csv_bytes(df), "inventory_aging.csv", "text/csv")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-family:\'IBM Plex Mono\',monospace;'
    'font-size:.67rem;color:#1c2a3e;padding:.4rem 0;">'
    'Supply Chain AI Agent &nbsp;·&nbsp; LangGraph + GPT-4o-mini &nbsp;·&nbsp; '
    '500 SKUs · 10 Stores · 180-day history &nbsp;·&nbsp; 12 tools</div>',
    unsafe_allow_html=True,
)