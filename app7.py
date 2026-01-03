import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Trading Risk Simulator", layout="wide")

st.markdown(
    """
    <style>
      .main { background-color: #0b0f17; }
      .stMetric {
        background-color: #1a2233;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        color: white;
        .risk-pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  margin-top: 6px;
}
.risk-low { background: rgba(34,197,94,0.18); color: #22c55e; border: 1px solid rgba(34,197,94,0.35); }
.risk-med { background: rgba(245,158,11,0.18); color: #f59e0b; border: 1px solid rgba(245,158,11,0.35); }
.risk-high { background: rgba(239,68,68,0.18); color: #ef4444; border: 1px solid rgba(239,68,68,0.35); }

      }
      section[data-testid="stSidebar"] { background-color: #0f1626; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ Survival First: Trading Risk Simulator")
st.markdown("### Monte Carlo simulation for **risk of ruin** & **path dependency** (Notebook-style)")

# ============================================================
# MATH HELPERS (consistent with simulator)
# ============================================================
def breakeven_win_rate(tp: float, sl: float, fee_round: float) -> float:
    """
    Payoff model per trade on allocated fraction f of capital:

      win multiplier  = 1 + f*(tp - fee_round)
      loss multiplier = 1 - f*(sl + fee_round)

    Breakeven for arithmetic expected return on stake (f cancels):
      p*(tp - fee_round) + (1-p)*(-sl - fee_round) = 0
      => p = (sl + fee_round) / (tp + sl)
    """
    denom = tp + sl
    if denom <= 0:
        return np.nan
    return (sl + fee_round) / denom


def kelly_fraction_exact(p: float, tp: float, sl: float, fee_round: float) -> float:
    """
    Exact Kelly for this simulator’s multiplicative model:

      g(f)= p ln(1 + f*(tp - fee)) + (1-p) ln(1 - f*(sl + fee))

    Let W = tp - fee, L = sl + fee (both should be positive).
    Closed-form optimum:
      f* = (pW - (1-p)L) / (W*L)

    Returns 0 if no edge or invalid inputs.
    """
    q = 1 - p
    W = tp - fee_round
    L = sl + fee_round
    if W <= 0 or L <= 0:
        return 0.0
    f_star = (p * W - q * L) / (W * L)
    return float(max(f_star, 0.0))


def expected_log_growth_per_trade(p: float, f: float, tp: float, sl: float, fee_round: float) -> float:
    q = 1 - p
    up = 1 + f * (tp - fee_round)
    dn = 1 - f * (sl + fee_round)
    if up <= 0 or dn <= 0:
        return float("-inf")
    return float(p * np.log(up) + q * np.log(dn))


@st.cache_data(show_spinner=False)
def run_monte_carlo(
    sims: int,
    trades: int,
    init_money: float,
    p: float,
    tp: float,
    sl: float,
    f: float,
    fee_round: float,
    ruin_dd: float,
    seed: int,
):
    """
    Vectorized across simulations; loops only over trades.

    Capital update each trade:
      cap <- cap * (1 + f*(tp - fee))   if win
      cap <- cap * (1 - f*(sl + fee))   if loss

    Ruin event if cap ever <= init_money*(1-ruin_dd)
    """
    rng = np.random.default_rng(int(seed))
    ruin_limit = init_money * (1.0 - ruin_dd)

    wins = rng.random((sims, trades)) < p  # bool matrix

    caps = np.full(sims, init_money, dtype=np.float64)
    paths = np.empty((sims, trades + 1), dtype=np.float32)
    paths[:, 0] = caps.astype(np.float32)

    ruined = np.zeros(sims, dtype=bool)

    win_mult = 1.0 + f * (tp - fee_round)
    loss_mult = 1.0 - f * (sl + fee_round)

    # If either multiplier is non-positive, log/ruin interpretation breaks; still simulate with clamp at 0.
    for t in range(trades):
        mult = np.where(wins[:, t], win_mult, loss_mult)
        caps = caps * mult
        caps = np.maximum(caps, 0.0)
        ruined |= (caps <= ruin_limit)
        paths[:, t + 1] = caps.astype(np.float32)

    risk_of_ruin = 100.0 * ruined.mean()
    ruin_count = int(ruined.sum())
    return paths, caps, ruined, risk_of_ruin, ruin_count, ruin_limit

# ============================================================
# SIDEBAR INPUTS (defaults match your NOTEBOOK)
# ============================================================
with st.sidebar:
    st.header("1. Simulation Settings")
    initial_money = st.number_input("Initial Capital ($)", value=1000.0, step=100.0, min_value=1.0)
    trades_count = st.number_input("Number of Trades", value=365, step=10, min_value=1)
    simulations = st.number_input("Number of Universes (Simulations)", value=1000, step=100, min_value=100)
    seed = st.number_input("Random Seed", value=42, step=1, min_value=0)

    st.divider()
    st.header("2. Strategy Edge")
    win_rate = st.slider("Win Rate (%)", 0.0, 100.0, 52.0) / 100.0

    tp = st.number_input("Take Profit (Win % move)", value=2.0, step=0.1, min_value=0.0) / 100.0
    sl = st.number_input("Stop Loss (Loss % move)", value=2.0, step=0.1, min_value=0.0) / 100.0

    st.divider()
    st.header("3. Risk Management")

    # In your notebook this was 0.10
    position_size = st.slider("Position Size (% of Capital)", 0.0, 200.0, 10.0) / 100.0

    # ✅ FIXED: default 0.15% round-trip fee (matches notebook fee_per_trade = 0.0015)
    fee_pct_round = st.number_input(
        "Trading Fee (round-trip, % of position)",
        value=0.15, step=0.01, min_value=0.0, format="%.4f"
    )
    fee_round = fee_pct_round / 100.0

    ruin_threshold = st.slider("Ruin Threshold (% Drawdown)", 0, 100, 40) / 100.0

    st.divider()
    st.header("4. Kelly (model-consistent)")
    be = breakeven_win_rate(tp, sl, fee_round)
    if np.isfinite(be):
        st.info(f"**Break-even Win Rate:** {be*100:.2f}%")
    else:
        st.info("**Break-even Win Rate:** N/A")

    kelly = kelly_fraction_exact(win_rate, tp, sl, fee_round)

    if kelly <= 0:
        st.error("No mathematical edge under this model (Kelly ≤ 0).")
        kelly_capped = 0.0
    else:
        # Spot cap (100%) is often what people EXPECT; allow user to choose.
        max_f = st.slider("Max bet cap (for Kelly display)", 0.0, 5.0, 1.0, step=0.1)
        kelly_capped = min(kelly, max_f)

        st.write(f"**Kelly (uncapped):** {kelly*100:.2f}%")
        st.write(f"**Kelly (capped at {max_f:.1f}×):** {kelly_capped*100:.2f}%")
        st.write(f"**Your Allocation:** {position_size / kelly:.2f}× Kelly (uncapped)")

# ============================================================
# RUN
# ============================================================
paths, final_caps, ruined, risk_of_ruin, ruin_count, ruin_limit = run_monte_carlo(
    sims=int(simulations),
    trades=int(trades_count),
    init_money=float(initial_money),
    p=float(win_rate),
    tp=float(tp),
    sl=float(sl),
    f=float(position_size),
    fee_round=float(fee_round),
    ruin_dd=float(ruin_threshold),
    seed=int(seed),
)

final_balances = final_caps.astype(np.float64)
median_end = float(np.median(final_balances))
avg_end = float(np.mean(final_balances))

upper95 = None
if ruin_count == 0:
    upper95 = 3.0 / float(simulations) * 100.0  # rule of 3

g = expected_log_growth_per_trade(win_rate, position_size, tp, sl, fee_round)

# ============================================================
# METRICS
# ============================================================
col1, col2, col3, col4 = st.columns(4)
delta_label = "High Risk" if risk_of_ruin > 10 else "Low/Moderate Risk"
col1.metric("Risk of Ruin", f"{risk_of_ruin:.2f}%", delta=delta_label, delta_color="inverse")
col2.metric("Median Final Balance", f"${median_end:,.2f}")
col3.metric("Expected Value (Mean)", f"${avg_end:,.2f}")
col4.metric("Survival Rate", f"{100.0 - risk_of_ruin:.2f}%")

if upper95 is not None:
    st.caption(
        f"Observed **0** ruin events in {int(simulations)} simulations. "
        f"95% upper bound on ruin probability ≈ **{upper95:.2f}%** (rule of 3)."
    )

st.caption(
    f"Ruin line = ${ruin_limit:,.2f} (initial × (1 − {ruin_threshold:.2f})) | "
    f"Expected log-growth per trade ≈ {g:.6f}"
)

# ============================================================
# VISUALIZATION
# ============================================================
st.divider()
viz_col1, viz_col2 = st.columns([2, 1])

with viz_col1:
    st.subheader("Equity Path Simulation (Log Scale)")
    fig = go.Figure()

    sims = int(simulations)
    sample_size = int(min(60, sims))
    idx = np.linspace(0, sims - 1, sample_size).astype(int) if sims > sample_size else np.arange(sims)

    for i in idx:
        fig.add_trace(go.Scatter(y=paths[i], mode="lines", opacity=0.25, line=dict(width=1), showlegend=False))

    median_path = np.median(paths, axis=0)
    fig.add_trace(go.Scatter(y=median_path, name="Median Path", line=dict(width=3)))

    fig.add_hline(y=ruin_limit, line_dash="dash", annotation_text="Ruin Threshold")

    fig.update_layout(
        yaxis_type="log",
        xaxis_title="Trade Number",
        yaxis_title="Capital ($)",
        height=520,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

with viz_col2:
    st.subheader("Outcome Distribution")
    hist_fig = go.Figure(data=[go.Histogram(x=final_balances, nbinsx=35)])
    hist_fig.update_layout(
        xaxis_title="Final Capital",
        yaxis_title="Frequency",
        height=520,
        template="plotly_dark",
    )
    st.plotly_chart(hist_fig, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.subheader("📌 Simulation Assumptions & Limitations")
st.caption(
    """
This simulation is an idealized mathematical model:

- **Perfect Execution:** Stops/targets fill exactly at the chosen prices.
- **Stationary Probability:** Win probability stays constant throughout the sequence.
- **Continuous Liquidity:** No gaps, no news jumps.
- **Zero Slippage:** Fills are not degraded by market impact/spread.
- **Independence:** Trades are IID (no clustering of wins/losses).
- **No Fat Tails:** Losses are limited exactly to Stop Loss %.
"""
)

with st.expander("📝 How to interpret the Kelly Criterion"):
    st.write(
        """
- **1.0× Kelly:** maximizes long-run expected log growth *under the model assumptions*.
- **> 1.0× Kelly:** over-betting—higher drawdowns and often worse long-run outcomes.
- **Fractional Kelly (½, ¼):** common professional practice to reduce drawdowns and model error risk.
        """
    )
