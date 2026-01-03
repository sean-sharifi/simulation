import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ============================================================
# CONFIGURATION & STYLING
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
      }
      section[data-testid="stSidebar"] { background-color: #0f1626; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TITLE
# ============================================================
st.title("🛡️ Survival First: Trading Risk Simulator")
st.markdown("### Monte Carlo simulation for **risk of ruin** & **path dependency** (Notebook-style)")

# ============================================================
# HELPERS
# ============================================================
def breakeven_win_rate(win_ret: float, loss_ret: float, fee: float) -> float:
    """
    Matches the simulator's payoff model:
      Win net on stake  = (win_ret - fee)
      Loss net on stake = -(loss_ret + fee)

    Breakeven occurs when expected arithmetic return on stake is 0:
      p*(win_ret - fee) + (1-p)*(-loss_ret - fee) = 0
    => p = (loss_ret + fee) / (win_ret + loss_ret)
    """
    denom = win_ret + loss_ret
    if denom <= 0:
        return np.nan
    return (loss_ret + fee) / denom


def kelly_fraction_exact(win_rate: float, win_ret: float, loss_ret: float, fee: float) -> float:
    """
    EXACT Kelly for this simulator's multiplicative model:

      Win multiplier  = 1 + f*(win_ret - fee)
      Loss multiplier = 1 - f*(loss_ret + fee)

    Let:
      W = win_ret - fee
      L = loss_ret + fee  (positive)

    Maximize expected log growth:
      g(f) = p ln(1 + fW) + q ln(1 - fL)

    Closed-form optimum:
      f* = (pW - qL) / (W*L)

    If <= 0 => no mathematical edge.
    """
    p = win_rate
    q = 1 - p
    W = win_ret - fee
    L = loss_ret + fee
    if W <= 0 or L <= 0:
        return 0.0
    f_star = (p * W - q * L) / (W * L)
    return float(max(f_star, 0.0))


def expected_log_growth_per_trade(win_rate: float, f: float, win_ret: float, loss_ret: float, fee: float) -> float:
    """
    Expected log growth per trade under the simulator model.
    """
    p = win_rate
    q = 1 - p
    up = 1 + f * (win_ret - fee)
    dn = 1 - f * (loss_ret + fee)
    if up <= 0 or dn <= 0:
        return float("-inf")
    return float(p * np.log(up) + q * np.log(dn))


@st.cache_data(show_spinner=False)
def run_monte_carlo(
    sims: int,
    trades: int,
    init_money: float,
    win_rate: float,
    win_ret: float,
    loss_ret: float,
    pos_size: float,
    fee: float,
    ruin_dd: float,
    seed: int,
):
    """
    Notebook-style simulation:
      trade_amount = current_cap * pos_size
      result       = trade_amount * (+win_ret or -loss_ret)
      fee_paid     = trade_amount * fee
      current_cap += result - fee_paid

    Ruin event:
      capital ever <= init_money * (1 - ruin_dd)
    """
    rng = np.random.default_rng(seed)

    ruin_limit = init_money * (1 - ruin_dd)

    # Store paths (float32 for memory friendliness)
    paths = np.empty((sims, trades + 1), dtype=np.float32)
    ruined = np.zeros(sims, dtype=bool)

    for i in range(sims):
        cap = float(init_money)
        paths[i, 0] = cap
        hit_ruin = False

        # Generate all trade outcomes up front
        # outcome is +win_ret with prob p, -loss_ret with prob q
        u = rng.random(trades)
        is_win = u < win_rate

        for t in range(trades):
            if cap <= 0:
                cap = 0.0
                paths[i, t + 1] = 0.0
                hit_ruin = True
                continue

            trade_amount = cap * pos_size

            # PnL on stake
            ret = win_ret if is_win[t] else -loss_ret
            pnl = trade_amount * ret

            # Fee on stake (every trade)
            fee_paid = trade_amount * fee

            cap = cap + pnl - fee_paid
            if cap < 0:
                cap = 0.0

            if cap <= ruin_limit:
                hit_ruin = True

            paths[i, t + 1] = cap

        ruined[i] = hit_ruin

    risk_of_ruin = 100.0 * ruined.mean()
    ruin_count = int(ruined.sum())
    return paths, risk_of_ruin, ruin_count, ruin_limit


# ============================================================
# SIDEBAR INPUTS
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

    win_return = st.number_input("Take Profit (Win %) ", value=2.0, step=0.1, min_value=0.0) / 100.0
    loss_return = st.number_input("Stop Loss (Loss %) ", value=2.0, step=0.1, min_value=0.0) / 100.0

    st.divider()
    st.header("3. Risk Management")
    position_size = st.slider("Position Size (% of Capital)", 0.0, 100.0, 10.0) / 100.0

    # fee with 4 decimal precision in percent units
    fee_input = st.number_input("Trading Fee per Trade (%)", value=0.0015, step=0.0001, format="%.4f")
    fee_rate = fee_input / 100.0

    ruin_threshold = st.slider("Ruin Threshold (% Drawdown)", 0, 100, 40) / 100.0

    # --- Correct Break-even Win Rate (uses ACTUAL fee_rate)
    be_wr = breakeven_win_rate(win_return, loss_return, fee_rate)
    if np.isfinite(be_wr):
        st.info(f"**Break-even Win Rate:** {be_wr*100:.2f}%")
    else:
        st.info("**Break-even Win Rate:** N/A")

    # --- Correct Kelly (exact for this model)
    kelly_pct = kelly_fraction_exact(win_rate, win_return, loss_return, fee_rate)

    if kelly_pct <= 0:
        st.error("Strategy has no mathematical edge (Negative Kelly).")
        fraction_of_kelly = np.nan
    else:
        fraction_of_kelly = position_size / kelly_pct if kelly_pct > 0 else np.nan
        st.write(f"**Kelly Optimal Size:** {kelly_pct*100:.2f}%")
        st.write(f"**Your Allocation:** {fraction_of_kelly:.2f}x Kelly")

# ============================================================
# RUN SIMULATION
# ============================================================
paths, risk_of_ruin, ruin_count, ruin_limit = run_monte_carlo(
    simulations,
    trades_count,
    initial_money,
    win_rate,
    win_return,
    loss_return,
    position_size,
    fee_rate,
    ruin_threshold,
    seed,
)

final_balances = paths[:, -1].astype(np.float64)
median_end = float(np.median(final_balances))
avg_end = float(np.mean(final_balances))

# Confidence bound when you observe 0 events
upper95 = None
if ruin_count == 0:
    # "Rule of 3": if 0 events in N trials, 95% upper bound ≈ 3/N
    upper95 = 3.0 / float(simulations) * 100.0

# Growth diagnostic
g = expected_log_growth_per_trade(win_rate, position_size, win_return, loss_return, fee_rate)

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
    st.caption(f"Observed **0** ruin events in {simulations} simulations. 95% upper bound on ruin prob ≈ **{upper95:.2f}%** (rule of 3).")

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
    sample_size = int(min(60, simulations))
    # sample evenly spaced indices for stable display
    if simulations > sample_size:
        idx = np.linspace(0, simulations - 1, sample_size).astype(int)
    else:
        idx = np.arange(simulations)

    for i in idx:
        fig.add_trace(
            go.Scatter(
                y=paths[i],
                mode="lines",
                opacity=0.25,
                line=dict(width=1),
                showlegend=False,
            )
        )

    median_path = np.median(paths, axis=0)
    fig.add_trace(
        go.Scatter(
            y=median_path,
            name="Median Path",
            line=dict(width=3),
        )
    )

    fig.add_hline(
        y=ruin_limit,
        line_dash="dash",
        annotation_text="Ruin Threshold",
    )

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
# FOOTER & ASSUMPTIONS
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
- **No Adverse Selection:** You are not being “picked off” by smarter flows.
- **Static Regime:** No volatility/regime shifts.
- **Zero Latency:** Instant execution & data.
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
