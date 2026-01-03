import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Trading Risk Simulator", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric {background-color: #333333; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- APP TITLE ---
st.title("🛡️ Survival First: Trading Risk Simulator")
st.markdown("### Monte Carlo Simulation for Risk of Ruin & Path Dependency")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. Simulation Settings")
    initial_money = st.number_input("Initial Capital ($)", value=1000, step=100)
    trades_count = st.number_input("Number of Trades", value=365, step=10)
    simulations = st.number_input("Number of Universes (Simulations)", value=1000, step=100)
    
    st.divider()
    st.header("2. Strategy Edge")
    win_rate = st.slider("Win Rate (%)", 0.0, 100.0, 52.0) / 100
    win_return = st.number_input("Take Profit (Win %) ", value=2.0, step=0.1) / 100
    loss_return = st.number_input("Stop Loss (Loss %) ", value=2.0, step=0.1) / 100
    
    # Calculation for Break-even Edge
    # Formula: p*(Win - Fee) - (1-p)*(Loss + Fee) = 0
    # p = (Loss + Fee) / (Win + Loss)
    # Note: We use the fee_rate defined below, but since it's in the sidebar, we calculate it here.
    temp_fee = 0.0015 / 100 
    be_win_rate = (loss_return + temp_fee) / (win_return + loss_return)
    st.info(f"**Break-even Win Rate:** {be_win_rate*100:.2f}%")
    
    st.divider()
    st.header("3. Risk Management")
    position_size = st.slider("Position Size (% of Capital)", 0.0, 100.0, 10.0) / 100
    
    # Trading Fee with 4 decimal precision
    fee_input = st.number_input("Trading Fee per Trade (%)", value=0.0015, step=0.0001, format="%.4f")
    fee_rate = fee_input / 100
    
    ruin_threshold = st.slider("Ruin Threshold (% Drawdown)", 0, 100, 40) / 100

    # Kelly Criterion Calculation
    # Adjusted for fees: b = (Win - Fee) / (Loss + Fee)
    net_win = win_return - fee_rate
    net_loss = loss_return + fee_rate
    if net_loss > 0:
        b_ratio = net_win / net_loss
        kelly_pct = (win_rate * b_ratio - (1 - win_rate)) / b_ratio
    else:
        kelly_pct = 0

    if kelly_pct <= 0:
        st.error("Strategy has no mathematical edge (Negative Kelly).")
    else:
        fraction_of_kelly = position_size / kelly_pct
        st.write(f"**Kelly Optimal Size:** {kelly_pct*100:.2f}%")
        st.write(f"**Your Allocation:** {fraction_of_kelly:.2f}x Kelly")

# --- SIMULATION ENGINE ---
@st.cache_data
def run_monte_carlo(sims, trades, init_money, w_rate, w_ret, l_ret, p_size, f_rate, r_thresh):
    all_paths = []
    ruin_count = 0
    ruin_limit = init_money * (1 - r_thresh)
    
    for i in range(sims):
        capital = [float(init_money)]
        current_cap = float(init_money)
        
        # Random outcomes
        results = np.random.choice(
            [w_ret, -l_ret], 
            size=trades, 
            p=[w_rate, 1 - w_rate]
        )
        
        path_ruined = False
        for res in results:
            if current_cap <= 1: 
                current_cap = 0
                capital.append(0)
                path_ruined = True
                continue
            
            trade_amount = current_cap * p_size
            net_change = (trade_amount * res) - (trade_amount * f_rate)
            current_cap += net_change
            
            if current_cap <= ruin_limit:
                path_ruined = True
                
            capital.append(max(current_cap, 0))
        
        if path_ruined:
            ruin_count += 1
        all_paths.append(capital)
            
    return np.array(all_paths), (ruin_count / sims) * 100

# --- EXECUTION ---
paths, risk_of_ruin = run_monte_carlo(simulations, trades_count, initial_money, win_rate, win_return, loss_return, position_size, fee_rate, ruin_threshold)
final_balances = paths[:, -1]

# Stats
median_end = np.median(final_balances)
avg_end = np.mean(final_balances)

# --- UI LAYOUT ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Risk of Ruin", f"{risk_of_ruin:.1f}%", delta="High Risk" if risk_of_ruin > 10 else "Safe", delta_color="inverse")
col2.metric("Median Final Balance", f"${median_end:,.2f}")
col3.metric("Expected Value (Avg)", f"${avg_end:,.2f}")
col4.metric("Survival Rate", f"{100 - risk_of_ruin:.1f}%")

# --- VISUALIZATION ---
st.divider()
viz_col1, viz_col2 = st.columns([2, 1])

with viz_col1:
    st.subheader("Equity Path Simulation (Log Scale)")
    fig = go.Figure()
    sample_size = min(50, simulations)
    for i in range(sample_size):
        fig.add_trace(go.Scatter(y=paths[i], mode='lines', opacity=0.3, line=dict(width=1), showlegend=False))
    
    fig.add_trace(go.Scatter(y=np.median(paths, axis=0), name='Median Path', line=dict(color='gold', width=3)))
    fig.add_hline(y=initial_money * (1 - ruin_threshold), line_dash="dash", line_color="red", annotation_text="Ruin Threshold")
    fig.update_layout(yaxis_type="log", xaxis_title="Trade Number", yaxis_title="Capital ($)", height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with viz_col2:
    st.subheader("Outcome Distribution")
    hist_fig = go.Figure(data=[go.Histogram(x=final_balances, nbinsx=30, marker_color='#636EFA')])
    hist_fig.update_layout(xaxis_title="Final Capital", yaxis_title="Frequency", height=500, template="plotly_dark")
    st.plotly_chart(hist_fig, use_container_width=True)

# --- FOOTER & ASSUMPTIONS ---
st.divider()
st.subheader("📌 Simulation Assumptions & Limitations")
st.caption("""
This simulation is a mathematical model based on the following idealized conditions:
- **Perfect Execution:** You can always place the stop loss and take profit at the selected amount and get filled exactly there.
- **Stationary Probability:** The probability of hitting TP vs SL is stable (e.g., exactly 50/50) throughout the entire sequence.
- **Continuous Liquidity:** There are no price gaps (e.g., overnight gaps or news spikes).
- **Zero Slippage:** Market orders are filled exactly at the target price despite position size.
- **No Adverse Selection:** Your orders do not impact the market price, and you aren't being picked off by HFTs.
- **Static Regime:** No regime changes; market volatility and behavior remain constant.
- **Zero Latency:** Trade execution and data feeds are instantaneous.
- **Independence:** No correlation between trades; each trade outcome is an independent event.
- **Normal Tails:** There are no 'Black Swan' or extreme tail events beyond the win/loss parameters provided.
""")

with st.expander("📝 How to interpret the Kelly Criterion"):
    st.write(f"""
    - **1.0x Kelly:** The mathematically optimal risk size to maximize long-term wealth growth.
    - **> 1.0x Kelly:** "Over-betting." This significantly increases your Risk of Ruin and leads to lower long-term wealth than 1.0x Kelly.
    - **0.5x Kelly:** Often called "Fractional Kelly," this is a common professional standard that provides a safety buffer while capturing most of the growth potential.
    """)
