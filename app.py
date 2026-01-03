import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Trading Risk Simulator", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #dddddd; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- APP TITLE ---
st.title("🛡️ Survival First: Trading Risk Simulator")
st.markdown("### Monte Carlo Simulation for Risk of Ruin & Path Dependency")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Simulation Settings")
    
    initial_money = st.number_input("Initial Capital ($)", value=1000, step=100)
    trades_count = st.number_input("Number of Trades", value=365, step=10)
    simulations = st.number_input("Number of Universes (Simulations)", value=1000, step=100)
    
    st.divider()
    st.header("Strategy Edge eee")
    
    win_rate = st.slider("Win Rate (%)", 0, 100, 52) / 100
    win_return = st.number_input("Take Profit (Win %) ", value=2.0, step=0.1) / 100
    loss_return = st.number_input("Stop Loss (Loss %) ", value=2.0, step=0.1) / 100
    
    st.divider()
    st.header("Risk Management")
    
    position_size = st.slider("Position Size (% of Capital)", 0.0, 100.0, 10.0) / 100
    fee_rate = st.number_input("Trading Fee per Trade (%)", value=0.015, step=0.001) / 100
    ruin_threshold = st.slider("Ruin Threshold (% Drawdown)", 0, 100, 40) / 100

# --- SIMULATION ENGINE ---
def run_monte_carlo():
    # Progress bar
    progress_bar = st.progress(0)
    
    all_paths = []
    ruin_count = 0
    ruin_limit = initial_money * (1 - ruin_threshold)
    
    for i in range(simulations):
        capital = [float(initial_money)]
        current_cap = float(initial_money)
        
        # Vectorized random choices for speed
        results = np.random.choice(
            [win_return, -loss_return], 
            size=trades_count, 
            p=[win_rate, 1 - win_rate]
        )
        
        path_ruined = False
        for res in results:
            if current_cap <= 1: # Absolute floor
                current_cap = 0
                capital.append(0)
                path_ruined = True
                continue
            
            # Profit/Loss Calculation
            trade_amount = current_cap * position_size
            net_change = (trade_amount * res) - (trade_amount * fee_rate)
            current_cap += net_change
            
            # Track Ruin
            if current_cap <= ruin_limit:
                path_ruined = True
                
            capital.append(max(current_cap, 0))
        
        if path_ruined:
            ruin_count += 1
            
        all_paths.append(capital)
        if i % (simulations // 10) == 0:
            progress_bar.progress(i / simulations)
            
    progress_bar.empty()
    return np.array(all_paths), (ruin_count / simulations) * 100

# --- RUN & RESULTS ---
paths, risk_of_ruin = run_monte_carlo()
final_balances = paths[:, -1]

# Calculate Summary Stats
median_end = np.median(final_balances)
avg_end = np.mean(final_balances)
max_drawdown = ((np.max(paths) - np.min(paths)) / np.max(paths)) * 100

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
    
    # Plot a sample of paths (to keep browser performance high)
    sample_size = min(50, simulations)
    for i in range(sample_size):
        fig.add_trace(go.Scatter(y=paths[i], mode='lines', opacity=0.3, line=dict(width=1), showlegend=False))
    
    # Highlight Median Path
    fig.add_trace(go.Scatter(y=np.median(paths, axis=0), name='Median Path', line=dict(color='gold', width=3)))
    
    # Add Ruin Line
    fig.add_hline(y=initial_money * (1 - ruin_threshold), line_dash="dash", line_color="red", annotation_text="Ruin Threshold")
    
    fig.update_layout(yaxis_type="log", xaxis_title="Trade Number", yaxis_title="Capital ($)", height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with viz_col2:
    st.subheader("Outcome Distribution")
    hist_fig = go.Figure(data=[go.Histogram(x=final_balances, nbinsx=30, marker_color='#636EFA')])
    hist_fig.update_layout(xaxis_title="Final Capital", yaxis_title="Frequency", height=500, template="plotly_dark")
    st.plotly_chart(hist_fig, use_container_width=True)

# --- EDUCATIONAL INSIGHT ---
with st.expander("📝 How to interpret this data"):
    st.write(f"""
    1. **Expected Value vs. Median:** If your Average is much higher than your Median, your strategy is "lottery-like"—a few lucky paths make the average look good, but most traders actually end up with the lower median amount.
    2. **Risk of Ruin:** At a {position_size*100}% position size, you have a **{risk_of_ruin:.1f}%** chance of hitting a {ruin_threshold*100}% drawdown. 
    3. **The Fee Trap:** You are paying {fee_rate*100}% per trade. Over {trades_count} trades, your friction costs are significant.
    """)
