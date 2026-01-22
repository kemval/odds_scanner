import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="⚽ Sports Betting Market Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light mode
st.markdown("""
<script>
    var elements = window.parent.document.querySelectorAll('.stApp');
    elements[0].classList.add('light-mode');
</script>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚽ Sports Betting Market Efficiency Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# OPTIMIZED DATA LOADING with caching and selective loading
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_processed_data():
    """Load processed odds data with optimized column selection."""
    try:
        # Only load essential columns for faster loading
        essential_cols = [
            'HomeTeam', 'AwayTeam', 'FTR', 'League', 'Season',
            'B365H', 'B365D', 'B365A', 
            'B365_margin', 'B365_true_prob_H', 'B365_true_prob_D', 'B365_true_prob_A',
            'Match_Type', 'Home_Favorite', 'Favorite_Strength'
        ]
        
        df = pd.read_csv('data/processed/odds_processed.csv')
        
        # Filter to only essential columns that exist
        available_cols = [col for col in essential_cols if col in df.columns]
        df = df[available_cols]
        
        # Optimize data types for memory efficiency
        if 'League' in df.columns:
            df['League'] = df['League'].astype('category')
        if 'Season' in df.columns:
            df['Season'] = df['Season'].astype('category')
        if 'FTR' in df.columns:
            df['FTR'] = df['FTR'].astype('category')
        if 'Match_Type' in df.columns:
            df['Match_Type'] = df['Match_Type'].astype('category')
            
        return df
    except FileNotFoundError:
        st.error("❌ Processed data not found. Please run `python src/data_processing.py` first.")
        return None

@st.cache_data(ttl=3600)
def load_backtest_results():
    """Load backtest results."""
    try:
        df = pd.read_csv('results/backtest_results.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_data(ttl=3600)
def load_inefficiencies():
    """Load market inefficiency analysis."""
    try:
        with open('results/inefficiencies.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def filter_data(df, league='All', season='All', outcome='All'):
    """Cached filtering function to avoid redundant operations."""
    filtered = df.copy()
    
    if league != 'All' and 'League' in df.columns:
        filtered = filtered[filtered['League'] == league]
    if season != 'All' and 'Season' in df.columns:
        filtered = filtered[filtered['Season'] == season]
    if outcome != 'All' and 'FTR' in df.columns:
        filtered = filtered[filtered['FTR'] == outcome]
    
    return filtered

@st.cache_data
def get_league_stats(df):
    """Pre-calculate league statistics."""
    if 'League' not in df.columns:
        return None
    
    stats = df.groupby('League').agg({
        'FTR': 'count',
        'B365_margin': 'mean' if 'B365_margin' in df.columns else lambda x: 0
    }).reset_index()
    stats.columns = ['League', 'Matches', 'Avg_Margin']
    return stats

# Load data with spinner
with st.spinner('Loading data...'):
    df = load_processed_data()
    backtest_df = load_backtest_results()
    inefficiencies = load_inefficiencies()

if df is None:
    st.stop()

# Initialize session state for performance
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = df

# Sidebar
st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.markdown("---")

# Page selection
page = st.sidebar.radio(
    "Select View",
    ["📊 Overview", "🔍 Data Explorer", "📈 Strategy Performance", "⚖️ Market Inefficiencies", "🎲 Match Analysis", "📸 Visualizations"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Dataset Info")
st.sidebar.metric("Total Matches", f"{len(df):,}")
st.sidebar.metric("Leagues", df['League'].nunique() if 'League' in df.columns else "N/A")
st.sidebar.metric("Seasons", df['Season'].nunique() if 'League' in df.columns else "N/A")

# Performance toggle
st.sidebar.markdown("---")
use_sampling = st.sidebar.checkbox("Use data sampling (faster)", value=False, help="Sample data for faster rendering")
if use_sampling:
    sample_size = st.sidebar.slider("Sample size", 100, 2000, 500)

# ============================================================================
# PAGE 1: OVERVIEW (Optimized)
# ============================================================================
if page == "📊 Overview":
    st.header("📊 Market Efficiency Overview")
    
    # Use sampled data if enabled
    display_df = df.sample(n=min(sample_size, len(df)), random_state=42) if use_sampling else df
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", f"{len(df):,}", help="Total number of matches analyzed")
    
    with col2:
        avg_margin = df['B365_margin'].mean() if 'B365_margin' in df.columns else 0
        st.metric("Avg Bookmaker Margin", f"{avg_margin:.2%}", help="Average bookmaker overround (vig)")
    
    with col3:
        if backtest_df is not None and len(backtest_df) > 0:
            best_roi = backtest_df['roi'].max()
            st.metric("Best Strategy ROI", f"{best_roi:.2%}", delta=f"{best_roi:.2%}", delta_color="inverse")
        else:
            st.metric("Best Strategy ROI", "N/A")
    
    with col4:
        if inefficiencies:
            p_val = inefficiencies.get('home_bias', {}).get('p_value', 1.0)
            st.metric("Home Bias p-value", f"{p_val:.3f}")
        else:
            st.metric("Home Bias p-value", "N/A")
    
    st.markdown("---")
    
    # Outcome distribution (optimized)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Match Outcome Distribution")
        if 'FTR' in display_df.columns:
            outcome_counts = display_df['FTR'].value_counts()
            fig = px.pie(
                values=outcome_counts.values,
                names=['Home Win', 'Draw', 'Away Win'],
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, width="stretch", key="outcome_pie")
    
    with col2:
        st.subheader("League Distribution")
        if 'League' in display_df.columns:
            league_counts = display_df['League'].value_counts()
            league_map = {'E0': 'Premier League', 'SP1': 'La Liga', 'I1': 'Serie A', 'D1': 'Bundesliga'}
            league_names = [league_map.get(l, l) for l in league_counts.index]
            
            fig = px.bar(x=league_names, y=league_counts.values, color=league_counts.values, color_continuous_scale='Blues')
            fig.update_layout(xaxis_title="League", yaxis_title="Matches", showlegend=False, height=350, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, width="stretch", key="league_bar")
    
    # Odds distribution (optimized with sampling)
    st.markdown("---")
    st.subheader("📊 Odds Distribution Analysis")
    
    # Sample data for histograms to improve performance
    hist_sample = display_df.sample(n=min(1000, len(display_df)), random_state=42) if len(display_df) > 1000 else display_df
    
    col1, col2, col3 = st.columns(3)
    
    for col, outcome, color in zip([col1, col2, col3], ['H', 'D', 'A'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        with col:
            odds_col = f'B365{outcome}'
            if odds_col in hist_sample.columns:
                fig = px.histogram(hist_sample, x=odds_col, nbins=30, 
                                 title=f"{'Home' if outcome == 'H' else 'Draw' if outcome == 'D' else 'Away'} Odds",
                                 color_discrete_sequence=[color])
                fig.update_layout(xaxis_title="Odds", yaxis_title="Frequency", showlegend=False, height=250, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig, width="stretch", key=f"odds_hist_{outcome}")

# ============================================================================
# PAGE 2: DATA EXPLORER (Optimized)
# ============================================================================
elif page == "🔍 Data Explorer":
    st.header("🔍 Data Explorer")
    
    # Filters
    st.subheader("🎚️ Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        leagues = ['All'] + sorted(df['League'].unique().tolist()) if 'League' in df.columns else ['All']
        selected_league = st.selectbox("League", leagues, key="explorer_league")
    
    with col2:
        seasons = ['All'] + sorted(df['Season'].unique().tolist()) if 'Season' in df.columns else ['All']
        selected_season = st.selectbox("Season", seasons, key="explorer_season")
    
    with col3:
        outcomes = ['All', 'H', 'D', 'A']
        selected_outcome = st.selectbox("Outcome", outcomes, key="explorer_outcome")
    
    # Use cached filtering
    filtered_df = filter_data(df, selected_league, selected_season, selected_outcome)
    st.markdown(f"**Showing {len(filtered_df):,} matches**")
    
    # Display data (limit rows for performance)
    st.subheader("📋 Match Data")
    display_cols = [col for col in ['HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D', 'B365A', 'B365_margin', 'Match_Type'] if col in filtered_df.columns]
    
    if display_cols:
        # Show only first 100 rows for performance
        st.dataframe(filtered_df[display_cols].head(100), width="stretch", height=400)
        
        if len(filtered_df) > 100:
            st.info(f"ℹ️ Showing first 100 of {len(filtered_df):,} matches. Download full data using the button below.")
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Filtered Data Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if 'FTR' in filtered_df.columns:
        with col1:
            st.metric("Home Win %", f"{(filtered_df['FTR'] == 'H').mean():.1%}")
        with col2:
            st.metric("Draw %", f"{(filtered_df['FTR'] == 'D').mean():.1%}")
        with col3:
            st.metric("Away Win %", f"{(filtered_df['FTR'] == 'A').mean():.1%}")
    
    with col4:
        if 'B365_margin' in filtered_df.columns:
            st.metric("Avg Margin", f"{filtered_df['B365_margin'].mean():.2%}")

# ============================================================================
# PAGE 3: STRATEGY PERFORMANCE (Optimized)
# ============================================================================
elif page == "📈 Strategy Performance":
    st.header("📈 Backtesting Strategy Performance")
    
    if backtest_df is None or len(backtest_df) == 0:
        st.warning("⚠️ No backtest results available. Run `python src/backtesting.py` first.")
    else:
        backtest_sorted = backtest_df.sort_values('roi', ascending=False)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Strategy", backtest_sorted.iloc[0]['strategy'])
        with col2:
            best_roi = backtest_sorted.iloc[0]['roi']
            st.metric("Best ROI", f"{best_roi:.2%}", delta=f"{best_roi:.2%}", delta_color="inverse")
        with col3:
            profitable = (backtest_sorted['roi'] > 0).sum()
            st.metric("Profitable Strategies", f"{profitable} / {len(backtest_sorted)}")
        with col4:
            st.metric("Median ROI", f"{backtest_sorted['roi'].median():.2%}")
        
        st.markdown("---")
        
        # ROI comparison (optimized)
        st.subheader("📊 Strategy ROI Comparison")
        fig = px.bar(backtest_sorted, x='strategy', y='roi', color='roi',
                    color_continuous_scale=['red', 'yellow', 'green'], color_continuous_midpoint=0)
        fig.update_layout(xaxis_title="Strategy", yaxis_title="ROI (%)", xaxis_tickangle=-45, 
                         height=400, showlegend=False, margin=dict(t=20, b=100, l=50, r=20))
        fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Breakeven")
        st.plotly_chart(fig, width="stretch", key="roi_comparison")
        
        # Win rate vs ROI (optimized)
        st.markdown("---")
        st.subheader("🎯 Win Rate vs ROI Analysis")
        fig = px.scatter(backtest_sorted, x='win_rate', y='roi', size='total_bets', color='roi',
                        hover_data=['strategy', 'wins', 'total_bets'],
                        color_continuous_scale=['red', 'yellow', 'green'], color_continuous_midpoint=0)
        fig.update_layout(xaxis_title="Win Rate", yaxis_title="ROI (%)", height=400, margin=dict(t=20, b=20, l=50, r=20))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, width="stretch", key="winrate_scatter")
        
        # Detailed table
        st.markdown("---")
        st.subheader("📋 Detailed Strategy Results")
        display_df = backtest_sorted[['strategy', 'total_bets', 'wins', 'win_rate', 'roi', 'total_profit', 'final_bankroll']].copy()
        display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.2%}")
        display_df['roi'] = display_df['roi'].apply(lambda x: f"{x:.2%}")
        display_df['total_profit'] = display_df['total_profit'].apply(lambda x: f"${x:.2f}")
        display_df['final_bankroll'] = display_df['final_bankroll'].apply(lambda x: f"${x:.2f}")
        st.dataframe(display_df, width="stretch", height=400)

# ============================================================================
# PAGE 4 & 5: Keep existing implementation (already optimized)
# ============================================================================
elif page == "⚖️ Market Inefficiencies":
    st.header("⚖️ Market Inefficiency Analysis")
    
    if inefficiencies is None:
        st.warning("⚠️ No inefficiency analysis available. Run `python src/inefficiency_scanner.py` first.")
    else:
        # Home bias
        st.subheader("🏠 Home Bias Analysis")
        home_bias = inefficiencies.get('home_bias', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Expected Home Wins", f"{home_bias.get('expected_wins', 0):.0f}")
        with col2:
            st.metric("Actual Home Wins", f"{home_bias.get('actual_wins', 0):.0f}")
        with col3:
            st.metric("Bias", f"{home_bias.get('bias_percentage', 0):.2%}")
        with col4:
            p_val = home_bias.get('p_value', 1.0)
            is_sig = home_bias.get('statistically_significant', False)
            st.metric("p-value", f"{p_val:.3f}", delta="Significant" if is_sig else "Not Significant", delta_color="normal" if is_sig else "off")
        
        if is_sig:
            st.success(f"✅ Home bias is statistically significant (p < 0.05)")
        else:
            st.info(f"ℹ️ Home bias is NOT statistically significant (p = {p_val:.3f})")
        
        # Draw bias by league
        st.markdown("---")
        st.subheader("⚖️ Draw Bias by League")
        
        draw_bias = inefficiencies.get('draw_bias', {})
        by_league = draw_bias.get('by_league', [])
        
        if by_league:
            league_df = pd.DataFrame(by_league)
            league_map = {'E0': 'Premier League', 'SP1': 'La Liga', 'I1': 'Serie A', 'D1': 'Bundesliga'}
            league_df['League_Name'] = league_df['League'].map(league_map)
            
            fig = px.bar(league_df, x='League_Name', y='Bias', color='Bias',
                        color_continuous_scale=['green', 'yellow', 'red'], color_continuous_midpoint=0)
            fig.update_layout(xaxis_title="League", yaxis_title="Draw Bias", height=350, margin=dict(t=20, b=20, l=50, r=20))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, width="stretch", key="draw_bias")
            
            # Table
            display_df = league_df[['League_Name', 'Implied_Draw_Prob', 'Actual_Draw_Rate', 'Bias', 'Matches']].copy()
            display_df['Implied_Draw_Prob'] = display_df['Implied_Draw_Prob'].apply(lambda x: f"{x:.2%}")
            display_df['Actual_Draw_Rate'] = display_df['Actual_Draw_Rate'].apply(lambda x: f"{x:.2%}")
            display_df['Bias'] = display_df['Bias'].apply(lambda x: f"{x:.2%}")
            display_df.columns = ['League', 'Implied Prob', 'Actual Rate', 'Bias', 'Matches']
            st.dataframe(display_df, width="stretch")

elif page == "🎲 Match Analysis":
    st.header("🎲 Individual Match Analysis")
    st.subheader("🔍 Search for a Match")
    
    # Search filters
    col1, col2 = st.columns(2)
    
    with col1:
        if 'HomeTeam' in df.columns:
            teams = sorted(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))
            selected_team = st.selectbox("Select Team", ['All'] + teams, key="match_team")
        else:
            selected_team = 'All'
    
    with col2:
        if 'League' in df.columns:
            leagues = ['All'] + sorted(df['League'].unique().tolist())
            selected_league_match = st.selectbox("League", leagues, key='match_league')
        else:
            selected_league_match = 'All'
    
    # Filter matches
    match_df = df.copy()
    if selected_team != 'All' and 'HomeTeam' in df.columns:
        match_df = match_df[(match_df['HomeTeam'] == selected_team) | (match_df['AwayTeam'] == selected_team)]
    if selected_league_match != 'All' and 'League' in df.columns:
        match_df = match_df[match_df['League'] == selected_league_match]
    
    st.markdown(f"**Found {len(match_df):,} matches**")
    
    if len(match_df) > 0:
        match_df['match_label'] = match_df.apply(
            lambda row: f"{row.get('HomeTeam', 'Home')} vs {row.get('AwayTeam', 'Away')} - {row.get('FTR', 'N/A')}", axis=1)
        
        selected_match_idx = st.selectbox("Select Match", range(min(100, len(match_df))),
                                         format_func=lambda i: match_df.iloc[i]['match_label'])
        
        match = match_df.iloc[selected_match_idx]
        
        # Display match details
        st.markdown("---")
        st.subheader("📋 Match Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🏠 Home Team")
            st.markdown(f"**{match.get('HomeTeam', 'N/A')}**")
            if 'B365H' in match:
                st.metric("Odds", f"{match['B365H']:.2f}")
            if 'B365_true_prob_H' in match:
                st.metric("True Probability", f"{match['B365_true_prob_H']:.1%}")
        
        with col2:
            st.markdown("### ⚽ Match Result")
            result = match.get('FTR', 'N/A')
            result_map = {'H': '🏠 Home Win', 'D': '🤝 Draw', 'A': '✈️ Away Win'}
            st.markdown(f"## {result_map.get(result, 'N/A')}")
            if 'B365_margin' in match:
                st.metric("Bookmaker Margin", f"{match['B365_margin']:.2%}")
        
        with col3:
            st.markdown("### ✈️ Away Team")
            st.markdown(f"**{match.get('AwayTeam', 'N/A')}**")
            if 'B365A' in match:
                st.metric("Odds", f"{match['B365A']:.2f}")
            if 'B365_true_prob_A' in match:
                st.metric("True Probability", f"{match['B365_true_prob_A']:.1%}")
        
        # Probability visualization
        st.markdown("---")
        st.subheader("📊 Probability Distribution")
        
        if all(col in match.index for col in ['B365_true_prob_H', 'B365_true_prob_D', 'B365_true_prob_A']):
            probs = {'Home Win': match['B365_true_prob_H'], 'Draw': match['B365_true_prob_D'], 'Away Win': match['B365_true_prob_A']}
            fig = px.bar(x=list(probs.keys()), y=list(probs.values()), color=list(probs.values()), color_continuous_scale='Blues')
            fig.update_layout(xaxis_title="Outcome", yaxis_title="Probability", showlegend=False, height=350, margin=dict(t=20, b=20, l=50, r=20))
            st.plotly_chart(fig, width="stretch", key="match_prob")

# ============================================================================
# PAGE 6: VISUALIZATIONS
# ============================================================================
elif page == "📸 Visualizations":
    st.header("📸 Analysis Visualizations")
    st.markdown("Comprehensive visual analysis of market inefficiencies and strategy performance")
    
    # Complete Inefficiency Analysis
    st.markdown("---")
    st.subheader("🔍 Complete Market Inefficiency Analysis")
    st.markdown("6-panel comprehensive analysis showing favorite-longshot bias, home advantage, and league-specific draw biases")
    
    try:
        st.image("results/complete_inefficiency_analysis.png", 
                caption="Complete market inefficiency analysis across all dimensions",
                use_column_width=True)
    except FileNotFoundError:
        st.warning("⚠️ Image not found. Run `notebooks/03_inefficiency_visualization.ipynb` to generate.")
    
    # Strategy Performance
    st.markdown("---")
    st.subheader("📊 Strategy Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Backtest Results Comparison**")
        try:
            st.image("results/backtest_results.png",
                    caption="ROI comparison, win rate analysis, and bet distribution across all strategies",
                    use_column_width=True)
        except FileNotFoundError:
            st.warning("⚠️ Image not found. Run `python src/backtesting.py` to generate.")
    
    with col2:
        st.markdown("**Bankroll Evolution**")
        try:
            st.image("results/bankroll_evolution.png",
                    caption="Bankroll progression over time for top 5 strategies",
                    use_column_width=True)
        except FileNotFoundError:
            st.warning("⚠️ Image not found. Run `python src/backtesting.py` to generate.")
    
    # Bias Analysis
    st.markdown("---")
    st.subheader("⚖️ Market Bias Visualization")
    st.markdown("Detailed analysis of systematic biases in betting markets")
    
    try:
        st.image("results/bias_analysis.png",
                caption="Market bias patterns and statistical significance testing",
                use_column_width=True)
    except FileNotFoundError:
        st.warning("⚠️ Image not found. Run `notebooks/02_bias_analysis.ipynb` to generate.")
    
    # Download section
    st.markdown("---")
    st.subheader("💾 Download Visualizations")
    st.markdown("All visualizations are saved in the `results/` directory as high-resolution PNG files (300 DPI)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📊 Inefficiency Analysis**")
        st.code("results/complete_inefficiency_analysis.png")
    
    with col2:
        st.markdown("**📈 Backtest Results**")
        st.code("results/backtest_results.png")
    
    with col3:
        st.markdown("**💰 Bankroll Evolution**")
        st.code("results/bankroll_evolution.png")
    
    with col4:
        st.markdown("**⚖️ Bias Analysis**")
        st.code("results/bias_analysis.png")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>⚽ Sports Betting Market Efficiency Analysis</strong></p>
    <p>Built with Streamlit • Data from football-data.co.uk</p>
    <p><em>For educational purposes only. This demonstrates market efficiency - not betting advice.</em></p>
</div>
""", unsafe_allow_html=True)
