# Executive Summary: Sports Betting Market Efficiency Analysis

**Project:** Market Inefficiency Scanner for European Soccer Betting  
**Objective:** Validate market efficiency through rigorous statistical analysis and backtesting  
**Business Context:** R&D for high-frequency sports betting models  
**Outcome:** Confirmed market efficiency - 2.4% Mean Absolute Error in calibration  

---

## 🎯 Project at a Glance

| Metric | Value |
|--------|-------|
| **Matches Analyzed** | 4,337 |
| **Leagues Covered** | 4 (Premier League, La Liga, Serie A, Bundesliga) |
| **Time Period** | 2021-2024 seasons |
| **Strategies Tested** | 12 |
| **Calibration Error** | 2.4% (Highly Accurate) |
| **Best ROI Observed** | -0.84% (Draw Specialist - Serie A) |
| **Avg Bookmaker Margin** | 5.4% |

---

## 📊 Key Findings

### 1. Market Efficiency Confirmed
All tested strategies resulted in negative ROI, validating that the European soccer market is highly efficient and resistant to simple statistical exploitations. This confirms that bookmakers' pricing accurately reflects risk and the margin (vig) is the primary hurdle.

```text
Efficiency Validation Summary:
├─ Median Strategy ROI: -9.26%
├─ Baseline (Always Home): -5.53%
├─ Margin Impact: -5.4% (avg)
└─ Conclusion: No low-hanging fruit; high-precision R&D required
```

### 2. Statistical Calibration Precision
The analysis demonstrates that bookmaker-implied probabilities match actual historical frequencies with remarkable precision. A **2.4% Calibration Error** suggests that the market is a "strong-form" efficient market.

### 3. Home & Draw Bias
- **Home Bias**: Found a 0.4% undervaluation, but it was statistically insignificant ($p = 0.594$).
- **Draw Bias**: Varies by league; Serie A shows the most pricing inertia, though still not enough to overcome the 5.4% margin.

---

## 💼 Strategic Value for Plannatech

This project serves as a proof-of-concept for the technical skills required in the **Sports Data Scientist** role:

### Technical Excellence
- **Data Pipeline**: Modular ETL processing 4,000+ match records with integrated data-quality diagnostics (`src/diagnose.py`).
- **Feature Engineering**: Mathematical removal of "overround" to find true probabilities—a core requirement for feeding "Train Station" style Neural Networks.
- **Statistical Rigor**: Use of Chi-Square testing and bucketed probability bins to avoid p-hacking.

### R&D Mindset
- **Intellectual Honesty**: Reporting "negative" ROI as a success in market validation.
- **Critical Awareness**: Understanding that in efficient markets, 1-2% model improvements are the difference between loss and profit.
- **Actionable Visualizations**: A high-performance Streamlit dashboard designed for both technical review and trader interaction.

---

## 🔬 Methodology Highlights

### The R&D Pipeline
1. **Robust Ingestion**: Historical 1X2 and odds data cleaning.
2. **True Prob Analysis**: Implementation of margin-removal algorithms.
3. **Inefficiency Scanner**: Multi-panel statistical bias detection.
4. **Monte Carlo Strategy Validation**: Backtesting against 12 independent hypotheses.

---

## 🎯 Recommendations for Plannatech R&D

**1. Focus on "Secondary" Markets**
The 1X2 market is highly efficient. Higher R&D ROI is likely found in **Asian Handicaps** or **Player Props**, where liquidity is lower and pricing is less automated.

**2. Leverage Time-Series Decay**
Analyzing "Line Movement" rather than just Closing Odds could reveal how the market absorbs information—essential for pre-match trading models.

**3. In-Play Calibration**
Developing live-monitoring tools for "Model Drift" during matches when "Train Station" internal NNs are active.

---

## 📞 Contact Information

**Created by**: Kembly Munoz Valencia  
**LinkedIn**: [linkedin.com/in/kemvalxx1](https://www.linkedin.com/in/kemvalxx1)  
**Project Repo**: [Odds Scanner GitHub](https://github.com/kemval/odds_scanner)  

**Built for**: Sports Data Scientist Application - Plannatech  
**Date**: January 2026  
**Status**: Ready for Technical Review  