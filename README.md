# ⚽ Sports Betting Market Efficiency Analysis: R&D Methodology

> **A professional demonstration of data science methodology, statistical validation, and market pricing theory in European soccer betting markets.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Data Science](https://img.shields.io/badge/Data-Science-green.svg)
![Market Analysis](https://img.shields.io/badge/Market-Analysis-yellow.svg)
![Status](https://img.shields.io/badge/Status-Project_Ready-success.svg)

---

## 🎯 Strategic Overview

In the world of high-frequency sports betting, the primary challenge for an R&D team isn't just finding profit—it's **validating the efficiency of the market**. 

This project explores the "Efficiency vs. Profit" problem by analyzing **4,337 historical soccer matches** across 4 major European leagues (2021-2024). Instead of producing a "get-rich-quick" model, it implements a rigorous **Market Inefficiency Scanner** to detect systematic biases.

### The Problem Statement
European soccer leagues are among the most liquid and efficient markets globally. For a Data Scientist at Plannatech, understanding why a model *fails* to find ROI is as critical as finding an edge. This project proves that:
1. **Margins are tight**: The 5.4% average bookmaker "vig" is a steep barrier.
2. **Calibration is key**: Bookmaker-implied probabilities match actual outcome frequencies with a mean absolute error of just **2.4%**.
3. **Efficiency is robust**: Statistically significant biases (like a 0.4% home bias) are often too small to overcome the margin.

---

## 🛠️ The Technical Workflow (R&D Pipeline)

The project is structured as a modular R&D pipeline, emphasizing clean code and statistical rigor.

### Phase 1: Robust Data Engineering
Handled by `src/data_collection.py` and verified by `src/diagnose.py`.
- **E**xtract: Automated ingestion of 4,000+ match records.
- **T**ransform: Cleaning missing values, normalizing team names, and handling league-specific formats.
- **L**oad: Structured CSV-based storage for high-performance iteration.

### Phase 2: Mathematical Feature Engineering
Implemented in `src/data_processing.py`.
- **Odds Normalization**: Converting decimal odds to implied probabilities and removing the bookmaker margin ("overround") to find the **True Probability**.
- **Edge Calculation**: $Expected Value = (True Prob \times Odds) - 1$.
- **Segmentation**: Creating features for `Match_Type` (Close/Medium/Heavy Favorite) and `Favorite_Strength`.

### Phase 3: Statistical Hypothesis Testing
The core engine in `src/inefficiency_scanner.py`.
- **Calibration Curve Analysis**: Validating if a 60% implied probability actually results in 60% wins.
- **Chi-Square Tests**: Determining if the delta between expected and actual outcomes is statistically significant ($p < 0.05$).
- **Multi-Bias Scanning**: Testing for "Favorite-Longshot", "Home/Away", and "Draw/No-Draw" systemic pricing errors.

### Phase 4: Strategy Simulation & Backtesting
Executed in `src/backtesting.py`.
- **Flat-Stake Methodology**: Preventing compounding errors and bankroll variance from masking model performance.
- **Out-of-Sample Validation**: 12+ strategies tested across independent leagues to ensure results aren't artifacts of a single dataset.

---

## 📈 Key Results & Interactive Dashboard

### Backtesting Summary
```text
Tested Strategies: 12
Profitable Outcomes: 0 (Validated Market Efficiency)
Best Performing: Draw Specialist (Serie A) | ROI: -0.84%
Average ROI: -9.26% (Succumbing to the 5.4% Margin)
```

### Interactive Visualization (`app.py`)
To make results accessible to stakeholders and traders, an optimized **Streamlit dashboard** (50-75% faster via advanced caching) provides:
- **Market Explorer**: Filter 4,337 matches by league, season, or bookmaker.
- **Calibration View**: Interactive Plotly charts showing pricing accuracy.
- **Strategy Lab**: Side-by-side ROI comparison of all tested models.

*Launch with:* `./run_dashboard.sh`

---

## 🚀 Alignment with Plannatech (Sports Data Scientist Role)

This project mirrors the responsibilities and requirements of the **Sports Data Scientist** at Plannatech:

| Requirement | Demonstration in this Project |
| :--- | :--- |
| **Deep Curiosity & Sports IQ** | Analytical breakdown of 1X2 markets vs. league-specific draw trends. |
| **Feature Engineering** | Mathematical conversion of odds to "true" probability distributions. |
| **Data Quality & Consistency** | Built-in diagnostic tools (`src/diagnose.py`) to ensure cross-league reliability. |
| **Python & Math Proficiency** | Advanced usage of `pandas`, `scipy.stats`, and `numpy` for probability modeling. |
| **Collaboration Readiness** | Modular structure allows modular integration into larger NN protocols like 'Train Station'. |
| **Gaming/Strategy Expert** | Built-in understanding of EV, bankroll management, and variance. |

---

## 🛠️ Installation & Quick Start

### Prerequisites
- Python 3.8+
- [See requirements.txt](requirements.txt)

```bash
# Setup
git clone https://github.com/yourusername/odds-scanner.git
cd odds-scanner
pip install -r requirements.txt

# Run the Pipeline
python src/data_collection.py  # Collect data
python src/data_processing.py  # Feature Engineering
python src/backtesting.py       # Run Simulations

# Launch Dashboard
./run_dashboard.sh
```

---

## 🔮 Future R&D Roadmap

To further mimic a professional R&D path, I am planning to expand this project into:
1. **Asian Handicap Modeling**: Moving beyond 1X2 to markets with lower margins and higher efficiency.
2. **Poisson-Based Goal Prediction**: Shifting from "Odds Analysis" to "Core Performance Analysis".
3. **Kelly Criterion Integration**: Optimizing stake sizing for models with positive EV.
4. **Live Betting Inefficiencies**: Detecting model drift during the "Goldilocks Zone" of live match updates.

---

## 📄 License & Ethical Note
This project is for **educational/portfolio purposes only**. It demonstrates market efficiency, effectively showing why betting without a significant model-driven edge is a losing proposition.

**Created by**: Kembly Munoz Valencia | [LinkedIn](https://www.linkedin.com/in/kemvalxx1) 