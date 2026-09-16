# Benchmarks and Environments for Evaluating AI/LLM Agents

A controlled comparison of Proximal Policy Optimisation (PPO), Advantage Actor-Critic (A2C) and GPT-OSS-120B in a shared FinRL portfolio-management environment.

This repository contains the source code, validated outputs, trained DRL models and Streamlit dashboard developed for my MSc Data Science and Artificial Intelligence dissertation at the University of Liverpool.

> Academic research only. This project does not provide financial advice or investment recommendations.

## Project Overview

The project investigates how deep reinforcement learning and large language model agents can be compared fairly when they use fundamentally different decision mechanisms.

The benchmark evaluates:

- PPO
- A2C
- GPT-OSS-120B via Groq
- Buy and Hold
- Equal Weight
- DJIA Index

The focus is not only on portfolio return, but also on risk, statistical uncertainty, market-regime behaviour, action patterns, reliability and LLM inference cost.

## Experimental Setup

- 28 fixed DJIA constituents
- Training period: 2 January 2009 – 30 June 2020
- Test period: 1 July 2020 – 27 May 2022
- Initial capital: $1,000,000
- Transaction cost: 0.1%
- Long-only trading
- 169-dimensional state space
- 28-dimensional action space
- Four technical indicators: MACD, RSI-30, CCI-30 and DX-30
- Separate turbulence-based market-risk control
- One new agent decision every five trading days
- 482 daily portfolio valuations
- 481 environment transitions
- 97 decision transitions
- 384 scheduled holds

All active agents use the same out-of-sample decision schedule.

## Key Results

Over the full test period:

| Strategy | Cumulative Return | Sharpe Ratio | Max Drawdown |
|---|---:|---:|---:|
| Equal Weight | 34.89% | 1.124 | -13.25% |
| Buy and Hold | 33.69% | 1.086 | -13.43% |
| A2C | 29.60% | 0.916 | -20.23% |
| DJIA Index | 28.93% | 0.946 | -15.07% |
| PPO | 24.90% | 0.952 | -13.50% |
| GPT-OSS-120B | 24.06% | 0.808 | -13.41% |

Equal Weight achieved the strongest overall full-period performance. A2C produced the highest AI-agent return, PPO achieved the strongest AI-agent Sharpe ratio, and GPT-OSS-120B recorded the smallest maximum drawdown among the three AI agents.

Bootstrap Sharpe-ratio confidence intervals overlap, so these rankings should be interpreted as descriptive rather than statistically confirmed superiority.

## LLM Pipeline

GPT-OSS-120B receives the current portfolio state, prices, technical indicators and turbulence value and returns a structured JSON response containing:

- one action for each of the 28 stocks
- a concise rationale

Responses are validated before execution.

The final implementation also includes checkpointing, replay and failure handling so that unsuccessful API calls cannot silently enter the experiment as genuine trading decisions.

## Project Structure

```text
MSc-dissertation/
├── agents/
├── dashboard/
├── data/
├── evaluation/
├── models/
├── results/
├── config.py
├── requirements.txt
└── README.md
```

## Running the Project

Create and activate a Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Fresh LLM inference requires a Groq API key:

```text
GROQ_API_KEY=your_api_key_here
```

To launch the dashboard:

```bash
streamlit run dashboard/dashboard.py
```

## Reproducibility

The repository includes fixed processed datasets, saved PPO and A2C models, action records, equity curves, benchmark outputs, LLM rationales and validation outputs.

The original DRL training runs did not record explicit random seeds, so complete retraining is not guaranteed to reproduce identical model weights. The saved model artefacts correspond to the policies used for the reported dissertation results.

## Limitations

This is a research benchmark rather than a deployable trading system.

The experiment is limited by:

- one historical test period
- one saved PPO and A2C policy
- no multi-seed DRL evaluation
- simplified transaction costs
- no short selling or leverage
- no slippage, liquidity or market-impact modelling
- no company fundamentals or financial news
- no live-market deployment

## Author

**Aysha Ziaulhaque**  
MSc Data Science and Artificial Intelligence  
University of Liverpool
