# 📊 Data Analysis Agent

A high-performance, multi-modal autonomous agent designed for professional data exploration, statistical modeling, and interactive visualization. The agent leverages a unified SQL/Python/R engine to deliver executive-grade insights with absolute data precision.

## 🚀 Key Capabilities

### 1. Multi-Modal Analytical Engine
Seamlessly switch between different analytical paradigms depending on your technical needs:
- **SQL Mode**: Direct database-style querying for high-speed aggregation.
- **Python Mode**: Advanced Machine Learning (AutoML) and deep statistical analysis using pandas, scikit-learn, and XGBoost.
- **R Mode**: Tidyverse-powered statistical modeling and data manipulation.
- **Generate Graphs**: A three-step linear pipeline (Extraction -> Architect -> Coder) for creating high-quality interactive visualizations.
- **Ask Questions**: A ReAct-based conversational interface for ad-hoc data inquiries.

### 2. High-Precision Graph Pipeline
- **SQL-Driven Extraction**: Visualizations are backed by a hidden SQL aggregation engine to ensure 100% accurate metrics.
- **Metadata Parity**: Axis labels, legends, and tooltips are automatically synced with your dataset headers.
- **Interactive HTML**: Export charts as standalone interactive files for executive presentations.

### 3. Executive Reporting & Layman Insights
- **Structured Summaries**: Results are formatted as professional markdown reports with clean headers and tabular data.
- **Layman Translations**: Technical metrics (like R² or RMSE) are automatically translated into plain-English sentences for non-technical stakeholders.
- **Key Insights**: Every analysis concludes with 3-4 bulleted takeaways explaining the real-world impact of the data.

### 4. Persistent Workspace
- **Multi-Mode History**: Your chat history, CSV results, and interactive graphs persist across mode switches, allowing you to build a comprehensive research thread without losing previous work.
- **Auto-Snapshots**: Every dataset upload is automatically summarized with AI-driven onboarding reports and visual data previews.

---

## 🛠 Mode Selection: "Enter X... get Y..."

| Mode | What you enter | What you get |
| :--- | :--- | :--- |
| **SQL Code** | "Show me total revenue by month" | Valid SQL code + CSV result preview + Download button |
| **Python Code** | "Build a model to predict churn" | Robust ML code + Layman evaluation + Predictive CSV |
| **R Code** | "Run a linear regression on price" | Clean R tidyverse code + Statistical summary |
| **Generate Graphs** | "Plot sales vs marketing spend" | High-quality interactive Plotly/Chart.js visualization |
| **Ask Questions** | "What is the average age in table X?" | Direct answer + Markdown tables for structured lists |

---

## ⚙️ Technical Stack
- **Framework**: Streamlit (Front-end), LangChain (Agentic Orchestration).
- **Models**: Meta Llama 3.1-70B (Architect/Coder), Nemotron-3-Nano (ReAct).
- **Processing**: pandas, scikit-learn, XGBoost, LightGBM, Statsmodels, SQLite3.
- **Visualization**: Plotly, Chart.js, Tailwind CSS (Design System).

## 🏃 How to Run
```bash
streamlit run Data_Analysis_Agent.py --server.port 8506
```

---
*Built for data-driven precision and executive clarity.*
