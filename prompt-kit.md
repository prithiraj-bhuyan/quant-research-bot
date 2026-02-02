# Phase 1 Prompt Kit: Quantitative Asset Allocation & Factor Optimization

This kit establishes the benchmarking baseline for our financial RAG system. It compares a conversational baseline (Prompt A) against a highly constrained, structured output (Prompt B) to evaluate model precision and readiness for Phase 2 automation.

---

## Task 1: Paper Triage (The Summarizer)
**Goal:** Quickly ingest research papers and categorize their technical framework for our database.

### Prompt A (Baseline)
> "Read the attached arXiv paper and summarize it for a portfolio manager. Tell me the main contribution, the methods used, the data sources, the findings, and any limitations."

### Prompt B (Advanced)
> "You are an expert Quantitative Researcher specializing in Portfolio Management. Analyze the attached research paper (arXiv ID provided in text) and output a structured summary in strictly valid JSON format.
> 
> **Required JSON Keys:**
> * `strategy_class`: Categorize as 'Asset Allocation', 'Factor Investing', or 'Optimization'.
> * `mathematical_framework`: Name the specific model (e.g., 'Deep RL', 'Mean-Variance', 'Prospect Theory').
> * `primary_alpha_claim`: One sentence on the core competitive advantage.
> * `quantitative_results`: A list of the top 3 specific metrics (e.g., Sharpe Ratio, Correlation). Render all math/formulas in LaTeX (e.g., $\sigma$ or $\mathbb{E}[R]$).
> * `data_source`: Specify the period and universe (e.g., '1990-2024, S&P 500').
> 
> **Constraint:** If data is missing, return 'NULL'. Return ONLY the JSON object. No preamble."

### Why these constraints exist:
* **Role Prompting ("Expert Quant"):** Sets the persona to ensure the model uses professional terminology and avoids overly simplistic explanations.
* **JSON Output:** Ensures the response can be programmatically parsed and stored in a database during Phase 2 without human intervention.
* **LaTeX Requirement:** Forces mathematical rigor. Standard text variables (like 'sigma') can be ambiguous; LaTeX ($\sigma$) ensures the bot's engine renders formulas correctly in the final UI.
* **NULL Constraint:** A critical guardrail to prevent hallucinations. It forces the model to admit when data is missing rather than fabricating dates or assets.
* **No Preamble:** Eliminates "conversational noise" (e.g., "Sure, here is your summary") which breaks automated code parsers.

---

## Task 2: Claim–Evidence Extraction (The Fact-Checker)
**Goal:** Extract verifiable data points to ground the bot's advice in peer-reviewed evidence.

### Prompt A (Baseline)
> "Read the attached arXiv paper. What claims does this paper make about returns and risks? List the evidence it provides for them."

### Prompt B (Advanced)
> "Act as a Financial Fact-Checker. Read the attached arXiv paper. Extract the top 5 quantitative claims regarding risk-adjusted returns or market regimes.
> 
> **Output Format:** Provide a Markdown table with the following columns:
> | Claim | LaTeX_Formula | Exact_Quote | Location |
> | :--- | :--- | :--- | :--- |
> 
> **Strict Rules:**
> * Use LaTeX for all variables ($r$, $\beta$, $\alpha$).
> * Distinguish between Arithmetic and Geometric means.
> * If the claim is based on a visual chart, specify 'Estimated from Figure [X]'.
> * If the claim is statistically significant, include the p-value or t-stat."

### Why these constraints exist:
* **Evidence-Based Grounding:** By requiring "Exact_Quote" and "Location," we force the model to look back at its context window, significantly reducing the chance of made-up claims.
* **Technical Distinction:** Mandating the difference between Arithmetic and Geometric means tests if the model truly understands financial math or is just summarizing text.
* **Source Attribution:** Requiring Table/Figure numbers ensures the user can manually verify the bot’s data, which is essential for "pure research" credibility.
* **Statistical Significance:** Forcing the inclusion of p-values or t-stats ensures the bot prioritizes "proven" research over mere anecdotal observations in a paper.