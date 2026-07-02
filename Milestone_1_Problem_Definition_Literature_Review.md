# Milestone 1: Problem Definition and Literature Review

Project: Talk to Your Database - A Natural-Language Analytics Copilot

Course: Data Science and AI Lab

Team Members:

- Siddhant Hitesh Mantri (21f3002218)
- Anirudh Komanduri (22f1000522)
- Vishal S (23f2003089)
- Smrutishikta Das (21f1006009)
- Walunila Aier (21f3002564)
- Sambhav Jha (22f3003227)

## 1. Problem Definition

Relational databases contain much of an organization's useful data, but querying them usually requires SQL knowledge. This blocks non-technical users such as managers, operations teams, product teams, and domain experts from directly answering data questions. Our project addresses this by building a natural-language analytics copilot: a user asks a question in plain English, the system identifies relevant schema context, generates a read-only SQL query, executes it safely, and returns the result as a table and, when appropriate, a visualization.

The problem is not only SQL generation. A usable system must handle the full workflow: understand the question, retrieve relevant tables and columns, generate valid SQL, execute it safely, and present the answer in a form users can inspect. Based on TA feedback, we narrow the required scope to the core workflow: natural language to SQL generation, schema-aware retrieval, SQL execution, and automatic visualization. Agentic self-correction, multi-turn conversations, multi-model comparison, and broad benchmarking will be treated as stretch goals.

## 2. Stakeholders and Scope

Primary stakeholders are non-technical business users and domain experts who need database answers without writing SQL. Secondary stakeholders include analysts, data engineers, database administrators, and course evaluators.

In scope:

- Single-database querying using SQLite or PostgreSQL.
- Single-turn natural-language analytical questions.
- Read-only `SELECT` queries.
- Schema-aware retrieval over tables, columns, keys, and optionally sampled values.
- SQL generation using feasible small open-source models, such as Phi-4-mini, Qwen-family small models, and other compact language models.
- SQL validation, row limits, timeouts, and read-only execution.
- Result display as SQL, table, and automatic chart.
- Evaluation using execution accuracy, SQL validity, retrieval quality, latency, and visualization validity.

Out of scope for the core deliverable:

- Write operations such as `INSERT`, `UPDATE`, `DELETE`, `DROP`, or `ALTER`.
- Multi-database federation.
- Production-grade enterprise access control beyond read-only sandboxing.
- Training a model from scratch.
- Full multi-turn conversation memory.
- Large-scale model comparison or exhaustive benchmarking.

Stretch goals prioritize execution-guided self-correction and multi-turn follow-up questions. Extra model baselines and broader Spider/BIRD evaluation will be attempted only if time permits.

## 3. Why the Problem Matters

In many organizations, users depend on dashboards, analysts, or spreadsheet exports. Dashboards work for repeated metrics but cannot cover every new question. Analysts can write custom SQL, but this creates delay and repeated manual work. A natural-language database interface can reduce this friction while still showing SQL for transparency.

The problem is technically important because realistic text-to-SQL is still unsolved. Real databases contain many tables, unclear column names, foreign-key relationships, dirty values, domain-specific terminology, and ambiguous questions. Benchmarks such as BIRD and Spider 2.0 were introduced because older datasets often understate these real-world difficulties.

## 4. Current Approaches and Literature Review

Organizations currently address database access through BI tools such as Tableau, Power BI, Looker, Metabase, and Superset, plus SQL notebooks and analyst-mediated workflows. These tools are useful, but someone still has to define SQL, joins, dashboards, or metrics in advance. General-purpose LLMs can draft SQL, but they are brittle without accurate schema context, table relationships, value examples, and database constraints.

Early systems such as Seq2SQL and SQLNet showed that neural models could generate executable SQL at scale on WikiSQL (Zhong et al., 2017; Xu et al., 2017). However, WikiSQL is mostly single-table, so it does not capture multi-table organizational databases.

Spider became the standard cross-domain benchmark because it requires generalization to unseen databases. It contains 10,181 questions, 5,693 unique SQL queries, 200 databases, and 138 domains (Yu et al., 2018). This shifted attention toward schema linking: the model must identify which tables and columns match the user's wording. RAT-SQL uses relation-aware schema encoding for this problem (Wang et al., 2020), and RESDSQL separates schema linking from SQL skeleton generation (Li et al., 2023), supporting our decision to make schema retrieval an explicit module.

Other work focuses on validity and execution. Execution-guided decoding prunes SQL candidates using database feedback (Wang et al., 2018), while PICARD constrains decoding through incremental parsing (Scholak et al., 2021). These methods motivate our validation and safe execution layer.

LLM-based text-to-SQL research emphasizes prompt design, schema representation, and in-context examples. Large language models can be strong baselines (Rajkumar et al., 2022), and prompt/schema formatting materially affects performance (Nan et al., 2023). DIN-SQL adds decomposition and self-correction (Pourreza and Rafiei, 2023), but that is more complex than our minimum feasible system.

Realistic benchmarks highlight the remaining gap. BIRD contains 12,751 question-SQL pairs over 95 databases totaling 33.4 GB across 37 domains (Li et al., 2023). In the original BIRD paper, ChatGPT achieved 40.08% execution accuracy while human experts achieved 92.96%, showing that realistic database-grounded text-to-SQL remains difficult. KaggleDBQA and Spider 2.0 similarly emphasize messy real data, documentation, enterprise schemas, and workflow complexity (Lee et al., 2021; Lei et al., 2024).

Finally, the output should be usable, not just executable. Natural-language-to-visualization work such as nvBench motivates adding a visualization layer (Luo et al., 2021). We will keep this feasible with rule-based chart selection: scalar outputs become text/metric answers, categorical-plus-numeric results become bar charts, time-plus-numeric results become line charts, and other outputs remain tables.

## 5. Metrics and Evidence of Success

Common text-to-SQL metrics include:

- Exact Match: whether predicted SQL matches the reference query after normalization.
- Component Matching: Spider-style comparison of clauses such as `SELECT`, `WHERE`, `GROUP BY`, and `ORDER BY`.
- Execution Accuracy: whether predicted SQL returns the same result as the gold SQL.
- Test-Suite Accuracy: execution over multiple generated databases to reduce accidental correctness.
- Valid Efficiency Score: BIRD-style scoring that considers correctness and query efficiency.

For our project, success will be measured with:

- SQL parse-validity rate.
- Read-only safety pass rate.
- Execution success rate.
- Execution accuracy on a curated evaluation set from Spider, BIRD mini-dev, or our demo database.
- Schema retrieval recall for gold tables/columns when gold SQL is available.
- Median and p95 latency.
- Visualization validity based on result shape.

Evidence will include a working FastAPI-based demo with a frontend where a user enters a question and receives schema-grounded SQL, safe execution, a result table, and a chart. Quantitatively, we will compare a baseline prompt-to-SQL pipeline against the schema-aware version using automated evaluation where gold SQL or executable answers are available. If human review is needed, the team will manually evaluate approximately 100 examples, depending on sample complexity and review speed. If time permits, we will add value grounding or self-correction as an ablation.

## 6. Gaps and Project Contribution

Current solutions still struggle with schema linking, value grounding, invalid SQL, dialect issues, and ambiguous wording. Benchmarks often evaluate SQL alone, while real users need a complete workflow from question to trusted answer. Advanced agentic systems may help, but they add complexity, latency, and implementation risk.

Our contribution is not a new model architecture or state-of-the-art benchmark result. It is a feasible, transparent analytics interface integrating schema-aware retrieval, read-only SQL generation, safe execution, and automatic visualization.

## 7. Milestone-Aligned Plan

The recommended build order is:

1. Build a baseline prompt-to-SQL-to-execution pipeline on a small demo database.
2. Test feasible small open-source model baselines and identify the practical inference bottleneck.
3. Add SQL validation, read-only constraints, row limits, and timeouts.
4. Add schema retrieval over tables, columns, keys, and optional descriptions.
5. Build the FastAPI backend and frontend result view.
6. Add result formatting and rule-based visualization.
7. Evaluate with automated metrics and a curated human-reviewed set if required.
8. Add self-correction and multi-turn follow-ups only after the core workflow is stable.

This plan directly addresses the TA feedback by prioritizing the core system and keeping agentic correction, multi-turn interaction, and broad model comparisons optional.

## 8. Implementation Decisions

For the milestone submission, we make the following implementation assumptions:

- Demo database: a small e-commerce-style SQLite or PostgreSQL database with customers, products, orders, payments, and shipments, because it supports intuitive business analytics questions and useful charts. If dataset preparation becomes a bottleneck, Chinook will be used as a low-risk fallback.
- Baseline models: small open-source language models, including Phi-4-mini, Qwen-family small models, and other feasible compact models.
- Demo stack: FastAPI backend with a frontend interface.
- Main expected bottleneck: LLM inference speed and deployment feasibility.
- Evaluation: automated evaluation using executable SQL or gold answers where possible; if human review is needed, the team will manually check around 100 examples, with possible expansion depending on review speed.
- Stretch priority: execution-guided self-correction first, followed by multi-turn follow-up questions.

## References

1. Zhong, V., Xiong, C., and Socher, R. (2017). Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning. https://arxiv.org/abs/1709.00103
2. Xu, X., Liu, C., and Song, D. (2017). SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning. https://arxiv.org/abs/1711.04436
3. Yu, T., et al. (2018). Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task. https://aclanthology.org/D18-1425/
4. Wang, C., Brockschmidt, M., and Singh, R. (2018). Robust Text-to-SQL Generation with Execution-Guided Decoding. https://arxiv.org/abs/1807.03100
5. Wang, B., Shin, R., Liu, X., Polozov, O., and Richardson, M. (2020). RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers. https://aclanthology.org/2020.acl-main.677/
6. Zhong, R., Yu, T., and Klein, D. (2020). Semantic Evaluation for Text-to-SQL with Distilled Test Suites. https://arxiv.org/abs/2010.02840
7. Scholak, T., Schucher, N., and Bahdanau, D. (2021). PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models. https://arxiv.org/abs/2109.05093
8. Lee, C.-H., Polozov, O., and Richardson, M. (2021). KaggleDBQA: Realistic Evaluation of Text-to-SQL Parsers. https://arxiv.org/abs/2106.11455
9. Li, J., et al. (2023). Can LLM Already Serve as a Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs. https://arxiv.org/abs/2305.03111
10. Rajkumar, N., Li, R., and Bahdanau, D. (2022). Evaluating the Text-to-SQL Capabilities of Large Language Models. https://arxiv.org/abs/2204.00498
11. Nan, L., et al. (2023). Enhancing Few-shot Text-to-SQL Capabilities of Large Language Models. https://arxiv.org/abs/2305.12586
12. Pourreza, M., and Rafiei, D. (2023). DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction. https://arxiv.org/abs/2304.11015
13. Li, H., Zhang, J., Li, C., and Chen, H. (2023). RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL. https://arxiv.org/abs/2302.05965
14. Luo, Y., Tang, J., and Li, G. (2021). nvBench: A Large-Scale Synthesized Dataset for Cross-Domain Natural Language to Visualization Task. https://arxiv.org/abs/2112.12926
15. Lei, F., et al. (2024). Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows. https://arxiv.org/abs/2411.07763
