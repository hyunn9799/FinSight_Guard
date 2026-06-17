# Feature Specification: Walk-Forward Strategy Optimization

**Feature Branch**: `[002-walk-forward-optimization]`

**Created**: 2026-06-17

**Status**: Draft

**Input**: User description: "총수익률 단독 최적화를 금지하고, MDD, Sharpe, Sortino, 승률, profit factor, 거래 횟수, 평균 보유 기간, 수수료/슬리피지 반영 후 수익률, 시장 국면별 성능을 함께 평가한다. 특정 시기 전체에 한 번 최적화해 적용하는 대신 walk-forward optimization으로 train/test를 시간 순서대로 나누고, out-of-sample 성과, 최악 구간 성능, 안정성, 장세별 성과를 기준으로 robust parameter를 선택한다. MVP는 weighted risk-adjusted score, train/test 분리, walk-forward, regime별 성과 리포트, 이후 regime별 파라미터 적용 순서로 진행한다."

## Clarifications

### Session 2026-06-17

- Q: What guardrails should determine whether an optimized candidate can be labeled robust? → A: Minimum 30 trades and MDD 25% or lower.
- Q: What default weighting policy should the weighted robust score use? → A: Risk-balanced: 30% out-of-sample return, 25% risk-adjusted return, 20% drawdown control, 15% worst-fold resilience, 10% stability/turnover penalty.
- Q: How many valid test folds are required before walk-forward output can produce a robust candidate? → A: Minimum 3 valid test folds.
- Q: When should regime-specific performance be marked low-confidence? → A: Fewer than 10 trades or fewer than 60 trading days in that regime.
- Q: What default transaction cost assumptions should robust optimization use? → A: 0.05% one-way fee and 0.05% one-way slippage, user-adjustable.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Optimize With Risk Controls (Priority: P1)

As a research user, I want strategy optimization to reject total-return-only ranking, so that a historically high return does not hide drawdown, excessive trading, weak validation, or unstable behavior.

**Why this priority**: The current optimization flow can select parameters by one headline return from one period. That creates overfitting risk and conflicts with the project's research-only positioning.

**Independent Test**: Can be tested by running optimization on deterministic sample data where the highest-return candidate has unacceptable drawdown or too few trades and confirming it is not selected as the robust candidate.

**Acceptance Scenarios**:

1. **Given** multiple parameter candidates are evaluated, **When** one candidate has the highest total return but breaches drawdown or trade-sample limits, **Then** the system excludes or clearly downgrades that candidate.
2. **Given** a candidate has positive total return but poor risk-adjusted behavior, **When** results are ranked, **Then** the ranking favors a more stable candidate over the highest-return-only candidate.
3. **Given** transaction cost assumptions are configured, **When** results are displayed, **Then** all return metrics reflect costs and the cost assumptions are visible.

---

### User Story 2 - Validate With Walk-Forward Evaluation (Priority: P2)

As a research user, I want optimization to use time-ordered train/test windows, so that parameters are evaluated only on periods that occur after the period used to select them.

**Why this priority**: Financial time series must avoid future leakage. Walk-forward evaluation gives a more realistic view than optimizing once across a full selected period.

**Independent Test**: Can be tested by creating deterministic dated sample data and confirming every test window starts after its train window, every fold result is reported, and aggregate results include median, worst-fold, and stability measures.

**Acceptance Scenarios**:

1. **Given** enough historical data is available, **When** the user runs robust optimization, **Then** the system produces multiple time-ordered train/test fold results.
2. **Given** one fold performs much worse than the others, **When** the final summary is shown, **Then** the weak fold affects the robustness score and appears in the fold breakdown.
3. **Given** historical data is too short for the requested walk-forward configuration, **When** optimization is requested, **Then** the system gives a limitation message and does not present the output as robust.

---

### User Story 3 - Explain Regime-Specific Strengths and Weaknesses (Priority: P3)

As a research user, I want the optimized candidate to be evaluated by market regime, so that I can see whether it only works in one type of market or remains acceptable across different conditions.

**Why this priority**: A parameter set that performs well only in a bull market may be fragile in bear, sideways, or high-volatility conditions.

**Independent Test**: Can be tested by labeling deterministic sample periods by regime and confirming the output summarizes performance separately for bull, bear, sideways, high-volatility, and low-volatility segments when those segments exist.

**Acceptance Scenarios**:

1. **Given** market regime labels exist for evaluated periods, **When** robust optimization completes, **Then** the result shows performance by regime.
2. **Given** a candidate performs well overall but poorly in a bear or sideways regime, **When** the result is summarized, **Then** the system highlights that weakness as a risk.
3. **Given** a regime has too little data, **When** regime performance is shown, **Then** the system marks that regime result as low-confidence instead of overinterpreting it.

---

### User Story 4 - Keep Optimization Research-Only (Priority: P4)

As a research user, I want robust optimization output to remain clearly educational and non-prescriptive, so that it supports scenario comparison without becoming trading advice.

**Why this priority**: The project constitution prohibits stock recommendations, trading instructions, guaranteed return claims, brokerage integration, and automated order execution.

**Independent Test**: Can be tested by checking the final output for required limitations, no-recommendation framing, and absence of forbidden advice language.

**Acceptance Scenarios**:

1. **Given** robust parameters are selected, **When** the user reviews the output, **Then** the system labels them as historical simulation candidates rather than recommended trading settings.
2. **Given** robust optimization evidence is included in a Korean research report, **When** the report is evaluated, **Then** it includes source-grounded metrics, limitations, scenario framing, and the required disclaimer.

### Edge Cases

- The highest-return candidate has fewer than the minimum meaningful number of trades.
- A candidate has attractive average performance but unacceptable maximum drawdown.
- A candidate is strong in one fold but poor in the worst fold.
- A candidate is strong in bull regimes but weak in bear or sideways regimes.
- A regime segment has too few observations to support a reliable conclusion.
- Slippage or fee assumptions make a previously profitable candidate unprofitable.
- Walk-forward windows cannot be created because the selected date range is too short.
- A fold has no trades, no valid signals, or missing price data.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST prohibit total return as the sole optimization objective for strategy parameter selection.
- **FR-002**: System MUST evaluate candidate parameters using return, drawdown, risk-adjusted return, win rate, profit factor, trade count, average holding period, transaction-cost-adjusted return, and turnover or trading-frequency measures.
- **FR-003**: System MUST require at least 30 completed trades and maximum drawdown of 25% or lower before a candidate can be labeled robust.
- **FR-004**: System MUST apply transaction costs before presenting optimized return metrics, defaulting to 0.05% one-way fee and 0.05% one-way slippage while allowing users to change those assumptions.
- **FR-005**: System MUST support a weighted robust score that defaults to a risk-balanced policy: 30% out-of-sample return, 25% risk-adjusted return, 20% drawdown control, 15% worst-fold resilience, and 10% stability or turnover penalty.
- **FR-006**: System MUST report the score components used to rank candidates so users can understand why a candidate was selected.
- **FR-007**: System MUST split historical data in time order for validation, ensuring each validation period occurs after its corresponding selection period.
- **FR-008**: System MUST provide fold-level walk-forward results, including train period, test period, selected candidate summary, out-of-sample metrics, and warnings.
- **FR-009**: System MUST aggregate fold results using stability-aware summaries, including median out-of-sample return, worst fold return, drawdown, and fold-to-fold variability.
- **FR-010**: System MUST identify and disclose insufficient data when the requested walk-forward evaluation cannot produce at least 3 valid test folds.
- **FR-011**: System MUST classify evaluated periods into market regimes when enough data exists, including bull, bear, sideways, high-volatility, and low-volatility conditions.
- **FR-012**: System MUST report strategy behavior separately by market regime and mark regime results as low-confidence when that regime has fewer than 10 completed trades or fewer than 60 trading days.
- **FR-013**: System MUST initially select one robust all-regime parameter candidate before introducing separate regime-specific parameter sets.
- **FR-014**: System MUST allow future regime-specific parameter evaluation only when each regime-specific candidate has enough out-of-sample evidence and risk disclosure.
- **FR-015**: System MUST compare robust optimized candidates against the current manual parameter configuration and a simple passive baseline over the same evaluation periods.
- **FR-016**: System MUST store or expose enough optimization summary data for reports to cite numeric claims with traceable evidence.
- **FR-017**: System MUST keep all optimization tests deterministic and independent of live external providers.

### Safety, Evidence & Reliability Requirements *(mandatory for research workflow changes)*

- **SER-001**: System MUST avoid direct buy/sell/hold recommendations, trading instructions, guaranteed return claims, guaranteed target claims, and order execution behavior.
- **SER-002**: System MUST back important numeric or factual report claims with `EvidenceItem` records when the feature affects research output.
- **SER-003**: Final Korean reports MUST include the required education-only, no-recommendation disclaimer exactly as defined in the constitution.
- **SER-004**: System MUST disclose unavailable or degraded market, fundamental, or news data instead of fabricating missing facts.
- **SER-005**: Workflow-affecting features MUST define deterministic behavior for validation failure, provider failure, evaluator failure, and rewrite limits.
- **SER-006**: Runtime-affecting features MUST preserve structured logs, report storage, health checks, and metrics required by the constitution.
- **SER-007**: Optimization output MUST state that backtest and out-of-sample historical evaluation reduce but do not eliminate overfitting risk.
- **SER-008**: Optimization output MUST not describe selected parameters as recommendations, signals to trade, or future-return expectations.
- **SER-009**: Evaluator checks MUST fail any report that presents robust parameters as guaranteed, advice-like, or appropriate for automatic execution.

### Key Entities *(include if feature involves data)*

- **Optimization Run**: A single user-requested evaluation with ticker, date range, cost assumptions, scoring policy, fold setup, and summary status.
- **Parameter Candidate**: A parameter set evaluated by return, risk, trade behavior, cost-adjusted performance, fold stability, and warnings.
- **Walk-Forward Fold**: A time-ordered selection period and subsequent validation period with out-of-sample metrics.
- **Robust Score**: A transparent score combining out-of-sample performance, downside control, fold stability, worst-period resilience, and turnover penalties.
- **Market Regime Segment**: A labeled market condition used to summarize candidate behavior by bull, bear, sideways, high-volatility, or low-volatility periods.
- **Optimization Evidence Item**: A traceable evidence record for numeric optimization claims used in reports or agent outputs.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of robust optimization result views include more than total return when enough data is available.
- **SC-002**: 100% of robust optimization result views include transaction-cost-adjusted performance.
- **SC-003**: 100% of valid walk-forward runs report every fold's train period, test period, and out-of-sample metrics.
- **SC-004**: At least 95% of deterministic test cases correctly reject candidates with fewer than 30 completed trades or maximum drawdown above 25%.
- **SC-005**: At least 95% of deterministic test cases correctly penalize candidates with strong average performance but weak worst-fold or unstable fold results.
- **SC-006**: Users can compare robust optimized parameters, manual parameters, and passive baseline performance in one result view.
- **SC-007**: 100% of report outputs containing optimization metrics include no-recommendation framing and the required disclaimer.
- **SC-008**: All new optimization, walk-forward, regime, and safety tests run without live external API calls.

## Assumptions

- The first implementation should prioritize one robust all-regime parameter candidate before adding regime-specific parameter application.
- Weighted robust scoring is the MVP default because it is easier to explain than a full multi-objective Pareto workflow.
- Multi-objective candidate exploration may be considered later if the result presentation remains understandable and safety-compliant.
- Regime labels are used for research explanation and validation, not for automated order execution.
- Persistent storage beyond local project storage is out of scope for the current portfolio MVP unless a later plan explicitly expands project infrastructure.
- Optimization output is evidence for scenario comparison only and must not become financial advice or a trading system.
