# Feature Specification: Robust Backtest Validation

**Feature Branch**: `[001-robust-backtest-validation]`

**Created**: 2026-06-17

**Status**: Draft

**Input**: User description: "optuna를 쓰는데 특정 시기의 가장좋은 수익률을 토대로 최적파라미터를 뽑아서 적용시키는데 이 방법 별로 안좋은것 같아 어떻게하면 좋을까 지금 프로젝트 코드 이해한다음에 답변해줘"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Validate Optimization Robustness (Priority: P1)

As a research user, I want optimization results to show whether a parameter set remains reasonable outside the period it was selected from, so that I do not mistake a single-period best result for a reliable strategy.

**Why this priority**: The current optimization experience can overemphasize the best historical simulation return from one selected period, which creates a high risk of overfitting and misleading interpretation.

**Independent Test**: Can be fully tested by running an optimization on deterministic historical sample data and confirming the result includes separate selection-period and validation-period outcomes with a clear robustness status.

**Acceptance Scenarios**:

1. **Given** a user runs strategy optimization over a historical range, **When** the optimized result is displayed, **Then** the result separates the period used to select parameters from at least one later or held-out validation period.
2. **Given** the selected parameters perform well in the selection period but materially worse in validation, **When** the result is displayed, **Then** the system labels the result as fragile or overfit-risk rather than presenting it as the best parameter set to apply.
3. **Given** the available history is too short to support a meaningful validation split, **When** the user requests optimization, **Then** the system discloses the limitation and avoids presenting the result as robust.

---

### User Story 2 - Compare Risk-Adjusted Outcomes (Priority: P2)

As a research user, I want optimized strategy results to include downside and stability measures, so that I can compare scenarios using more than headline return.

**Why this priority**: A parameter set with the highest historical return may be unacceptable if it depends on high drawdown, too few trades, excessive churn, or unstable performance across subperiods.

**Independent Test**: Can be tested independently by evaluating a completed optimization result and confirming that return, drawdown, trade count, validation degradation, and warnings are visible together.

**Acceptance Scenarios**:

1. **Given** optimization produces candidate parameter sets, **When** the user reviews results, **Then** the system shows both return-focused and risk-focused metrics for the selected candidate.
2. **Given** two candidates have similar returns but different downside profiles, **When** the result is summarized, **Then** the safer candidate is distinguishable from the more fragile candidate.
3. **Given** a candidate has too few trades to be meaningful, **When** it is shown, **Then** the system warns that the sample is insufficient.

---

### User Story 3 - Preserve Research-Only Framing (Priority: P3)

As a research user, I want optimization and backtest output to remain clearly educational and scenario-based, so that the product does not imply automated trading advice.

**Why this priority**: The project constitution prohibits direct buy, sell, hold, guaranteed return, or trading-instruction behavior.

**Independent Test**: Can be tested independently by checking optimization output text and report evidence to ensure it contains no direct recommendation language and includes required limitations.

**Acceptance Scenarios**:

1. **Given** a user views optimized parameters, **When** the result is rendered, **Then** the system describes them as historical simulation candidates, not recommended trading settings.
2. **Given** optimized backtest evidence is included in a research report, **When** the report is evaluated, **Then** it includes evidence, limitations, and the required no-recommendation disclaimer.

### Edge Cases

- Available data is shorter than the minimum period required for both selection and validation.
- The best selection-period candidate has no trades or only one completed trade.
- The validation period contains no qualifying signals.
- Optimization finds a very high return with extreme drawdown or concentrated gains.
- Data loading partially fails for one validation segment.
- User changes the date range so validation would extend beyond available market data.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST distinguish between parameter selection performance and out-of-sample validation performance in optimization results.
- **FR-002**: System MUST avoid choosing final displayed parameters solely by the highest return in one user-selected period when validation data is available.
- **FR-003**: Users MUST be able to see the date ranges used for parameter selection and validation.
- **FR-004**: System MUST show robustness warnings when validation performance materially degrades relative to selection-period performance.
- **FR-005**: System MUST include risk and reliability indicators alongside return, including at minimum downside loss exposure, number of trades, and validation degradation.
- **FR-006**: System MUST flag candidate results with insufficient trade samples as low-confidence.
- **FR-007**: System MUST explain when available data is insufficient for validation and continue with a clearly limited historical simulation view.
- **FR-008**: System MUST allow users to compare the current manual parameter result against the optimized candidate using the same evaluation periods.
- **FR-009**: System MUST label all optimized outputs as historical simulation candidates and not as recommended parameters.
- **FR-010**: System MUST preserve deterministic behavior in tests by using fixed sample data rather than live market or news providers.

### Safety, Evidence & Reliability Requirements *(mandatory for research workflow changes)*

- **SER-001**: System MUST avoid direct buy/sell/hold recommendations, trading instructions, guaranteed return claims, guaranteed target claims, and order execution behavior.
- **SER-002**: System MUST back important numeric or factual report claims with `EvidenceItem` records when the feature affects research output.
- **SER-003**: Final Korean reports MUST include the required education-only, no-recommendation disclaimer exactly as defined in the constitution.
- **SER-004**: System MUST disclose unavailable or degraded market, fundamental, or news data instead of fabricating missing facts.
- **SER-005**: Workflow-affecting features MUST define deterministic behavior for validation failure, provider failure, evaluator failure, and rewrite limits.
- **SER-006**: Runtime-affecting features MUST preserve structured logs, report storage, health checks, and metrics required by the constitution.
- **SER-007**: Backtest optimization output MUST disclose that historical simulation performance does not guarantee future performance.
- **SER-008**: Backtest optimization output MUST not trigger brokerage actions, order placement, or automated portfolio changes.

### Key Entities *(include if feature involves data)*

- **Optimization Candidate**: A set of strategy parameters with selection-period return, validation-period return, trade count, downside metrics, and warnings.
- **Validation Segment**: A historical time range reserved for checking whether candidate behavior persists outside the selection period.
- **Robustness Summary**: A user-facing classification and explanation of candidate stability, degradation, sample sufficiency, and known limitations.
- **Backtest Evidence Item**: A traceable record for numeric claims from historical simulations, including metric name, value, ticker, collection time, and source description.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of optimization result views show both selection-period and validation-period outcomes when enough history is available.
- **SC-002**: 100% of optimization result views disclose a limitation when validation cannot be performed due to insufficient data.
- **SC-003**: At least 95% of deterministic test cases correctly classify fragile candidates when validation return materially underperforms selection return.
- **SC-004**: Users can compare manual parameters and optimized candidates across the same evaluation periods in one result view.
- **SC-005**: Evaluator checks reject reports that present optimized parameters as guaranteed, recommended, or advice-like settings.
- **SC-006**: All new optimization tests run without live external API calls.

## Assumptions

- The feature improves interpretation of historical simulation results rather than creating a trading or recommendation engine.
- Validation should use available historical data only and must not imply future predictive certainty.
- If only a small amount of data is available, the product should prefer a limitation warning over a false sense of precision.
- Existing backtest strategy behavior can remain available for manual experimentation while optimized results receive stronger robustness framing.
- Existing project safety language and report disclaimer remain mandatory wherever optimization output is included in final research reports.
