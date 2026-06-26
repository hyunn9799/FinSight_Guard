# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]

**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

[Extract from feature spec: primary requirement + technical approach from research]

## 학습 단계 브리핑

<!--
  필수 작성: plan을 채우기 전에 이번 단계에서 어떤 판단을 할지 먼저 설명한다.
  범위, 기술 경계, evidence/safety 영향, 검토할 절충안, assumptions,
  예상 변경 파일을 포함한다.
-->

**이번 단계의 목적**: [학습자가 이 plan을 왜 읽어야 하는지 설명]

**이번 단계에서 확인할 판단**:
- [범위 결정]
- [아키텍처 또는 의존성 결정]
- [안전성/evidence/관측성 관련 결정]

**Assumptions**:
- [애매한 요구사항을 임의로 확정하지 말고 assumptions로 기록]

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: [e.g., Python 3.11, Swift 5.9, Rust 1.75 or NEEDS CLARIFICATION]

**Primary Dependencies**: [e.g., FastAPI, UIKit, LLVM or NEEDS CLARIFICATION]

**Storage**: [if applicable, e.g., PostgreSQL, CoreData, files or N/A]

**Testing**: [e.g., pytest, XCTest, cargo test or NEEDS CLARIFICATION]

**Target Platform**: [e.g., Linux server, iOS 15+, WASM or NEEDS CLARIFICATION]

**Project Type**: [e.g., library/cli/web-service/mobile-app/compiler/desktop-app or NEEDS CLARIFICATION]

**Performance Goals**: [domain-specific, e.g., 1000 req/s, 10k lines/sec, 60 fps or NEEDS CLARIFICATION]

**Constraints**: [domain-specific, e.g., <200ms p95, <100MB memory, offline-capable or NEEDS CLARIFICATION]

**Scale/Scope**: [domain-specific, e.g., 10k users, 1M LOC, 50 screens or NEEDS CLARIFICATION]

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Learning-oriented Spec Kit governance: 산출물 생성 전에 계획 판단을
  설명하고, assumptions를 명시하며, 최종 plan 요약에 핵심 결정, 대안,
  선택 이유, 반드시 읽을 섹션이 포함될 때만 PASS.
- Evidence grounding: important numeric or factual claims map to `EvidenceItem`
  records and the report includes an evidence summary.
- Financial safety: no buy/sell/hold recommendation, no trading/order execution,
  no guaranteed returns or target guarantees, and the required Korean disclaimer
  is preserved for final reports.
- LangGraph workflow: role-based agents, conditional routing, retry/fallback
  behavior, Evaluator pass/fail routing, and max two Rewrite attempts are
  represented when the feature touches workflow behavior.
- Deterministic quality: relevant tests avoid live external APIs and cover safety,
  evaluator, routing, indicators, storage, and API behavior as applicable.
- Observability: structured logs, report persistence, health checks, and metrics
  remain available for affected workflow paths.

**Constitution Result**: [PASS/FAIL with notes and any justified exceptions]

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
# [REMOVE IF UNUSED] Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: [Document the selected structure and reference the real
directories captured above]

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |

## 단계 완료 리뷰

<!--
  필수 작성: plan과 설계 산출물을 만들거나 수정한 뒤 학습 관점의 결과를
  요약한다.
-->

- **이번 단계의 목적**: [이 단계에서 결정하려던 것]
- **새로 생긴 파일 또는 변경된 파일**: [생성/수정 파일]
- **핵심 설계 결정**: [결정 목록]
- **검토한 대안**: [대안과 절충점]
- **현재 설계를 선택한 이유**: [선택 근거]
- **반드시 읽어야 할 부분**: [학습자가 확인할 구체적인 섹션]
