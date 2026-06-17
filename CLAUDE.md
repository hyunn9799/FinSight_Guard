# FinSight Guard — Claude Code Guidelines

## gstack

All web browsing must use the `/browse` skill from gstack. Never use `mcp__claude-in-chrome__*` tools directly.

### Available gstack skills

- `/office-hours` — open-ended pair-programming session
- `/plan-ceo-review` — review plan from CEO perspective
- `/plan-eng-review` — review plan from engineering perspective
- `/plan-design-review` — review plan from design perspective
- `/design-consultation` — design consultation session
- `/design-shotgun` — generate multiple design directions fast
- `/design-html` — produce an HTML prototype
- `/review` — code review
- `/ship` — ship a change end-to-end
- `/land-and-deploy` — land and deploy to production
- `/canary` — canary deploy
- `/benchmark` — run benchmarks
- `/browse` — browse the web (use this for ALL web browsing)
- `/connect-chrome` — connect to a running Chrome instance
- `/qa` — full QA pass
- `/qa-only` — QA without code changes
- `/design-review` — design review
- `/setup-browser-cookies` — configure browser cookies for auth
- `/setup-deploy` — configure deployment
- `/setup-gbrain` — configure gbrain
- `/retro` — run a retrospective
- `/investigate` — investigate a bug or issue
- `/document-release` — document a release
- `/document-generate` — generate documentation
- `/codex` — codex-style research
- `/cso` — chief security officer review
- `/autoplan` — automatically generate a plan
- `/plan-devex-review` — review plan from devex perspective
- `/devex-review` — developer experience review
- `/careful` — careful, high-stakes change workflow
