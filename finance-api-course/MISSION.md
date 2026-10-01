# Mission: Make the Personal Finance API production-ready

## Why
I want to land a backend engineering job. A finished-looking CRUD API does not
prove I am hireable; being able to *find* a real flaw in my own code, explain
why it is dangerous, and fix it the way a production team would is exactly what
interviewers probe for. This repo is the artifact I will walk an interviewer
through.

## Success looks like
- I can take any endpoint in `personal_finance_api` and name how it fails on
  security, performance, reliability, testing, or observability — before being
  told.
- For each weakness I can first *demonstrate the exploit or failure* (a request
  that should be refused but isn't, a query that falls over), then ship the fix
  and a test that would catch a regression.
- The repo has: enforced per-user authorization, a hardened auth/token setup,
  input and rate limits, a real test suite, a CI/CD pipeline that blocks broken
  merges, and structured logs + health checks.
- I can answer, out loud and from memory, "how is this production-ready?" across
  all five pillars in an interview.

## Constraints
- Learning by breaking first: every lesson proves the loophole before fixing it.
- Grounded in the `Backend-Engineers-Guide` repo plus high-trust external
  sources (OWASP, official docs) — not my own guesses or the model's.
- Stack is fixed: TypeScript, Express 5, Prisma, PostgreSQL. No rewrites.
- Short lessons, one tangible win each.

## Out of scope (for now)
- Rewriting to another framework/language (NestJS, Go, etc.).
- Frontend work.
- The AI/LLM endpoints as a feature — we touch them only where they create a
  security/cost/reliability risk.
- Kubernetes and multi-region infra. We stop at "deployable and observable."
