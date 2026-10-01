# Notes

## Learner preferences
- Goal: land a backend engineering job. Frame every lesson around what an
  interviewer would ask / be impressed by.
- Teaching style: **break it first, then fix it.** Never present a fix before
  the learner has seen the loophole exploited or the failure happen.
- Ground claims in cited sources (OWASP, Backend-Engineers-Guide, official
  docs). Avoid unsourced assertions.

## Repo facts (so we don't re-derive each session)
- `personal_finance_api` clone is READ-ONLY here
  (`/home/user/adebimpeabdulhamideniola/personal_finance_api`) — cloned without
  the learner's GitHub login, so we cannot push to it from this session.
- Stack: TypeScript, Express 5, Prisma 7, PostgreSQL. Path alias `@/` -> `src/`.
- `test` script is a placeholder (`exit 1`) — no tests exist yet.
- Money stored as `Float` (precision risk, later lesson).
- Node 22 + `psql` (Postgres 16) + `docker` are available in this container, so
  we can actually run the API against a throwaway DB to prove exploits.

## Backlog of proven/ suspected loopholes (ordered)
1. **BOLA** — transaction update/delete look up by `id` only, never check
   ownership. Any user can edit/delete anyone's transactions. (LESSON 1)
2. JWT secret falls back to `"default_secret"` if env unset -> token forgery.
3. No rate limit on `/login` -> password brute force.
4. `/api/reports/*` mounted without `authenticate` middleware -> always 401
   (reliability bug) AND relies on `req.userId` that is never set.
5. No pagination on `GET /transactions`.
6. `console.log(process.env.DB_URL)` on boot -> leaks DB creds to logs.
7. Float money.
8. `cors()` wide open.
9. No tests, no CI, no health check, no structured logging.

## Session log
- S1: Mission set (backend job). Explored repo + guide + OWASP. Built workspace.
  Shipped Lesson 1 (BOLA).
