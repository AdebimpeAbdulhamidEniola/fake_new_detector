# Production-Ready Backend API Resources

## Knowledge

- [OWASP API Security Top 10 (2023)](https://owasp.org/API-Security/editions/2023/en/0x11-t10/)
  The industry-standard list of how APIs get breached. Use for: naming and
  prioritising security weaknesses. API1 (Broken Object Level Authorization)
  and API2 (Broken Authentication) are our first two targets.
- [OWASP: Broken Object Level Authorization (API1:2023)](https://owasp.org/API-Security/editions/2023/en/0xa1-broken-object-level-authorization/)
  The exact flaw in our transaction update/delete endpoints. Use for: the
  definition, attack scenarios, and the "How To Prevent" checklist.
- [OWASP Authorization Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html)
  Deny-by-default, enforce-on-every-request authorization patterns. Use for:
  how to structure the fix, not just where.
- `Backend-Engineers-Guide/notes/08_security/01_auth.md` (local clone)
  Authentication vs authorization, and "never trust frontend-only checks." Use
  for: grounding the mental model behind every security lesson.
- `Backend-Engineers-Guide/notes/08_security/04_security_best_practices_and_measures.md`
- `Backend-Engineers-Guide/projects/security_controls_lab/` (local clone)
  Runnable demos for hashing, JWT signing/expiry, and rate limiting. Use for:
  seeing the defensive primitive in isolation before applying it to our code.
- [Prisma docs](https://www.prisma.io/docs) — pagination, relation filters,
  transactions. Use for: performance and the correct way to scope queries by user.
- [Express 5 docs](https://expressjs.com/) — middleware, error handling.

## Wisdom (Communities)

- [r/ExperiencedDevs](https://reddit.com/r/ExperiencedDevs) and
  [r/node](https://reddit.com/r/node) — for "is this production-ready?" critique
  of a design or PR. Use for: sanity-checking decisions before an interview.
- [The Pragmatic Engineer](https://blog.pragmaticengineer.com/) — what hiring
  teams actually expect from backend engineers.

## Gaps
- No vetted source yet for CI/CD specifics on this stack (GitHub Actions +
  Prisma migrations). To find before the CI/CD lesson.
- No vetted observability source yet (structured logging + health checks in
  Express). To find before the observability lesson.
