# Decisions

1. **Data strategy:** committed parquet snapshot built by a script, not live Elexon calls at runtime. Reason: deterministic deploys, independence from API uptime, fixed January 2026 scope.
2. **Decision cutoff:** 09:00 UK local time on D-1, configurable. Assumption, to be verified against GB day-ahead auction timing.
3. **Hosting:** Vercel (static front end + serverless FastAPI). Docker used for local dev, CI and portability only, because Vercel doesn't run Dockerfiles.
4. **Error sign convention:** forecast minus actual (positive = over-forecast).
5. **TypeScript pinned to 6.0.x:** typescript-eslint doesn't yet support TypeScript 7 (peer range `<6.1.0`).