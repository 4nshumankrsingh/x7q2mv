# Decisions

1. **Data strategy:** committed parquet snapshot built by a script, not live Elexon calls at runtime. Reason: deterministic deploys, independence from API uptime, fixed January 2026 scope.
2. **Decision cutoff:** 09:20 UK local time on D-1, configurable. This is the gate closure of the EPEX GB day-ahead hourly auction, the earliest of the main GB day-ahead auctions (N2EX closes at 09:50; a later EPEX half-hour auction runs at 15:30). Sensitivity to other cutoffs is reported by `scripts/analyze`.
3. **Hosting:** Vercel (static front end + serverless FastAPI). Docker used for local dev, CI and portability only, because Vercel doesn't run Dockerfiles.
4. **Error sign convention:** forecast minus actual (positive = over-forecast).
5. **TypeScript pinned to 6.0.x:** typescript-eslint doesn't yet support TypeScript 7 (peer range `<6.1.0`).
6. **Selection rule:** for each target, the latest forecast with publishTime <= cutoff (inclusive). Targets with no eligible forecast are dropped and counted.
7. **Alignment:** WINDFOR is hourly, FUELHH half-hourly. Default compares the forecast at hour H with the actual at startTime H; the mean of the two half-hours is reported as a sensitivity.
8. **Uncertainty band:** empirical P10/P90 of forecast error from the same month. This is in-sample, so it is somewhat optimistic.