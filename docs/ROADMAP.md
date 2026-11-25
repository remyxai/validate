# What this alpha does, and what comes next

## In this release

| Area | State |
| --- | --- |
| Change Evidence Spec v0.1 | Draft. Five JSON Schemas, canonical hashing rules, identities. The deployment/outcome schema is a stub with no producer. |
| Paired comparison | Pairing by example id or by seed, coverage against declared seeds, t and bootstrap intervals, targets and guardrails, per-seed controls and activation witnesses. |
| Plan and approval | Refs resolved to commits, harness digested, one plan digest to approve. A changed plan is refused with the field that changed. |
| Runner | Local subprocess backend for trusted code. Docker backend, experimental. Per-arm checkouts and environments, matched seeds, retry ledger, timeout, cancel, wall-clock cap. |
| Run order and caches | On by default for timing criteria. Measurement protocol: seed-matched crossover schedule frozen in the plan, a new empty cache directory before every measured attempt, cold and warm profiles, reset hooks, and an order-effect check with a declared tolerance. See [MEASUREMENT.md](MEASUREMENT.md). |
| Evidence | `evidence.json`, `report.md`, manifest, observations, harness snapshot, checksums. `verify` recomputes the comparison and identities. Decisions are recorded separately. |
| Export | Private copy, or a redacted public derivative from an allowlist of files. |
| Examples | An offline retrieval demo with fourteen fixtures, and two recorded CPU comparisons on a PEFT pull request: one from 0.1.0a0 and one with both run orders. |

## Known gaps

- Environment setup is whatever your `setup` commands do. Nothing reads a
  repository's lockfile or Dockerfile for you, and transitive dependencies are
  not locked.
- No spend estimate and no hard spend limit. A cost cap is a check on what the
  harness reports after each attempt.
- The local backend shares one environment directory across seeds of an arm.
  Under a measurement protocol each attempt gets its own cache directories;
  without one, only the docker backend gives each attempt fresh writable state.
- The measurement protocol does not clear the OS page cache, hardware state,
  GPU driver caches or remote services, and says so in every report. Host,
  load and GPU readings are recorded and shown, not gated. The drift
  diagnostic compares a seed's two sequences; it is not a model of drift
  across the whole run.
- Timing criteria are recognised by metric name and unit. A timing with an
  unusual name needs `performance: true`.
- Reset hooks are commands. There is no built-in reset for a particular
  database, model server or provider.
- Slices are reported, not gated.
- Relative thresholds use the baseline point estimate.
- The standalone runner and managed Validate are separate implementations.
  There is no bundle import into the managed product.
- No signing. A bundle is self-reported.
- No recorded API or GPU recipe yet. The examples make no paid calls. The
  measurement guide has notes for both; neither has been run end to end.

## Next

1. Agent guidance and a local MCP server over the same plan, run and report
   steps, with approval still bound to a plan digest.
2. One application recipe that reuses an existing evaluator, and one adapter
   from an existing evaluation framework's results to observations.
3. An advisory CI integration and published conformance fixtures.
   `run --strict` exits zero on WARN today, so it is not a required-pass gate.
4. Compatible semantics between this runner and managed Validate on shared
   fixtures.
5. Optional, consented sharing of evidence history. Off by default.
6. Linking authorized deployment outcomes, and testing prospectively whether
   past evidence improves the choice of the next change.

Items 4 to 6 depend on what early users ask for. None of them is a claim
about this release.
