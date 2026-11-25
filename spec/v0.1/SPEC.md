# Change Evidence Spec v0.1 (draft)

A data format for one question: did this change to an AI system help, on
agreed criteria, compared with a baseline. The runner in this repo is one
implementation. Anything that writes these documents is a producer.

Schemas are in `schemas/`. YAML is the authoring format for declarations;
everything else is JSON.

## Objects

| object | file | identity |
|---|---|---|
| Change | inside `evidence.json` | `change_id` |
| Declaration | `impact.yaml`, resolved copy in `declaration.json` | `declaration_hash` |
| Execution | inside `evidence.json`, full plan in `manifest.json` | `spec_hash`, `plan_digest`, `run_id`, `attempt_id` |
| Observation | one line of `observations.jsonl` | (arm, seed, unit, metric), plus (replicate, sequence) under a measurement protocol |
| Comparison | inside `evidence.json` | derived |
| Decision | one line of `decisions.jsonl` | `decision_id` |
| Deployment / outcome | `outcome.schema.json` | draft only; no producer yet |

## Canonical form

Hashes are sha256 over canonical JSON: UTF-8, object keys sorted, no
whitespace, a float holding an integral value written as an integer, NaN and
Infinity rejected. Defaults are applied before hashing, so an omitted default
and a spelled-out default are the same document.

## Identities

- `declaration_hash`: objective, criteria (sorted by id), method, validity
  checks and data scope. It answers "what was asked". Stable across reruns
  and across candidates. A threshold edit changes it.
- `env_key` (per arm): arm name, commit sha, setup commands, the arm's
  environment variables, backend, image and image content id. It answers
  "what must be built". Two arms never share one, even at the same commit. A
  threshold edit does not change it.
- `spec_hash`: resolved shas, harness digest, setup, command, arm
  environments, seeds, execution settings and budget. It answers "what gets
  executed". A threshold edit does not change it. Under a measurement
  protocol it also covers the protocol and the run schedule.
- `change_id`: repo identity, baseline sha, candidate sha, arm environments.
  A new candidate commit is a new change under the same declaration.
- `plan_digest`: the full resolved plan. This is what a person approves.
- `run_id`: one per execution of an approved plan.
- `attempt_id`: one per (arm, seed, try), or per (arm, seed, replicate,
  sequence, try) under a measurement protocol. A retry is a new attempt.
- `bundle_digest`: sha256 over the bundle's file list and file digests.

## Declaration

Required: `spec_version`, `change` (repo, baseline, candidate), `run`
(command), `criteria`. See `schemas/declaration.schema.json` and the demo
(`remyx-validate demo`) for a complete file.

A criterion has an `id`, a `metric`, a `direction` (`higher` or `lower`), an
`aggregate` (`mean`, `sum`, `min`, `max`, `median`, `pNN`) and a `role`:

- `target`: the improvement, in the good direction, must be at least
  `min_improvement` (default 0, meaning strictly better).
- `guardrail`: `max` or `min` on the candidate value, and/or
  `max_regression` against the baseline. A breach vetoes.
- `info`: reported, never gates.

`relative: true` reads `min_improvement` and `max_regression` as a fraction
of the baseline point estimate. The baseline's own uncertainty is not
propagated into the threshold.

A bound that does not apply to a criterion's role is an error, not ignored.

`pairing: seed` or `pairing: example` on a criterion overrides `method.unit`
for that criterion. A gating criterion never changes unit on its own.

`method.unit` names the independent sampling unit: `example` (observations
carry unit ids and are paired by id) or `seed` (one value per run, paired by
seed). `method.uncertainty` is `interval` (default) or `point`.

`validity.activation` lists metrics that must differ between arms (a witness
that the change took effect). `validity.held_constant` lists metrics that
must be equal. Both are judged on every declared seed, arm against arm
within the seed, and a reading absent from any seed is a missing control.
`validity.baseline_expect` pins the baseline to a known value within a
tolerance.

## Observations and coverage

An observation is identified by (arm, seed, unit, metric). A second value for
the same identity, a seed the declaration does not list, or one unit carrying
two slice labels makes the observations malformed.

Coverage is counted against what was declared, and nothing absent is filled
in:

- `seed` pairing: one pair per declared seed. A run-level value is used as
  reported. If a seed has only per-unit values, both arms are restricted to
  the units both reported before aggregating.
- `example` pairing: a (seed, unit) cell counts only when both arms reported
  it. A unit's value is the mean over its matched seeds. The expected number
  of cells is declared seeds times units, where units is `data.expected_units`
  if declared and otherwise the units either arm reported. Without
  `data.expected_units`, a unit missing from both arms cannot be noticed.

`method.max_missing_rate` (default 0) is the share of expected measurements
that may be unpaired.

## Measurement protocol

A criterion is a timing criterion when it says `performance: true`, or when
it does not say and its metric name or unit reads as a timing (the producer
records the inference as `performance: true` in the resolved declaration,
with the extension flag `x-performance-inferred`). `performance: false` opts
out. A declaration with a timing criterion and no `measurement` block gets
the default block: `design: crossover`, one replicate, `schedule_seed: 0`,
`cache.profile: cold`.

A declaration with no timing criterion and no `measurement` block has the
v0.1.0a0 behaviour and the same hashes: one pass per seed, order alternating
by the seed's position, no cache reset. Evidence produced that way is not
order-controlled, and a producer must say so.

With a `measurement` block, declared or defaulted:

- **Schedule.** A is the baseline, B the candidate. A block is one (seed,
  replicate). `design: crossover` (default) runs both sequences, AB and BA,
  in every block: four measured attempts. `design: single` runs one sequence
  per block. Block order and which sequence leads are drawn from
  `schedule_seed` with SHA-256 in counter mode and are balanced across
  blocks. Workload seeds are sorted first and never feed the scheduler. The
  schedule is part of the plan and of `spec_hash`.
- **Slots.** Each attempt records `seed`, `replicate`, `block`, `sequence`,
  `period` (its position in the sequence), `planned_ordinal` and `ordinal`
  (the order it ran in). Observations carry `replicate`, `sequence` and
  `period`.
- **Retries.** The unit retried is the sequence. If either arm fails, both
  are run again. Attempts from the abandoned try stay in the ledger with
  `discarded: true` and contribute no observations.
- **Cache isolation.** Before every measured attempt the producer creates a
  new, empty directory it owns and points the mutable cache locations at it.
  `cache.profile: warm` then runs the command `warmup_runs` times untimed for
  that arm. `reset_hooks` run before every measured attempt. A failed reset,
  hook or warm-up ends the attempt with status `reset_failed` or
  `warmup_failed` before anything is measured. The protocol record in
  `execution.protocol` lists what was isolated and what was not controlled.
- **Nesting.** Runs of a seed are averaged within each independent unit
  before any estimate. A unit absent from any of its runs is excluded and
  counts toward the missing rate. Controls and witnesses are judged in every
  measured pair, and a held-constant metric must also be equal across all
  runs of the same seed.
- **Order effect.** For a criterion with `performance: true` under the
  crossover, let dAB and dBA be candidate minus baseline in each sequence.
  The order effect is dAB − dBA, estimated over units with the criterion's
  aggregate (t interval for `mean`, unit bootstrap otherwise). A gating
  performance criterion must declare `order_tolerance`. Interval inside
  ±tolerance: judged normally. Interval entirely outside: the criterion's
  status is `unassessed` and the check `order_bias` is recorded as failed.
  Otherwise, or with no interval: the criterion is undecided. Under
  `uncertainty: point` the point estimate decides. The first-position and
  second-position contrasts and the drift between a block's two sequences
  are reported and not gated.
- **Fingerprints.** A result may carry `"fingerprints": {name: text}`. Each
  becomes the run-level metric `fingerprint.<name>`: the first 48 bits of the
  SHA-256 of the text, as a number.
- **Conditions.** `execution.protocol.host` and each attempt's `conditions`
  are self-reported readings. Nothing is computed from them.
- **Exploratory.** Under `design: single`, or when `cache.uncontrolled` lists
  shared state, a gating performance criterion that would be met or clear is
  reported as `exploratory` instead.
- A gating timing criterion under the crossover without `order_tolerance`
  is invalid.

See [docs/MEASUREMENT.md](../../docs/MEASUREMENT.md) for guidance.

## Estimators

- `mean`: Student t interval on per-unit paired differences.
- other aggregates: paired percentile bootstrap, 2000 resamples, seeded from
  the declaration hash and criterion id so a recomputation matches. Needs at
  least 10 paired units.
- one paired unit: point estimate, no interval.

Under `uncertainty: interval`, a target is met when the interval's lower
bound clears the threshold, not met when its upper bound is below it, and
undecided otherwise. A guardrail is breached when the interval is entirely on
the wrong side of the limit. No interval means undecided. Under `point`, the
point estimate decides.

## Verdict

First match wins:

1. INCOMPLETE: execution did not complete, an integrity check failed, or
   the observations are malformed.
2. NO_SIGNAL: a held-constant metric drifted, or the baseline is outside its
   expected range.
3. INCOMPLETE: a gating metric, control or witness is missing, or more
   measurements are unpaired than `method.max_missing_rate` allows.
4. NO_SIGNAL: an activation witness is identical across arms on any seed, or
   the arms are identical on every gating metric and no witness is declared.
5. FAIL: a guardrail breached or a target not met. An `unassessed` criterion
   never causes a FAIL.
6. NO_SIGNAL: a gating performance criterion is `unassessed` because its
   order effect is outside its tolerance.
7. WARN: a gating criterion undecided or exploratory.
8. PASS.

Identical arms with a passing activation witness is a measured zero on the
declared data. It is judged like any other result and carries the reason
`measured_zero`. It is not a proof of equivalence: there is no equivalence
margin in v0.1.

Intervals are per criterion at the declared confidence. They are not a joint
statement over all criteria.

A decision is separate from the verdict and from any merge. A deployment
outcome is linked later and is never inferred from a merge.

## Verification

A consumer holding a bundle can check, without running anything:

- every file against `checksums.json`, and that no file is unlisted
- the schemas, with non-finite numbers rejected
- that the plan in `manifest.json` hashes to `plan_digest`, and that the
  approval is for that digest
- that the declaration is identical in `manifest.json`, `declaration.json`
  and `evidence.json`, and hashes to `declaration_hash`
- that `spec_hash`, `change_id` and each `env_key` recompute from the plan
- that every observation belongs to a counted, successful attempt with its
  arm, seed and run order
- under a measurement protocol: that the recorded schedule is the one the
  declaration derives, that every counted attempt is one of its slots and
  started from a clean cache, and that the two arms of a kept sequence ran
  back to back
- that the full comparison (estimates, intervals, counts, statuses, checks,
  reasons, limitations, verdict) recomputes from the observations
- that `report.md` is what `evidence.json` renders to

The integrity findings recorded by the runner are inputs to the comparison
and are taken as recorded. Verification shows internal consistency. It is
not authentication and it is not reproduction.

A redacted public export is a derivative with its own digest. It records the
digest of the bundle it came from and the redaction policy. A producer must
refuse to redact anything an identity is computed from.

## Extensions

Keys starting with `x-` are allowed where the schema says so. Producers and
consumers preserve them, do not interpret keys they do not know, and leave
them out of every hash. Use a namespace: `x-yourorg-field`.

## Compatibility

`spec_version` is `major.minor`. A minor version may add optional fields. A
consumer ignores optional fields it does not know and rejects a major
version it does not know. Hash inputs do not change within a major version.
v0.x is a draft and may break between minors until 1.0.

## Open questions for review

- Whether `objective` belongs in `declaration_hash` (a wording fix currently
  changes the hash).
- Canonicalization is described in prose and implemented in Python. An
  independent producer needs normative test vectors, or a move to a published
  canonical JSON scheme, before the hashes can be called interoperable.
- `x-` keys are left out of every hash. That is right for annotations and
  wrong for anything that changes what runs. The rule for telling them apart
  is not written yet.
- How these identities line up with build-cache and declaration identities
  in other tools, including managed Validate. Field names are not reconciled.
- An equivalence margin, so "no worse than" can be a declared claim.
- An expected-unit manifest (ids, not a count) from the harness.
- Relative thresholds that account for baseline uncertainty.
- Slices are reported as point estimates and are not gated.
- The deployment/outcome object is a stub.
