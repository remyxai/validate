# Comparing latency and throughput

Quality metrics mostly do not care which version ran first. Timing metrics
do. A cache filled by the first run, a JIT that already compiled, a warmer
CPU or a service that remembered a prefix can all hand the second run a head
start. If each seed runs the two versions in one order only, that head start
is counted as part of the change.

This was raised on the launch thread: ["Does the report show which version
ran first? A warm cache could give whichever version runs second a head start
on latency."](https://huggingface.co/posts/salma-remyx/800639086221839#6b06424baeec2abfe33cf45a)

Validate controls for it by default. Any criterion whose metric name or unit
reads as a timing (`latency`, `throughput`, `duration`, `ttft`, names ending
in `_ms`, `_s`, `per_s`, units like `ms` or `req/s`) is compared under the
measurement protocol below, with no `measurement` block needed. The plan
tells you when that happened. Two switches, both on the criterion:

- `performance: true` for a timing the name does not give away
- `performance: false` to say a metric is not a timing, or to accept that
  run order is not controlled for it. The plan and the report then say so.

## What the protocol does

```yaml
criteria:
  - id: p95_latency
    metric: latency_ms
    role: guardrail
    direction: lower
    aggregate: p95
    max: 800
    order_tolerance: 5      # ms: the order effect you would call unimportant

measurement:                # optional: these are the defaults
  design: crossover         # `single` is cheaper and exploratory only
  replicates: 1
  schedule_seed: 0          # orders the runs. Never passed to the workload
  cache:
    profile: cold           # or `warm`
```

**Both orders on every seed.** Under `crossover`, each seed and replicate
runs the baseline then the candidate (AB) and the candidate then the baseline
(BA). Each arm is measured once in first position and once in second position
on the same seed: four measured attempts per seed per replicate. Which order
goes first, and the order of the seeds, come from `schedule_seed`. Nothing is
derived from a workload seed's value or its position in the list.

**A schedule you approve.** The full list of attempts is in the plan, with
the attempt count and the number of untimed warm-up runs. It is covered by
the plan digest, so a changed schedule needs a new approval. The report lists
the attempts in the order they ran next to the order that was planned.

**An empty cache before every measured attempt.** Each attempt runs in a new
process with a new, empty directory for `XDG_CACHE_HOME`, `TMPDIR`,
`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `NUMBA_CACHE_DIR`,
`CUDA_CACHE_PATH` and `MPLCONFIGDIR`, plus any application cache variables you
list under `cache.isolate`. The directory lives inside the attempt's folder
and is deleted afterwards, including after a failure, a timeout or a cancel.
Preflight gets a throwaway cache of its own. Retries get a new one too.

**Model downloads are not repeated.** Moving `XDG_CACHE_HOME` would relocate
the Hugging Face and PyTorch download caches, so on the local backend the
runner points `HF_HOME` and `TORCH_HOME` at where they already are. Set
`cache.assets: isolated` if the thing you are timing is a cold start that
includes the download.

**The order effect is measured, and you set its tolerance in advance.** For
each `performance` criterion the report shows the change measured in AB, the
change measured in BA, and their difference with an interval. A gating
criterion must declare `order_tolerance`. `plan` refuses a timing target or
guardrail without one, because a tolerance chosen after the run is not a
tolerance. Informational criteria do not need one.

| order effect | what happens |
| --- | --- |
| interval inside ±tolerance | the criterion is judged normally |
| interval entirely outside | that criterion is not assessed: it can neither pass nor fail. Reason `order_bias:<criterion>` |
| anything else, or too few units for an interval | the criterion is undecided, so the verdict is at best WARN |

A difference that is not statistically significant is not treated as no
order effect. With `relative: true` the tolerance is a fraction of the
baseline.

An unassessed performance criterion does not take the quality criteria down
with it. If a quality target or guardrail fails on its own evidence, the
verdict is FAIL. If nothing else fails, the verdict is NO_SIGNAL, because a
criterion you declared as gating could not be judged.

The same table reports two more numbers, neither of them gated:

- **first-position and second-position**: the candidate against the baseline
  when both ran first, and when both ran second. If these differ a lot, one
  arm benefits from running second more than the other does.
- **drift**: how much the metric moved between the first and the second pair
  of runs on a seed, both arms averaged. That is time passing (a warming
  machine, a filling page cache, a neighbour's job), not the change.

## Cold and warm

`profile: cold` times each attempt from the reset state. Use it for startup,
first-request latency and anything where cache fill is part of the cost.

`profile: warm` resets, then runs your command `warmup_runs` times untimed
for that attempt's own arm, then runs it once more for the measurement. Warm
state never passes from one arm to the other. Warm-up output is kept in the
bundle and is not part of any metric.

A new process starts for every run, so state held in memory (a loaded model,
a compiled graph, a connection pool) is gone between the warm-up and the
timed run. If steady state for you means a warm process, warm up inside your
harness and time only the steady part. The harness sees `VALIDATE_PHASE`
(`preflight`, `reset`, `warmup` or `run`), `VALIDATE_SEQUENCE`,
`VALIDATE_PERIOD`, `VALIDATE_REPLICATE` and `VALIDATE_CACHE_DIR`.

If a cache is the feature you are evaluating, keep that cache inside the
workload (build it in the harness, for both arms, from the same starting
point). Then the benefit you measure is the candidate's own caching and not
warmth inherited from the other arm.

## State the runner cannot reach

A database, a retrieval index, a KV or prefix cache, a model server: declare
a reset hook. It runs before every measured attempt, through the same backend
as the workload, and it is part of the approved plan.

```yaml
measurement:
  reset_hooks:
    - id: flush-retrieval-cache
      command: ./scripts/flush_cache.sh
      timeout_s: 60
```

A hook that fails or times out stops the attempt before anything is timed.
Hooks are trusted commands and cannot be combined with `trust: untrusted`.

If there is shared state you cannot reset, such as a provider-side prompt
cache, say so:

```yaml
measurement:
  cache:
    uncontrolled: ["provider prompt cache"]
```

Performance criteria are then reported as exploratory. They can still fail.
They cannot pass.

## What is recorded about the machine

The report carries a self-reported description of the host (OS, CPU model,
core count, GPUs and driver if `nvidia-smi` is present, no hostname), the
1-minute load average before and after every measured attempt, and GPU
temperature and utilisation where they can be read. Attempts run one at a
time, never side by side. These are there so a reader can spot a noisy run.
They are not gated.

## What is not controlled

Validate never claims these were cleared, because it cannot verify them:

- the OS page cache and filesystem buffers
- CPU and GPU thermal, frequency and power state
- GPU driver, device and kernel-level caches
- remote services and provider-side caches
- other load on the same machine

The crossover balances any effect that adds the same amount to whichever arm
runs second. It does not remove carryover that helps one arm more than the
other, or drift that is not steady across the run. The first-position
comparison in the report (the candidate when it ran first against the
baseline when it ran first) is the closest thing to an uncontaminated
contrast, and a large gap between it and the second-position comparison is
worth reading before you trust the number. On a shared or noisy machine,
expect a wider order-effect interval and more undecided results.

## How many seeds and replicates

The independent unit is still the seed (or the example, if you declared
`unit: example`). The four runs of a seed are averaged inside that unit. They
do not add units and they do not narrow the interval the way more seeds do.

- One seed gives no interval for the effect or for the order effect.
- Five or more seeds is a reasonable floor for a t interval on timing data.
- `replicates` reduces noise inside each seed. Raise it when a single run of
  a seed is noisy. Raise the seed count when seeds differ from each other.

Cost: `4 x seeds x replicates` measured attempts, plus `warmup_runs` untimed
runs per attempt under the warm profile. The plan prints the total before you
approve it.

## Matched work

Seeds alone do not guarantee the two arms did the same work, and GPU
execution is not always deterministic for a fixed seed. Have the harness
report a number that identifies the work (a dataset size, a token count, a
hash of the inputs folded to a number) and list it under
`validity.held_constant`. Under the protocol it must be equal between the
arms in every measured pair, and equal across all runs of the same seed.

When the thing to match is not a number (a dataset revision, a hash of the
prompts, a request-id set), report it as a fingerprint:

```json
{"metrics": {"latency_ms": 41.2}, "fingerprints": {"inputs": "sha256:9f2c…"}}
```

Each fingerprint becomes the metric `fingerprint.<name>`, so
`held_constant: [fingerprint.inputs]` checks it the same way.

For workloads whose size varies (generation length, batch size), report the
size as a metric too. Validate compares what you declare. It does not
normalise latency by output length for you.

## Workload notes

**CPU.** Pin thread counts in the arm environment (`OMP_NUM_THREADS`,
`MKL_NUM_THREADS`) for both arms. Avoid running anything else during the
comparison. `remyx-validate demo --scenario crossover` and
`--scenario warm` are offline CPU examples of the cold and warm profiles.

**API.** Remote latency includes the network and the provider's own caches.
List provider caches under `cache.uncontrolled` unless you can disable them
per request, and vary a request id per attempt if the provider caches by
prompt. Expect exploratory results.

**GPU.** Call `torch.cuda.synchronize()` (or your framework's equivalent)
before reading the clock, because GPU work is asynchronous. Time with a
monotonic clock inside the harness. The first call after a process starts
pays for CUDA context creation and kernel selection, so decide whether that
belongs in your metric, and use the warm profile plus in-process warm-up if
it does not. Run one arm at a time on the device.

## Try it

```bash
remyx-validate demo --scenario crossover    # both orders, empty caches: PASS
remyx-validate demo --scenario warm         # each arm warmed by itself: PASS
remyx-validate demo --scenario carryover    # a shared service favours the second run: NO_SIGNAL
remyx-validate demo --scenario reset-hook   # the same service, reset by a hook: PASS
```

## Without the protocol

A declaration with no timing criterion and no `measurement` block runs one
pass per seed, the order alternating by the seed's position in the list, with
no cache reset. That is what 0.1.0a0 did for everything. Reports say so under
"Run order and caches". Bundles written by 0.1.0a0 still verify, and `verify`
notes that they carry no order control.

Upgrading from 0.1.0a0: a declaration with a timing criterion now plans twice
the measured attempts, and a timing target or guardrail needs an
`order_tolerance` before `plan` accepts it. Its `declaration_hash` changes, so
an earlier approval no longer applies.
