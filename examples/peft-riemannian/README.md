# PEFT: Riemannian-preconditioned LoRA optimizer

Target: [huggingface/peft#3382](https://github.com/huggingface/peft/pull/3382),
which adds `create_riemannian_optimizer`.

- **baseline arm**: the commit the PR branched from, LoRA trained with AdamW
- **candidate arm**: the PR head, the same LoRA trained with
  `create_riemannian_optimizer(model, AdamW, ...)`

The harness (`harness/bench.py`) is a synthetic teacher-student task: a
frozen two-layer MLP gets a hidden rank-4 update, and a rank-8 LoRA adapter
is trained for 400 steps to match it. Both arms share base weights, teacher,
LoRA init and batches for each seed. It runs on CPU and downloads no data.

This is a small synthetic benchmark. It says whether the optimizer helps on
this task at this learning rate and step budget. It is not the MetaMathQA
result from the PR thread and does not stand in for it.

## Run

Needs `uv` and network for the first setup (clones PEFT, installs CPU torch
into one virtualenv per arm; a few minutes). Later runs reuse both
environments.

```bash
cd examples/peft-riemannian
remyx-validate plan
remyx-validate approve <digest>
remyx-validate run
remyx-validate report
```

The plan lists 20 measured attempts: five seeds, both run orders, two arms.
About seven minutes on a 4-core CPU once the environments are built.

## Adapting it to another PEFT draft PR

```bash
mkdir ../my-pr && cd ../my-pr
remyx-validate init --pr https://github.com/huggingface/peft/pull/<N>
cp ../peft-riemannian/harness/bench.py harness/
```

Then in `impact.yaml`: copy the `setup` and `run` blocks from this example,
set `arms.candidate.env` to whatever switches the new behaviour on, and name
the metric and threshold under `criteria`. If the change is off by default
and you forget the switch, the activation witness turns the result into
`NO_SIGNAL` instead of a confident zero.

`init --pr` leaves `execution.trust: review-required`. Read the pull request
and the setup commands, then choose. This example is `trusted` and `local`
because the PR is merged upstream and we read it; that choice does not carry
over to another pull request. The docker backend is experimental, gives
untrusted code no network by default, and is not a security boundary for
hostile code.

## Recorded run

`sample-evidence/` is a redacted public export of one run on a 4-core Linux
CPU machine (2026-11-24, torch 2.15.0+cpu, transformers 5.24.1, Python 3.12).
Logs and raw outputs are left out. From the repository root:

```bash
remyx-validate verify examples/peft-riemannian/sample-evidence
```

That checks the record against itself. It is not a reproduction, and nobody
outside this machine has rerun it yet.

| | AdamW (baseline) | Riemannian (candidate) | paired difference | 95% interval, 5 seeds |
|---|---:|---:|---:|---|
| final eval loss | 0.004214 | 0.006574 | +0.00236 | [0.001489, 0.003230] |
| steps per second | 828.9 | 539.9 | -289.0 | [-289.9, -288.1] |

Verdict: `FAIL (target not met)`. On this task, at lr 2e-3 for 400 steps,
the preconditioned optimizer ended at a higher loss on every seed. All
validity checks passed, including the witness that the candidate arm used
the preconditioned optimizer on every seed.

### Does run order explain the throughput gap?

That first run was recorded with 0.1.0a0. Each seed ran in one order and
caches were not reset, so its report cannot say whether the slower arm was
slower because of when it ran. Its declaration is inside the bundle.

`impact.yaml` now runs the comparison the way Validate treats any timing
metric by default: every seed runs AdamW then Riemannian and Riemannian then
AdamW, and each of the 20 timed attempts starts in a new process with empty
cache directories. `sample-evidence-crossover/` is the redacted export of
that run (2026-11-25, same machine and packages, remyx-validate 0.1.0a1).

```bash
remyx-validate verify examples/peft-riemannian/sample-evidence-crossover   # from the repository root
```

| steps per second, Riemannian minus AdamW | value | 95% interval, 5 seeds |
|---|---:|---|
| both orders averaged | -295.4 | [-303.7, -287.0] |
| AdamW first, Riemannian second | -293.3 | |
| Riemannian first, AdamW second | -297.4 | |
| difference between the two orders | +4.1 | [-6.9, 15.0] |
| each arm measured when it ran first | -290.4 | |
| each arm measured when it ran second | -300.3 | |

AdamW averaged 836.9 steps/s and Riemannian 541.6. The declaration allowed
the two orders to differ by 5% of baseline throughput (41.85 steps/s) and
fixed that before the run. The measured difference is inside it: the
candidate is about 295 steps/s slower whichever arm goes first, so run order
does not account for the gap. The loss result is the same as before, since
loss is deterministic per seed.
[The report](sample-evidence-crossover/report.md) lists all 20 attempts in
the order they ran, with the machine's load before each one.

What this does and does not settle:

- The interval for the order difference includes zero and reaches 15
  steps/s, under 2% of baseline. An order effect that small cannot be ruled
  out with five seeds. It is far too small to explain a 35% gap.
- The 5% tolerance was our choice. It was written down before this run, but
  we had already seen the size of the gap in the first one.
- The runner emptied the cache directories it owns and started a new process
  for every attempt. It did not clear the OS page cache or control CPU
  frequency or other load on this shared machine. Running both orders
  balances those effects; it does not remove them. The report lists them as
  not controlled.
- Throughput is an informational criterion here. Nothing was gated on it.

See [Comparing latency and throughput](../../docs/MEASUREMENT.md).

What this does not show: anything about other learning rates, real models or
the paper's benchmarks. The paper's main claim is stability across learning
rates, which this declaration does not test. A learning-rate sweep is the
next declaration to write. Throughput is wall-clock on a shared CPU and
moves between runs; the loss values are deterministic for a given seed and
package set.

The main packages are pinned in `impact.yaml`. Their transitive dependencies
are not locked, so a rebuild months from now may differ.
