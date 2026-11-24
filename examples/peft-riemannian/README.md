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

What this does not show: anything about other learning rates, real models or
the paper's benchmarks. The paper's main claim is stability across learning
rates, which this declaration does not test. A learning-rate sweep is the
next declaration to write. Throughput is wall-clock on a shared CPU and
moves between runs; the loss values are deterministic for a given seed and
package set.

The main packages are pinned in `impact.yaml`. Their transitive dependencies
are not locked, so a rebuild months from now may differ.
