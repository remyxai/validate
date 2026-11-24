# Change evidence

Completed · valid evidence · **FAIL** (target not met)

**Objective.** Decide whether the Riemannian-preconditioned optimizer trains LoRA to a lower loss than AdamW at the same step budget.

**Hypothesis.** With identical init, data and step count, r x r preconditioning of the LoRA gradients reaches a lower eval loss than plain AdamW.

| | ref | commit |
|---|---|---|
| baseline | `3116c96b5d870851e8477eb99dfcf8471bac0e72` | `3116c96b5d87` |
| candidate | `f90cc44e81dfbc9a91741f62b8d87d5d963843e7` | `f90cc44e81df` |

- baseline arm env: `OPTIMIZER=adamw`
- candidate arm env: `OPTIMIZER=riemannian`

## Criteria

| criterion | role | baseline | candidate | delta | 95% interval | pairs (matched/expected) | result |
|---|---|---:|---:|---:|---|---:|---|
| final_eval_loss (`final_eval_loss`, mean, lower is better) | target | 0.004214 | 0.006574 | +0.00236 | [0.001489, 0.00323] | 5 (5/5) | target not met |
| throughput (`steps_per_s`, mean, higher is better) | info | 828.9 | 539.9 | −289 | [-289.9, -288.1] | 5 (5/5) | informational |

- `final_eval_loss`: rule "improve" violated

## Validity

- pass · harness_unchanged: harness files match the approved plan
- pass · declaration_unchanged: criteria match the approved plan
- pass · checkout_pinned_baseline: baseline checkout is at the approved commit
- pass · checkout_clean_baseline: baseline tracked files match the approved commit
- pass · checkout_pinned_candidate: candidate checkout is at the approved commit
- pass · checkout_clean_candidate: candidate tracked files match the approved commit
- pass · held_constant: `num_trainable_params` is equal between arms on all 5 seed(s)
- pass · held_constant: `train_steps` is equal between arms on all 5 seed(s)
- pass · held_constant: `initial_eval_loss` is equal between arms on all 5 seed(s)
- pass · activation: witness `optimizer_preconditioned` differs between arms on all 5 seed(s)

## Execution

- status: COMPLETED; backend: local; wall time: 202.495 s
- cost: unknown

| attempt | arm | seed | try | status | seconds |
|---|---|---:|---:|---|---:|
| `att_baseline_s0_t1` | baseline | 0 | 1 | ok | 16.141 |
| `att_candidate_s0_t1` | candidate | 0 | 1 | ok | 16.294 |
| `att_candidate_s1_t1` | candidate | 1 | 1 | ok | 16.443 |
| `att_baseline_s1_t1` | baseline | 1 | 1 | ok | 16.243 |
| `att_baseline_s2_t1` | baseline | 2 | 1 | ok | 16.242 |
| `att_candidate_s2_t1` | candidate | 2 | 1 | ok | 16.543 |
| `att_candidate_s3_t1` | candidate | 3 | 1 | ok | 16.243 |
| `att_baseline_s3_t1` | baseline | 3 | 1 | ok | 15.944 |
| `att_baseline_s4_t1` | baseline | 4 | 1 | ok | 16.092 |
| `att_candidate_s4_t1` | candidate | 4 | 1 | ok | 16.352 |

## Limitations

- Self-reported local evidence: produced and checksummed on the machine that ran it. The checksums detect later edits; they do not prove who ran what. This is not a remote attestation.
- Local subprocess backend: the workload ran with the user's file and network access and could have written its own results.
- Independent unit is the seed; 5 seed(s) were run.
- An offline comparison on the declared data. It does not predict a production outcome.

## Identities

- change_id: `chg_8795830f595b6b4e`
- declaration_hash: `bcd7a9283b7e9222fd4cba52eb8c0c41a48f87dd7a73d7b7e08b5dc9a2be9636`
- spec_hash: `34e2e3521b71f49e3d338aa71461690693928b3dc285f4ce607892d0165cab7d`
- plan_digest: `67343350a913f9bf95e57c081f27afb0fd1e989240eb2700b7c7a9ebdf8eab24`
- run_id: `run_20261124T194037Z_6a6f83`
- approved by: salma-remyx <<EMAIL>> at 2026-11-24T19:40:36Z

## Decision

The verdict is not a decision. Record one with `remyx-validate record run_20261124T194037Z_6a6f83 --decision ship|revise|hold|drop --reason "…"`.
