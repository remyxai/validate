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
| final_eval_loss (`final_eval_loss`, mean, lower is better) | target | 0.004214 | 0.006574 | +0.00236 | [0.001489, 0.00323] | 5 (10/10) | target not met |
| throughput (`steps_per_s`, mean, higher is better) | info | 836.9 | 541.6 | −295.4 | [-303.7, -287] | 5 (10/10) | informational |

- `final_eval_loss`: rule "improve" violated

## Validity

- pass · harness_unchanged: harness files match the approved plan
- pass · declaration_unchanged: criteria match the approved plan
- pass · checkout_pinned_baseline: baseline checkout is at the approved commit
- pass · checkout_clean_baseline: baseline tracked files match the approved commit
- pass · checkout_pinned_candidate: candidate checkout is at the approved commit
- pass · checkout_clean_candidate: candidate tracked files match the approved commit
- pass · held_constant: `num_trainable_params` is equal between arms on all 10 measured pair(s)
- pass · held_constant: `train_steps` is equal between arms on all 10 measured pair(s)
- pass · held_constant: `initial_eval_loss` is equal between arms on all 10 measured pair(s)
- pass · activation: witness `optimizer_preconditioned` differs between arms on all 10 measured pair(s)

## Run order and caches

- design: crossover. Every seed and replicate ran both orders, so each arm was measured in first and in second position on the same seed
- 1 replicate(s), 4 measured attempts per seed; order drawn from scheduler seed 0, independent of the workload seeds, and fixed in the approved plan
- cache profile: cold. Each measured attempt ran in a new process with a new, empty directory for: `CUDA_CACHE_PATH`, `MPLCONFIGDIR`, `NUMBA_CACHE_DIR`, `TMPDIR`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME`
- downloaded model assets: kept shared so they are not downloaded inside a timed attempt (`HF_HOME`, `TORCH_HOME`)
- retries: a failed attempt restarts its whole sequence (both arms) from a fresh cache; superseded attempts are kept in the ledger and excluded from the comparison
- not controlled by the runner: OS page cache and filesystem buffers; CPU and GPU thermal, frequency and power state; GPU driver, device and kernel-level caches; remote services and provider-side caches; other load on the same machine
- machine (self-reported): Linux x86_64, Intel(R) Xeon(R) Platinum 8175M CPU @ 2.50GHz, 4 logical CPUs; GPUs: none detected
- one measured attempt at a time; arms never run side by side; metrics are timed by the harness; `seconds` in the attempt table is the runner's wall time for the whole process and is not a metric

Change measured in each order (candidate minus baseline). AB is the baseline first; BA is the candidate first. The order effect is the AB value minus the BA value. First-position is the candidate when it ran first minus the baseline when it ran first.

| criterion | AB | BA | order effect | 95% interval | tolerance | result | first-position | second-position | drift | units |
|---|---:|---:|---:|---|---:|---|---:|---:|---|---:|
| throughput | −293.3 | −297.4 | +4.083 | [-6.86, 15.03] | ±41.85 | inside tolerance | −290.4 | −300.3 | −3.475 [-12.17, 5.219] | 5 |

Drift is how much the metric moved between the first and the second pair of runs on a seed, both arms averaged. It is time passing, not the change, and is reported without a gate.

## Execution

- status: COMPLETED; backend: local; wall time: 402.386 s
- cost: unknown

Attempts in the order they ran. `planned` is the position in the approved schedule. A discarded attempt belongs to a sequence that was run again and is not part of the comparison.

| ran | planned | attempt | arm | seed | replicate | order | position | try | status | cache | warm-up | load before | seconds |
|---:|---:|---|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|
| 1 | 1 | `att_baseline_s1_r0_AB_t1` | baseline | 1 | 0 | AB | 1 | 1 | ok | new, empty | 0 | 1.6 | 15.843 |
| 2 | 2 | `att_candidate_s1_r0_AB_t1` | candidate | 1 | 0 | AB | 2 | 1 | ok | new, empty | 0 | 1.47 | 16.093 |
| 3 | 3 | `att_candidate_s1_r0_BA_t1` | candidate | 1 | 0 | BA | 1 | 1 | ok | new, empty | 0 | 1.41 | 16.093 |
| 4 | 4 | `att_baseline_s1_r0_BA_t1` | baseline | 1 | 0 | BA | 2 | 1 | ok | new, empty | 0 | 1.52 | 15.841 |
| 5 | 5 | `att_candidate_s4_r0_BA_t1` | candidate | 4 | 0 | BA | 1 | 1 | ok | new, empty | 0 | 1.41 | 16.047 |
| 6 | 6 | `att_baseline_s4_r0_BA_t1` | baseline | 4 | 0 | BA | 2 | 1 | ok | new, empty | 0 | 1.35 | 15.747 |
| 7 | 7 | `att_baseline_s4_r0_AB_t1` | baseline | 4 | 0 | AB | 1 | 1 | ok | new, empty | 0 | 1.25 | 15.801 |
| 8 | 8 | `att_candidate_s4_r0_AB_t1` | candidate | 4 | 0 | AB | 2 | 1 | ok | new, empty | 0 | 1.2 | 16.041 |
| 9 | 9 | `att_baseline_s0_r0_AB_t1` | baseline | 0 | 0 | AB | 1 | 1 | ok | new, empty | 0 | 1.26 | 15.792 |
| 10 | 10 | `att_candidate_s0_r0_AB_t1` | candidate | 0 | 0 | AB | 2 | 1 | ok | new, empty | 0 | 1.19 | 16.094 |
| 11 | 11 | `att_candidate_s0_r0_BA_t1` | candidate | 0 | 0 | BA | 1 | 1 | ok | new, empty | 0 | 1.37 | 16.041 |
| 12 | 12 | `att_baseline_s0_r0_BA_t1` | baseline | 0 | 0 | BA | 2 | 1 | ok | new, empty | 0 | 1.29 | 15.792 |
| 13 | 13 | `att_candidate_s3_r0_BA_t1` | candidate | 3 | 0 | BA | 1 | 1 | ok | new, empty | 0 | 1.35 | 16.141 |
| 14 | 14 | `att_baseline_s3_r0_BA_t1` | baseline | 3 | 0 | BA | 2 | 1 | ok | new, empty | 0 | 1.33 | 15.743 |
| 15 | 15 | `att_baseline_s3_r0_AB_t1` | baseline | 3 | 0 | AB | 1 | 1 | ok | new, empty | 0 | 1.26 | 15.795 |
| 16 | 16 | `att_candidate_s3_r0_AB_t1` | candidate | 3 | 0 | AB | 2 | 1 | ok | new, empty | 0 | 1.25 | 16.096 |
| 17 | 17 | `att_candidate_s2_r0_BA_t1` | candidate | 2 | 0 | BA | 1 | 1 | ok | new, empty | 0 | 1.18 | 16.143 |
| 18 | 18 | `att_baseline_s2_r0_BA_t1` | baseline | 2 | 0 | BA | 2 | 1 | ok | new, empty | 0 | 1.37 | 15.793 |
| 19 | 19 | `att_baseline_s2_r0_AB_t1` | baseline | 2 | 0 | AB | 1 | 1 | ok | new, empty | 0 | 1.28 | 15.791 |
| 20 | 20 | `att_candidate_s2_r0_AB_t1` | candidate | 2 | 0 | AB | 2 | 1 | ok | new, empty | 0 | 1.2 | 16.041 |

## Limitations

- Self-reported local evidence: produced and checksummed on the machine that ran it. The checksums detect later edits; they do not prove who ran what. This is not a remote attestation.
- Local subprocess backend: the workload ran with the user's file and network access and could have written its own results.
- Independent unit is the seed; 5 seed(s) were run.
- Run order: every seed ran both orders, baseline first and candidate first, on a schedule fixed before approval. That balances position effects that add the same amount to whichever arm runs second. It does not remove carryover that differs by arm, or drift that is not steady over the run.
- Each seed was measured 2 time(s) per arm. Those runs are averaged within the independent unit and are not counted as extra units.
- Caches: every measured attempt started from a new, empty directory for the cache locations listed in the execution protocol, in a new process. The OS page cache, hardware thermal and frequency state, GPU driver caches, remote services and other load on the machine were not controlled.
- An offline comparison on the declared data. It does not predict a production outcome.

## Identities

- change_id: `chg_8795830f595b6b4e`
- declaration_hash: `fb61ff0f9160c9a20697a894a5eb92639fccc00c73da6aceaf79f3970023e795`
- spec_hash: `83edc46af0c384115f6144728eaf752126f7bb2650064c585004dab6fa9e946c`
- plan_digest: `99344d7f8c4c30763ed4bc91af3556ec4dfba76b7fc5ad130e7af4fa2232c013`
- run_id: `run_20261125T032831Z_157bd4`
- approved by: salma-remyx <<EMAIL>> at 2026-11-25T03:28:30Z

## Decision

The verdict is not a decision. Record one with `remyx-validate record run_20261125T032831Z_157bd4 --decision ship|revise|hold|drop --reason "…"`.
