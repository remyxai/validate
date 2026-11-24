# Threat model (alpha)

Written alongside the code. It has not had an independent security review.

**Supported use in this release: code, setup commands and harnesses you have
read and trust.** The docker backend is experimental and is not a security
boundary for hostile code. Do not run an unreviewed pull request on a machine
that holds credentials you care about.

## Who is trusted

| Party | Trusted for |
| --- | --- |
| The person running the tool | the declaration, the approval, the machine |
| The supervisor (`remyx_validate` process) | criteria, plan digest, attempt ledger, checksums, verdict |
| The harness directory | measuring correctly. It is your code, digested into the plan and copied per run |
| The baseline and candidate checkouts | nothing, under `trust: untrusted` |

## What the supervisor does

| Threat | Control |
| --- | --- |
| A pull request is run as trusted by default | `init --pr` writes `trust: review-required`; `plan` refuses until a person chooses |
| Criteria edited after approval | criteria are frozen in the approved plan; the declaration is re-hashed after the run |
| Plan drifts after approval (moved branch, edited harness) | `run` recomputes the plan and refuses a digest with no approval |
| Saved plan edited before approval | approval recomputes the digest |
| Harness modified during the run | the workload gets a per-run copy whose digest is rechecked; read-only mount under docker |
| Harness reaches outside the project | the path must resolve inside the project and may not be or contain a symbolic link |
| Source edited with HEAD unchanged | tracked files are compared with the approved commit after setup and after the run |
| Two arms share state | each arm has its own checkout and environment, keyed by commit, setup, arm settings and image id |
| Bundle edited later | checksums; `verify` recomputes the identities and the whole comparison from the observations |
| Malformed bundle or archive | size and count limits, no links or devices, no duplicate paths; a readable failure, not an exception |
| Secret access | local: environment rebuilt from an allowlist. docker: no host environment, no home mount. Untrusted declarations cannot use `pass_env`. Credential-shaped values in a declaration are rejected |
| Uncontrolled network | untrusted docker runs have no network in setup or run unless explicitly acknowledged |
| Runaway retry or time | retries capped at 5 by schema; the wall-clock cap covers setup, preflight and attempts |
| Malformed output | finite numbers only, size and count limits, duplicates rejected, negative cost rejected |
| Missing suite reported as a result | suite files checked on both arms before any attempt; exit 0 with no results is INCOMPLETE |
| Silent upload | the package has one outbound HTTP call (`init --pr` to api.github.com) |

Each row has a regression test in `tests/`.

## What it does not do

- **A local bundle is not an attestation.** Whoever holds the files can
  rerun, re-edit and re-checksum. `verify` shows that a bundle is internally
  consistent. It does not show who ran it, on what hardware, or who approved
  it. Evidence carries `trust_level: self_reported_local`.
- **The harness and the candidate share a process.** Candidate code imported
  by the harness can return anything, including wrong answers that score
  well. The supervisor protects the criteria and the ledger, not the
  measurement inside the workload. A scorer in a separate process is not
  built.
- **The local backend is not a sandbox.** The workload runs as you, with your
  network, and can read any file you can, including credentials on disk.
  Only environment variables are scrubbed. Seeds of the same arm share one
  environment directory.
- **Docker is container isolation.** A kernel or runtime escape defeats it,
  and docker group membership is root-equivalent. If you acknowledge network
  for untrusted code, candidate-controlled setup runs with network access.
- **A cost cap is not a spending limit.** It is checked after each attempt
  against what the harness reports. A failed attempt has unknown cost. Set
  provider-side limits for paid calls.
- **Redaction is pattern matching.** A public export removes what the
  patterns recognize. Read the files before sharing them.
- **Planning uses the network.** Fetching the repository, and whatever your
  setup commands download, happen outside the comparison.
- **No signing** of releases or bundles yet.

## Open items

- A scorer process outside the candidate's container.
- A VM or gVisor option for hosts where container isolation is not enough.
- Setup for untrusted code on a disposable builder, so dependencies can be
  installed without giving candidate code a network.
- Fresh writable state per attempt on the local backend.
- Signed releases and signed bundles.
