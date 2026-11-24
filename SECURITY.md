# Security policy

Validate (the open-source local runner) is a developer alpha. Read [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)
before running code you did not write: the local backend is not a sandbox and
the docker backend is experimental.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability.

Report it privately through GitHub: open the repository's **Security** tab and
choose **Report a vulnerability**. That creates a private advisory visible only
to the maintainers.

Include what you ran, what you expected the tool to prevent, and what happened.
A small declaration or bundle that reproduces it helps.

## In scope

- A workload changing the criteria, the plan or the verdict the supervisor records
- A bundle that passes `remyx-validate verify` while disagreeing with its own observations or identities
- `verify` or `export` writing outside their destination, or executing content from a bundle
- Network traffic from the tool itself beyond the documented git fetch and `init --pr` lookup
- A public export that keeps a path, email address or credential the redaction is documented to remove

## Known limits, not vulnerabilities

- A local bundle is self-reported. Whoever holds the files can rerun and re-create them.
- The harness and the candidate code share a process, so candidate code can affect what is measured.
- The local backend runs with your user's file and network access.
- Container isolation can be defeated by a kernel or runtime escape.
