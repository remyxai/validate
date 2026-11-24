# Contributing

This is a developer alpha. The most useful contributions right now are reports
from running it on a change you were already considering.

## What to open an issue for

- **A run report.** What you compared, which evaluation tools you used, what the
  setup cost you, and what the evidence was missing. You do not need to share
  private data or results.
- **A spec question or change.** The [Change Evidence Spec](spec/v0.1/SPEC.md) is
  a draft. Its open questions are listed at the end. Say which producer or
  consumer you have in mind.
- **A fixture.** A small case that exercises the contract: a declaration, the
  observations, and the verdict you expect. Cases the current rules get wrong
  are the most valuable.
- **An adapter proposal.** An evaluation framework whose results you want to
  record as observations. Describe its result format before writing code.
- **A bug.** Include the command, the declaration (or the part that matters),
  and the output. `--json` output is easiest to read.

For vulnerabilities, see [SECURITY.md](SECURITY.md).

## Evidence you contribute

Say what kind of result it is:

- **synthetic**: generated data or simulated measurements, like the bundled demo
- **recorded**: a run you performed, with the environment it ran on
- **reproduced**: an independent rerun of someone else's recorded result

Only contribute data and results you have the right to share. Do not include
credentials, private repository contents or other people's unpublished results.
Null and negative results are welcome and are kept.

## Development

```bash
git clone https://github.com/remyxai/validate.git
cd validate
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
remyx-validate demo --all --dir /tmp/validate-fixtures
```

Tests that need a docker daemon are skipped without one; `pytest -rs` lists them.

Keep changes small. A change to comparison or verdict rules needs a fixture in
`tests/` that fails before it and passes after, and a matching edit to
`spec/v0.1/SPEC.md`. A change to anything that feeds a hash (see the spec's
Identities section) is a format change: say so in the pull request.

The runner stays local-only. Changes that add network calls, telemetry or an
account requirement will not be merged.

## License

Contributions are accepted under the repository's license (see [LICENSE](LICENSE)).
