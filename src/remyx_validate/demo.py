"""A deterministic, offline demo and the fixture set behind it.

It builds a small support-article retriever in a throwaway git repo, a harness
with a fixed query set, and a declaration. Each scenario is a different
candidate commit (or a different declaration) chosen to land on one verdict.

Quality is measured for real: the retriever ranks articles and the harness
checks the top hit. Latency and cost are simulated from an operation count,
so every machine gets the same numbers and the demo needs no clock, network,
GPU or API key.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import textwrap

RANKER = '''\
"""Toy support-article retriever. CONFIG is what a change edits."""
import math
import os
import time
from collections import Counter

CONFIG = %(config)s

# Fault injection for the demo's failure fixtures. Empty in a normal commit.
FIXTURE = %(fixture)s


def effective_config():
    cfg = dict(CONFIG)
    if cfg.pop("rerank_flag", False):
        # A default-off feature flag: the rerank only runs when the env says so.
        cfg["rerank_depth"] = cfg["rerank_depth"] if os.environ.get("ENABLE_RERANK") == "1" else 0
    return cfg


class Ranker:
    def __init__(self, docs):
        self.cfg = effective_config()
        self.docs = docs
        self.sets = [set(d["words"]) for d in docs]
        df = Counter(w for s in self.sets for w in s)
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
        if FIXTURE.get("tamper"):
            # A hostile candidate: write to the harness's query file mid-run.
            path = os.path.join(os.environ["VALIDATE_HARNESS_DIR"], "queries.jsonl")
            try:
                os.chmod(path, 0o644)
                with open(path, "a") as fh:
                    fh.write("\\n")
            except OSError:
                pass
        if FIXTURE.get("sleep_s"):
            time.sleep(FIXTURE["sleep_s"])

    def describe(self):
        return dict(self.cfg)

    def search(self, query):
        """Returns (top article id, operation count)."""
        if FIXTURE.get("crash_over_len") and len(query) > FIXTURE["crash_over_len"]:
            raise RuntimeError("query too long")
        cfg = self.cfg
        ops = 0
        words = query[: cfg["max_query_words"]] if cfg.get("max_query_words") else query
        use_idf = cfg["weighting"] == "idf"
        scores = []
        for s in self.sets:
            score = 0.0
            for w in words:
                ops += 1
                if w in s:
                    score += self.idf.get(w, 0.0) if use_idf else 1.0
            scores.append(score)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        depth = cfg.get("rerank_depth", 0)
        if depth > 1:
            # Second pass: reward adjacent query-word pairs that are also
            # adjacent in the article. Costs a scan of each shortlisted article.
            bigrams = list(zip(words, words[1:]))
            best, best_score = order[0], None
            for i in order[:depth]:
                seq = self.docs[i]["words"]
                bonus = 0.0
                for a, b in bigrams:
                    for j in range(len(seq) - 1):
                        ops += 1
                        if seq[j] == a and seq[j + 1] == b:
                            bonus += 1.0
                total = scores[i] + 3.0 * bonus
                if best_score is None or total > best_score:
                    best, best_score = i, total
            return self.docs[best]["id"], ops
        return self.docs[order[0]]["id"], ops
'''

HARNESS = '''\
"""Demo harness: runs every query through the checked-out retriever.

Writes one observation per query to $VALIDATE_OUTPUT_DIR/results.json.
latency_ms and cost_usd are simulated from the retriever's operation count.
"""
import hashlib
import json
import os
import sys
import zlib

sys.path.insert(0, os.getcwd())
from app.ranker import Ranker  # noqa: E402

harness = os.environ["VALIDATE_HARNESS_DIR"]
out = os.environ["VALIDATE_OUTPUT_DIR"]

with open(os.path.join(harness, "docs.json")) as fh:
    docs = json.load(fh)
with open(os.path.join(harness, "queries.jsonl")) as fh:
    queries = [json.loads(line) for line in fh if line.strip()]
limit = int(os.environ.get("DEMO_LIMIT", "0"))
if limit:
    queries = queries[:limit]

ranker = Ranker(docs)

# Order and cache fixtures. With DEMO_WARM_MS set, a process that finds state
# left by an earlier one is "warm" and every query is that much faster.
#   * a marker under $XDG_CACHE_HOME: a cache the runner isolates per attempt
#   * DEMO_CARRY: a file outside it, standing in for a shared service that
#     speeds up whichever arm runs second on the same seed
warm_ms = float(os.environ.get("DEMO_WARM_MS", "0"))
warm = False
if warm_ms:
    marker = os.path.join(os.environ.get("XDG_CACHE_HOME", out), "demo-index")
    warm = os.path.exists(marker)
    open(marker, "w").close()
    carry = os.environ.get("DEMO_CARRY")
    if carry:
        me = os.environ["VALIDATE_ARM"] + ":" + os.environ["VALIDATE_SEED"]
        other = ("candidate" if os.environ["VALIDATE_ARM"] == "baseline" else "baseline") \
            + ":" + os.environ["VALIDATE_SEED"]
        last = open(carry).read() if os.path.exists(carry) else ""
        warm = warm or last == other
        with open(carry, "w") as fh:
            fh.write("spent" if last == other else me)
fingerprint = int(hashlib.sha256(
    json.dumps(ranker.describe(), sort_keys=True).encode()).hexdigest()[:8], 16)

observations, errors = [], 0
for q in queries:
    try:
        top, ops = ranker.search(q["words"])
    except Exception:
        errors += 1
        continue
    observations.append({
        "unit": q["id"],
        "slice": q["slice"],
        "metrics": {
            "resolved": int(top == q["gold"]),
            # Simulated: a fixed cost per operation plus per-query jitter that is
            # the same for both arms, so the pairing is exact.
            "latency_ms": 120 + 0.25 * ops + (zlib.crc32(q["id"].encode()) % 61) - 30
            - (warm_ms if warm else 0),
            "cost_usd": ops * 2e-6,
        },
    })

with open(os.path.join(out, "results.json"), "w") as fh:
    json.dump({
        "metrics": {"dataset_size": len(queries), "pipeline_fingerprint": fingerprint,
                    "errors": errors},
        "observations": observations,
    }, fh)
print(f"evaluated {len(observations)} queries, {errors} errors")
'''

BASELINE_CONFIG = {"weighting": "idf", "rerank_depth": 0}

# name -> (candidate CONFIG, FIXTURE, declaration tweaks, expected verdict, one-line story)
SCENARIOS = {
    "guardrail": dict(
        config={"weighting": "idf", "rerank_depth": 8}, expect="FAIL",
        story="A deep phrase reranker lifts resolution rate but pushes p95 latency past the 800 ms limit."),
    "positive": dict(
        config={"weighting": "idf", "rerank_depth": 2}, expect="PASS",
        story="A shallow reranker lifts resolution rate and stays inside the latency limit."),
    "negative": dict(
        config={"weighting": "idf", "rerank_depth": 0, "max_query_words": 5}, expect="FAIL",
        story="Truncating queries to five words is faster and resolves fewer tickets."),
    "uncertain": dict(
        config={"weighting": "idf", "rerank_depth": 2}, expect="WARN",
        env_both={"DEMO_LIMIT": "12"},
        story="The shallow reranker measured on 12 queries: the interval cannot settle the target."),
    "null": dict(
        config={"weighting": "idf", "rerank_depth": 1}, expect="FAIL",
        story="A config change that is live (the fingerprint differs) and changes nothing: a measured zero."),
    "inactive": dict(
        config={"weighting": "idf", "rerank_depth": 8, "rerank_flag": True}, expect="NO_SIGNAL",
        story="The reranker ships behind a default-off flag nobody turned on for the candidate arm."),
    "missing": dict(
        config={"weighting": "idf", "rerank_depth": 2}, fixture={"crash_over_len": 10},
        expect="INCOMPLETE",
        story="The candidate crashes on long queries, so those units have no candidate value."),
    "drift": dict(
        config={"weighting": "idf", "rerank_depth": 2}, expect="NO_SIGNAL",
        env_candidate={"DEMO_LIMIT": "150"},
        story="The candidate arm was run on a different number of queries than the baseline."),
    "timeout": dict(
        config={"weighting": "idf", "rerank_depth": 2}, fixture={"sleep_s": 30},
        expect="INCOMPLETE", timeout_s=3,
        story="The candidate hangs past the per-attempt timeout."),
    "tamper": dict(
        config={"weighting": "idf", "rerank_depth": 2}, fixture={"tamper": True},
        expect="INCOMPLETE",
        story="The candidate writes to the harness's query file during the run."),
    # Run order and cache fixtures. All four use the shallow reranker under a
    # crossover: baseline then candidate, and candidate then baseline.
    "crossover": dict(
        config={"weighting": "idf", "rerank_depth": 2}, expect="PASS",
        env_both={"DEMO_WARM_MS": "40"}, measurement="cold",
        story="Both run orders, each attempt from an empty cache: a warm cache cannot favour "
              "whichever arm runs second."),
    "warm": dict(
        config={"weighting": "idf", "rerank_depth": 2}, expect="PASS",
        env_both={"DEMO_WARM_MS": "40"}, measurement="warm",
        story="Steady-state timing: each arm is warmed by its own untimed run before the "
              "timed one."),
    "carryover": dict(
        config={"weighting": "idf", "rerank_depth": 2}, expect="NO_SIGNAL",
        env_both={"DEMO_WARM_MS": "40", "DEMO_CARRY": "../../demo-carry.state"},
        measurement="cold",
        story="A shared service outside the runner's caches speeds up whichever arm runs "
              "second. The two orders disagree, so the latency result is not trusted."),
    "reset-hook": dict(
        config={"weighting": "idf", "rerank_depth": 2}, expect="PASS",
        env_both={"DEMO_WARM_MS": "40", "DEMO_CARRY": "../../demo-carry.state"},
        measurement="cold", reset="rm -f ../../demo-carry.state",
        story="The same shared service, reset by a declared hook before every attempt."),
}

DEFAULT_SCENARIO = "guardrail"


def _corpus(seed: int = 20261124):
    """120 articles in 40 topics of three near-duplicates, and 240 queries."""
    rng = random.Random(seed)
    common = [f"the{i:02d}" for i in range(60)]
    docs = []
    for t in range(40):
        topic = [f"t{t:02d}w{i}" for i in range(10)]
        for k in range(3):
            words = rng.sample(topic, 8) + rng.sample(common, 12)
            rng.shuffle(words)
            docs.append({"id": f"KB-{t:02d}{k}", "words": words})
    queries = []
    for i in range(240):
        gold = docs[rng.randrange(len(docs))]
        start = rng.randrange(len(gold["words"]) - 3)
        phrase = gold["words"][start:start + 3]
        noise = [rng.choice(common) for _ in range(rng.randint(2, 9))]
        cut = rng.randint(0, len(noise))
        words = noise[:cut] + phrase + noise[cut:]
        queries.append({"id": f"q{i:03d}", "gold": gold["id"], "words": words,
                        "slice": "short" if len(words) <= 8 else "long"})
    return docs, queries


def _git(args, cwd):
    env = dict(os.environ,
               GIT_AUTHOR_NAME="validate-demo", GIT_AUTHOR_EMAIL="demo@example.invalid",
               GIT_COMMITTER_NAME="validate-demo", GIT_COMMITTER_EMAIL="demo@example.invalid",
               GIT_AUTHOR_DATE="2026-11-24T00:00:00Z", GIT_COMMITTER_DATE="2026-11-24T00:00:00Z",
               GIT_CONFIG_NOSYSTEM="1")
    subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
                   cwd=cwd, check=True, capture_output=True, env=env)


def _write_ranker(repo: str, config: dict, fixture: dict) -> None:
    with open(os.path.join(repo, "app", "ranker.py"), "w", encoding="utf-8") as fh:
        fh.write(RANKER % {"config": json.dumps(config, sort_keys=True).replace("true", "True"),
                           "fixture": json.dumps(fixture, sort_keys=True).replace("true", "True")})


def declaration_text(scenario: str, protocol: bool = True) -> str:
    """`protocol=False` writes the declaration without run-order control, as
    0.1.0a0 did: the latency criterion is marked `performance: false`."""
    sc = SCENARIOS[scenario]
    env_b = dict(sc.get("env_both", {}))
    env_c = {**env_b, **sc.get("env_candidate", {})}

    expected_units = int(env_b.get("DEMO_LIMIT", 240))

    def env_block(env):
        return "{env: {" + ", ".join(f'{k}: "{v}"' for k, v in sorted(env.items())) + "}}"

    measurement = ""
    performance = ("\n            performance: false        # demo of the old behaviour: "
                   "run order not controlled")
    sc = dict(sc, measurement=sc.get("measurement", "cold")) if protocol else sc
    if sc.get("measurement"):
        performance = ("\n            performance: true         # run order and caches can move it"
                       "\n            order_tolerance: 5        # ms between the two run orders")
        hooks = (f"\n          reset_hooks:\n            - id: reset-shared-service\n"
                 f"              command: {sc['reset']}" if sc.get("reset") else "")
        warm = "\n            warmup_runs: 1" if sc["measurement"] == "warm" else ""
        measurement = (f"\n        measurement:\n          design: crossover       # every seed runs AB and BA"
                       f"\n          replicates: 1\n          schedule_seed: 7        # orders the runs, never the workload"
                       f"\n          cache:\n            profile: {sc['measurement']}{warm}{hooks}")

    return textwrap.dedent(f"""\
        # Demo declaration, scenario: {scenario}
        # {sc['story']}
        spec_version: "0.1"
        objective: Resolve more support tickets on the first article without slowing responses.
        change:
          family: retrieval
          hypothesis: The candidate retriever resolves more tickets within the latency limit.
          repo: ./app-repo
          baseline: {{ref: main}}
          candidate: {{ref: candidate}}
        harness:
          path: ./harness
        run:
          command: python3 {{harness}}/eval.py
          timeout_s: {sc.get('timeout_s', 120)}
        arms:
          baseline: {env_block(env_b)}
          candidate: {env_block(env_c)}
        method:
          unit: example          # each query is one paired unit
          seeds: [0]
          confidence: 0.95
          uncertainty: interval
        criteria:
          - id: resolution_rate
            metric: resolved
            role: target
            direction: higher
            aggregate: mean
            min_improvement: 0.02     # at least +2 points, judged on the interval
          - id: p95_latency
            metric: latency_ms
            role: guardrail
            direction: lower
            aggregate: p95
            unit: ms
            max: 800                  # veto above 800 ms{performance}
          - id: cost_per_request
            metric: cost_usd
            role: info
            direction: lower
            aggregate: mean
        validity:
          activation:
            - metric: pipeline_fingerprint   # must differ between arms
          held_constant: [dataset_size]{measurement}
        execution:
          backend: local
          trust: trusted
        budget:
          max_wall_s: 300
          max_cost_usd: 5
        data:
          identity: demo-support-queries-v1
          expected_units: {expected_units}
          description: 240 synthetic support queries over 120 articles, generated with a fixed seed.
        """)


def create(dest: str, scenario: str = DEFAULT_SCENARIO, protocol: bool = True) -> str:
    """Build the demo project for one scenario. Returns the declaration path."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {', '.join(SCENARIOS)}")
    sc = SCENARIOS[scenario]
    dest = os.path.abspath(dest)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    repo, harness = os.path.join(dest, "app-repo"), os.path.join(dest, "harness")
    os.makedirs(os.path.join(repo, "app"))
    os.makedirs(harness)

    docs, queries = _corpus()
    with open(os.path.join(harness, "docs.json"), "w", encoding="utf-8") as fh:
        json.dump(docs, fh)
    with open(os.path.join(harness, "queries.jsonl"), "w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q) + "\n")
    with open(os.path.join(harness, "eval.py"), "w", encoding="utf-8") as fh:
        fh.write(HARNESS)

    open(os.path.join(repo, "app", "__init__.py"), "w").close()
    _git(["init", "--quiet"], repo)
    _git(["symbolic-ref", "HEAD", "refs/heads/main"], repo)
    _write_ranker(repo, BASELINE_CONFIG, {})
    _git(["add", "-A"], repo)
    _git(["commit", "--quiet", "-m", "baseline: IDF-weighted retriever"], repo)
    _git(["checkout", "--quiet", "-b", "candidate"], repo)
    _write_ranker(repo, sc["config"], sc.get("fixture", {}))
    _git(["commit", "--quiet", "-am", f"candidate ({scenario}): {sc['story']}"], repo)
    if scenario == "guardrail":
        # A follow-up candidate for the "second change, same declaration" walk-through.
        _git(["checkout", "--quiet", "-b", "candidate-2", "main"], repo)
        _write_ranker(repo, SCENARIOS["positive"]["config"], {})
        _git(["commit", "--quiet", "-am", "candidate-2: shallow reranker"], repo)
    _git(["checkout", "--quiet", "main"], repo)

    path = os.path.join(dest, "impact.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(declaration_text(scenario, protocol))
    with open(os.path.join(dest, ".gitignore"), "w", encoding="utf-8") as fh:
        fh.write(".validate/\n")
    return path
