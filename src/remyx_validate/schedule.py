"""The measured-attempt schedule for a declared measurement protocol.

A = baseline, B = candidate. A *block* is one (seed, replicate). Under the
`crossover` design every block runs both sequences, AB and BA, so each arm is
measured once in first position and once in second position on the same
seed: four measured attempts per block. Under `single` a block runs one
sequence, which costs half as much and cannot separate the change from the
order it ran in.

Which sequence a block runs first, and the order of the blocks, come from a
scheduler seed that is separate from the workload seeds. Nothing is derived
from a workload seed's value or position, so reordering or renumbering seeds
does not decide who runs first. The stream below is SHA-256 in counter mode,
so the same declaration gives the same schedule on every Python version.
"""
from __future__ import annotations

import hashlib

ARM_OF = {"A": "baseline", "B": "candidate"}
SEQUENCES = ("AB", "BA")


class _Stream:
    def __init__(self, seed: int, label: str):
        self.prefix = f"remyx-validate/schedule/v1/{seed}/{label}/".encode("utf-8")
        self.n = 0

    def below(self, k: int) -> int:
        """Uniform integer in [0, k), by rejection so there is no modulo bias."""
        limit = (1 << 64) - ((1 << 64) % k)
        while True:
            raw = hashlib.sha256(self.prefix + str(self.n).encode("ascii")).digest()
            self.n += 1
            value = int.from_bytes(raw[:8], "big")
            if value < limit:
                return value % k

    def shuffle(self, items: list) -> list:
        items = list(items)
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]
        return items


def _balanced(n: int, stream: _Stream) -> list:
    """n sequence labels, as close to half AB and half BA as n allows, in
    random order. With an odd n the extra one is chosen by the stream."""
    labels = [SEQUENCES[i % 2] for i in range(n)]
    if n % 2 and stream.below(2):
        labels[-1] = "BA" if labels[-1] == "AB" else "AB"
    return stream.shuffle(labels)


def build(seeds: list, measurement: dict) -> list:
    """Every measured attempt, in the order it is planned to run."""
    seed = measurement["schedule_seed"]
    blocks = [(s, r) for r in range(measurement["replicates"]) for s in sorted(seeds)]
    blocks = _Stream(seed, "blocks").shuffle(blocks)
    first = _balanced(len(blocks), _Stream(seed, "sequences"))
    crossover = measurement["design"] == "crossover"
    slots = []
    for index, ((workload_seed, replicate), lead) in enumerate(zip(blocks, first)):
        sequences = (lead, "BA" if lead == "AB" else "AB") if crossover else (lead,)
        for sequence in sequences:
            for period, letter in enumerate(sequence, start=1):
                slots.append({"ordinal": len(slots) + 1, "block": index, "seed": workload_seed,
                              "replicate": replicate, "sequence": sequence, "period": period,
                              "arm": ARM_OF[letter]})
    return slots


def groups(slots: list) -> list:
    """Slots grouped into sequences: the unit that is run, and re-run, whole."""
    out = []
    for slot in slots:
        key = (slot["block"], slot["sequence"])
        if not out or out[-1][0] != key:
            out.append((key, []))
        out[-1][1].append(slot)
    return [members for _, members in out]


def slot_key(row: dict) -> tuple:
    return (row["arm"], row["seed"], row.get("replicate"), row.get("sequence"))


def problems(slots: list, seeds: list, measurement: dict) -> list:
    """Balance properties a schedule must have. Used by tests and by verify."""
    out = []
    crossover = measurement["design"] == "crossover"
    want = {(s, r) for s in seeds for r in range(measurement["replicates"])}
    seen = {}
    for slot in slots:
        seen.setdefault((slot["seed"], slot["replicate"]), []).append(slot)
    if set(seen) != want:
        out.append("the schedule does not cover every seed and replicate")
    for block, members in sorted(seen.items()):
        cells = sorted((m["sequence"], m["period"], m["arm"]) for m in members)
        full = sorted((q, p, ARM_OF[q[p - 1]]) for q in SEQUENCES for p in (1, 2))
        if crossover and cells != full:
            out.append(f"seed {block[0]} replicate {block[1]} does not run both orders")
        if not crossover and cells not in ([c for c in full if c[0] == "AB"],
                                           [c for c in full if c[0] == "BA"]):
            out.append(f"seed {block[0]} replicate {block[1]} is not one whole sequence")
    if [s["ordinal"] for s in slots] != list(range(1, len(slots) + 1)):
        out.append("schedule ordinals are not 1..n in order")
    return out
