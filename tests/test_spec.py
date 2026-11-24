"""Schema, canonical form and identities."""
import copy
import os

import pytest
import yaml

from remyx_validate import declaration as decl, demo, plan as planmod
from remyx_validate.canonical import canonical_json, digest


def base():
    return yaml.safe_load(demo.declaration_text("guardrail"))


def test_demo_declaration_is_valid():
    assert decl.schema_problems(base(), "declaration") == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d.pop("criteria"), "criteria"),
    (lambda d: d.pop("spec_version"), "spec_version"),
    (lambda d: d["change"].pop("baseline"), "baseline"),
    (lambda d: d["criteria"][0].pop("direction"), "direction"),
    (lambda d: d["criteria"][0].update(role="goal"), "goal"),
    (lambda d: d.update(surprise=1), "surprise"),
    (lambda d: d["run"].update(retries=50), "50"),
    (lambda d: d["method"].update(confidence=1.5), "1.5"),
])
def test_schema_rejects(mutate, needle):
    d = base()
    mutate(d)
    problems = decl.schema_problems(d, "declaration")
    assert problems and any(needle in p for p in problems)


def test_semantic_rules(tmp_path):
    def problems(mutate):
        d = base()
        mutate(d)
        return decl.semantic_problems(decl.normalize(d))

    assert any("guardrail needs" in p for p in problems(lambda d: d["criteria"][1].pop("max")))
    assert any("at least one target or guardrail" in p
               for p in problems(lambda d: d.update(criteria=[d["criteria"][2]])))
    assert any("untrusted" in p for p in problems(lambda d: d["execution"].update(trust="untrusted")))
    assert any("placeholder" in p for p in problems(lambda d: d["criteria"][0].update(metric="TODO_metric")))
    assert any("set by the runner" in p
               for p in problems(lambda d: d["arms"]["candidate"]["env"].update(VALIDATE_ARM="baseline")))


def test_equivalent_declarations_hash_identically():
    a = decl.normalize(base())
    b = base()
    b["criteria"] = list(reversed(b["criteria"]))          # order of criteria
    b["method"] = dict(reversed(list(b["method"].items())))  # key order
    b["criteria"][0]["x-note"] = "extension fields are ignored"
    b["criteria"][1]["max"] = 800.0                        # 800 vs 800.0
    b["method"].pop("uncertainty")                         # an omitted default
    assert decl.declaration_hash(a) == decl.declaration_hash(decl.normalize(b))


def test_threshold_edit_changes_declaration_hash_not_env_key():
    a = decl.normalize(base())
    b = copy.deepcopy(a)
    b["criteria"][1]["max"] = 950
    sha = "a" * 40
    assert decl.declaration_hash(a) != decl.declaration_hash(b)
    for arm in ("baseline", "candidate"):
        assert decl.env_key(a, arm, sha) == decl.env_key(b, arm, sha)
    assert decl.spec_hash(a, {"baseline": sha}, None) == decl.spec_hash(b, {"baseline": sha}, None)


def test_setup_edit_changes_env_key_not_declaration_hash():
    a = decl.normalize(base())
    b = copy.deepcopy(a)
    b["setup"] = ["pip install something"]
    assert decl.declaration_hash(a) == decl.declaration_hash(b)
    assert decl.env_key(a, "baseline", "a" * 40) != decl.env_key(b, "baseline", "a" * 40)


def test_arms_at_the_same_commit_never_share_an_environment():
    d = decl.normalize(base())
    sha = "a" * 40
    assert decl.env_key(d, "baseline", sha) != decl.env_key(d, "candidate", sha)
    d["arms"]["candidate"]["env"]["FLAG"] = "1"
    before = decl.env_key(d, "candidate", sha)
    d["arms"]["candidate"]["env"]["FLAG"] = "2"
    assert decl.env_key(d, "candidate", sha) != before


def test_image_content_id_is_part_of_the_environment_identity():
    d = decl.normalize(base())
    d["execution"].update(backend="docker", trust="untrusted", network="none")
    sha = "a" * 40
    assert decl.env_key(d, "baseline", sha, "sha256:aaa") != decl.env_key(d, "baseline", sha, "sha256:bbb")


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "impact.yaml"
    path.write_text(demo.declaration_text("guardrail").replace(
        "    max: 800 ", "    max: 800\n    max: 99999 "))
    with pytest.raises(decl.DeclarationError) as exc:
        decl.load(str(path))
    assert "duplicate key" in str(exc.value)


def test_more_semantic_rules():
    def problems(mutate):
        d = base()
        mutate(d)
        bad = decl.schema_problems(d, "declaration")
        return bad or decl.semantic_problems(decl.normalize(d))

    assert problems(lambda d: d.update(preflight={}))                      # needs a command
    assert any("no value can satisfy" in p for p in problems(lambda d: d["criteria"][1].update(min=900)))
    assert any("only applies to a guardrail" in p for p in problems(lambda d: d["criteria"][0].update(max=1)))
    assert any("only applies to a guardrail" in p for p in problems(lambda d: d["criteria"][2].update(max_regression=1)))
    assert any("not chosen how to run" in p
               for p in problems(lambda d: d["execution"].update(trust="review-required")))
    untrusted = dict(backend="docker", trust="untrusted")
    assert problems(lambda d: d["execution"].update(untrusted)) == []      # network defaults to none
    assert any("pass_env" in p for p in problems(lambda d: d["execution"].update(untrusted, pass_env=["HF_TOKEN"])))
    assert any("network access" in p for p in problems(lambda d: d["execution"].update(untrusted, network="setup-only")))
    assert problems(lambda d: d["execution"].update(untrusted, network="setup-only",
                                                    acknowledge_untrusted_network=True)) == []
    assert any("looks like a credential" in p
               for p in problems(lambda d: d["arms"]["candidate"]["env"].update(KEY="ghp_" + "a" * 36)))
    assert any("embeds credentials" in p
               for p in problems(lambda d: d["change"].update(repo="https://user:secret@example.com/r.git")))
    assert any("cannot start with" in p for p in problems(lambda d: d["change"]["candidate"].update(ref="--upload-pack=x")))


def test_untrusted_docker_defaults_to_no_network():
    d = base()
    d["execution"].update(backend="docker", trust="untrusted")
    assert decl.normalize(d)["execution"]["network"] == "none"


def test_canonical_json():
    assert canonical_json({"b": 1.0, "a": [2, {"z": None, "y": True}]}) == '{"a":[2,{"y":true,"z":null}],"b":1}'
    assert digest({"a": 1, "x-ext": 5}) == digest({"a": 1})
    with pytest.raises(ValueError):
        canonical_json({"a": float("nan")})


def test_new_candidate_is_a_new_change_under_the_same_declaration(tmp_path):
    path = demo.create(str(tmp_path / "p"), "guardrail")
    first = planmod.build(path)
    with open(path, encoding="utf-8") as fh:
        text = fh.read().replace("ref: candidate}", "ref: candidate-2}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    second = planmod.build(path)
    assert first["declaration_hash"] == second["declaration_hash"]
    assert first["change"]["change_id"] != second["change"]["change_id"]
    assert first["env_keys"]["baseline"] == second["env_keys"]["baseline"]
    assert first["plan_digest"] != second["plan_digest"]


def test_all_schemas_are_valid_json_schema():
    import jsonschema
    for name in ("declaration", "evidence", "observation", "decision", "outcome"):
        jsonschema.Draft202012Validator.check_schema(decl.load_schema(name))
