"""Every verdict has a deterministic fixture."""
import pytest

from remyx_validate import demo
from conftest import load_evidence


@pytest.mark.parametrize("name", list(demo.SCENARIOS))
def test_scenario_verdict(run_scenario, name):
    """As the demo ships it: every seed in both orders, empty caches."""
    summary, _ = run_scenario(name, protocol=True)
    assert summary["verdict"] == demo.SCENARIOS[name]["expect"], summary["reasons"]
    ev = load_evidence(summary)
    assert ev["execution"]["protocol"]["design"] == "crossover"
    assert all(a["isolation"]["fresh"] for a in ev["execution"]["attempts"])


@pytest.mark.parametrize("name", ["guardrail", "positive", "missing", "drift", "inactive"])
def test_scenario_verdict_without_the_protocol(run_scenario, name):
    summary, _ = run_scenario(name, protocol=False)
    assert summary["verdict"] == demo.SCENARIOS[name]["expect"], summary["reasons"]
    assert load_evidence(summary)["execution"]["protocol"]["order_controlled"] is False


def test_guardrail_veto_from_the_design_brief(run_scenario):
    summary, _ = run_scenario("guardrail")
    ev = load_evidence(summary)
    by_id = {c["id"]: c for c in ev["comparison"]["criteria"]}
    assert ev["execution"]["status"] == "COMPLETED"
    assert ev["comparison"]["validity"]["status"] == "VALID"
    assert by_id["resolution_rate"]["status"] == "met"
    assert by_id["p95_latency"]["status"] == "breached"
    assert by_id["p95_latency"]["candidate_interval"][0] > 800
    assert by_id["cost_per_request"]["status"] == "info"
    assert summary["reasons"] == ["guardrail_veto:p95_latency"]


def test_execution_validity_and_verdict_are_separate(run_scenario):
    for name, execution, validity in [("timeout", "TIMEOUT", "NOT_ASSESSED"),
                                      ("inactive", "COMPLETED", "INVALID"),
                                      ("tamper", "COMPLETED", "INVALID"),
                                      ("null", "COMPLETED", "VALID")]:
        summary, _ = run_scenario(name)
        assert (summary["execution_status"], summary["validity"]) == (execution, validity)


def test_measured_zero_is_not_an_inactive_treatment(run_scenario):
    null, _ = run_scenario("null")
    inactive, _ = run_scenario("inactive")
    assert "measured_zero" in null["reasons"]
    assert inactive["reasons"] == ["inactive_treatment:pipeline_fingerprint"]


def test_identical_arms_without_a_witness_is_no_signal(run_scenario):
    def drop_activation(text):
        return text.replace("  activation:\n    - metric: pipeline_fingerprint   # must differ between arms\n", "")
    summary, _ = run_scenario("null", edit=drop_activation)
    assert summary["verdict"] == "NO_SIGNAL"
    assert summary["reasons"] == ["identical_arms_unverified"]


def test_point_rule_decides_what_interval_rule_cannot(run_scenario):
    summary, _ = run_scenario("uncertain", edit=lambda t: t.replace("uncertainty: interval", "uncertainty: point"))
    assert summary["verdict"] == "PASS"
    ev = load_evidence(summary)
    assert any("point" in text for text in ev["comparison"]["limitations"])


def test_runs_are_reproducible(run_scenario):
    a, _ = run_scenario("guardrail")
    ea = load_evidence(a)
    b, _ = run_scenario("guardrail")
    eb = load_evidence(b)
    assert ea["change"]["change_id"] == eb["change"]["change_id"]
    assert ea["observations"]["sha256"] == eb["observations"]["sha256"]
    for ca, cb in zip(ea["comparison"]["criteria"], eb["comparison"]["criteria"]):
        assert ca["delta"] == cb["delta"] and ca["delta_interval"] == cb["delta_interval"]
