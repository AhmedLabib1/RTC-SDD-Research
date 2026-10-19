"""Local evaluation invariants vs the official RTC-SDD metric definition."""

import pytest

from rtcsdd.data.protocol import Example
from rtcsdd.eval.competition import (
    choose_better_checkpoint,
    competition_proxy,
    dev_online_macro_f1,
)
from rtcsdd.eval.metrics import evaluate, evaluate_subgroups


def dev(ids, labels, *, view="online"):
    return [
        Example(f"{view}/en/{id}.wav", "dev", view, "en", label)
        for id, label in zip(ids, labels)
    ]


def test_threshold_and_hand_checked_macro_f1():
    examples = dev(["a", "b", "c", "d"], [0, 0, 1, 1])
    # Predictions: spoof, bona, spoof, bona; each class has P=R=0.5.
    scores = dict(zip((e.utterance_id for e in examples), [.5, .1, .8, .2]))
    metrics = evaluate(examples, scores)
    assert metrics["threshold"] == .5
    assert metrics["confusion"]["true_spoof_pred_spoof"] == 1
    assert metrics["confusion"]["true_spoof_pred_bonafide"] == 1
    assert metrics["macro_f1"] == pytest.approx(.5)
    assert dev_online_macro_f1(evaluate_subgroups(examples, scores)) == pytest.approx(.5)


def test_single_predicted_class_macro_f1_zero_for_missing_class():
    examples = dev(["a", "b", "c", "d"], [0, 0, 1, 1])
    scores = {e.utterance_id: 1.0 for e in examples}
    # Spoof F1=2*2/(2*2 + 2 + 0)=2/3, Bonafide F1=0 => macro 1/3.
    assert evaluate(examples, scores)["macro_f1"] == pytest.approx(1 / 3)


def test_compare_sklearn_f1_if_available():
    sklearn = pytest.importorskip("sklearn.metrics")
    examples = dev(["a", "b", "c", "d", "e", "f"], [0, 1, 0, 1, 0, 1])
    probs = [.5, .49, .3, .9, .7, .3]
    pred = [0 if p >= .5 else 1 for p in probs]
    scores = dict(zip((e.utterance_id for e in examples), probs))
    reference = sklearn.f1_score(
        [e.label for e in examples], pred,
        labels=[0, 1], average="macro", zero_division=0,
    )
    assert evaluate(examples, scores)["macro_f1"] == pytest.approx(reference)


def test_online_only_is_primary_not_all_dev():
    online = dev(["a", "b"], [0, 1])
    offline = dev(["c", "d"], [0, 1], view="offline")
    both = online + offline
    scores = {
        online[0].utterance_id: .9, online[1].utterance_id: .1,
        offline[0].utterance_id: .1, offline[1].utterance_id: .9,
    }
    metrics = evaluate_subgroups(both, scores)
    assert dev_online_macro_f1(metrics) == 1.0
    assert metrics["all_dev"]["macro_f1"] < 1.0


def test_best_checkpoint_selection_maximizes_online_f1():
    assert choose_better_checkpoint(.85, .42, .82, .30)
    assert not choose_better_checkpoint(.80, .10, .82, .30)
    assert choose_better_checkpoint(.82, .21, .82, .30)
    assert not choose_better_checkpoint(.82, .35, .82, .30)


def test_noisy_proxy_is_explicitly_not_official():
    online = dev(["a", "b"], [0, 1])
    clean = {online[0].utterance_id: .9, online[1].utterance_id: .1}
    noisy = {online[0].utterance_id: .1, online[1].utterance_id: .1}
    outcome = competition_proxy(online, clean, online, noisy)
    assert outcome["clean_online_macro_f1"] == 1.0
    assert outcome["synthetic_noisy_online_macro_f1"] == pytest.approx(1 / 3)
    assert outcome["proxy_weighted_macro_f1"] == pytest.approx(.3 + .7 / 3)
    assert "NOT_official" in outcome["metric"]
    with pytest.raises(ValueError, match="online DEV"):
        competition_proxy(dev(["a"], [0], view="offline"), {}, online, noisy)


def test_invalid_or_missing_scores_rejected():
    online = dev(["a", "b"], [0, 1])
    with pytest.raises(ValueError, match="Score IDs mismatch"):
        evaluate(online, {online[0].utterance_id: .2})
    with pytest.raises(ValueError, match="Invalid spoof score"):
        evaluate(online, {online[0].utterance_id: float("nan"), online[1].utterance_id: .1})
