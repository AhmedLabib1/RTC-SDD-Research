import pytest

from rtcsdd.data.protocol import Example
from rtcsdd.eval.metrics import evaluate, evaluate_subgroups
from rtcsdd.eval.submission import read_score_file, write_submission


def test_binary_macro_f1():
    examples = [
        Example("online/en/a.wav", "dev", "online", "en", 0),
        Example("online/en/b.wav", "dev", "online", "en", 1),
        Example("offline/zh/c.wav", "dev", "offline", "zh", 0),
        Example("offline/zh/d.wav", "dev", "offline", "zh", 1),
    ]
    scores = {"online/en/a.wav": 0.9, "online/en/b.wav": 0.1,
              "offline/zh/c.wav": 0.8, "offline/zh/d.wav": 0.2}
    result = evaluate(examples, scores)
    assert result["macro_f1"] == 1.0
    assert evaluate_subgroups(examples, scores)["view/online"]["macro_f1"] == 1.0


def test_strict_submission_id_order(tmp_path):
    examples = [
        Example("online/en/b.wav", "progress", "online", "en", None),
        Example("online/en/a.wav", "progress", "online", "en", None),
    ]
    scores = {"online/en/a.wav": 0.1, "online/en/b.wav": 0.8}
    p = tmp_path / "out.txt"
    write_submission(examples, scores, p)
    assert p.read_text().splitlines() == [
        "online/en/b.wav 0.8000000000", "online/en/a.wav 0.1000000000"
    ]
    assert read_score_file(p) == scores


def test_reject_nan_and_extra(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("online/en/a.wav nan\n")
    with pytest.raises(ValueError, match="finite"):
        read_score_file(path)
    example = [Example("online/en/a.wav", "progress", "online", "en", None)]
    with pytest.raises(ValueError, match="ID mismatch"):
        write_submission(example, {"online/en/different.wav": 0.8}, tmp_path / "out.txt")
