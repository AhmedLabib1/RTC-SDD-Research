import pytest

from rtcsdd.data.protocol import read_protocol


def test_labeled_train(tmp_path):
    p = tmp_path / "train_label.txt"
    p.write_text("offline/en/Train_000001.wav spoof\nonline/zh/Train_000002.wav bonafide\n")
    examples = read_protocol(p, "train")
    assert [e.label for e in examples] == [0, 1]
    assert [e.view for e in examples] == ["offline", "online"]


def test_refuse_labeled_progress(tmp_path):
    p = tmp_path / "progress.txt"
    p.write_text("online/en/test.wav spoof\n")
    with pytest.raises(ValueError, match="expected 1"):
        read_protocol(p, "progress")


def test_refuse_traversal_and_duplicates(tmp_path):
    p = tmp_path / "train_label.txt"
    p.write_text("offline/en/../x.wav spoof\n")
    with pytest.raises(ValueError):
        read_protocol(p, "train")
    p.write_text("offline/en/x.wav spoof\noffline/en/x.wav spoof\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_protocol(p, "train")
