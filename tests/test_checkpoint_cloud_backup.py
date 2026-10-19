"""Offline tests: no network/GPU, including recovery and failed uploads."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rtcsdd.storage import huggingface as hf
from rtcsdd.storage.backup import CheckpointBackupManager, validate_backup_config
from rtcsdd.training.config import fingerprint, read_config


class FakeHubApi:
    def __init__(self, *, private=True, bad_size=False, error=False, exists=True):
        self.private = private
        self.bad_size = bad_size
        self.error = error
        self.exists = exists
        self.remote_files = {}
        self.commits = 0
        self.creates = 0

    def repo_info(self, **kwargs):
        if not self.exists:
            raise FileNotFoundError('repository missing')
        return SimpleNamespace(private=self.private)

    def create_repo(self, **kwargs):
        self.creates += 1
        self.private = kwargs['private']
        self.exists = True

    def create_commit(self, *, operations, **kwargs):
        if self.error:
            raise ConnectionError('temporary network failure')
        # All items appear together (simulated atomic commit).
        staged = {}
        for op in operations:
            path = op.path_or_fileobj
            staged[op.path_in_repo] = Path(path).read_bytes() if isinstance(path, (str, Path)) else path
        self.remote_files.update(staged)
        self.commits += 1

    def get_paths_info(self, *, paths, **kwargs):
        result = []
        for name in paths:
            if name not in self.remote_files:
                continue
            payload = self.remote_files[name]
            result.append(SimpleNamespace(
                path=name,
                size=len(payload) + (1 if self.bad_size else 0),
                lfs=SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()),
            ))
        return result


def store(fake):
    return hf.HuggingFaceCheckpointStore('team/private-model', token='test-token', api=fake)


def test_remote_path_validation():
    for invalid in ('', '/absolute', '../evil', 'a/../b', 'a//b', 'a\\b', ' a/x'):
        with pytest.raises(ValueError):
            hf.validate_remote_path(invalid)
    assert hf.validate_remote_path('experiments/p01/last.pt') == 'experiments/p01/last.pt'


def test_upload_atomic_two_files_and_hash(tmp_path):
    checkpoint = tmp_path / 'last.pt'
    checkpoint.write_bytes(b'fake-training-state-001')
    api = FakeHubApi()
    receipt = store(api).upload_checkpoint(
        checkpoint, 'experiments/p01/last.pt', metadata={'epoch': 1, 'global_step': 1000},
    )
    assert api.commits == 1
    assert api.remote_files['experiments/p01/last.pt'] == checkpoint.read_bytes()
    assert json.loads(api.remote_files['experiments/p01/last.pt.sha256.json']) == receipt
    assert receipt['sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_private_rejects_public(tmp_path):
    p = tmp_path / 'last.pt'; p.write_bytes(b'data')
    api = FakeHubApi(private=False)
    with pytest.raises(hf.CheckpointStorageError, match='not verified PRIVATE'):
        store(api).upload_checkpoint(p, 'e/last.pt')
    assert api.commits == 0


def test_private_repo_created_if_missing(tmp_path):
    p = tmp_path / 'last.pt'; p.write_bytes(b'data')
    api = FakeHubApi(exists=False)
    store(api).upload_checkpoint(p, 'e/last.pt')
    assert api.creates == 1 and api.private is True


def test_remote_size_mismatch_fails_and_local_intact(tmp_path):
    p = tmp_path / 'last.pt'; p.write_bytes(b'data')
    api = FakeHubApi(bad_size=True)
    with pytest.raises(hf.CheckpointStorageError, match='size mismatch'):
        store(api).upload_checkpoint(p, 'e/last.pt')
    assert p.read_bytes() == b'data'


def test_disabled_manager_does_not_require_token(tmp_path, monkeypatch):
    monkeypatch.delenv('HF_TOKEN', raising=False)
    manager = CheckpointBackupManager({}, tmp_path)
    manager.saved_last(epoch=1, global_step=1000, reason='periodic')
    assert not (tmp_path / 'backup_history.jsonl').exists()


def test_enabled_config_validates_and_never_allows_deletion():
    settings = {'enabled': True, 'repo_id': 'alice/backup', 'remote_prefix': 'experiments/p01'}
    assert validate_backup_config(settings)['enabled']
    with pytest.raises(ValueError, match='forbidden'):
        validate_backup_config({**settings, 'delete_local_after_upload': True})
    with pytest.raises(ValueError, match='requires repo_id'):
        validate_backup_config({'enabled': True, 'remote_prefix': 'abc'})


class FakeStore:
    def __init__(self, fail=False):
        self.uploaded = []
        self.fail = fail
        self.token = 'secret-token'

    def upload_checkpoint(self, path, remote_path, *, metadata=None):
        self.uploaded.append((Path(path).name, remote_path, metadata))
        if self.fail:
            raise ConnectionError('secret-token network error')
        return {'size_bytes': Path(path).stat().st_size}


def manager(tmp_path, store, *, cadence=1000):
    opts = dict(enabled=True, repo_id='team/private', remote_prefix='experiments/p01',
                upload_every_optimizer_steps=cadence,
                max_attempts=1)
    return CheckpointBackupManager({'checkpoint_backup': opts}, tmp_path, store=store)


def test_hook_cadence_and_pause_and_best(tmp_path):
    (tmp_path/'last.pt').write_bytes(b'x')
    (tmp_path/'best_dev_loss.pt').write_bytes(b'y')
    (tmp_path/'config.json').write_text('{}')
    fake = FakeStore()
    mgr = manager(tmp_path, fake)
    mgr.saved_last(epoch=1, global_step=999, reason='periodic')
    assert not fake.uploaded
    mgr.saved_last(epoch=1, global_step=1000, reason='periodic')
    mgr.saved_last(epoch=1, global_step=1100, reason='pause')
    mgr.saved_best('best_dev_loss.pt', epoch=1, global_step=4737)
    assert [x[0] for x in fake.uploaded] == [
        'config.json','last.pt','last.pt','best_dev_loss.pt'
    ]
    assert len((tmp_path/'backup_history.jsonl').read_text().splitlines()) == 4


def test_network_failure_nonfatal_and_local_intact(tmp_path):
    file = tmp_path/'last.pt';file.write_bytes(b'valuable-state')
    fake = FakeStore(fail=True)
    mgr = manager(tmp_path, fake)
    mgr.saved_last(epoch=1, global_step=1000, reason='periodic')
    assert file.read_bytes() == b'valuable-state'
    text = (tmp_path/'backup_history.jsonl').read_text()
    assert 'success": false' in text.lower()
    assert 'secret-token' not in text


def test_backup_changes_do_not_invalidate_existing_p01_fingerprint():
    cfg = read_config(Path(__file__).parent.parent/'src/rtcsdd/configs/p01_xlsr_aasist_official_like_fp32.json')
    baseline = dict(cfg)
    baseline.pop('checkpoint_backup')
    assert fingerprint(cfg) == fingerprint(baseline)
    updated = dict(cfg)
    updated['checkpoint_backup'] = {**cfg['checkpoint_backup'],
        'enabled': True, 'repo_id':'user/repo'}
    assert fingerprint(cfg) == fingerprint(updated)


def test_restore_verifies_sha_and_refuses_existing(tmp_path, monkeypatch):
    api = FakeHubApi()
    s = store(api)
    src=tmp_path/'src.pt';src.write_bytes(b'important-model')
    s.upload_checkpoint(src, 'experiments/p01/last.pt')

    import huggingface_hub
    def fake_download(*, filename, local_dir, **kwargs):
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(api.remote_files[filename])
        return str(destination)

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_download)
    dest=tmp_path/'last.pt'
    assert s.download_checkpoint('experiments/p01/last.pt',dest)==dest
    assert dest.read_bytes()==src.read_bytes()
    with pytest.raises(FileExistsError):
        s.download_checkpoint('experiments/p01/last.pt',dest)
    api.remote_files['experiments/p01/last.pt']=b'CORRUPTED'
    with pytest.raises(hf.CheckpointStorageError,match='(size|SHA-256)'):
        s.download_checkpoint('experiments/p01/last.pt', tmp_path/'other.pt')


def test_config_imports_training_runner_without_hf_network():
    from rtcsdd.training import trainer
    from rtcsdd.storage import cli
    assert callable(trainer.train)
    assert callable(cli.main)
