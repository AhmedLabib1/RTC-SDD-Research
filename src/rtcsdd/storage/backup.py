"""Optional best-effort cloud backup orchestration for ANY experiment.

Every remote upload follows a completed LOCAL atomic save. A failed upload must
never delete/corrupt a resume checkpoint or crash the GPU training loop. Backup
settings do not influence gradients and can change safely between resumes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rtcsdd.storage.huggingface import HuggingFaceCheckpointStore, validate_remote_path


DEFAULTS = {
    'enabled': False,
    'repo_id': None,
    'remote_prefix': None,
    'upload_every_optimizer_steps': 1000,
    'upload_on_pause': True,
    'upload_after_epoch': True,
    'upload_best_after_epoch': True,
    'max_attempts': 2,
    'retry_delay_seconds': 2,
    'create_private': True,
    'delete_local_after_upload': False,
}


def validate_backup_config(raw: dict | None) -> dict:
    if raw is not None and not isinstance(raw, dict):
        raise ValueError('checkpoint_backup must be a JSON object')
    raw = raw or {}
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown checkpoint_backup options: {sorted(unknown)}')
    settings = {**DEFAULTS, **raw}
    for key in ('enabled', 'upload_on_pause', 'upload_after_epoch',
                'upload_best_after_epoch', 'create_private', 'delete_local_after_upload'):
        if not isinstance(settings[key], bool):
            raise ValueError(f'checkpoint_backup.{key} must be boolean')
    if settings['delete_local_after_upload']:
        raise ValueError('Automatic deletion is forbidden: last.pt is needed for fast resume')
    for key in ('upload_every_optimizer_steps', 'max_attempts', 'retry_delay_seconds'):
        value = settings[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f'checkpoint_backup.{key} must be an integer')
    if settings['upload_every_optimizer_steps'] < 0 or not 1 <= settings['max_attempts'] <= 5:
        raise ValueError('Invalid checkpoint backup cadence or retry count')
    if not 0 <= settings['retry_delay_seconds'] <= 30:
        raise ValueError('Invalid checkpoint_backup.retry_delay_seconds')
    if settings['enabled']:
        repo_id = settings['repo_id']
        if not isinstance(repo_id, str) or repo_id.count('/') != 1:
            raise ValueError('Enabled checkpoint_backup requires repo_id namespace/model')
        prefix = settings['remote_prefix']
        if not isinstance(prefix, str):
            raise ValueError('Enabled checkpoint_backup requires explicit unique remote_prefix')
        validate_remote_path(prefix)
    return settings


class CheckpointBackupManager:
    """One experiment's optional Hugging Face storage hooks."""

    def __init__(self, config: dict, output: Path, *, store=None):
        self.cfg = validate_backup_config(config.get('checkpoint_backup'))
        self.output = Path(output)
        self.enabled = self.cfg['enabled']
        self._store = store
        self._manifest_uploaded = False
        if self.enabled:
            print('[HF BACKUP] enabled: '
                  f"repo={self.cfg['repo_id']} prefix={self.cfg['remote_prefix']}; "
                  'LOCAL CHECKPOINTS WILL NOT BE DELETED', flush=True)
        else:
            print('[HF BACKUP] disabled (set checkpoint_backup.enabled=true to opt in)',
                  flush=True)

    def _storage(self):
        if self._store is None:
            self._store = HuggingFaceCheckpointStore(
                self.cfg['repo_id'], create_private=self.cfg['create_private'],
            )
        return self._store

    def _record(self, *, local: Path, remote: str, ok: bool,
                reason: str, error: str | None = None) -> None:
        # Persist status without serializing any token or cloud credentials.
        data = {
            'local_file': local.name,
            'remote_path': remote,
            'reason': reason,
            'success': ok,
            'error': error,
            'timestamp_unix': int(time.time()),
        }
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / 'backup_history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(data, sort_keys=True) + '\n')
        if not ok:
            print(f'[HF BACKUP FAILED] {local.name}: {error}; local file kept', flush=True)

    def upload(self, path: Path, *, reason: str, epoch: int,
               global_step: int) -> bool:
        if not self.enabled:
            return False
        path = Path(path)
        remote = validate_remote_path(f"{self.cfg['remote_prefix']}/{path.name}")
        if not path.is_file():
            raise FileNotFoundError(path)  # error in trainer, not remote network failure
        for attempt in range(1, self.cfg['max_attempts'] + 1):
            try:
                receipt = self._storage().upload_checkpoint(
                    path, remote,
                    metadata={'experiment': self.output.name, 'epoch': epoch,
                              'global_step': global_step, 'reason': reason},
                )
                self._record(local=path, remote=remote, ok=True, reason=reason)
                print(f"[HF BACKUP OK] {path.name} -> {self.cfg['repo_id']}/{remote} "
                      f"({receipt['size_bytes']/1024**3:.2f} GiB)", flush=True)
                return True
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                # Do not dump an HTTP exception body or an access token.
                error = error.replace(str(getattr(self._store, 'token', None) or ''), '[redacted]') \
                    if getattr(self._store, 'token', None) else error
                if attempt < self.cfg['max_attempts']:
                    print(f'[HF BACKUP RETRY] {path.name} attempt {attempt} failed', flush=True)
                    time.sleep(self.cfg['retry_delay_seconds'])
                else:
                    self._record(local=path, remote=remote, ok=False,
                                 reason=reason, error=error[:300])
        return False

    def _ensure_manifest(self, *, epoch: int, global_step: int) -> None:
        """A restored last.pt is not resumable without the ORIGINAL config.json."""
        if self._manifest_uploaded:
            return
        manifest = self.output / 'config.json'
        if not manifest.is_file():
            print('[HF BACKUP WARNING] output/config.json missing: recovery metadata unavailable',
                  flush=True)
            return
        self._manifest_uploaded = self.upload(
            manifest, reason='experiment_manifest', epoch=epoch, global_step=global_step,
        )

    def saved_last(self, *, epoch: int, global_step: int, reason: str) -> None:
        """Call AFTER last.pt atomic save. Pause/epoch uploads ignore cadence."""
        if not self.enabled:
            return
        every = self.cfg['upload_every_optimizer_steps']
        due = ((reason == 'pause' and self.cfg['upload_on_pause']) or
               (reason in ('pre_eval', 'epoch_end') and self.cfg['upload_after_epoch']) or
               (reason == 'periodic' and every > 0 and global_step % every == 0))
        if due:
            self._ensure_manifest(epoch=epoch, global_step=global_step)
            self.upload(self.output / 'last.pt', reason=reason,
                        epoch=epoch, global_step=global_step)
            if reason == 'epoch_end' and epoch > 1:
                epoch_metrics = self.output / f'dev_epoch_{epoch-1:03d}.json'
                if epoch_metrics.is_file():
                    self.upload(epoch_metrics, reason='epoch_dev_metrics',
                                epoch=epoch-1, global_step=global_step)

    def saved_best(self, name: str, *, epoch: int, global_step: int) -> None:
        if not self.enabled or not self.cfg['upload_best_after_epoch']:
            return
        if name not in ('best_dev_loss.pt', 'best_online_f1.pt'):
            raise ValueError(f'Unexpected best checkpoint name: {name}')
        self._ensure_manifest(epoch=epoch, global_step=global_step)
        self.upload(self.output / name, reason='improved_dev',
                    epoch=epoch, global_step=global_step)
