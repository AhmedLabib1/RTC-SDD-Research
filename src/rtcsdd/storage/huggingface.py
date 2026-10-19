"""Upload/restore generic checkpoints in a PRIVATE Hugging Face Model repository.

The caller owns when to save. This module never deletes local files and never
loads an untrusted pickle. No network access occurs until an operation is called.
Credentials come from HF_TOKEN, NOT source code or experiment configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


class CheckpointStorageError(RuntimeError):
    """A remote backup cannot be trusted as complete."""


def validate_remote_path(value: str) -> str:
    """Use a relative POSIX path that cannot escape a remote experiment prefix."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Remote path must be a nonempty string without surrounding spaces")
    path = PurePosixPath(value)
    if (path.is_absolute() or '\\' in value or '//' in value or
            any(part in ('', '.', '..') for part in value.split('/')) or
            str(path) != value):
        raise ValueError(f"Unsafe HF repository path: {value!r}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _required_token() -> str:
    token = os.environ.get('HF_TOKEN', '').strip()
    if token:
        return token
    # Works directly in Kaggle if a secret named HF_TOKEN has been added
    # and Notebook access was granted; no notebook upload/auth code required.
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        token = None
    if not token:
        raise CheckpointStorageError(
            'No Hugging Face token available. Provide HF_TOKEN as an environment '
            'variable or an authorized Kaggle Secret named HF_TOKEN; never store '
            'tokens in JSON, Python code, or Git.'
        )
    return str(token).strip()


class HuggingFaceCheckpointStore:
    """One API instance for a private model repo; injectable for offline tests."""

    def __init__(self, repo_id: str, *, api: Any = None, token: str | None = None,
                 create_private: bool = True):
        if not isinstance(repo_id, str) or repo_id.count('/') != 1 or any(
            part in ('', '.', '..') for part in repo_id.split('/')
        ):
            raise ValueError('repo_id must be namespace/repository-name')
        self.repo_id = repo_id
        self.token = token if token is not None else _required_token()
        if not self.token:
            raise CheckpointStorageError('An HF access token is required')
        if api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise CheckpointStorageError(
                    'Install huggingface_hub into the SAME Python 3.10 venv used for training'
                ) from exc
            api = HfApi(token=self.token)
        self.api = api
        self.create_private = create_private
        self._private_verified = False

    def ensure_private_repo(self) -> None:
        if self._private_verified:
            return
        try:
            info = self.api.repo_info(
                repo_id=self.repo_id, repo_type='model', token=self.token
            )
        except Exception as exc:
            # A private repo without permission can also present as 404. Only
            # attempt creation when explicitly enabled; never bypass auth errors.
            if not self.create_private:
                raise CheckpointStorageError(
                    'Cannot access HF repo. Check repo_id, write permission, and token.'
                ) from exc
            try:
                self.api.create_repo(
                    repo_id=self.repo_id, repo_type='model', private=True,
                    token=self.token, exist_ok=True,
                )
                info = self.api.repo_info(
                    repo_id=self.repo_id, repo_type='model', token=self.token
                )
            except Exception as create_exc:
                raise CheckpointStorageError(
                    'Cannot access/create private HF model repository; check HF_TOKEN permissions.'
                ) from create_exc
        if info.private is not True:
            raise CheckpointStorageError(
                f'Refusing upload/download: {self.repo_id} is not verified PRIVATE'
            )
        self._private_verified = True

    def _remote_info(self, remote_path: str):
        objects = self.api.get_paths_info(
            repo_id=self.repo_id, repo_type='model',
            paths=[remote_path], token=self.token,
        )
        for info in objects:
            if info.path == remote_path:
                return info
        raise CheckpointStorageError(f'Remote object not found after upload: {remote_path}')

    def upload_checkpoint(self, local_path: str | Path, remote_path: str,
                          *, metadata: dict | None = None) -> dict:
        local = Path(local_path)
        remote_path = validate_remote_path(remote_path)
        if not local.is_file() or local.stat().st_size == 0:
            raise FileNotFoundError(f'Checkpoint is missing or empty: {local}')
        self.ensure_private_repo()
        original = local.stat()
        checksum = sha256_file(local)
        if (local.stat().st_size != original.st_size or
                local.stat().st_mtime_ns != original.st_mtime_ns):
            raise CheckpointStorageError('Checkpoint changed during hashing; retry after save finishes')

        receipt = {
            'version': 1,
            'repo_id': self.repo_id,
            'remote_path': remote_path,
            'size_bytes': original.st_size,
            'sha256': checksum,
            'details': metadata or {},
        }
        # The checkpoint and its integrity receipt are committed ATOMICALLY;
        # an interrupted upload cannot leave an overwritten last.pt paired with
        # an old receipt from a previous commit.
        from huggingface_hub import CommitOperationAdd
        self.api.create_commit(
            repo_id=self.repo_id, repo_type='model', token=self.token,
            operations=[
                CommitOperationAdd(path_in_repo=remote_path,
                                   path_or_fileobj=str(local)),
                CommitOperationAdd(path_in_repo=remote_path + '.sha256.json',
                                   path_or_fileobj=json.dumps(
                                       receipt, sort_keys=True, indent=2
                                   ).encode()),
            ],
            commit_message=f'Backup + SHA256: {remote_path}',
        )
        remote = self._remote_info(remote_path)
        if remote.size != original.st_size:
            raise CheckpointStorageError(
                f'Remote size mismatch: local {original.st_size}, remote {remote.size}'
            )
        lfs = getattr(remote, 'lfs', None)
        remote_sha = getattr(lfs, 'sha256', None) if lfs else None
        if remote_sha and remote_sha.lower() != checksum:
            raise CheckpointStorageError('Remote LFS SHA-256 differs from local checkpoint')
        if (local.stat().st_size != original.st_size or
                local.stat().st_mtime_ns != original.st_mtime_ns):
            raise CheckpointStorageError('Local checkpoint changed during upload')
        self._remote_info(remote_path + '.sha256.json')
        return receipt

    def download_checkpoint(self, remote_path: str, destination: str | Path,
                            *, overwrite: bool = False) -> Path:
        """Restore only an integrity-manifested checkpoint; never deserialize it."""
        remote_path = validate_remote_path(remote_path)
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not overwrite:
            raise FileExistsError(f'Refusing to overwrite existing checkpoint: {dest}')
        self.ensure_private_repo()
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise CheckpointStorageError('Install huggingface_hub for restore') from exc
        with tempfile.TemporaryDirectory(prefix='.hf_checkpoint_restore_', dir=dest.parent) as temp:
            kwargs = dict(repo_id=self.repo_id, repo_type='model', local_dir=temp,
                          token=self.token)
            receipt_file = Path(hf_hub_download(
                filename=remote_path + '.sha256.json', **kwargs
            ))
            receipt = json.loads(receipt_file.read_text(encoding='utf-8'))
            if (receipt.get('repo_id') != self.repo_id or
                    receipt.get('remote_path') != remote_path or
                    not isinstance(receipt.get('size_bytes'), int) or
                    not isinstance(receipt.get('sha256'), str)):
                raise CheckpointStorageError('Invalid or mismatched remote integrity receipt')
            downloaded = Path(hf_hub_download(filename=remote_path, **kwargs))
            if downloaded.stat().st_size != receipt['size_bytes']:
                raise CheckpointStorageError('Downloaded checkpoint size does not match receipt')
            if sha256_file(downloaded) != receipt['sha256']:
                raise CheckpointStorageError('Downloaded checkpoint SHA-256 mismatch')
            if dest.exists() and not overwrite:
                raise FileExistsError(f'Checkpoint appeared during restore: {dest}')
            # For older HF Hub versions that may use local symlinks, materialize
            # the verified checkpoint before the TemporaryDirectory is cleaned.
            if downloaded.is_symlink():
                materialized = Path(temp) / '_materialized_checkpoint.pt'
                shutil.copyfile(downloaded, materialized)
                os.replace(materialized, dest)
            else:
                os.replace(downloaded, dest)
        return dest
