"""Temporary Git authentication without storing tokens in URLs."""

from __future__ import annotations

import os
import shlex
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


def _secret_value(name: str) -> str | None:
    value = os.environ.get(name)

    if value:
        return value

    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None

    try:
        return UserSecretsClient().get_secret(name) or None
    except Exception:
        return None


@contextmanager
def git_environment(
    *,
    secret_name: str | None = None,
    username: str = "x-access-token",
):
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    if not secret_name:
        yield env
        return

    token = _secret_value(secret_name)

    if not token:
        raise RuntimeError(
            "Requested Git credential is unavailable"
        )

    with tempfile.TemporaryDirectory(
        prefix="research-git-"
    ) as tmp:

        folder = Path(tmp)
        helper_py = folder / "askpass.py"

        helper_py.write_text(
            'import os,sys\n'
            'p=" ".join(sys.argv[1:]).lower()\n'
            'sys.stdout.write('
            'os.environ["RESEARCH_GIT_USERNAME"] '
            'if "username" in p '
            'else os.environ["RESEARCH_GIT_TOKEN"])\n',
            encoding="utf-8",
        )

        if os.name == "nt":
            helper = folder / "askpass.cmd"

            helper.write_text(
                f'@echo off\n'
                f'"{sys.executable}" "{helper_py}" %*\n',
                encoding="utf-8",
            )
        else:
            helper = folder / "askpass.sh"

            helper.write_text(
                '#!/bin/sh\n'
                f'exec {shlex.quote(sys.executable)} '
                f'{shlex.quote(str(helper_py))} "$@"\n',
                encoding="utf-8",
            )

            helper.chmod(0o700)

        env["GIT_ASKPASS"] = str(helper)
        env["RESEARCH_GIT_TOKEN"] = token
        env["RESEARCH_GIT_USERNAME"] = username

        yield env
