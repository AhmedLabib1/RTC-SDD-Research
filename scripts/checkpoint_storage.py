#!/usr/bin/env python3
"""Convenient alternative entry point to `python -m rtcsdd.storage.cli`."""

from rtcsdd.storage.cli import main


if __name__ == '__main__':
    raise SystemExit(main())
