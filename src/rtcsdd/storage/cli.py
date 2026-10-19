"""Generic manual HF checkpoint backup/restore CLI.

Examples:
 python -m rtcsdd.storage.cli upload --checkpoint OUT/last.pt \
   --repo-id org/private-ckpts --remote-path experiments/p01/last.pt
 python -m rtcsdd.storage.cli download --repo-id org/private-ckpts \
   --remote-path experiments/p01/last.pt --destination OUT/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rtcsdd.storage.huggingface import HuggingFaceCheckpointStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    upload = sub.add_parser('upload', help='Back up a COMPLETED checkpoint')
    upload.add_argument('--checkpoint', required=True, type=Path)
    upload.add_argument('--repo-id', required=True)
    upload.add_argument('--remote-path', required=True)
    restore = sub.add_parser('download', help='Restore verified checkpoint (trusted repo only)')
    restore.add_argument('--repo-id', required=True)
    restore.add_argument('--remote-path', required=True)
    restore.add_argument('--destination', required=True, type=Path)
    restore.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    store = HuggingFaceCheckpointStore(args.repo_id)
    if args.action == 'upload':
        receipt = store.upload_checkpoint(args.checkpoint, args.remote_path)
        print(f"[HF BACKUP OK] {args.repo_id}/{args.remote_path} "
              f"size={receipt['size_bytes']/1024**3:.2f} GiB sha256={receipt['sha256']}")
    else:
        result = store.download_checkpoint(
            args.remote_path, args.destination, overwrite=args.overwrite,
        )
        print(f'[HF RESTORE VERIFIED] {result}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
