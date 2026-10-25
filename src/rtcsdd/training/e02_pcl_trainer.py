"""E02: Independent, fail-closed CE + phoneme-aligned PCL trainer.

Uses original XLS-R/AASIST weights, paired batches, all labeled recordings,
cache-bound hashes, explicit per-window gradient normalization, atomic resume,
clean full Dev and the existing HF backup manager.

This file is CREATED by the Kaggle integration cell, not run until the launch cell.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rtcsdd.data.protocol import read_protocol
from rtcsdd.data.dataset import RTCFakePackedDataset
from rtcsdd.data.paired_batches import PairPreservingBatchSampler
from rtcsdd.data.phoneme_cache import pair_key_sha256
from rtcsdd.models.xlsr_aasist import load_xlsr_aasist
from rtcsdd.training.phoneme_contrastive import phoneme_contrastive_loss
from rtcsdd.training.config import fingerprint
from rtcsdd.training.checkpointing import (
    save_resume, read_resume, save_best_model, write_json_atomic,
)
from rtcsdd.training.trainer import rng_snapshot, restore_rng
from rtcsdd.eval.dev_evaluator import evaluate_full_dev
from rtcsdd.eval.competition import dev_online_macro_f1
from rtcsdd.storage.backup import CheckpointBackupManager

EXPECTED_COUNTS = {'train': (75785, 37125, 1535), 'dev': (15204, 7465, 274)}
REQUIRED_CACHE_FIELDS = {'pair_hash', 'pair_index', 'offline_frames', 'online_frames', 'matches'}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_id(value: str, side: str, split: str) -> str:
    """Trust only an exact trailing side/language/filename path, not a basename."""
    raw = str(value).strip().replace('\\', '/')
    if not raw or raw.lower() in {'nan', 'none', 'null'}:
        raise ValueError(f'Missing {side} identity')
    parts = PurePosixPath(raw).parts
    found = []
    for i in range(len(parts) - 2):
        if parts[i] == side and parts[i+1] in {'en', 'zh'}:
            if i+3 == len(parts) and parts[i+2].endswith('.wav'):
                found.append('/'.join(parts[i:i+3]))
    if len(found) != 1:
        raise ValueError(f'Invalid or ambiguous {split}/{side} path, suffix={raw[-90:]}')
    return found[0]


def read_pair_groups(dataset_root: Path, split: str, examples):
    csv_path = dataset_root / f'{split}_offline_online_pairs.csv'
    uid_map = {ex.utterance_id: ex for ex in examples}
    if len(uid_map) != len(examples):
        raise ValueError('Duplicate train/dev protocol IDs')
    groups = []
    complete = {}
    singleton_count = 0
    with csv_path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['offline_id', 'online_id']:
            raise ValueError(f'Unexpected {split} paired CSV schema: {reader.fieldnames}')
        for rowno, row in enumerate(reader, 2):
            off_id = canonical_id(row['offline_id'], 'offline', split)
            off = uid_map.get(off_id)
            if off is None or off.view != 'offline':
                raise ValueError(f'{split} CSV row {rowno}: unknown offline recording')
            value = str(row['online_id'] or '').strip()
            if not value:
                groups.append(SimpleNamespace(offline=off, online=None))
                singleton_count += 1
                continue
            on_id = canonical_id(value, 'online', split)
            on = uid_map.get(on_id)
            if on is None or on.view != 'online':
                raise ValueError(f'{split} CSV row {rowno}: unknown online recording')
            key = pair_key_sha256(off_id, on_id)
            if key in complete:
                raise ValueError(f'{split}: duplicate complete-pair hash')
            complete[key] = (off_id, on_id)
            groups.append(SimpleNamespace(offline=off, online=on))
    expected_n, expected_pairs, expected_singletons = EXPECTED_COUNTS[split]
    if (len(examples), len(complete), singleton_count) != (
        expected_n, expected_pairs, expected_singletons
    ):
        raise ValueError(
            f'{split} unexpected counts: samples={len(examples)}, '
            f'complete_pairs={len(complete)}, offline_only={singleton_count}'
        )
    return groups, complete, singleton_count, digest(csv_path)


def validate_cache(cache_root: Path, split: str, manifest_sha: str, full_pairs: dict):
    """SHA verify and join ALL records by identity hash; forbid guessing indices."""
    paths = sorted((cache_root / split).glob('*.json.gz'))
    n_expected = EXPECTED_COUNTS[split][1]
    if len(paths) != math.ceil(n_expected / 256):
        raise ValueError(f'{split}: unexpected cache shard count {len(paths)}')
    seen = {}
    cursor = 0
    ref_metadata = None
    for path in paths:
        sidecar = Path(str(path) + '.sha256')
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        claimed = sidecar.read_text(encoding='utf-8').strip().split()[0].lower()
        if len(claimed) != 64 or digest(path) != claimed:
            raise ValueError(f'Cache SHA256 mismatch: {path.name}')
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            payload = json.load(stream)
        metadata, records = payload['metadata'], payload['records']
        if metadata['manifest_sha256'] != manifest_sha:
            raise ValueError(f'{split}: cache/CSV manifest checksum mismatch')
        if metadata['start'] != cursor or metadata['stop'] != cursor + len(records):
            raise ValueError(f'{split}: shard indices not contiguous at {path.name}')
        stable = {k: v for k, v in metadata.items() if k not in {'start', 'stop'}}
        if ref_metadata is None:
            ref_metadata = stable
        elif ref_metadata != stable:
            raise ValueError(f'{split}: cache generation metadata inconsistency')
        for record in records:
            if set(record) != REQUIRED_CACHE_FIELDS:
                raise ValueError(f'{split}: unexpected cache record schema')
            if record['pair_index'] != cursor:
                raise ValueError(f'{split}: noncontiguous record pair_index={record["pair_index"]}')
            cursor += 1
            key = record['pair_hash']
            if key in seen or key not in full_pairs:
                raise ValueError(f'{split}: unknown or duplicate pair hash {key[:14]}')
            if record['offline_frames'] != 201 or record['online_frames'] != 201:
                raise ValueError('Unexpected feature frame count: cache expects 201')
            seen[key] = record
    if set(seen) != set(full_pairs) or cursor != n_expected:
        raise ValueError(f'{split}: pair/cache hash join incomplete or inconsistent')
    return seen, ref_metadata


def build_contract(repo: Path, data: Path, cache: Path):
    contexts = {}
    for split in ('train', 'dev'):
        examples = read_protocol(data / f'{split}_label.txt', split)
        groups, pairmap, alone, manifest_sha = read_pair_groups(data, split, examples)
        cached, meta = validate_cache(cache, split, manifest_sha, pairmap)
        contexts[split] = {
            'examples': examples, 'groups': groups, 'pairmap': pairmap,
            'cache': cached, 'offline_only': alone,
            'manifest_sha256': manifest_sha, 'metadata': meta,
        }
        sampler = PairPreservingBatchSampler(examples, groups, seed=1234, epoch=0)
        assert sampler.paired_batch_count == len(pairmap)
        assert sampler.ce_only_batch_count == math.ceil(alone / 2)
        actual = list(sampler)
        assert sum(map(len, actual)) == len(examples)
        assert len({i for batch in actual for i in batch}) == len(examples)
        print(f'[PASS] {split}: samples={len(examples):,} '
              f'paired={len(pairmap):,} offline_only={alone:,} '
              f'microbatches={len(sampler):,}; hash joins=100%', flush=True)
    assert contexts['train']['metadata']['audio_length'] == 64600
    return contexts


def e02_hash(base_config, contexts, args, module: Path) -> str:
    # Config includes all training controls; source hash prevents changed loss on resume.
    payload = {
        'experiment': 'e02_ce_pcl_fp32',
        'base_training_fingerprint': fingerprint(base_config),
        'train_config': base_config['training'],
        'data_config': base_config['data'],
        'lambda_pcl': args.lambda_pcl,
        'temperature': args.temperature,
        'max_matches': args.max_matches,
        'loss_sha256': digest(module),
        'trainer_sha256': digest(Path(__file__)),
        'pair_manifests': {
            split: contexts[split]['manifest_sha256'] for split in ('train', 'dev')
        },
        'phoneme_cache_metadata': {
            split: contexts[split]['metadata'] for split in ('train', 'dev')
        },
    }
    # Epoch count and backup settings are not included, to allow a genuine continuation.
    payload['train_config'] = {
        k: v for k, v in payload['train_config'].items()
        if k not in {'epochs', 'max_minutes'}
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode()).hexdigest()


def configure_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loader(dataset, batch_sampler=None, *, batch_size=None, workers=2, seed=1234):
    g = torch.Generator().manual_seed(seed)
    kw = {'num_workers': workers, 'pin_memory': True, 'generator': g}
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kw)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kw)


def run(args):
    repo = args.repo.resolve()
    data = args.data.resolve()
    cache = args.cache.resolve()
    config_path = args.config.resolve()
    pretrain = args.pretrained.resolve()
    out = args.out.resolve()
    module = repo / 'src/rtcsdd/training/phoneme_contrastive.py'
    if not (repo / 'third_party/official_rtc_sdd/xlsr_aasist/model/model.py').is_file():
        raise FileNotFoundError('Pinned AASIST model missing')
    if not module.is_file():
        raise FileNotFoundError('PCL module missing')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['experiment'] = 'e02_ce_pcl_fp32'
    config['training']['epochs'] = args.epochs
    config['training']['max_minutes'] = args.max_minutes
    config['training']['precision'] = 'fp32'
    config['checkpoint_backup'] = {
        'enabled': True,
        'repo_id': 'AhmedLabib1/rtc-sdd-research-checkpoints',
        'remote_prefix': 'experiments/e02_ce_pcl_fp32',
        'upload_every_optimizer_steps': 1000,
        'upload_on_pause': True,
        'upload_after_epoch': True,
        'upload_best_after_epoch': True,
        'max_attempts': 2,
        'retry_delay_seconds': 2,
        'create_private': False,
        'delete_local_after_upload': False,
    }
    if not (0 <= args.lambda_pcl <= 1):
        raise ValueError('lambda_pcl must be in [0,1]')
    if not (0 < args.temperature <= 1):
        raise ValueError('temperature must be in (0,1]')
    if args.max_matches < 2:
        raise ValueError('max_matches must be >=2')
    train_cfg = config['training']
    data_cfg = config['data']
    if train_cfg['batch_size'] != 2 or train_cfg['gradient_accumulation'] != 8:
        raise ValueError('Expected baseline microbatch 2 x accumulation 8')
    if train_cfg['class_weights'] != [0.1, 0.9] or float(train_cfg['learning_rate']) != 1e-6:
        raise ValueError('Unexpected E01-comparable loss weights or LR')
    if data_cfg['audio_length'] != 64600:
        raise ValueError('Unexpected input audio length')
    contexts = build_contract(repo, data, cache)
    for split in ('train','dev'):
        meta = contexts[split]['metadata']
        if data_cfg['crop'] != meta['crop'] or data_cfg['pad'] != meta['pad']:
            raise ValueError('E02 cache crop/pad differs from audio training config')
    config_hash = e02_hash(config, contexts, args, module)
    print('[PASS] E02 exact cache/manifest contract; config hash', config_hash, flush=True)
    if args.preflight_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for E02 fine-tuning')
    if not pretrain.is_file():
        raise FileNotFoundError(pretrain)
    if out.name != 'e02_ce_pcl_fp32':
        raise ValueError('E02 output directory must be named e02_ce_pcl_fp32')
    if args.resume:
        if not (out / 'last.pt').is_file() or not (out / 'config.json').is_file():
            raise FileNotFoundError('Cannot resume without original config.json and last.pt')
        old = json.loads((out/'config.json').read_text(encoding='utf-8'))
        if old['config_fingerprint'] != config_hash:
            raise RuntimeError('E02 fingerprint mismatch. Refusing unsafe resume.')
    else:
        if (out / 'last.pt').exists() or (out / 'config.json').exists():
            raise FileExistsError('Fresh E02 output exists; use --resume, not overwrite')
    out.mkdir(parents=True, exist_ok=True)
    configure_seed(train_cfg['seed'])
    print('[MODEL] Loading original pretrained XLS-R+AASIST', flush=True)
    model = load_xlsr_aasist(repo / 'third_party/official_rtc_sdd', pretrain, 'cuda:0')
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    ssl = [p for n,p in model.named_parameters() if n.startswith('ssl_model.')]
    head = [p for n,p in model.named_parameters() if not n.startswith('ssl_model.')]
    assert ssl and head
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'],
                                 weight_decay=train_cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    class_weights = torch.tensor([0.1,0.9], device='cuda:0', dtype=torch.float32)
    backup = CheckpointBackupManager(config, out)
    if not args.resume:
        write_json_atomic({
            'config': config, 'config_fingerprint': config_hash,
            'dataset_root': str(data), 'pretrained': str(pretrain),
            'pair_cache_root': str(cache),
            'objective': 'weighted_ce_plus_lambda_symmetric_phoneme_infonce',
            'lambda_pcl': args.lambda_pcl, 'temperature': args.temperature,
            'max_matches': args.max_matches,
        }, out/'config.json')
    epoch0 = batch_cursor = global_step = 0
    best_dev_loss = math.inf
    best_online_f1 = -1.0
    no_improve_epochs = 0
    saved_rng = None
    if args.resume:
        ckpt = read_resume(out/'last.pt', expected_fingerprint=config_hash)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        epoch0, batch_cursor, global_step = ckpt['epoch'], ckpt['next_batch'], ckpt['global_step']
        best_dev_loss, best_online_f1 = ckpt['best_dev_loss'], ckpt['best_online_macro_f1']
        no_improve_epochs, saved_rng = ckpt['no_improve_epochs'], ckpt['rng']
        del ckpt
        print(f'[RESUME] epoch={epoch0+1} next_batch={batch_cursor} step={global_step}', flush=True)
    dev_ctx = contexts['dev']
    dev_data = RTCFakePackedDataset(dev_ctx['examples'], data,
                                     length=64600, crop=data_cfg['crop'], pad=data_cfg['pad'],
                                     rawboost={'enabled': False}, seed=train_cfg['seed'])
    dev_loader = get_loader(dev_data, batch_size=train_cfg['eval_batch_size'],
                            workers=train_cfg['workers'], seed=train_cfg['seed']+1000000)
    train_ctx = contexts['train']
    train_data = RTCFakePackedDataset(train_ctx['examples'], data,
                                      length=64600, crop=data_cfg['crop'], pad=data_cfg['pad'],
                                      rawboost=data_cfg['rawboost'],
                                      upstream_root=repo / 'third_party/official_rtc_sdd',
                                      seed=train_cfg['seed'])
    max_save_interval = train_cfg.get('save_every_optimizer_steps', 1000)
    log_interval = train_cfg.get('log_every_optimizer_steps', 100)
    t0 = time.monotonic()
    print('[TRAIN] fp32 pair-preserving, CE + PCL; full Dev; checkpoint+HF backup enabled', flush=True)
    print('[TRAIN] original pretrained, not E01 fine-tuned weights', flush=True)

    def snapshot(ep, next_batch, reason):
        save_resume(out/'last.pt', model=model, optimizer=optimizer, scaler=scaler,
                    config_hash=config_hash, epoch=ep, next_batch=next_batch,
                    global_step=global_step, best_dev_loss=best_dev_loss,
                    best_online_macro_f1=best_online_f1, no_improve_epochs=no_improve_epochs,
                    rng_state=rng_snapshot(), accumulated_samples=0)
        print(f'[SAVED] last.pt epoch={ep+1} next_batch={next_batch} step={global_step}', flush=True)
        backup.saved_last(epoch=ep+1, global_step=global_step, reason=reason)

    for ep in range(epoch0, args.epochs):
        sampler = PairPreservingBatchSampler(train_ctx['examples'], train_ctx['groups'],
                                              seed=train_cfg['seed'], epoch=ep)
        microbatches = list(sampler)
        if ep != epoch0:
            batch_cursor = 0
        if not 0 <= batch_cursor <= len(microbatches):
            raise ValueError('Invalid resume microbatch cursor')
        train_data.set_epoch(ep)
        model.train()
        if batch_cursor < len(microbatches):
            # Pre-compute window denominators: correct CE and PCL normalization.
            sliced = microbatches[batch_cursor:]
            window_size = train_cfg['gradient_accumulation']
            train_loader = get_loader(train_data,
                batch_sampler=sliced, workers=train_cfg['workers'],
                seed=train_cfg['seed']+ep)
            iterator = iter(train_loader)
            if saved_rng is not None:
                restore_rng(saved_rng)
                saved_rng = None
            cursor = batch_cursor
            optimizer.zero_grad(set_to_none=True)
            all_examples = train_ctx['examples']
            cache_by_hash = train_ctx['cache']
            for win_start in range(0, len(sliced), window_size):
                window = sliced[win_start:win_start + window_size]
                denom = 0.0
                pcl_slots = 0
                for batch_indices in window:
                    denom += sum((0.1 if all_examples[i].label == 0 else 0.9) for i in batch_indices)
                    if len(batch_indices) == 2:
                        aa, bb = [all_examples[i].utterance_id for i in batch_indices]
                        if sampler.is_pair_batch((aa,bb)):
                            matched = cache_by_hash[pair_key_sha256(aa,bb)]['matches']
                            if len({m[0] for m in matched}) >= 2:
                                pcl_slots += 1
                if denom <= 0:
                    raise ValueError('Zero window loss denominator')
                ce_logged = pcl_logged = 0.0
                for batch_indices in window:
                    audio, labels, ids = next(iterator)
                    if tuple(ids) != tuple(all_examples[i].utterance_id for i in batch_indices):
                        raise RuntimeError('DataLoader batch order differs from pair sampler')
                    audio, labels = audio.to('cuda:0'), labels.to('cuda:0')
                    feats = []
                    def capture(module, inputs, output):
                        if not isinstance(output,dict) or 'x' not in output:
                            raise RuntimeError('Unexpected XLS-R feature interface')
                        feats.append(output['x'])
                    hook = model.ssl_model.model.register_forward_hook(capture)
                    try:
                        logits = model(audio)
                    finally:
                        hook.remove()
                    if len(feats) != 1 or tuple(feats[0].shape[1:]) != (201,1024):
                        raise RuntimeError(f'Unexpected XLS-R features: {[x.shape for x in feats]}')
                    ce_num = F.cross_entropy(logits.float(), labels,
                                             weight=class_weights, reduction='sum')
                    if not torch.isfinite(ce_num):
                        raise FloatingPointError('Nonfinite weighted CE')
                    loss = ce_num / denom
                    ce_logged += float(ce_num.detach())
                    if len(ids) == 2 and sampler.is_pair_batch(ids):
                        cache_rec = cache_by_hash[pair_key_sha256(ids[0],ids[1])]
                        if len({m[0] for m in cache_rec['matches']}) >= 2:
                            pcl = phoneme_contrastive_loss(
                                feats[0][0], feats[0][1], cache_rec['matches'],
                                offline_frames=cache_rec['offline_frames'],
                                online_frames=cache_rec['online_frames'],
                                temperature=args.temperature, max_matches=args.max_matches,
                            )
                            if not torch.isfinite(pcl):
                                raise FloatingPointError('Nonfinite PCL')
                            assert pcl_slots > 0
                            loss = loss + (args.lambda_pcl / pcl_slots) * pcl
                            pcl_logged += float(pcl.detach())
                    loss.backward()
                    cursor += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                    train_cfg['gradient_clip_norm'], error_if_nonfinite=True)
                if not math.isfinite(float(grad_norm)):
                    raise FloatingPointError('Nonfinite gradient norm')
                if global_step == 0:
                    ssl_grad = sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in ssl)
                    head_grad = sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head)
                    assert ssl_grad and head_grad, 'Missing model gradients'
                    print(f'[GRAD] XLS-R={ssl_grad} AASIST={head_grad}', flush=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step == 1 or global_step % log_interval == 0:
                    print(f'[TRAIN] epoch={ep+1}/{args.epochs} '
                          f'batch={cursor}/{len(microbatches)} step={global_step} '
                          f'CE={ce_logged/denom:.5f} '
                          f'PCL={pcl_logged/pcl_slots if pcl_slots else 0.0:.5f} '
                          f'grad_norm={float(grad_norm):.3f} '
                          f'elapsed_min={(time.monotonic()-t0)/60:.1f}', flush=True)
                time_exceeded = args.max_minutes > 0 and (time.monotonic()-t0)/60 >= args.max_minutes
                if global_step % max_save_interval == 0 or time_exceeded:
                    snapshot(ep, cursor, 'pause' if time_exceeded else 'periodic')
                if time_exceeded:
                    print('[PAUSED] Use --resume in a restored session.', flush=True)
                    return
            snapshot(ep, len(microbatches), 'pre_eval')
        else:
            if saved_rng is not None:
                restore_rng(saved_rng)
                saved_rng = None
            print(f'[RESUME] epoch={ep+1}: training already completed, evaluating Dev', flush=True)
        print('[DEV] Evaluating clean full Dev, CE and threshold=0.5', flush=True)
        metrics, loss = evaluate_full_dev(model, dev_loader, dev_ctx['examples'],
                                          device='cuda:0', amp=False,
                                          class_weights=train_cfg['class_weights'])
        online_f1 = dev_online_macro_f1(metrics)
        epoch_metrics = {
            'epoch': ep + 1, 'step': global_step, 'dev_weighted_loss': loss,
            'dev_online_macro_f1': online_f1,
            'official_hidden_weighted_f1': None,
            'metric_kind': 'clean_online_dev_only_not_competition_score',
            'subgroups': metrics,
            'training_objective': 'weighted_CE_plus_lambda_PCL',
            'lambda_pcl': args.lambda_pcl,
        }
        write_json_atomic(epoch_metrics, out/f'dev_epoch_{ep+1:03d}.json')
        write_json_atomic(epoch_metrics, out/'dev_metrics.json')
        print(f'[DEV] epoch={ep+1} loss={loss:.5f} online_macro_f1={online_f1:.5f} '
              f'all_macro_f1={metrics["all_dev"]["macro_f1"]:.5f}', flush=True)
        if loss < best_dev_loss:
            best_dev_loss = loss
            no_improve_epochs = 0
            save_best_model(out/'best_dev_loss.pt', model=model, epoch=ep+1,
                global_step=global_step, dev_loss=loss, online_macro_f1=online_f1,
                config_hash=config_hash, selection='min_dev_weighted_loss')
            backup.saved_best('best_dev_loss.pt', epoch=ep+1, global_step=global_step)
        else:
            no_improve_epochs += 1
        if online_f1 > best_online_f1:
            best_online_f1 = online_f1
            save_best_model(out/'best_online_f1.pt', model=model, epoch=ep+1,
                global_step=global_step, dev_loss=loss, online_macro_f1=online_f1,
                config_hash=config_hash, selection='max_clean_online_dev_macro_f1')
            backup.saved_best('best_online_f1.pt', epoch=ep+1, global_step=global_step)
        snapshot(ep+1, 0, 'epoch_end')
        if no_improve_epochs >= train_cfg['early_stopping_patience']:
            print('[EARLY STOP] No Dev loss improvement', flush=True)
            break
    print('[DONE] E02 CE + PCL epochs complete', flush=True)
    print('[OUTPUT]', out, flush=True)


def parser():
    p = argparse.ArgumentParser(description='E02 CE + PCL independent trainer')
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--pretrained', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max-minutes', type=float, default=540)
    p.add_argument('--lambda-pcl', type=float, default=0.1)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--max-matches', type=int, default=96)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
