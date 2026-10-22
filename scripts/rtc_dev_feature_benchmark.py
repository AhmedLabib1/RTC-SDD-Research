"""RTC-SDD Dev-only frozen SSL representation benchmark (no classifier training)."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

MODELS = [
    'facebook/wav2vec2-base',
    'facebook/hubert-base-ls960',
    'microsoft/wavlm-base-plus',
    'facebook/data2vec-audio-base',
    'facebook/wav2vec2-xls-r-300m',
]


def extract(args, model_name, examples, manifest):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from huggingface_hub import model_info
    from tqdm.auto import tqdm
    from rtcsdd.data.dataset import RTCFakePackedDataset

    sha = model_info(model_name).sha
    identity = {
        'protocol': 'dev_frozen_features_v1', 'model': model_name,
        'checkpoint_sha': sha, 'dev_manifest': manifest,
        'audio_length': 64600, 'crop': 'left', 'pad': 'repeat',
        'pooling': 'mean_std', 'dtype': 'float32',
    }
    fp = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = args.output_dir / model_name.replace('/', '__')
    directory.mkdir(parents=True, exist_ok=True)
    chunks = [(offset, examples[offset:offset + args.chunk_size])
              for offset in range(0, len(examples), args.chunk_size)]
    device = torch.device('cuda:0')
    model = None
    timer = time.perf_counter()
    bar = tqdm(total=len(examples), desc=model_name.split('/')[-1], unit='audio')
    arrays = []
    for offset, subset in chunks:
        path = directory / f'dev_{offset:06d}_{len(subset):04d}.npz'
        expected_ids = [e.utterance_id for e in subset]
        expected_y = [e.label for e in subset]
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                features = saved['features']
                ids = saved['ids'].tolist()
                labels = saved['labels'].tolist()
                saved_fp = str(saved['fingerprint'].item())
            if (ids != expected_ids or labels != expected_y or saved_fp != fp
                    or features.ndim != 2 or len(features) != len(subset)
                    or not np.isfinite(features).all()):
                raise RuntimeError(f'Cache mismatch: {path}; do not delete automatically')
        else:
            if model is None:
                print(f'Loading {model_name}@{sha[:12]}', flush=True)
                model = AutoModel.from_pretrained(
                    model_name, revision=sha, trust_remote_code=False
                ).to(device).eval()
                for param in model.parameters():
                    param.requires_grad_(False)
            data = RTCFakePackedDataset(subset, args.data_root, rawboost={'enabled': False})
            loader = DataLoader(data, batch_size=args.batch_size,
                                num_workers=args.workers, shuffle=False,
                                pin_memory=True)
            pieces, seen_ids, seen_y = [], [], []
            with torch.inference_mode():
                for waves, y, names in loader:
                    hidden = model(input_values=waves.to(device, non_blocking=True)).last_hidden_state
                    pooled = torch.cat((hidden.mean(1), hidden.std(1, unbiased=False)), dim=1)
                    pieces.append(pooled.float().cpu().numpy())
                    seen_y.extend(y.tolist())
                    seen_ids.extend(names)
            features = np.concatenate(pieces, axis=0)
            if seen_ids != expected_ids or seen_y != expected_y or not np.isfinite(features).all():
                raise RuntimeError('Extracted chunk validation failed')
            temp = path.with_name(path.name + '.partial')
            with temp.open('wb') as f:
                np.savez(f, features=features, ids=np.asarray(seen_ids, dtype=str),
                         labels=np.asarray(seen_y, dtype=np.int64), fingerprint=np.asarray(fp))
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, path)
        arrays.append(features)
        bar.update(len(subset))
    bar.close()
    if model is not None:
        del model
        torch.cuda.empty_cache()
    return np.concatenate(arrays, axis=0), sha, fp, time.perf_counter() - timer, directory


def metrics_for(x, y, seed):
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    n_per_class = min(1000, *(len(g) for g in groups))
    if n_per_class < 2:
        raise ValueError('Insufficient examples from either class')
    indices = np.concatenate([rng.choice(g, size=n_per_class, replace=False) for g in groups])
    rng.shuffle(indices)
    z = normalize(x[indices].astype(np.float32))
    labels = y[indices]
    sil = float(silhouette_score(z, labels, metric='cosine'))
    centers = [normalize(z[labels == label].mean(axis=0, keepdims=True))[0] for label in (0, 1)]
    center_cosine_distance = float(1 - np.dot(centers[0], centers[1]))
    compactness = {
        'spoof_mean_cosine_to_centroid': float(np.mean(z[labels == 0] @ centers[0])),
        'bonafide_mean_cosine_to_centroid': float(np.mean(z[labels == 1] @ centers[1])),
    }
    # Leave-self-out neighborhood agreement: this is not classifier training.
    nbrs = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute', n_jobs=2)
    nbrs.fit(z)
    nearest = nbrs.kneighbors(z, return_distance=False)
    agrees = []
    for i, neighbors in enumerate(nearest):
        other = [int(j) for j in neighbors if int(j) != i][:5]
        if len(other) != 5:
            raise RuntimeError('Could not find five non-self neighbors')
        agrees.append(float(np.mean(labels[other] == labels[i])))
    agreement = float(np.mean(agrees))
    return {
        'silhouette_cosine': sil,
        'centroid_cosine_distance': center_cosine_distance,
        'within_class_compactness': compactness,
        'neighbor_label_agreement_at_5': agreement,
        'balanced_sample_size': int(len(labels)),
        'per_class': int(n_per_class),
        'random_seed': seed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=Path('/kaggle/working/dev_feature_benchmark'))
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--chunk-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    args = p.parse_args()
    if args.batch_size < 1 or args.chunk_size < 1 or not args.data_root.is_dir():
        p.error('Invalid configuration')
    import torch
    from rtcsdd.data.protocol import read_protocol
    assert torch.cuda.is_available(), 'Select Kaggle T4 GPU'
    torch.set_num_threads(2)
    examples = read_protocol(args.data_root / 'dev_label.txt', 'dev')
    if len(examples) != 15204 or len({e.utterance_id for e in examples}) != len(examples):
        raise RuntimeError('Unexpected Dev manifest')
    y = np.asarray([e.label for e in examples])
    assert set(y.tolist()) == {0, 1}
    manifest = hashlib.sha256('\n'.join(e.utterance_id for e in examples).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        print('\nBENCHMARK:', model_name, flush=True)
        target = args.output_dir / model_name.replace('/', '__') / 'result.json'
        try:
            x, sha, fp, seconds, directory = extract(args, model_name, examples, manifest)
            if target.exists():
                prior = json.loads(target.read_text())
                if prior.get('cache_fingerprint') == fp:
                    print('Existing validated result:', target, flush=True)
                    continue
                raise RuntimeError('Result fingerprint mismatch')
            result = {
                'protocol': 'dev_frozen_features_v1',
                'checkpoint': model_name, 'checkpoint_commit': sha,
                'cache_fingerprint': fp, 'dev_manifest_sha256': manifest,
                'dev_count': len(examples), 'feature_dimensions': int(x.shape[1]),
                'seconds_this_run': round(seconds, 2),
                'evaluation': metrics_for(x, y, args.seed),
                'warning': 'Unsupervised geometry + label-based diagnostics; no classifier trained; not F1 or official score.',
            }
            tmp = target.with_suffix('.json.partial')
            tmp.write_text(json.dumps(result, indent=2), encoding='utf-8')
            os.replace(tmp, target)
            print('SAVED', target, flush=True)
            print(json.dumps(result['evaluation'], indent=2), flush=True)
        except Exception as exc:
            print('FAILED', model_name, repr(exc), flush=True)
            # Continue other models; incomplete cached chunks remain recoverable.
    print('\nFINISHED; inspect result.json files for completed models.')


if __name__ == '__main__':
    main()
