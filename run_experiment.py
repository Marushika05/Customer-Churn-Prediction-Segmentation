"""Run the complete six-way academic experiment. Original application is never imported."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from academic.ingestion import read_file, clean, normalise_columns
from academic.features import FEATURES, build_snapshots, chronological_split, InsufficientData
from academic.segmentation import RFMSegmenter, profiles
from academic.classification import preprocessor, encoder, estimator, threshold, metrics

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(path, cache_dir, use_cache=True):
    sha = digest(path)
    # Cache invalidates whenever the data/feature implementation changes.
    code_hash = hashlib.sha256(b''.join((ROOT/f).read_bytes() for f in
                              ['academic/ingestion.py', 'academic/features.py', 'academic/config.py', 'run_experiment.py'])).hexdigest()[:12]
    cache = cache_dir / f'snapshots_{sha[:16]}_{code_hash}.joblib'
    if use_cache and cache.exists():
        print('Loading verified input-specific snapshot cache', flush=True)
        return joblib.load(cache)
    sheet_info = {}
    if path.suffix.lower() == '.xlsx':
        wb = load_workbook(path, read_only=True)
        sheet_info = {s.title: s.max_row-1 for s in wb.worksheets}
        wb.close()
    print('Loading all source sheets and cleaning transactions...', flush=True)
    raw = normalise_columns(read_file(path))
    raw_rows = len(raw)
    raw_end = pd.to_datetime(raw['invoice_date'],
                             format='mixed', errors='coerce').max()
    tx, report = clean(raw)
    del raw
    gc.collect()
    # Ignore the final partially observed day; never extend observation to unobserved midnight.
    end = raw_end.normalize()
    h = pd.Timedelta(days=90)
    cuts = [end-3*h, end-2*h, end-h]
    if cuts[0] < tx.invoice_date.min().normalize()+pd.Timedelta(days=60):
        raise InsufficientData('Need at least 330 days: 60 history + three non-overlapping 90-day horizons')
    identified = tx[tx.customer_id.notna()].copy()
    identified['customer_id'] = identified.customer_id.astype('int64')
    print('Building full customer cohorts at', ', '.join(str(c.date()) for c in cuts), flush=True)
    snaps = build_snapshots(identified, cuts)
    official = (path.name == 'online_retail_II.xlsx' and raw_rows == 1067371 and len(sheet_info) == 2
                and tx.invoice_date.min().year == 2009 and tx.invoice_date.max().year == 2011)
    meta = {'dataset': 'UCI Online Retail II (complete workbook)' if official else 'PROVISIONAL / custom input; not verified complete Online Retail II',
            'provisional': not official, 'source_url': SOURCE if official else None,
            'input_path': str(path.resolve()), 'sha256': sha, 'sheets': sheet_info,
            'raw_rows': raw_rows, 'cleaning': report, 'observation_end_exclusive': str(end),
            'excluded_anonymous_rows_from_models': int(tx.customer_id.isna().sum()),
            'snapshot_code_hash': code_hash}
    del identified, tx
    gc.collect()
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump((snaps, meta), cache, compress=3)
    return snaps, meta


def run(snaps, metadata, out):
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te, split = chronological_split(snaps)
    h = pd.Timedelta(days=90)
    assert pd.Timestamp(split['train_cutoff'])+h <= pd.Timestamp(split['validation_cutoff'])
    assert pd.Timestamp(split['validation_cutoff'])+h <= pd.Timestamp(split['test_cutoff'])
    if 'observation_end_exclusive' in metadata:
        assert pd.Timestamp(split['test_label_end']) <= pd.Timestamp(metadata['observation_end_exclusive'])
    for name, part in zip(('train', 'validation', 'test'), (tr, va, te)):
        if part.churned.nunique() != 2:
            raise InsufficientData(f'{name} contains only one class; cannot perform the required comparison')
        assert not part.customer_id.duplicated().any()
        part.to_csv(out/f'{name}_observations.csv', index=False)
    pre = preprocessor().fit(tr[FEATURES])
    bases = [pre.transform(p[FEATURES]) for p in (tr, va, te)]
    predictions = te[['customer_id', 'cutoff', 'churned']].copy()
    tables, cluster_metrics, all_profiles, models = [], [], [], {}
    test_hash = hashlib.sha256(predictions.to_csv(index=False).encode()).hexdigest()
    validation_hash = hashlib.sha256(va[['customer_id','cutoff','churned']].to_csv(index=False).encode()).hexdigest()
    for variant in ('baseline', 'agglomerative', 'dbscan'):
        segmenter = enc = None
        x = bases
        labels = None
        if variant != 'baseline':
            print('Fitting training-only', variant, 'on', len(tr), 'customers', flush=True)
            segmenter = RFMSegmenter(variant).fit(tr)
            labels = [segmenter.labels_, segmenter.assign(va), segmenter.assign(te)]
            enc = encoder(labels[0])
            x = [np.c_[base, enc.transform(lab.reshape(-1,1))] for base, lab in zip(bases, labels)]
            predictions[f'{variant}_segment'] = labels[2]
            cluster_metrics.append(segmenter.metrics)
            pd.DataFrame(segmenter.search).to_csv(out/f'{variant}_parameter_search.csv', index=False)
            all_profiles.append(profiles(tr, labels[0], variant, 'training'))
        for model_name in ('Logistic Regression', 'Random Forest'):
            print('Training', model_name, '/', variant, flush=True)
            model = estimator(model_name).fit(x[0], tr.churned)
            pv = model.predict_proba(x[1])[:,1]
            pt = model.predict_proba(x[2])[:,1]
            thr = threshold(va.churned, pv)
            result = {'model': model_name, 'features': variant, 'threshold': thr,
                      **metrics(te.churned, pt, thr),
                      **{'validation_'+k:v for k,v in metrics(va.churned, pv, thr).items()},
                      'n_train': len(tr), 'n_validation': len(va), 'n_test': len(te),
                      'test_observations_sha256': test_hash,
                      'validation_observations_sha256': validation_hash}
            tables.append(result)
            key = model_name.lower().replace(' ', '_')+'__'+variant
            predictions[key] = pt
            models[key] = {'preprocessor': pre, 'segmenter': segmenter, 'encoder': enc,
                           'classifier': model, 'threshold': thr}
            if variant != 'baseline' and model_name == 'Random Forest':
                all_profiles.append(profiles(te, labels[2], variant, 'held-out test; RF risk', pt))
    table = pd.DataFrame(tables)
    for metric in ('precision', 'recall', 'f1', 'roc_auc'):
        baseline = table[table.features == 'baseline'].set_index('model')[metric]
        table['delta_'+metric] = table[metric]-table.model.map(baseline)
    assert table.test_observations_sha256.nunique() == 1
    table.to_csv(out/'classification_comparison.csv', index=False)
    pd.DataFrame(cluster_metrics).to_csv(out/'clustering_evaluation.csv', index=False)
    pd.concat(all_profiles, ignore_index=True).to_csv(out/'segment_profiles.csv', index=False)
    predictions.to_csv(out/'held_out_predictions.csv', index=False)
    joblib.dump(models, out/'models.joblib', compress=3)
    versions = {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','openpyxl','matplotlib','joblib']}
    manifest = {**metadata, 'split': split, 'seed': 42, 'versions': versions,
                'class_prevalence': {n:float(p.churned.mean()) for n,p in zip(('train','validation','test'),(tr,va,te))},
                'evaluation': 'all customers at each cutoff; same customer may recur across dates; no ID as predictor',
                'feature_columns': FEATURES, 'test_observations_sha256': test_hash,
                'fit_population': 'training cutoff only; no validation/test clustering or preprocessing fit',
                'deployment_refit': False}
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    from academic.plots import plot_results
    plot_results(table, predictions, tr, models, out)
    write_report(table, cluster_metrics, manifest, out)
    print(table[['model','features','precision','recall','f1','roc_auc','delta_roc_auc']].to_string(index=False), flush=True)
    return table


def write_report(table, clusters, manifest, out):
    columns = ['model','features','precision','recall','f1','roc_auc','delta_f1','delta_roc_auc']
    lines = ['# Executed experiment results', '', manifest['dataset'], '',
             '| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for row in table[columns].itertuples(index=False, name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
    lines += ['', '## Interpretation', '',
              'Differences are descriptive results on one held-out cohort, not statistical proof or guaranteed improvement.',
              'The threshold for each experiment maximises validation F1. ROC-AUC uses probability scores.',
              'Clusters are fitted on training RFM only. Test cluster labels are frozen inductive approximations.',
              '', '## Chronology', '', '```json', json.dumps(manifest['split'], indent=2), '```',
              '', '## Clustering metrics', '', '```json', json.dumps(clusters, indent=2), '```',
              '', 'Silhouette uses at most 1,500 training non-noise points with a fixed seed; DBI uses all training non-noise points.',
              'All eligible customer observations are retained for clustering fitting, classifiers and test evaluation.',
              '', '## Limitations', '',
              '- Churn means no positive purchase in 90 days, not proven permanent customer departure.',
              '- The 90-day horizon is a declared study assumption, not an optimised business threshold.',
              '- Gross monetary spend excludes refunds; cancellations contribute a separate historical return feature.',
              '- All historically observed buyers are eligible, including new and already inactive customers.',
              '- Customers may recur across dates; this estimates later behaviour of a retailer customer population, not customer-disjoint transfer.',
              '- Segmentation cannot add new raw information beyond RFM; gains reflect representation and model effects.',
              '- Cluster selection uses training-only internal metrics; these scores are descriptive and selection-biased.',
              '- Nearest-centroid Ward assignment does not reproduce refitted hierarchical clustering; nearest-core DBSCAN cannot create new clusters.',
              '- One temporal holdout and seasonal changes limit generalisation; no causal retention effectiveness or significance is established.',
              '- Final partial transaction day is excluded; complete recorded coverage before that boundary is assumed.',
              '- No dashboard, API or unrelated models are required by this workflow.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/raw/online_retail_II.xlsx')
    parser.add_argument('--output', type=Path, default=ROOT/'results/full')
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    snaps, metadata = prepare(args.data, ROOT/'data/processed', not args.no_cache)
    run(snaps, metadata, args.output)
    print(f'Finished in {time.perf_counter()-start:.1f}s. Results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
