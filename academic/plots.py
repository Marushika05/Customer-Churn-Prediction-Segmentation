"""Small, exportable result figures; no dashboard dependencies."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def plot_results(table, predictions, train, models, out):
    plt.rcParams.update({'figure.dpi':120, 'font.size':10})
    fig, axes = plt.subplots(1, 2, figsize=(12,4.5))
    for ax, metric in zip(axes, ('roc_auc','f1')):
        table.pivot(index='model', columns='features', values=metric)[['baseline','agglomerative','dbscan']].plot.bar(ax=ax, rot=0)
        ax.set_ylim(0,1)
        ax.set_title('Held-out '+metric.upper())
        ax.set_xlabel('')
        ax.legend(title='Features', fontsize=8)
    fig.tight_layout(); fig.savefig(out/'classification_metrics.png'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    integrated=table[table.features != 'baseline'].copy()
    names=integrated.model.str.replace('Logistic Regression','LR').str.replace('Random Forest','RF')+' + '+integrated.features
    for ax,metric in zip(axes,('delta_roc_auc','delta_f1')):
        values=integrated[metric].to_numpy()
        ax.barh(names,values,color=['#2a8c6b' if v >= 0 else '#b54b4b' for v in values])
        ax.axvline(0,color='black',linewidth=.8)
        ax.set_title(metric.replace('delta_','Change in ').upper())
        ax.set_xlabel('Integrated minus same-classifier baseline')
        ax.margins(x=.35)
        for i,v in enumerate(values):
            ax.text(v,i,f' {v:+.6f}',va='center',ha='left' if v>=0 else 'right',fontsize=8)
    fig.tight_layout(); fig.savefig(out/'performance_differences.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,5))
    for key in models:
        fpr,tpr,_ = roc_curve(predictions.churned, predictions[key])
        ax.plot(fpr,tpr,label=key.replace('__',' / ').replace('_',' '),linewidth=1.2)
    ax.plot([0,1],[0,1],'k--',alpha=.4)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Identical held-out cohort: ROC curves')
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'roc_curves.png'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    for ax, variant in zip(axes, ('agglomerative','dbscan')):
        segmenter = models['random_forest__'+variant]['segmenter']
        import numpy as np
        for label in np.unique(segmenter.labels_):
            part = train[segmenter.labels_ == label]
            ax.scatter(np.log1p(part.recency_days),np.log1p(part.monetary),s=5,alpha=.4,
                       label='Noise' if label == -1 else f'Segment {label}',color='gray' if label == -1 else None)
        ax.set(xlabel='log(1 + recency days)',ylabel='log(1 + gross spend GBP)',title=variant.title()+' — training customers')
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'customer_segments.png'); plt.close(fig)
