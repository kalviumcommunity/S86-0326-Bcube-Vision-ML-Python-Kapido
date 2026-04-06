import os
import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def summarize_numeric(df: pd.DataFrame, cols):
    percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    desc = df[cols].describe(percentiles=percentiles).T
    desc['skew'] = df[cols].skew()
    desc['pct_missing'] = df[cols].isna().mean()
    return desc


def summarize_categorical(df: pd.DataFrame, cols, top_n=20):
    out = {}
    for c in cols:
        vc = df[c].value_counts(dropna=False)
        out[c] = {
            'n_unique': int(df[c].nunique(dropna=True)),
            'pct_missing': float(df[c].isna().mean()),
            'top_counts': vc.head(top_n).to_dict(),
        }
    return out


def plot_numeric(df: pd.DataFrame, col: str, outdir: Path, target=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[col].dropna(), bins=30, kde=False, ax=ax)
    ax.set_title(f'Distribution: {col}')
    fig.tight_layout()
    fig.savefig(outdir / f'{col}__hist.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title(f'Boxplot: {col}')
    fig.tight_layout()
    fig.savefig(outdir / f'{col}__box.png')
    plt.close(fig)

    if target is not None and target in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=target, y=col, data=df, ax=ax)
        ax.set_title(f'{col} by {target}')
        fig.tight_layout()
        fig.savefig(outdir / f'{col}__by__{target}__box.png')
        plt.close(fig)


def plot_categorical(df: pd.DataFrame, col: str, outdir: Path, target=None, max_categories=40):
    vc = df[col].value_counts(dropna=False)
    top = vc.head(max_categories)
    fig, ax = plt.subplots(figsize=(6, max(2, 0.2 * len(top))))
    sns.barplot(x=top.values, y=top.index.astype(str), ax=ax)
    ax.set_title(f'Counts: {col}')
    fig.tight_layout()
    fig.savefig(outdir / f'{col}__counts.png')
    plt.close(fig)

    if target is not None and target in df.columns and df[col].nunique() <= 40:
        fig, ax = plt.subplots(figsize=(6, max(2, 0.2 * df[col].nunique())))
        sns.countplot(y=col, hue=target, data=df, order=top.index, ax=ax)
        ax.set_title(f'{col} by {target}')
        fig.tight_layout()
        fig.savefig(outdir / f'{col}__by__{target}__counts.png')
        plt.close(fig)


def inspect_dataframe(df: pd.DataFrame, target: str | None = None, output_dir: str = 'reports/feature_inspection'):
    outdir = Path(output_dir)
    _ensure_dir(outdir)

    report = {}
    report['shape'] = df.shape
    report['dtypes'] = {c: str(t) for c, t in df.dtypes.items()}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    report['numeric_summary'] = summarize_numeric(df, numeric_cols).to_dict(orient='index')
    report['categorical_summary'] = summarize_categorical(df, categorical_cols)

    # Save JSON summary
    with open(outdir / 'summary.json', 'w', encoding='utf8') as f:
        json.dump(report, f, indent=2)

    # Save CSV summaries for numeric
    if numeric_cols:
        pd.DataFrame(report['numeric_summary']).T.to_csv(outdir / 'numeric_summary.csv')

    # Generate plots
    plots_dir = outdir / 'plots'
    _ensure_dir(plots_dir)

    for col in numeric_cols:
        try:
            plot_numeric(df, col, plots_dir, target=target)
        except Exception:
            pass

    for col in categorical_cols:
        try:
            plot_categorical(df, col, plots_dir, target=target)
        except Exception:
            pass

    return outdir


def _parse_args():
    p = argparse.ArgumentParser(description='Feature distribution inspection')
    p.add_argument('csv', help='Path to CSV file')
    p.add_argument('--target', help='Optional target column to compare distributions', default=None)
    p.add_argument('--out', help='Output directory', default='reports/feature_inspection')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    df = pd.read_csv(args.csv)
    outdir = inspect_dataframe(df, target=args.target, output_dir=args.out)
    print('Inspection complete. Results written to', outdir)
