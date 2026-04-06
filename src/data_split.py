from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def chronological_split(df: pd.DataFrame, time_col: str, test_size: float = 0.2):
    df_sorted = df.sort_values(time_col)
    n = len(df_sorted)
    split_at = int(n * (1 - test_size))
    train = df_sorted.iloc[:split_at].reset_index(drop=True)
    test = df_sorted.iloc[split_at:].reset_index(drop=True)
    return train, test


def random_split(
    df: pd.DataFrame,
    target: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
):
    strat = None
    if stratify and target is not None and target in df.columns:
        strat = df[target]
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )
    # reset indices for convenience
    return train.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(train: pd.DataFrame, test: pd.DataFrame, outdir: Path, prefix: str = ''):
    outdir.mkdir(parents=True, exist_ok=True)
    train_path = outdir / f"{prefix}train.csv"
    test_path = outdir / f"{prefix}test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def summarize_target_distribution(train: pd.DataFrame, test: pd.DataFrame, target: Optional[str]):
    out = {}
    if target is None or target not in train.columns:
        return out
    out['train'] = (train[target].value_counts(dropna=False).to_dict())
    out['test'] = (test[target].value_counts(dropna=False).to_dict())
    return out


def split_command(
    csv_path: str,
    outdir: str = 'data/splits',
    target: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    time_column: Optional[str] = None,
    prefix: str = '',
):
    df = pd.read_csv(csv_path)
    outdir_path = Path(outdir)

    if time_column:
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not in dataset")
        train, test = chronological_split(df, time_column, test_size=test_size)
    else:
        train, test = random_split(
            df, target=target, test_size=test_size, random_state=random_state, stratify=stratify
        )

    train_path, test_path = save_splits(train, test, outdir_path, prefix=prefix)

    summary = {
        'train_shape': train.shape,
        'test_shape': test.shape,
        'target_distribution': summarize_target_distribution(train, test, target),
    }

    # save short report
    report_path = outdir_path / f"{prefix}split_report.txt"
    with open(report_path, 'w', encoding='utf8') as f:
        f.write(f"train_shape: {train.shape}\n")
        f.write(f"test_shape: {test.shape}\n")
        if target and target in df.columns:
            f.write("\ntrain target distribution:\n")
            f.write(str(train[target].value_counts(dropna=False).to_dict()))
            f.write("\n\ntest target distribution:\n")
            f.write(str(test[target].value_counts(dropna=False).to_dict()))

    return {'train_path': str(train_path), 'test_path': str(test_path), 'report_path': str(report_path), 'summary': summary}


def _parse_args():
    p = argparse.ArgumentParser(description='Create train/test splits safely')
    p.add_argument('csv', help='Path to input CSV file')
    p.add_argument('--out', default='data/splits', help='Output directory')
    p.add_argument('--target', default=None, help='Target column name (for stratified split)')
    p.add_argument('--test-size', default=0.2, type=float, help='Proportion to reserve for test')
    p.add_argument('--random-state', default=42, type=int, help='Random seed')
    p.add_argument('--no-stratify', dest='stratify', action='store_false', help='Disable stratified splitting')
    p.add_argument('--time-column', default=None, help='If set, perform chronological split using this column')
    p.add_argument('--prefix', default='', help='Prefix for saved files')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    result = split_command(
        args.csv,
        outdir=args.out,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=args.stratify,
        time_column=args.time_column,
        prefix=args.prefix,
    )
    print('Split complete. Files:')
    print(result['train_path'])
    print(result['test_path'])
    print('Report:', result['report_path'])
