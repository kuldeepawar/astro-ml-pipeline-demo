# src/data_pipeline.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "objects.csv"
PROC = ROOT / "data" / "processed_objects.csv"

def preprocess(infile=None, outfile=None):
    infile = Path(infile or RAW)
    outfile = Path(outfile or PROC)
    df = pd.read_csv(infile)
    # Basic cleaning
    df = df.dropna()
    # Feature engineering
    df['magnitude_norm'] = (df['magnitude'] - df['magnitude'].mean()) / df['magnitude'].std()
    # label mapping
    df['label_num'] = df['label'].map({'spiral': 0, 'elliptical': 1})
    # Save processed
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False)
    print(f'Wrote processed data to {outfile} (rows: {len(df)})')
    return outfile

if __name__ == '__main__':
    preprocess()
