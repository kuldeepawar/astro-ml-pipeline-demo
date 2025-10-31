# data/generate_big_dataset.py
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'objects_big.csv'

def generate(n=100000, seed=42, out=OUT):
    np.random.seed(seed)
    ids = [f"obj_{i:07d}" for i in range(n)]
    mag = np.random.normal(16, 1.5, size=n)
    color = np.random.normal(0.6, 0.3, size=n)
    size = np.abs(np.random.normal(2.5, 0.8, size=n))
    ell = np.random.beta(1,5,size=n)
    labels = np.random.choice(['spiral','elliptical'], size=n, p=[0.6,0.4])

    df = pd.DataFrame({
        'id': ids,
        'magnitude': np.round(mag,2),
        'color_index': np.round(color,2),
        'size': np.round(size,2),
        'ellipticity': np.round(ell,2),
        'label': labels
    })
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {n} rows")

if __name__ == '__main__':
    generate()
