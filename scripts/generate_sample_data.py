# scripts/generate_sample_data.py (smaller sample generator)
import pandas as pd
import numpy as np

n = 200
ids = [f"obj_{i:04d}" for i in range(n)]
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
df.to_csv('data/objects.csv', index=False)
print('Wrote data/objects.csv')
