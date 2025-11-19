import numpy as np
from collections import Counter
arr = np.load('data/processed/train_processed.npz', allow_pickle=True)
y = arr['y']
print('dtype:', y.dtype)
print('sample y[:20]:', y[:20])
unique = np.unique(y)
print('n_unique:', len(unique))
print('unique head:', unique[:50])
print('counts (top 30):')
print(dict(Counter(y.tolist()).most_common(30)))
