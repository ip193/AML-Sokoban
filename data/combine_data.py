import gzip
import marshal
import pickle
from glob import glob

from tqdm import tqdm

X_files = sorted(glob('X_*.pkl.gz'))
y_files = sorted(glob('y_*.pkl.gz'))

print(X_files)
print(y_files)

X = []
for X_file in X_files:
    with gzip.open(X_file, 'rb') as f:
        X.extend(pickle.load(f))

Y = []
for y_file in y_files:
    with gzip.open(y_file, 'rb') as f:
        Y.extend(pickle.load(f))

print(len(X), len(Y))

hashed_x_list = [marshal.dumps(x) for x in X]

hashed_x = {}
for x, y in tqdm(zip(X.copy(), Y.copy())):
    x_hash = marshal.dumps(x)
    if x_hash in hashed_x:
        if hashed_x[x_hash] != y:
            for idx, element in enumerate(hashed_x_list):
                if x_hash == element:
                    Y[idx] = min(hashed_x[x_hash], y)  # always use min distance for duplicate stages
            print("! Duplicate in Dataset!!!", "changed value", hashed_x[x_hash], y)
    else:
        hashed_x[x_hash] = y

name = 'test'
with gzip.open(f'X_{name}.pkl.gz', 'wb') as f:
    pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
with gzip.open(f'y_{name}.pkl.gz', 'wb') as f:
    pickle.dump(Y, f, pickle.HIGHEST_PROTOCOL)
