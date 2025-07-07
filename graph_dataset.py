from deepchem.data import Dataset

class GraphListDataset(Dataset):
    def __init__(self, X, y=None, ids=None):
        self._X = X
        self._y = y if y is not None else [0.0] * len(X)
        self._ids = ids if ids is not None else [str(i) for i in range(len(X))]

    def __len__(self): return len(self._X)
    def __getitem__(self, idx): return self._X[idx], self._y[idx], self._ids[idx]
    def ids(self): return self._ids
    def X(self): return self._X
    def y(self): return self._y
