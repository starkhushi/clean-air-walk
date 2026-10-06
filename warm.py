"""Build-time warm-up: download the TabPFN v2 regressor weights into
TABPFN_MODEL_CACHE_DIR and run one tiny fit so the first visitor does not wait."""
import time

import numpy as np

t = time.time()
from app.forecast import _fit_predict  # noqa: E402
import pandas as pd  # noqa: E402

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(64, 5)))
y = X[0].to_numpy() * 2 + rng.normal(scale=0.1, size=64)
q = _fit_predict(X.iloc[:48], y[:48], X.iloc[48:])
print(f"TabPFN warm-up ok: quantiles shape {q.shape}, {time.time() - t:.1f}s", flush=True)
