import sys
from functools import lru_cache

import joblib
import numpy as np


def _install_numpy_module_aliases() -> None:
    # Compatibility for pickles produced with NumPy layouts using numpy._core.* imports.
    import numpy.core as np_core
    import numpy.core.numeric as np_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def _load_model_with_numpy_compat(path: str):
    # Compatibility for pickles saved with environments that serialize BitGenerator as class objects.
    import numpy.random._pickle as np_pickle

    _install_numpy_module_aliases()
    original_ctor = np_pickle.__bit_generator_ctor

    def compat_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return original_ctor(bit_generator_name)

    np_pickle.__bit_generator_ctor = compat_ctor
    try:
        return joblib.load(path)
    finally:
        np_pickle.__bit_generator_ctor = original_ctor


@lru_cache(maxsize=2)
def load_model(path: str):
    try:
        return _load_model_with_numpy_compat(path)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("numpy._core"):
            return _load_model_with_numpy_compat(path)
        raise
    except ValueError as exc:
        if "is not a known BitGenerator module" not in str(exc):
            raise
        return _load_model_with_numpy_compat(path)


def predict_risk_probability(model, input_df) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None

    proba = model.predict_proba(input_df)[0]
    if len(proba) >= 2:
        return float(proba[1])
    return float(np.max(proba))
