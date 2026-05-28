import sys
from types import ModuleType

import joblib
import numpy
import numpy.random._pickle as nrp


MODEL_PATH = "models/model_diabetes_terbaik.pkl"


def install_numpy_core_aliases() -> None:
    if hasattr(numpy, "_core"):
        return

    core_module = ModuleType("numpy._core")
    sys.modules["numpy._core"] = core_module
    for attr in ["numeric", "multiarray", "umath", "records", "memmap", "defchararray"]:
        if hasattr(numpy.core, attr):
            attr_module = getattr(numpy.core, attr)
            setattr(core_module, attr, attr_module)
            sys.modules[f"numpy._core.{attr}"] = attr_module


def patch_bit_generator_ctor():
    original_ctor = nrp.__bit_generator_ctor

    def patched_ctor(bit_generator):
        if isinstance(bit_generator, type):
            try:
                return bit_generator()
            except Exception:
                pass
        return original_ctor(bit_generator)

    nrp.__bit_generator_ctor = patched_ctor
    return original_ctor


def main() -> None:
    install_numpy_core_aliases()
    original_ctor = patch_bit_generator_ctor()

    try:
        model = joblib.load(MODEL_PATH)
        print(f"OK {type(model)}")
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        nrp.__bit_generator_ctor = original_ctor


if __name__ == "__main__":
    main()
