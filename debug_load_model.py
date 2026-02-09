import traceback
import os
from pathlib import Path

import tensorflow as tf


import sys

# Allow passing a model filename as first arg, otherwise default to mobilenetv2
default_name = "abdominal_ct_mobilenetv2.keras"
arg = sys.argv[1] if len(sys.argv) > 1 else default_name
MODEL_PATH = Path(__file__).parent / "models" / arg


def inspect_h5(path):
    try:
        import h5py
    except Exception:
        print("h5py not available; cannot inspect HDF5 model file.")
        return
    with h5py.File(path, "r") as f:
        print("HDF5 top-level keys:", list(f.keys()))
        if "model_config" in f.attrs:
            print("model_config attribute present")


def main():
    print("Model path:", MODEL_PATH)
    if not MODEL_PATH.exists():
        print("Model file not found")
        return

    # If directory -> SavedModel
    if MODEL_PATH.is_dir():
        print("Model appears to be a SavedModel directory")
    else:
        print("Model appears to be a file; size:", MODEL_PATH.stat().st_size)

    # Try default load
    try:
        print("Attempting to load model with compile=False...")
        m = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully (default)")
        try:
            m.summary()
        except Exception:
            print("Could not print summary")
        return
    except Exception:
        print("Error during default load:")
        traceback.print_exc()

    # Try safe_mode if available
    try:
        print("Attempting to load model with safe_mode=True, compile=False...")
        m = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=True)
        print("Model loaded successfully (safe_mode=True)")
        try:
            m.summary()
        except Exception:
            print("Could not print summary")
        return
    except TypeError:
        print("safe_mode argument not supported by this TF/Keras version.")
    except Exception:
        print("Error during safe_mode load:")
        traceback.print_exc()

    # If file is HDF5, try to inspect contents
    if MODEL_PATH.suffix in [".h5", ".hdf5", ".keras"] and MODEL_PATH.is_file():
        print("Inspecting HDF5 contents:")
        inspect_h5(MODEL_PATH)


if __name__ == "__main__":
    main()
