"""
00_check_env.py - Verify the environment before running any experiment.
========================================================================
Run:  python scripts/00_check_env.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

OK, MISSING = [], []


def check(name, fn, critical=True):
    try:
        print(f"  [OK]   {name}: {fn()}")
        OK.append(name)
        return True
    except Exception as e:
        tag = "MISS" if critical else "OPTIONAL"
        print(f"  [{tag}] {name}: {type(e).__name__}: {str(e)[:120]}")
        if critical:
            MISSING.append(name)
        return False


def main():
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    print("\n-- Libraries --")
    check("Python", lambda: sys.version.split()[0])
    check("numpy", lambda: __import__("numpy").__version__)
    check("pandas", lambda: __import__("pandas").__version__)
    check("flwr", lambda: __import__("flwr").__version__)
    check("transformers", lambda: __import__("transformers").__version__)
    check("matplotlib", lambda: __import__("matplotlib").__version__)

    print("\n-- GPU / PyTorch --")

    def torch_info():
        import torch
        s = f"{torch.__version__}"
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            s += (f" | GPU: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})"
                  f" | PyTorch supports: {torch.cuda.get_arch_list()}")
        else:
            s += " | CUDA not found -> will run on CPU"
        return s
    check("torch", torch_info)

    def device_test():
        from src import model as M
        dev = M.get_device(verbose=True)
        return f"device that will actually be used = {dev}"
    check("Real GPU execution test", device_test)

    print("\n-- Data --")

    def data_info():
        if not os.path.exists(config.SOURCE_DATASET_PATH):
            raise FileNotFoundError(
                f"{os.path.basename(config.SOURCE_DATASET_PATH)} is missing from data/")
        mb = os.path.getsize(config.SOURCE_DATASET_PATH) / 1e6
        return f"{os.path.basename(config.SOURCE_DATASET_PATH)} ({mb:.1f} MB)"
    check("Source dataset", data_info)

    print("\n-- Ollama (only needed for the semantic trigger) --")

    def ollama_info():
        import requests
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        want = config.OLLAMA_MODEL_NAME
        status = "PRESENT" if any(want.split(":")[0] in m for m in models) else "NOT PULLED"
        return f"running | model '{want}': {status} | available: {models}"
    check(f"Ollama ({config.OLLAMA_MODEL_NAME})", ollama_info, critical=False)

    print("\n" + "=" * 70)
    if MISSING:
        print(f"{len(MISSING)} required component(s) missing: {MISSING}")
        print("Install with: pip install -r requirements.txt (see the README)")
    else:
        print("Environment ready. Next: python scripts/01_prepare_data.py")
    print("Note: if Ollama is unavailable the system falls back to a built-in pool of")
    print("      trigger sentences (the data stays valid, it is just not LLM-generated).")
    print("=" * 70)


if __name__ == "__main__":
    main()
