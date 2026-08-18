"""
io_utils.py - SAFE FILE WRITING
================================
On Windows, if a CSV file is currently OPEN IN EXCEL the operating system locks it and
pandas raises PermissionError. That used to CRASH the whole pipeline - catastrophic when
it happens after hours of training.

`safe_to_csv()` handles that case: it tries to write, and if the file is locked it falls
back to a timestamped alternative filename and prints a clear warning WITHOUT
interrupting the experiment.
"""

import os
import time
import json


def safe_to_csv(df, path: str, quiet: bool = False, **kwargs) -> str:
    """
    Write a DataFrame to CSV. Returns the path that was ACTUALLY written (possibly the
    fallback name). Never lets a PermissionError escape.
    """
    kwargs.setdefault("index", False)
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt = f"{base}_{time.strftime('%H%M%S')}{ext}"
        try:
            df.to_csv(alt, **kwargs)
            print(f"    [WARNING] Could not write '{os.path.basename(path)}' "
                  f"(is it open in Excel?).")
            print(f"              Wrote '{os.path.basename(alt)}' instead.")
            return alt
        except Exception as e:
            print(f"    [ERROR] Could not write the fallback file either: {e}")
            return ""
    except Exception as e:
        print(f"    [ERROR] Writing '{path}' failed: {type(e).__name__}: {e}")
        return ""


def safe_write_json(obj, path: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt = f"{base}_{time.strftime('%H%M%S')}{ext}"
        try:
            with open(alt, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
            print(f"    [WARNING] '{os.path.basename(path)}' is locked -> "
                  f"wrote '{os.path.basename(alt)}'.")
            return alt
        except Exception as e:
            print(f"    [ERROR] Could not write JSON: {e}")
            return ""
    except Exception as e:
        print(f"    [ERROR] Writing '{path}' failed: {type(e).__name__}: {e}")
        return ""


def safe_write_text(text: str, path: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt = f"{base}_{time.strftime('%H%M%S')}{ext}"
        try:
            with open(alt, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"    [WARNING] '{os.path.basename(path)}' is locked -> "
                  f"wrote '{os.path.basename(alt)}'.")
            return alt
        except Exception as e:
            print(f"    [ERROR] Could not write text: {e}")
            return ""
    except Exception as e:
        print(f"    [ERROR] Writing '{path}' failed: {type(e).__name__}: {e}")
        return ""
