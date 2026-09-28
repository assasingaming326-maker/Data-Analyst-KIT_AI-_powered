"""Fallback runner when pytest isn't installed: python tests/run_all.py"""
import sys, tempfile, inspect, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src")); sys.path.insert(0, str(Path(__file__).parent))
import test_core
ok = bad = 0
for n, f in vars(test_core).items():
    if n.startswith("test_"):
        try:
            f(Path(tempfile.mkdtemp())) if inspect.signature(f).parameters else f(); ok += 1; print("PASS", n)
        except Exception: bad += 1; print("FAIL", n); traceback.print_exc()
print(ok, "passed,", bad, "failed"); sys.exit(bad)
