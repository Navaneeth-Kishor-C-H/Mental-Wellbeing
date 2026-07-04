from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

if __name__ == "__main__":
    runpy.run_path(str(SRC / "mental_wellbeing" / "web" / "dashboard.py"), run_name="__main__")
