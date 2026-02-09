"""
Run 611 Script.ipynb and save executed notebook + log.
Execute from project root: py run_analysis.py
"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
base = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(base, "run_log.txt")
with open(log_path, "w", encoding="utf-8") as _:
    _.write("run_analysis started from " + base + "\n")

def log(msg):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

log("Starting notebook execution...")
try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbconvert.preprocessors import CellExecutionError
except ImportError as e:
    log("Missing dependency: " + str(e))
    log("Run: pip install nbformat nbconvert")
    sys.exit(1)

nb_path = "611 Script.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=1800)
try:
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(os.path.abspath(__file__))}})
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    log("Notebook executed successfully. Outputs saved in 611 Script.ipynb")
except CellExecutionError as e:
    log("Cell error: " + str(e))
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    sys.exit(1)
except Exception as e:
    log("Error: " + type(e).__name__ + " " + str(e))
    sys.exit(1)
