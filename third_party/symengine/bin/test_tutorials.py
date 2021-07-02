from pathlib import Path
import papermill as pm
import jupytext

mystDir = Path.cwd()  # Assume that we are running where the files are
tmpi = "tmp_inp.ipynb"  # Temporary input
tmpo = "tmp_out.ipynb"

for tut in list(mystDir.glob("**/*.myst.md")):
    print(f"Testing {tut.stem}.md")
    ntbk = jupytext.read(tut)
    jupytext.write(ntbk, tmpi)
    pm.execute_notebook(tmpi, tmpo, kernel_name="xcpp17")
