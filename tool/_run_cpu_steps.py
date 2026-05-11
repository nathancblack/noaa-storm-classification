"""Run Steps 3-7 cells from weather.ipynb on CPU. Picks them by cell id."""
import sys
from pathlib import Path
import nbformat

NB = Path(__file__).parent / 'weather.ipynb'
nb = nbformat.read(NB, as_version=4)

# Step cell ids in execution order (markdown cells skipped automatically)
TARGETS = ['a296354c', 'f6627606', '9e857cc2', '200da2bb', 'ac865d8d']

import os
os.chdir(NB.parent)  # reports paths are '../reports'

for cid in TARGETS:
    cell = next((c for c in nb.cells if c.get('id') == cid), None)
    if cell is None:
        print(f"[!] cell {cid} not found", file=sys.stderr); continue
    src = cell['source']
    if isinstance(src, list): src = ''.join(src)
    print(f"\n{'#'*72}\n# Running cell {cid}\n{'#'*72}")
    try:
        exec(compile(src, f'<cell {cid}>', 'exec'), {'__name__': '__main__'})
    except Exception as e:
        print(f"[!] cell {cid} raised {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
