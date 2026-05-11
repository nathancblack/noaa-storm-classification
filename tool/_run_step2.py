"""Execute weather.ipynb cells 0..24 (through the Step-2 save cell), skipping
the JSON-dump cell so the locked headline numbers in week4_results.json are
not overwritten. Saves the executed notebook to weather.executed.ipynb."""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

NB_IN  = Path(__file__).parent / 'weather.ipynb'
NB_OUT = Path(__file__).parent / 'weather.executed.ipynb'
LAST_CELL_ID = '90af4ce6'  # Step-2 save cell — stop after this one (inclusive)

nb = nbformat.read(NB_IN, as_version=4)

# Truncate to [0 .. last_cell_id]
keep = []
for c in nb.cells:
    keep.append(c)
    if c.get('id') == LAST_CELL_ID:
        break
else:
    raise SystemExit(f'cell id {LAST_CELL_ID} not found')
nb.cells = keep
print(f'will execute {len(keep)} cells (last id={LAST_CELL_ID})', flush=True)

t0 = time.time()
client = NotebookClient(nb, timeout=-1, kernel_name='python3', resources={'metadata': {'path': str(NB_IN.parent)}})
try:
    client.execute()
finally:
    nbformat.write(nb, NB_OUT)
    print(f'wrote {NB_OUT} in {time.time()-t0:.1f}s', flush=True)
