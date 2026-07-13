#!/usr/bin/env bash
# End-to-end smoke test: the repo is alive. Must exit 0.
# Deliberately fast and offline — no model weights, no dataset, no network.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> 1/5 package imports"
uv run python -c "import gdp; print('   gdp', gdp.__version__)"

echo "==> 2/5 config loads and validates"
uv run python -c "
from gdp import load_config
c = load_config('configs/default.yaml')
print('   seed', c.seed, '| classes', len(c.dataset.classes), '|', c.detector.model_id)
"

echo "==> 3/5 device selection"
uv run python -c "
from gdp import select_device
print('   device:', select_device('auto'))
"

echo "==> 4/5 fixture dataset loads"
uv run python -c "
from gdp import load_config
from gdp.data import load_dataset
c = load_config('configs/default.yaml')
ds = load_dataset(c.dataset.annotations, c.dataset.root)
n = sum(len(s.boxes) for s in ds.samples)
print(f'   {len(ds)} images, {n} boxes')
print('   prompt:', ds.prompt()[:60], '...')
"

echo "==> 5/5 CLI responds"
uv run gdp info -c configs/default.yaml > /dev/null
uv run gdp --help > /dev/null
echo "   gdp info + --help ok"

echo
echo "SMOKE OK"
