# Legacy DGL -> PyG model migration scripts

Provenance and reproducibility record for how `espaloma_charge`'s PyTorch
Geometric model weights (`espaloma_charge/tests/data/pyg_model_state_dict.pt`)
were derived from, and validated against, the original DGL-based released
model (v0.0.8). See `../../DGL_TO_PYG_MIGRATION.md` for the broader
migration plan this is part of.

Run the two scripts in order:

## 1. `capture_baseline.py`

Downloads the original released model and runs it on a fixed set of
molecules, recording their charges as ground truth, plus extracts the
model's raw weights as a plain `state_dict`.

Needs an environment with a working `dgl` install:

```
python capture_baseline.py
```

If your env's `dgl` is `>=2` against a modern `torchdata` release, `dgl`
won't import at all (torchdata removed the `datapipes` module that dgl's
`graphbolt` needs) -- `torchdata_shim.py` works around this in-process,
and `capture_baseline.py` already imports it first. If your env instead
pins `dgl==1.1.2` (what `requirements.txt` and the README's install
instructions use), you don't need the shim at all -- it predates the
graphbolt/torchdata machinery entirely.

Writes:
- `../../espaloma_charge/tests/data/parity_fixture.json` (committed)
- `old_model_state_dict.pt`, next to this script (gitignored, intermediate
  only -- consumed by step 2)

## 2. `port_to_pyg.py`

Remaps the extracted DGL weights into the new PyG `SAGEConv` layout and
validates the ported model against the fixture from step 1. Run this in
the project's normal PyG dev environment instead, e.g.:

```
pixi run -e dev python port_to_pyg.py
```

Read `port_to_pyg.py`'s module docstring before touching the remapping --
it documents a non-obvious finding: the original model's conv layers carry
a real, trained `bias` tensor that its own DGL forward pass (for
`aggregator_type='mean'`) never actually applies. Porting that bias into
PyG's `SAGEConv` (which would apply it) silently changes the model's
output and is what caused the first parity attempt to fail.

Writes:
- `../../espaloma_charge/tests/data/pyg_model_state_dict.pt` (committed --
  this is what `espaloma_charge/tests/test_parity.py` loads)
- `new_model_v0.1.0.pt`, next to this script (gitignored) -- a full
  pickled module, ready to attach by hand to a new GitHub release. Neither
  script publishes anything on its own.
