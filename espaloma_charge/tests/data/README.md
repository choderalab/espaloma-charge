# Test data provenance

- `parity_fixture.json` -- charges produced by the original DGL-based
  released model (v0.0.8) on a fixed set of molecules.
- `pyg_model_state_dict.pt` -- that same model's weights, ported into the
  PyTorch Geometric architecture used by `espaloma_charge.models`.

Both files were generated and validated by `scripts/legacy_dgl_migration/`
(repo root) -- see that directory's `README.md` for how to regenerate or
extend them. `espaloma_charge/tests/test_parity.py` is what actually
consumes them.
