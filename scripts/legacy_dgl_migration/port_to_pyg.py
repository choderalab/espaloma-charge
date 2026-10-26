"""
Step 2 of 2 in deriving the PyG model from the original DGL release.

Builds the new PyG-based architecture (Sequential+SAGEConv, ChargeReadout,
ChargeEquilibrium), loads the old model's state_dict (from
capture_baseline.py) into it via a key-remapping, and validates the result
against the parity fixture (also from capture_baseline.py) before writing
out the committed, ported weights.

Key finding this script encodes (confirmed by directly introspecting the
released model's actual conv layer -- see the git history of this file's
companion inspect_old_sageconv.py investigation, not just inferred from
state_dict key names):

  DGL's SAGEConv(aggregator_type='mean').forward() is, in the installed
  dgl 2.3.0/1.1.2 source (dgl/nn/pytorch/conv/sageconv.py):

      if self._aggre_type == "gcn":
          rst = h_neigh
          if self.bias is not None:
              rst = rst + self.bias          # bias only added here
      else:
          rst = self.fc_self(h_self) + h_neigh   # our case: 'mean' -- NO bias

  So despite the released checkpoint holding a real (non-zero, trained)
  `bias` tensor per conv layer, it is NEVER added by the 'mean' aggregator
  branch that this model actually uses -- it's dead weight left over from
  however the module was originally constructed. Porting it into PyG's
  SAGEConv (which WOULD apply it, via lin_l's bias) silently reintroduces
  a bias term the original model never had, which is what caused the
  initial parity check to fail with "many mismatches" before this was
  found. The fix: build the new SAGEConv layers with bias=False, and
  drop the old `bias` key during the remap entirely.

  PyG's SAGEConv(aggr='mean'), root_weight=True (torch_geometric/nn/conv/
  sage_conv.py): out = lin_l(mean(x_neighbors)) + lin_r(x_self). So:
    lin_r.weight <- fc_self.weight
    lin_l.weight <- fc_neigh.weight
    (no bias anywhere, with bias=False)

Run this in the project's normal PyG dev environment, e.g.:
    pixi run -e dev python scripts/legacy_dgl_migration/port_to_pyg.py

Inputs (both produced by capture_baseline.py):
  - ./old_model_state_dict.pt                                (intermediate)
  - ../../espaloma_charge/tests/data/parity_fixture.json      (committed)

Outputs:
  - ../../espaloma_charge/tests/data/pyg_model_state_dict.pt  (committed --
    this is what espaloma_charge/tests/test_parity.py loads)
  - ./new_model_v0.1.0.pt (local only, *.pt is gitignored) -- a full
    pickled module, ready to attach to a new GitHub release by hand. This
    script does NOT publish anything.
"""

import os
import json
from functools import partial

import torch
from rdkit import Chem
from torch_geometric.nn import SAGEConv

from espaloma_charge.utils import from_rdkit_mol
from espaloma_charge.models import Sequential, ChargeReadout, ChargeEquilibrium

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(HERE, "..", "..", "espaloma_charge", "tests", "data")

CONV_LAYER_NAMES = ("d0", "d2", "d4", "d6")


def remap_dgl_sageconv_state_dict(old_sd):
    new_sd = {}
    for key, value in old_sd.items():
        if ".f_in." in key or ".fc_params." in key:
            new_sd[key] = value  # plain nn.Linear in both architectures
            continue

        matched = False
        for layer_name in CONV_LAYER_NAMES:
            prefix = f"0._sequential.{layer_name}."
            if key == prefix + "fc_neigh.weight":
                new_sd[prefix + "lin_l.weight"] = value
                matched = True
            elif key == prefix + "bias":
                matched = True  # deliberately dropped, see module docstring
            elif key == prefix + "fc_self.weight":
                new_sd[prefix + "lin_r.weight"] = value
                matched = True
        if not matched:
            raise KeyError(f"Unexpected key in old state_dict: {key}")

    return new_sd


def build_new_model():
    config = [128, "relu"] * 4
    return torch.nn.Sequential(
        Sequential(layer=partial(SAGEConv, aggr="mean", bias=False), config=config),
        ChargeReadout(128),
        ChargeEquilibrium(),
    )


def main():
    with open(os.path.join(TEST_DATA_DIR, "parity_fixture.json")) as f:
        fixture = json.load(f)

    old_sd = torch.load(
        os.path.join(HERE, "old_model_state_dict.pt"), weights_only=True
    )
    new_sd = remap_dgl_sageconv_state_dict(old_sd)

    model = build_new_model()
    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    print("load_state_dict: missing =", missing, "unexpected =", unexpected)

    model.eval()
    max_abs_diff = 0.0
    with torch.no_grad():
        for smiles, expected in fixture.items():
            molecule = Chem.MolFromSmiles(smiles)
            graph = from_rdkit_mol(molecule)
            graph = model(graph)
            actual = graph.q.flatten().tolist()
            diffs = [abs(a - e) for a, e in zip(actual, expected)]
            max_abs_diff = max(max_abs_diff, max(diffs))
            status = "OK" if max(diffs) < 1e-4 else "MISMATCH"
            print(f"{status:9s} {smiles:15s} max_abs_diff={max(diffs):.3e}")

    print(f"\nOverall max_abs_diff across {len(fixture)} molecules: {max_abs_diff:.3e}")
    assert max_abs_diff < 1e-4, "Parity check FAILED"
    print("Parity check PASSED (atol=1e-4)")

    state_dict_path = os.path.join(TEST_DATA_DIR, "pyg_model_state_dict.pt")
    torch.save(new_sd, state_dict_path)
    print(f"\nWrote {state_dict_path}")

    release_asset_path = os.path.join(HERE, "new_model_v0.1.0.pt")
    torch.save(model, release_asset_path)
    print(f"Wrote {release_asset_path} (ready-to-upload release asset, not published)")


if __name__ == "__main__":
    main()
