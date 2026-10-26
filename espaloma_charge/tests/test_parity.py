"""Numeric parity between the original DGL-based released model (v0.0.8)
and the migrated PyTorch Geometric architecture.

`data/parity_fixture.json` holds charges produced by the original DGL
model on a fixed molecule set; `data/pyg_model_state_dict.pt` holds that
same model's weights, ported into the PyG architecture. See
`scripts/legacy_dgl_migration/` (repo root) for how both files were
derived and validated, including why the DGL checkpoint's unused `bias`
parameter must NOT be carried over.
"""

import json
import os
from functools import partial

import pytest
import torch
from rdkit import Chem
from torch_geometric.nn import SAGEConv

from espaloma_charge.models import Sequential, ChargeReadout, ChargeEquilibrium
from espaloma_charge.utils import from_rdkit_mol

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _build_model():
    config = [128, "relu"] * 4
    return torch.nn.Sequential(
        Sequential(layer=partial(SAGEConv, aggr="mean", bias=False), config=config),
        ChargeReadout(128),
        ChargeEquilibrium(),
    )


@pytest.fixture(scope="module")
def ported_model():
    model = _build_model()
    state_dict = torch.load(
        os.path.join(DATA_DIR, "pyg_model_state_dict.pt"), weights_only=True
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def test_parity_with_dgl_model(ported_model):
    with open(os.path.join(DATA_DIR, "parity_fixture.json")) as f:
        fixture = json.load(f)

    with torch.no_grad():
        for smiles, expected in fixture.items():
            molecule = Chem.MolFromSmiles(smiles)
            graph = from_rdkit_mol(molecule)
            graph = ported_model(graph)
            actual = graph.q.flatten().tolist()
            assert actual == pytest.approx(expected, abs=1e-4), smiles
