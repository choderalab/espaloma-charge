"""`local_model` points MODEL_PATH at a model built from the committed,
parity-tested state_dict (data/pyg_model_state_dict.pt), so these tests
never need a published GitHub release.
"""

import os
from functools import partial

import pytest
import torch
from torch_geometric.nn import SAGEConv

from espaloma_charge import app
from espaloma_charge.models import Sequential, ChargeReadout, ChargeEquilibrium

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _build_model():
    config = [128, "relu"] * 4
    model = torch.nn.Sequential(
        Sequential(layer=partial(SAGEConv, aggr="mean", bias=False), config=config),
        ChargeReadout(128),
        ChargeEquilibrium(),
    )
    state_dict = torch.load(
        os.path.join(DATA_DIR, "pyg_model_state_dict.pt"), weights_only=True
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@pytest.fixture(autouse=True)
def local_model(tmp_path, monkeypatch):
    model_path = tmp_path / "model.pt"
    torch.save(_build_model(), model_path)
    monkeypatch.setattr(app, "MODEL_PATH", str(model_path))


def test_charge():
    from rdkit import Chem; from espaloma_charge import charge
    molecule = Chem.MolFromSmiles("N#N")
    charge(molecule)

def test_charge_batch():
    from rdkit import Chem
    from espaloma_charge import charge
    molecule = Chem.MolFromSmiles("N#N")
    charge([molecule, molecule])
