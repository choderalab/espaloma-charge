import os
from functools import partial

from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
from openff.toolkit import ToolkitRegistry, ForceField, RDKitToolkitWrapper
from openff.units import unit
import pytest
import torch
from torch_geometric.nn import SAGEConv
from openff.toolkit.utils.exceptions import ChargeMethodUnavailableError
from openff.toolkit._tests.create_molecules import create_ethanol

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
    """Point MODEL_PATH at a model built from the committed, parity-tested
    state_dict (data/pyg_model_state_dict.pt), so these tests never need a
    published GitHub release.
    """
    model_path = tmp_path / "model.pt"
    torch.save(_build_model(), model_path)
    monkeypatch.setattr(app, "MODEL_PATH", str(model_path))


def test_ethanol_molecule_api_w_wrapper():
    ectkw = EspalomaChargeToolkitWrapper()
    offmol = create_ethanol()
    offmol.assign_partial_charges('espaloma-am1bcc',
                                  toolkit_registry=ectkw)
    # Ensure its not all 0s
    assert (abs(offmol.partial_charges.m_as(unit.elementary_charge)) > 0.01).any(), offmol.partial_charges

def test_ethanol_molecule_api_w_explicit_registry():
    ectkr = ToolkitRegistry([EspalomaChargeToolkitWrapper])
    offmol = create_ethanol()
    offmol.assign_partial_charges('espaloma-am1bcc',
                                  toolkit_registry=ectkr)
    # Ensure its not all 0s
    assert (abs(offmol.partial_charges.m_as(unit.elementary_charge)) > 0.01).any(), offmol.partial_charges

def test_ethanol_molecule_api_wo_registry():
    offmol = create_ethanol()
    # This should raise a ChargeMethodUnavailableError since the global toolkit registry won't
    # automatically add the espalomawrapper
    with pytest.raises(ValueError):
        offmol.assign_partial_charges('espaloma-am1bcc')


def test_ethanol_direct():
    ectkw = EspalomaChargeToolkitWrapper()
    offmol = create_ethanol()
    ectkw.assign_partial_charges(offmol, 'espaloma-am1bcc')
    # Ensure its not all 0s
    assert (abs(offmol.partial_charges.m_as(unit.elementary_charge)) > 0.01).any(), offmol.partial_charges

def test_ethanol_direct_bad_method():
    ectkw = EspalomaChargeToolkitWrapper()
    offmol = create_ethanol()
    with pytest.raises(ChargeMethodUnavailableError):
        ectkw.assign_partial_charges(offmol, 'not-a-supported-method')

def test_create_system():
    offmol = create_ethanol()
    ff = ForceField('openff-2.0.0.offxml')
    ff.deregister_parameter_handler('ToolkitAM1BCC')
    ff.get_parameter_handler('ChargeIncrementModel', {'version': '0.3',
                                                      'partial_charge_method': 'espaloma-am1bcc'})
    sys = ff.create_openmm_system(offmol.to_topology(),
                                  toolkit_registry=ToolkitRegistry([EspalomaChargeToolkitWrapper, RDKitToolkitWrapper]))

def test_create_interchange():
    offmol = create_ethanol()
    ff = ForceField('openff-2.0.0.offxml')
    ff.deregister_parameter_handler('ToolkitAM1BCC')
    ff.get_parameter_handler('ChargeIncrementModel', {'version': '0.3',
                                                      'partial_charge_method': 'espaloma-am1bcc'})
    interchange = ff.create_interchange(offmol.to_topology(),
                                        toolkit_registry=ToolkitRegistry([EspalomaChargeToolkitWrapper, RDKitToolkitWrapper]))


# TODO: Tests for user-provided use_conformers, strict_n_conformers, normalize_partial_charges
