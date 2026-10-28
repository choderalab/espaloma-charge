def test_import():
    from espaloma_charge.models import Sequential
    from espaloma_charge.models import ChargeEquilibrium

def test_sequential():
    from functools import partial
    from torch_geometric.nn import SAGEConv
    from rdkit import Chem
    from espaloma_charge.utils import from_rdkit_mol
    from espaloma_charge.models import Sequential
    sequential = Sequential(
        layer=partial(SAGEConv, aggr="mean"),
        config=[32, "relu", 32, "relu", 32, "relu"],
    )
    molecule = Chem.MolFromSmiles("C")
    graph = from_rdkit_mol(molecule)
    sequential(graph)

def test_readout_and_equilibrium():
    import torch
    from rdkit import Chem
    from espaloma_charge.utils import from_rdkit_mol
    from espaloma_charge.models import ChargeReadout, ChargeEquilibrium
    molecule = Chem.MolFromSmiles("C")
    molecule = Chem.AddHs(molecule)
    for atom in molecule.GetAtoms():
        print(atom.GetHybridization())
    graph = from_rdkit_mol(molecule)
    graph.h = torch.randn(5, 3)
    readout = ChargeReadout(3)
    graph = readout(graph)
    graph = ChargeEquilibrium()(graph)
