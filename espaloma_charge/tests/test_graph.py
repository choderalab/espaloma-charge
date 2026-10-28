import pytest
from espaloma_charge.utils import from_rdkit_mol

@pytest.mark.parametrize(
    "smiles", ["C" * idx for idx in range(1, 10)]
)
def test_from_rdkit(smiles):
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(smiles)
    graph = from_rdkit_mol(molecule)
    assert molecule.GetNumAtoms() == graph.num_nodes
    assert molecule.GetNumBonds() * 2 == graph.edge_index.shape[1]
