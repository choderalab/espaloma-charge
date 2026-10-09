import torch

SUPPORTED_ELEMENTS = [
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]


def fp_rdkit(atom):
    from rdkit import Chem

    element = atom.GetSymbol()
    if element not in SUPPORTED_ELEMENTS:
        raise ValueError(f"Element {element} is not supported.")

    HYBRIDIZATION_RDKIT = {
        Chem.rdchem.HybridizationType.SP: torch.tensor(
            [1, 0, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP2: torch.tensor(
            [0, 1, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3: torch.tensor(
            [0, 0, 1, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3D: torch.tensor(
            [0, 0, 0, 1, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3D2: torch.tensor(
            [0, 0, 0, 0, 1],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.S: torch.tensor(
            [0, 0, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.UNSPECIFIED: torch.tensor(
            [0, 0, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
    }
    return torch.cat(
        [
            torch.tensor(
                [
                    atom.GetTotalDegree(),
                    atom.GetTotalValence(),
                    atom.GetExplicitValence(),
                    # atom.GetFormalCharge(),
                    atom.GetIsAromatic() * 1.0,
                    atom.GetMass(),
                    atom.IsInRingSize(3) * 1.0,
                    atom.IsInRingSize(4) * 1.0,
                    atom.IsInRingSize(5) * 1.0,
                    atom.IsInRingSize(6) * 1.0,
                    atom.IsInRingSize(7) * 1.0,
                    atom.IsInRingSize(8) * 1.0,
                ],
                dtype=torch.get_default_dtype(),
            ),
            HYBRIDIZATION_RDKIT[atom.GetHybridization()],
        ],
        dim=0,
    )


def from_rdkit_mol(mol, use_fp=True):
    from torch_geometric.data import Data

    # enter nodes
    n_atoms = mol.GetNumAtoms()
    atomic_num = torch.Tensor(
        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    )
    q_ref = torch.Tensor(
        [[atom.GetFormalCharge()] for atom in mol.GetAtoms()]
    )
    h_v = torch.zeros(atomic_num.shape[0], 100, dtype=torch.float32)

    h_v[
        torch.arange(atomic_num.shape[0]),
        torch.squeeze(atomic_num).long(),
    ] = 1.0

    h_v_fp = torch.stack([fp_rdkit(atom) for atom in mol.GetAtoms()], axis=0)

    if use_fp == True:
        h_v = torch.cat([h_v, h_v_fp], dim=-1)  # (n_atoms, 117)

    # enter bonds
    bonds = list(mol.GetBonds())
    bonds_begin_idxs = [bond.GetBeginAtomIdx() for bond in bonds]
    bonds_end_idxs = [bond.GetEndAtomIdx() for bond in bonds]

    # NOTE: edges are directional, so both directions are added explicitly
    edge_index = torch.tensor(
        [
            bonds_begin_idxs + bonds_end_idxs,
            bonds_end_idxs + bonds_begin_idxs,
        ],
        dtype=torch.long,
    )

    return Data(
        h0=h_v,
        q_ref=q_ref,
        edge_index=edge_index,
        num_nodes=n_atoms,
    )
