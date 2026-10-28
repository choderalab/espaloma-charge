import pandas as pd
import torch
from rdkit import Chem
from openff.toolkit.topology import Molecule

def run():
    from espaloma_charge.utils import from_rdkit_mol
    molecule = Molecule.from_smiles("CCl")
    molecule.assign_partial_charges("am1bcc")
    print(molecule.partial_charges)
    g = from_rdkit_mol(molecule.to_rdkit())
    model = torch.load("model.pt", weights_only=False)
    model(g)
    print(g.q)


if __name__ == "__main__":
    run()
