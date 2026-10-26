"""
Step 1 of 2 in deriving the PyG model from the original DGL release.

Runs the ORIGINAL released DGL-based model (v0.0.8) on a fixed molecule
set and records the output charges -- this is the ground truth the PyG
port (port_to_pyg.py) is validated against -- plus extracts a plain
state_dict from the pickled model so its weights aren't tied to
DGL-importability for the next step.

Must run in an environment with a working `dgl` install. If that env's
dgl is >=2 against a modern torchdata, load torchdata_shim.py first (see
its docstring); dgl==1.1.2 (what requirements.txt pins) does not need it.

Outputs (both written next to this script):
  - old_model_state_dict.pt   -- intermediate only, consumed by
                                  port_to_pyg.py next, not committed
                                  (matches the repo's blanket `*.pt`
                                  gitignore rule)
  - ../../espaloma_charge/tests/data/parity_fixture.json
                                -- committed test fixture
"""

import os
import json

import torchdata_shim  # noqa: F401  -- must precede any dgl/espaloma_charge import

import torch
from rdkit import Chem
from espaloma_charge import charge
from espaloma_charge.app import MODEL_PATH, MODEL_URL

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(HERE, "..", "..", "espaloma_charge", "tests", "data")

SMILES = [
    # alkane sweep, matches tests/test_graph.py's "C".."CCCCCCCCC"
    *["C" * n for n in range(1, 10)],
    "c1ccccc1",       # benzene (aromatic ring)
    "C1CCCCC1",       # cyclohexane (aliphatic ring)
    "CCO",            # ethanol
    "CC(=O)O",        # acetic acid
    "c1ccncc1",       # pyridine (heteroatom in ring)
    "CC(=O)N",        # acetamide
    "c1ccc(cc1)O",    # phenol
    "CCCl",           # chloroethane
    "c1ccc(cc1)F",    # fluorobenzene
    "[NH4+]",         # formal +1 charge
    "[O-]C(=O)C",     # acetate, formal -1 charge
    "CSC",            # dimethyl sulfide
    "CCBr",           # bromoethane
    "CI",             # iodomethane
    "CP(=O)(O)O",     # methylphosphonic acid
]


def main():
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    results = {}
    for smiles in SMILES:
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None, f"RDKit failed to parse {smiles!r}"
        results[smiles] = charge(molecule).tolist()

    fixture_path = os.path.join(TEST_DATA_DIR, "parity_fixture.json")
    with open(fixture_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {fixture_path} with {len(results)} molecules.")

    print("MODEL_URL:", MODEL_URL.strip())
    print("MODEL_PATH (downloaded to):", MODEL_PATH)

    old_model = torch.load(MODEL_PATH, weights_only=False)
    state_dict = old_model.state_dict()
    state_dict_path = os.path.join(HERE, "old_model_state_dict.pt")
    torch.save(state_dict, state_dict_path)
    print(f"Extracted state_dict with {len(state_dict)} tensors -> {state_dict_path}")


if __name__ == "__main__":
    main()
