import pandas as pd
import torch
from openff.toolkit.topology import Molecule

def run():

    from espaloma_charge.utils import from_rdkit_mol
    molecules = Molecule.from_file("spice.oeb", allow_undefined_stereo=True)

    import pandas as pd
    df = pd.DataFrame(columns=["SMILES"])
    for molecule in molecules:
        df = df.append(
            {
                'SMILES': molecule.to_smiles(),
            },
            ignore_index=True,
        )


    df.to_csv("inspect.csv")
    from rdkit.Chem import PandasTools
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES", "MOL")
    open("inspect_all.html", "w").write(df.to_html())



if __name__ == "__main__":
    run()
