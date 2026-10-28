import pandas as pd
import torch
from openff.toolkit.topology import Molecule
class ChargeDataset(torch.utils.data.Dataset):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

def run():
    from espaloma_charge.utils import from_rdkit_mol
    molecules = Molecule.from_file("spice.oeb", allow_undefined_stereo=True)
    from collections import defaultdict
    name2graph = defaultdict(lambda: [])
    name2smiles = defaultdict(lambda: [])
    failures = []
    for molecule in molecules:
        try:
            name = molecule.name
            graph = from_rdkit_mol(molecule.to_rdkit())
            graph.q_ref = torch.tensor(molecule.partial_charges.magnitude).unsqueeze(-1)
            name2graph[name].append(graph)
            name2smiles[name].append(molecule.to_smiles())
        except:
            failures.append(molecule.to_smiles())

    with open("failures.txt", "w") as file_handle:
        file_handle.writelines(failures)

    names = list(name2graph.keys())
    import random
    random.seed(2666)
    random.shuffle(names)
    n_te = int(0.1*len(names))
    names_train = names[:8*n_te]
    names_valid = names[8*n_te:9*n_te]
    names_test = names[9*n_te:]

    dataset_train = ChargeDataset([item for item in name2graph[name] for name in names_train])
    dataset_valid = ChargeDataset([item for item in name2graph[name] for name in names_valid])
    dataset_test = ChargeDataset([item for item in name2graph[name] for name in names_test])
    smiles_test = [item for item in name2smiles[name] for name in names_test]

    model = torch.load("model.pt", map_location="cpu", weights_only=False)

    import pandas as pd
    df = pd.DataFrame(columns=["SMILES", "RMSE"])

    q = []
    q_ref = []
    for name in names_test:
        print(name)
        for smiles, g in zip(name2smiles[name], name2graph[name]):
            with torch.no_grad():
                g = model(g)
            q.append(g.q.flatten().detach())
            q_ref.append(g.q_ref.flatten().detach())
            print(g.q_ref.sum(), g.q.sum())
            df = df.append(
                {
                    'NAME': name,
                    'SMILES': smiles,
                    'RMSE': (g.q_ref.flatten() - g.q.flatten()).pow(2).mean().pow(0.5).item(),
                    'Q': g.q.flatten().detach().numpy(),
                    'Q_REF': g.q_ref.flatten().detach().numpy(),
                    'TOTAL_CHARGE': g.q_ref.sum().item(),
                },
                ignore_index=True,
            )

    q = torch.cat(q)
    q_ref = torch.cat(q_ref)
    rmse = (q - q_ref).pow(2).mean().pow(0.5)
    print("rmse", rmse)

    df = df.sort_values(by="RMSE", ascending=False)
    df.to_csv("out.csv")
    from rdkit.Chem import PandasTools
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES", "MOL")
    open("inspect_charge_te.html", "w").write(df.to_html())



if __name__ == "__main__":
    run()
