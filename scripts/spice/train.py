import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from openff.toolkit.topology import Molecule
class ChargeDataset(torch.utils.data.Dataset):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

def run(args):
    from espaloma_charge.utils import from_rdkit_mol
    molecules = Molecule.from_file(args.path, allow_undefined_stereo=True)
    from collections import defaultdict
    name2graph = defaultdict(lambda: [])
    failures = []
    for molecule in molecules:
        try:
            name = molecule.name
            graph = from_rdkit_mol(molecule.to_rdkit())
            graph.q_ref = torch.tensor(molecule.partial_charges.magnitude).unsqueeze(-1)
            name2graph[name].append(graph)
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

   
    dataset_train = []
    dataset_valid = []
    dataset_test = []

    for name in names_train:
        dataset_train += name2graph[name]
    for name in names_valid:
        dataset_valid += name2graph[name]
    for name in names_test:
        dataset_test += name2graph[name]

    dataset_train = ChargeDataset(dataset_train)
    dataset_valid = ChargeDataset(dataset_valid)
    dataset_test = ChargeDataset(dataset_test)

    
    config = [args.width, args.activation] * args.depth
    from espaloma_charge.models import (
        Sequential, ChargeReadout, ChargeEquilibrium
    )
    from torch_geometric.nn import SAGEConv
    from functools import partial
    model = torch.nn.Sequential(
        Sequential(
            layer=partial(SAGEConv, aggr="mean", bias=False),
            config=config,
        ),
        ChargeReadout(args.width),
        ChargeEquilibrium(),
    )

    dataloader = DataLoader(dataset_train, batch_size=args.batch_size, pin_memory=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size)
  
    rmse_vl_best = 9999.9
    rmse_te_best = 9999.9

    if torch.cuda.is_available(): model = model.cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for idx_epoch in range(args.n_epochs):
        print(idx_epoch, flush=True)
        for g in dataloader:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                g = g.to("cuda:0")
            g = model(g)
            loss = torch.nn.MSELoss()(
                g.q_ref,
                g.q,
            )

            loss.backward()
            optimizer.step()

        
        if idx_epoch % 10 ==  0:
            with torch.no_grad():
                q_vl = []
                q_vl_ref = []
                q_te = []
                q_te_ref = []
                for g in dataloader_valid:
                    if torch.cuda.is_available():
                        g = g.to("cuda:0")
                    g = model(g)
                    q_vl.append(g.q)
                    q_vl_ref.append(g.q_ref)


                for g in dataloader_test:
                    if torch.cuda.is_available():
                        g = g.to("cuda:0")
                    g = model(g)
                    q_te.append(g.q)
                    q_te_ref.append(g.q_ref)

                q_vl = torch.cat(q_vl)
                q_vl_ref = torch.cat(q_vl_ref)
                q_te = torch.cat(q_te)
                q_te_ref = torch.cat(q_te_ref)

                rmse_vl = (q_vl - q_vl_ref).pow(2).mean().pow(0.5).item()
                rmse_te = (q_te - q_te_ref).pow(2).mean().pow(0.5).item()
                
                print(rmse_vl)

                if rmse_vl <= rmse_vl_best:
                    print(rmse_vl, rmse_vl_best)
                    print("saving")
                    rmse_vl_best = rmse_vl
                    rmse_te_best = rmse_te
                    torch.save(model, "model.pt") 
             
    import json
    import pandas as pd
    key = dict(vars(args))
    key.pop("out")
    key = json.dumps(key)

    df = pd.DataFrame.from_dict(
        {key: [rmse_vl_best, rmse_te_best]},
        orient="index",
        columns=["vl", "te"]
    )
    df.to_csv(args.out + ".csv", mode="a", header=False)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--path", type=str, default="spice.oeb")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out", type=str, default="_out")
    args = parser.parse_args()
    run(args)
