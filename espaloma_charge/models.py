"""Neural network components of espaloma charge."""

import torch

class _Sequential(torch.nn.Module):
    """Sequentially staggered neural networks."""

    def __init__(
        self,
        layer,
        config,
        in_features,
        model_kwargs={},
    ):
        super(_Sequential, self).__init__()

        self.exes = []

        # init dim
        dim = in_features

        # parse the config
        for idx, exe in enumerate(config):

            try:
                exe = float(exe)

                if exe >= 1:
                    exe = int(exe)
            except BaseException:
                pass

            # int -> feedfoward
            if isinstance(exe, int):
                setattr(self, "d" + str(idx), layer(dim, exe, **model_kwargs))

                dim = exe
                self.exes.append("d" + str(idx))

            # str -> activation
            elif isinstance(exe, str):
                if exe == "bn":
                    setattr(self, "a" + str(idx), torch.nn.BatchNorm1d(dim))

                else:
                    activation = getattr(torch.nn.functional, exe)
                    setattr(self, "a" + str(idx), activation)

                self.exes.append("a" + str(idx))

            # float -> dropout
            elif isinstance(exe, float):
                dropout = torch.nn.Dropout(exe)
                setattr(self, "o" + str(idx), dropout)

                self.exes.append("o" + str(idx))

    def forward(self, x, edge_index=None, **kwargs):
        for exe in self.exes:
            if exe.startswith("d"):
                if edge_index is not None:
                    x = getattr(self, exe)(x, edge_index)
                else:
                    x = getattr(self, exe)(x)
            else:
                x = getattr(self, exe)(x)

        return x


class Sequential(torch.nn.Module):
    """Sequential neural network with input layers.

    Parameters
    ----------
    layer : torch.nn.Module
        PyTorch Geometric graph convolution layer class.

    config : List
        A sequence of numbers (for units) and strings (for activation functions)
        denoting the configuration of the sequential model.

    feature_units : int(default=117)
        The number of input channels.

    Methods
    -------
    forward(g, x)
        Forward pass.
    """

    def __init__(
        self,
        layer,
        config,
        feature_units=116,
        input_units=128,
        model_kwargs={},
    ):
        super(Sequential, self).__init__()

        # initial featurization
        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(feature_units, input_units), torch.nn.Tanh()
        )

        self._sequential = _Sequential(
            layer, config, in_features=input_units, model_kwargs=model_kwargs
        )

    def forward(self, data, x=None, **kwargs):
        """Forward pass.

        Parameters
        ----------
        data : `torch_geometric.data.Data` or `torch_geometric.data.Batch`
            input graph

        Returns
        -------
        data : `torch_geometric.data.Data` or `torch_geometric.data.Batch`
            output graph
        """
        if x is None:
            # get node attributes
            x = data.h0
            x = self.f_in(x)

        # message passing on homo graph
        x = self._sequential(x, data.edge_index)

        # put attribute back in the graph
        data.h = x

        return data

def get_charges(e, s, sum_e_s_inv, sum_s_inv, sum_q):
    """ Solve the function to get the absolute charges of atoms in a
    molecule from parameters.
    Parameters
    ----------
    e : torch.Tensor, dtype = torch.float32,
        electronegativity.
    s : torch.Tensor, dtype = torch.float32,
        hardness.
    sum_q : torch.Tensor, dtype = torch.float32,
        total charge of a molecule, broadcast to every atom in it.
    We use Lagrange multipliers to analytically give the solution.
    $$
    U({\bf q})
    &= \sum_{i=1}^N \left[ e_i q_i +  \frac{1}{2}  s_i q_i^2\right]
        - \lambda \, \left( \sum_{j=1}^N q_j - Q \right) \\
    &= \sum_{i=1}^N \left[
        (e_i - \lambda) q_i +  \frac{1}{2}  s_i q_i^2 \right
        ] + Q
    $$
    This gives us:
    $$
    q_i^*
    &= - e_i s_i^{-1}
    + \lambda s_i^{-1} \\
    &= - e_i s_i^{-1}
    + s_i^{-1} \frac{
        Q +
         \sum\limits_{i=1}^N e_i \, s_i^{-1}
        }{\sum\limits_{j=1}^N s_j^{-1}}
    $$
    """
    return -e * s**-1 + (s**-1) * torch.div(sum_q + sum_e_s_inv, sum_s_inv)

class ChargeReadout(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc_params = torch.nn.Linear(in_features, 2)

    def forward(self, data, **kwargs):
        h = self.fc_params(data.h)
        e, s = h.split(1, -1)
        data.e, data.s = e, s
        return data

class ChargeEquilibrium(torch.nn.Module):
    """Charge equilibrium within batches of molecules."""

    def __init__(self):
        super(ChargeEquilibrium, self).__init__()

    def forward(self, data, total_charge=0.0):
        """apply charge equilibrium to all molecules in batch"""
        from torch_geometric.utils import scatter

        # calculate $s ^ {-1}$ and $ es ^ {-1}$
        s_inv = data.s ** -1
        e_s_inv = data.e * s_inv

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.s.device)
        num_graphs = int(batch.max()) + 1

        if "q_ref" in data:
            total_charge = scatter(data.q_ref, batch, dim=0, dim_size=num_graphs, reduce="sum")
        else:
            total_charge = torch.ones(num_graphs, 1, device=data.s.device) * total_charge

        sum_q = total_charge[batch]

        sum_s_inv = scatter(s_inv, batch, dim=0, dim_size=num_graphs, reduce="sum")
        sum_e_s_inv = scatter(e_s_inv, batch, dim=0, dim_size=num_graphs, reduce="sum")
        sum_s_inv = sum_s_inv[batch]
        sum_e_s_inv = sum_e_s_inv[batch]

        data.q = get_charges(data.e, data.s, sum_e_s_inv, sum_s_inv, sum_q)

        return data
