"""
In-process compatibility shim, needed only to run the OLD DGL-based model
(capture_baseline.py) in an environment with a modern torchdata release.

torchdata>=0.10 removed the `datapipes` module (and the old
`dataloader2.graph` utilities). dgl>=2's `graphbolt` package still does
`from torchdata.datapipes.iter import IterDataPipe, Mapper` and
`import torchdata.dataloader2.graph as dp_utils` unconditionally at import
time, which breaks `import dgl` under any modern torchdata release.

`IterDataPipe` and `Mapper` still exist, unchanged, in `torch.utils.data`
itself (torchdata used to just re-export them), so this shim restores
only those two names. The `dataloader2.graph` helpers (`find_dps`,
`replace_dp`, `traverse_dps`) are stubbed to raise if actually called --
espaloma_charge only uses dgl's old `dgl.dataloading.GraphDataLoader`
API, never dgl's graphbolt-based DataLoader, so these stubs are never
exercised; they exist purely so the module-level `import ... as dp_utils`
statement in dgl's graphbolt/dataloader.py succeeds.

NOTE: if your environment pins dgl==1.1.2 (as `requirements.txt` does, and
as the README's documented install instructions use), you likely don't
need this shim at all -- dgl 1.1.2 predates the graphbolt/torchdata
machinery entirely. This shim is only needed for dgl>=2 (e.g. what
`environment.yml` currently pins) against a modern torchdata.

This is NOT installed anywhere -- it only registers modules in
`sys.modules` for the lifetime of the current process, and must be
imported before the first `import dgl` (or anything that imports dgl,
e.g. `espaloma_charge`).
"""

import sys
import types

from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import Mapper


def _unimplemented(name):
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"torchdata.dataloader2.graph.{name} shim: real datapipe-graph "
            "utilities were removed upstream in torchdata>=0.10; this stub "
            "only exists to satisfy dgl.graphbolt's import-time "
            "requirements. espaloma_charge does not use dgl's "
            "graphbolt-based DataLoader, so this should never actually run."
        )

    return _fn


datapipes_iter = types.ModuleType("torchdata.datapipes.iter")
datapipes_iter.IterDataPipe = IterDataPipe
datapipes_iter.Mapper = Mapper


class Prefetcher(IterDataPipe):
    def __init__(self, source_datapipe, buffer_size=2):
        self.source_datapipe = source_datapipe
        self.buffer_size = buffer_size

    def __iter__(self):
        yield from self.source_datapipe


datapipes_iter.Prefetcher = Prefetcher

datapipes = types.ModuleType("torchdata.datapipes")
datapipes.iter = datapipes_iter

dataloader2_graph = types.ModuleType("torchdata.dataloader2.graph")
dataloader2_graph.find_dps = _unimplemented("find_dps")
dataloader2_graph.replace_dp = _unimplemented("replace_dp")
dataloader2_graph.traverse_dps = _unimplemented("traverse_dps")

dataloader2 = types.ModuleType("torchdata.dataloader2")
dataloader2.graph = dataloader2_graph

sys.modules["torchdata.datapipes"] = datapipes
sys.modules["torchdata.datapipes.iter"] = datapipes_iter
sys.modules["torchdata.dataloader2"] = dataloader2
sys.modules["torchdata.dataloader2.graph"] = dataloader2_graph
