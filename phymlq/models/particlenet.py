import torch
import torch_geometric.nn


class ParticleDynamicEdgeGATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels: list, k=7):
        super(ParticleDynamicEdgeGATConv, self).__init__()

        self.k = k

        self.conv_list = torch.nn.ModuleList()
        self.conv_list.append(torch_geometric.nn.GATConv(in_channels, int(out_channels[0] / 4), heads=4))
        self.conv_list.append(torch_geometric.nn.GATConv(out_channels[0], int(out_channels[0] / 4), heads=4))
        self.conv_list.append(torch_geometric.nn.GATConv(out_channels[1], int(out_channels[2] / 4), heads=4))

        self.act_list = torch.nn.ModuleList()
        self.act_list.append(torch.nn.ReLU())
        self.act_list.append(torch.nn.ReLU())
        self.act_list.append(torch.nn.ReLU())

        self.skip_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
        )
        self.act = torch.nn.ReLU()

    def forward(self, pts, fts, batch=None):
        edges = torch_geometric.nn.knn_graph(pts, self.k, batch, loop=False, flow="target_to_source")

        x = fts
        for idx, layer in enumerate(self.conv_list):
            x = layer(x, edges)
            x = self.act_list[idx](x)

        skip = self.skip_mlp(fts)
        out = torch.add(x, skip)
        return self.act(out)
