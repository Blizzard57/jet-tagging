import torch
import torch_geometric.nn

from phymlq.layers.lieconv import SO2Group, LieConv


class LieNet(torch.nn.Module):

    def __init__(self):
        super(LieNet, self).__init__()
        local_nn_1 = torch.nn.Sequential(
            torch.nn.Linear(5, 12),
            torch_geometric.nn.BatchNorm(12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 8),
            torch_geometric.nn.BatchNorm(8),
            torch.nn.ReLU()
        )
        local_nn_2 = torch.nn.Sequential(
            torch.nn.Linear(35, 32),
            torch_geometric.nn.BatchNorm(32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch_geometric.nn.BatchNorm(32),
            torch.nn.ReLU()
        )
        local_nn_3 = torch.nn.Sequential(
            torch.nn.Linear(67, 64),
            torch_geometric.nn.BatchNorm(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch_geometric.nn.BatchNorm(64),
            torch.nn.ReLU()
        )
        global_nn_1 = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch_geometric.nn.BatchNorm(32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch_geometric.nn.BatchNorm(32),
            torch.nn.ReLU()
        )
        global_nn_2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch_geometric.nn.BatchNorm(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch_geometric.nn.BatchNorm(64),
            torch.nn.ReLU()
        )
        global_nn_3 = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch_geometric.nn.BatchNorm(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch_geometric.nn.BatchNorm(64),
            torch.nn.ReLU()
        )
        weight_net_1 = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch_geometric.nn.BatchNorm(8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch_geometric.nn.BatchNorm(4),
            torch.nn.ReLU()
        )
        weight_net_2 = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch_geometric.nn.BatchNorm(8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch_geometric.nn.BatchNorm(4),
            torch.nn.ReLU()
        )
        weight_net_3 = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch_geometric.nn.BatchNorm(8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch_geometric.nn.BatchNorm(4),
            torch.nn.ReLU()
        )
        self.conv = LieConv(0.75, 32, SO2Group(),
                             local_nn=local_nn_1, global_nn=global_nn_1, weight_net=weight_net_1, is_first=True)
        self.conv2 = LieConv(0.75, 32, SO2Group(),
                             local_nn=local_nn_2, global_nn=global_nn_2, weight_net=weight_net_2)
        self.conv3 = LieConv(0.6, 64, SO2Group(),
                             local_nn=local_nn_3, global_nn=global_nn_3, weight_net=weight_net_3)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.Dropout(0.25),
            torch.nn.ReLU(),
        )
        self.bn = torch_geometric.nn.BatchNorm(4)
        self.linear = torch.nn.Linear(64, 2)
        self.act = torch.nn.Softmax(dim=1)

    def forward(self, batch_data):
        batch = batch_data.batch
        pos = batch_data.pos
        out = batch_data.x
        out = self.bn(out)

        out, fps_idx = self.conv(pos, out, batch)
        pos = pos[fps_idx]
        batch = batch[fps_idx]

        out, fps_idx = self.conv2(pos, out, batch)
        pos = pos[fps_idx]
        batch = batch[fps_idx]

        out, fps_idx = self.conv3(pos, out, batch)
        batch = batch[fps_idx]

        out = torch_geometric.nn.global_mean_pool(out, batch)
        out = self.mlp(out)
        out = self.linear(out)
        return self.act(out)
