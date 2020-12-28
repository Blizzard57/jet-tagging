import typing

import torch
import torch_geometric.nn, torch_geometric.typing
import torch_scatter.utils
from torch import Tensor
from torch_sparse import SparseTensor

from phymlq.hyperparams import DEVICE


class DensityNet(torch_geometric.nn.MessagePassing):

    def __init__(self, k=100, nn=None, kernel="gaussian", kernel_params=None):
        super(DensityNet, self).__init__(aggr='mean', flow='target_to_source')
        self.k = k
        self.nn = nn
        self.kernel = kernel
        self.kernel_params = kernel_params

    def forward(self, pos, batch, fps_idx):
        knn_edges = torch_geometric.nn.knn_graph(pos, k=self.k, batch=batch, flow=self.flow)

        pos_density = self.propagate(knn_edges, pos=pos)
        inv_density = 1 / pos_density

        sampled_inv_density = inv_density[fps_idx]

        if self.nn is not None:
            sampled_inv_density = self.nn(sampled_inv_density)

        return sampled_inv_density

    # noinspection PyMethodOverriding
    def message(self, x_i: Tensor, x_j: Tensor):
        sq_dist = torch.norm(x_j - x_i, dim=1, keepdim=True)

        if self.kernel == "gaussian":
            bandwidth = self.kernel_params[0]
            density = torch.exp(-sq_dist / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
        else:
            raise NotImplementedError

        return density

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass


class PointConv(torch_geometric.nn.MessagePassing):

    def __init__(self, ratio, k, weight_net=None, local_nn=None, global_nn=None):
        super(PointConv, self).__init__(aggr=None, flow='target_to_source')
        self.ratio = ratio
        self.k = k
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.weight_net = weight_net

    def forward(self, pos, x, batch):
        # pos: [N, P]
        # x: [N, F]
        # edge_index: [2, E]
        # batch: [N]
        knn_edges, fps_idx, knn_idx = self.sample_and_knn(pos, x, batch, self.ratio, self.k)

        out = self.propagate(knn_edges, x=x, pos=pos, batch=batch, knn_idx=knn_idx, fps_idx=fps_idx,
                             fps_idx_shape=fps_idx.shape[0]).to(DEVICE)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out, fps_idx

    # noinspection PyMethodOverriding
    def message(self, x, edge_index, pos_i, pos_j, x_j):
        grouped_norm = (pos_j - pos_i)
        if x_j is not None:
            msg = torch.cat([grouped_norm.clone(), x_j], dim=1)
        else:
            msg = grouped_norm.clone()

        if self.weight_net is not None:
            grouped_norm = self.weight_net(grouped_norm)
        if self.local_nn is not None:
            return self.local_nn(msg), msg, grouped_norm
        return msg, msg, grouped_norm

    # noinspection PyMethodOverriding
    def aggregate(self, msg_output, knn_idx, fps_idx_shape, density_scale=None):
        new_points, msg, weights = msg_output  # weights: nn(msg), msg: new_points, grouped_norm: grouped_norm
        return self.aggr_int(fps_idx_shape, self.k, self.node_dim, knn_idx, new_points, weights,
                             density_scale=density_scale)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def sample_and_knn(cls, pos, _x, batch, ratio, k):
        """
        Applies FPS sampling to each point in the point clouds and
        """
        fps_idx = torch_geometric.nn.fps(pos, batch, ratio=ratio)
        fps_idx = torch.sort(fps_idx)[0]  # TODO: verify if this is correct
        # fps_idx = fps(pos, batch, ratio=ratio)

        sampled_pos = pos[fps_idx]  # new_xyz
        sampled_batch = batch[fps_idx]
        knn_edges = torch_geometric.nn.knn(pos, sampled_pos, k, batch_x=batch, batch_y=sampled_batch)  # [2, E]
        # knn_edges = knn(pos, sampled_pos, k, batch, sampled_batch) # VERIFY KNN

        # FIXME: Check for error here
        knn_idx = knn_edges[0].clone()
        knn_edges[0] = fps_idx[knn_edges[0]]
        return knn_edges, fps_idx, knn_idx

    @staticmethod
    @torch.jit.script
    def aggr_int(dim_size: int, k: int, node_dim: int, index, new_points, weights,
                 density_scale: typing.Optional[torch_geometric.typing.OptTensor] = None):
        uniq, counts = torch.unique(index, return_counts=True)
        # ASSUMPTION: INDEX IS SORTED-> SO UNIQ AND INDEX ARE THE SAME
        assert torch.all(torch.eq(index, torch.sort(index)[0])) == 1
        size_new_points = [dim_size * k, int(list(new_points.size())[-1])]
        integers = torch.arange(counts[0], device=DEVICE).view(-1, 1)

        prev_i = counts[0]
        integers_dim = 0
        for i in counts[1:]:
            if i == prev_i:
                integers_dim += 1
            else:
                integers = torch.vstack((integers, torch.arange(prev_i, device=DEVICE)
                                         .repeat(integers_dim, 1).view(-1, 1)))
                prev_i = i
                integers_dim = 1
        integers = torch.vstack((integers, torch.arange(prev_i, device=DEVICE).repeat(integers_dim, 1).view(-1, 1)))
        integers = integers.repeat(1, size_new_points[-1])

        index = torch_scatter.utils.broadcast(index, new_points, node_dim)
        idx_clone = (index * k + integers).clone()
        out_new_points = torch.zeros(size_new_points, device=DEVICE).scatter_(node_dim, idx_clone, new_points).view(
            [-1, k, size_new_points[-1]])
        size_weights = [dim_size * k, int(list(weights.size())[-1])]
        out_weights = torch.zeros(size_weights, device=DEVICE).scatter_(node_dim, idx_clone[:, :size_weights[-1]],
                                                                        weights).view([-1, k, size_weights[-1]])
        if density_scale is not None:
            out_new_points *= density_scale.view(-1, 1, 1)
        return out_new_points.permute(0, 2, 1).matmul(out_weights).view([dim_size, -1])


class PointConvDensity(PointConv):

    def __init__(self, ratio, k, weight_net=None, local_nn=None, global_nn=None, density_nn=None, kernel_params=None):
        super(PointConvDensity, self).__init__(ratio, k, weight_net=weight_net, local_nn=local_nn,
                                               global_nn=global_nn)
        self.density_net = DensityNet(nn=density_nn, kernel_params=kernel_params)

    # noinspection PyMethodOverriding
    def aggregate(self, msg_output, knn_idx, pos, batch, fps_idx, fps_idx_shape):
        density_scale = self.density_net.forward(pos, batch, fps_idx)
        return super(PointConvDensity, self).aggregate(msg_output, knn_idx, fps_idx_shape,
                                                       density_scale=density_scale)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError
