import abc
import torch
from torch import Tensor
from torch_sparse import SparseTensor

from phymlq.hyperparams import DEVICE
from phymlq.layers.pointconv import PointConvSetAbstraction


class LieGroup(abc.ABC):
    """
    Abstract class to represent all Lie Groups
    """

    rep_dim = NotImplemented  # dimension on which G acts (eg. 2 for SO(2)) (=initial dims)
    lie_dim = NotImplemented  # dimension of the lie algebra of G (eg. 1 for SO(2))
    q_dim = NotImplemented  # dimension which the quotient space X/G is embedded. (eg. 1 for SO(2) acting on R2)

    def __init__(self, alpha=0.2):
        """
        :param alpha: Weighting parameter for distance
        """
        super(LieGroup, self).__init__()
        self.alpha = alpha

    def exp(self, a):
        """
        Matrix exponentiation on Lie Algebra, i.e out = exp(sum a_i A_i) where
        A_i is the exponential generator of G.
        https://arxiv.org/pdf/2002.12880.pdf page 3 - paragraph 3

        :param a: Matrix of shape (-1, lie_dims)
        :return: Matrix of shape (-1, rep_dims, rep_dims), with value exp(a)
        """
        raise NotImplementedError

    def log(self, u):
        """
        Matrix logarithm for collection of matrices and converts to lie algebra basis
        :param u: [u (*, rep_dim, rep_dim)]
        :return: [coefficients of log(u) in basis (*, d)]
        """
        raise NotImplementedError

    def lifted_elements(self, pos, n_samples):
        """
        Takes in the coordinates pos, and lifts them to the Lie algebra elements
        in basis (a) and embedded orbit identifiers q. For groups where lifting is
        multivalued specify n_samples > 1 as lifts do for each point
        :param pos: [pos (*, n, rep_dim)]
        :param n_samples: TODO find out what it is
        :return: [a (*, n * n_samples, lie_dim)], [q (*, n * n_samples, q_dim)]
        """
        raise NotImplementedError

    def inv(self, g):
        """
        Computes the inverse of elements of g (*, rep_dim, rep_dim) as exp(-log(g))
        :param g: TODO
        """
        return self.exp(-self.log(g))

    def distance(self, abq_pairs):
        """
        Computes distance of size(*) from [abq_pairs (*, lie_dim + 2*q_dim)]
        Simply computes alpha*norm(log(v^{-1}u)) + (1 - alpha) * norm(q_a - q_b),
        combined distance from group element distance and orbit distance
        """
        ab_dist = torch.norm(abq_pairs[..., :self.lie_dim], dim=-1)
        qa = abq_pairs[..., self.lie_dim:self.lie_dim + self.q_dim]
        qb = abq_pairs[..., self.lie_dim + self.q_dim: self.lie_dim + 2 * self.q_dim]
        qa_qb_dist = torch.norm(qa - qb, dim=-1)
        return self.alpha * ab_dist + (1 - self.alpha) * qa_qb_dist

    def lift(self, pos, x, n_samples):
        """
        Assumes pos (*, n, 2), x (*, n, c)
        returns (a, x) with shape [(*, n * n_samples, lie_dim), (*, n * n_samples, c)]
        """

        expanded_a, expanded_q = self.lifted_elements(pos, n_samples)  # (*, n * n_samples, ld), (*, n * n_samples, qd)

        # Expand x as num_samples
        expanded_x = x[..., None, :].repeat((1,) * len(x.shape[:-1] + (n_samples, 1)))
        expanded_x = expanded_x.reshape(*expanded_a.shape[:-1], x.shape[-1])

        paired_a = self.quotient(expanded_a)

        if expanded_q is not None:
            q_in = expanded_q.unsqueeze(-2).expand(*paired_a.shape[:-1], 1)
            q_out = expanded_q.unsqueeze(-3).expand(*paired_a.shape[:-1], 1)
            embedded_locations = torch.cat([paired_a, q_in, q_out], dim=-1)
        else:
            embedded_locations = paired_a

        return embedded_locations, expanded_x

    def quotient(self, a):
        """
        Computes log(e^{-b} e^a) for all a b pairs along n dimension of the input
        :param a: Tensor of shape (*, n, lie_dims)
        :returns: pairs_ab, Tensor of shape (*, n, n, lie_dims)
        """
        v = self.exp(-a.unsqueeze(2))  # inverse of the b vector
        u = self.exp(a.unsqueeze(1))  # the a vector transposed
        return self.log(v @ u)


class SO2Group(LieGroup):
    """
    The Lie algebra is theta
    The Lie group is the set of rotations angle theta
    The Quotient group is the set of shifts by radius r
    """
    lie_dim = 1
    rep_dim = 2
    q_dim = 1

    def exp(self, a):
        """
        Take the element from the Lie algebra to the Lie group
        :param a: Element of the SO2 algebra
        :returns: Element of the SO2 group
        """
        r = torch.zeros(*a.shape[:-1], 2, 2, device=a.device, dtype=a.dtype)
        sin = a[..., 0].sin()
        cos = a[..., 0].cos()
        r[..., 0, 0] = cos
        r[..., 1, 1] = cos
        r[..., 0, 1] = -sin
        r[..., 1, 0] = sin
        return r

    def log(self, r):
        """
        Take the element from the Lie group to the Lie algebra
        :param r: Element of the SO2 group
        :returns: Element of the SO2 algebra
        """
        return torch.atan2(r[..., 1, 0] - r[..., 0, 1], r[..., 0, 0] + r[..., 1, 1])[..., None]  # atan(a/b)

    def lifted_elements(self, pos, n_samples=1):
        """
        Take the position in x,y and gives r,theta
        :param pos: Tensor, position in x, y plane
        :param n_samples: Not Used here, number of times to sample in non-Abelian groups
        """
        assert n_samples == 1, "Abelian Group, no need for n_samples"
        assert pos.shape[-1] == 2, "Lifting from R^2 to SO(2) supported only"
        r = torch.norm(pos, dim=-1).unsqueeze(-1)
        theta = torch.atan2(pos[..., 1], pos[..., 0]).unsqueeze(-1)
        return theta, r

    def distance(self, ab_pairs):
        """
        Gives the weighted sum of distances between the a-b pairs in the quotient group and the lie group
        (distance between radial coordinates)
        :param ab_pairs: tensor, pairs of points, in radial coordinates
        :returns: tensor, distance between the points
        """
        angle_pairs = ab_pairs[..., 0]
        ra = ab_pairs[..., 1]
        rb = ab_pairs[..., 2]
        return self.alpha * angle_pairs.abs() + (1 - self.alpha) * (ra - rb).abs() / (ra + rb + 1e-3)


class LieConv(PointConvSetAbstraction):

    def __init__(self, ratio, k, group, weight_net=None, local_nn=None, global_nn=None):
        super(LieConv, self).__init__(ratio, k, weight_net=weight_net, local_nn=local_nn, global_nn=global_nn)
        self.group = group

    def forward(self, pos, x, batch):
        fps_idx, knn_idx, knn_edges, knn_abq_pairs, vals = self.process(pos, x, batch, self.ratio, self.k, self.group)
        out = self.propagate(knn_edges, x=vals, knn_abq_pairs=knn_abq_pairs, knn_idx=knn_idx, fps_idx=fps_idx,
                             fps_idx_shape=fps_idx.shape[0])
        if self.global_nn is not None:
            out = self.global_nn(out)

        return out, fps_idx

    # noinspection PyMethodOverriding
    def message(self, knn_abq_pairs, x_j):
        grouped_norm = knn_abq_pairs
        msg = torch.cat([grouped_norm.clone(), x_j], dim=1) if x_j is not None else grouped_norm
        if self.weight_net is not None:
            grouped_norm = self.weight_net(grouped_norm)
        if self.local_nn is not None:
            return self.local_nn(msg), msg, grouped_norm
        return msg, msg, grouped_norm

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def fps(ratio, dists):
        num_nodes = dists.shape[0]
        m = int(torch.round(torch.Tensor([ratio * num_nodes])))
        dists += -1e8 * torch.eye(num_nodes, device=DEVICE)

        fps_idx = torch.zeros(m, dtype=torch.long)
        fps_idx[0] = torch.randint(low=0, high=num_nodes, size=(1,))

        for k in range(1, m):
            dists[:, fps_idx[k - 1]] = -1e-8
            fps_idx[k] = torch.argmax(dists[fps_idx[k - 1]])

        fps_idx = torch.sort(fps_idx)[0]

        return fps_idx.long()

    @staticmethod
    def knn(abq_pairs, q_idx, k: int, dists):
        k = int(torch.min(torch.Tensor([k, abq_pairs.shape[0]])))
        _, top_k_idx = torch.topk(dists[q_idx], k, dim=-1, largest=False, sorted=False)

        new_abq = abq_pairs[q_idx]
        assert abq_pairs.shape[1] == dists[q_idx].shape[1]

        knn_abq_pairs = torch.zeros([new_abq.shape[0], k, new_abq.shape[-1]])

        for i in range(new_abq.shape[-1]):
            knn_abq_pairs[:, :, i] = torch.gather(new_abq[:, :, i], 1, top_k_idx)

        integers = torch.arange(q_idx.shape[0], device=DEVICE).view(-1, 1).repeat(1, k).view(1, -1)
        knn_edges = torch.vstack((integers, top_k_idx.view(1, -1)))

        return knn_edges.long(), knn_abq_pairs.view(-1, knn_abq_pairs.shape[-1])

    @classmethod
    def process(cls, pos, x, batch, ratio, k, group):
        batch_size = int(batch.max() + 1)
        deg = pos.new_zeros(batch_size, dtype=torch.long)
        deg.scatter_add_(0, batch, torch.ones_like(batch))
        ptr = deg.new_zeros(batch_size + 1)
        torch.cumsum(deg, 0, out=ptr[1:])

        fps_count = 0

        batch_fps_idx, batch_knn_idx, batch_knn_edges, batch_knn_abq_pairs, batch_vals = \
            torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

        for i in range(batch_size):
            pos_new = pos[ptr[i - 1]: ptr[i]]
            x_new = x[ptr[i - 1]: ptr[i]]

            abq_pairs, vals = group.lift(pos_new.unsqueeze(0), x_new.unsqueeze(0), 1)
            abq_pairs = abq_pairs.squeeze(0).to(DEVICE)
            vals = vals.squeeze(0).to(DEVICE)
            dists = group.distance(abq_pairs).to(DEVICE)
            fps_idx = cls.fps(ratio, dists.clone()).to(DEVICE)  # Tensor of shape (ratio * n,)
            knn_edges, knn_abq_pairs = cls.knn(abq_pairs, fps_idx, k, dists)  # (ratio * n, k, lie_dims + 2 * q_dim)

            fps_idx += ptr[i - 1]
            knn_idx = (knn_edges[0] + fps_count).clone()
            fps_count += fps_idx.shape[0]
            knn_edges[0] = fps_idx[knn_edges[0]]
            knn_edges[1] += ptr[i - 1]

            batch_fps_idx = torch.hstack((batch_fps_idx, fps_idx))
            batch_knn_idx = torch.hstack((batch_knn_idx, knn_idx))
            batch_knn_edges = torch.hstack((batch_knn_edges, knn_edges))
            batch_knn_abq_pairs = torch.vstack((batch_knn_abq_pairs, knn_abq_pairs))
            batch_vals = torch.vstack((batch_vals, vals))

        return batch_fps_idx, batch_knn_idx, batch_knn_edges, batch_knn_abq_pairs, batch_vals
