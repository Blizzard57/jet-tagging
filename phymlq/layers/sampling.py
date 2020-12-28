import typing
import torch


def fps(pos, batch, ratio, distance: typing.Optional[typing.Callable] = None):
    """
    :param pos:
    :param batch:
    :param ratio:
    :param distance:
    """
    batch_size = int(batch.max() + 1)

    deg = pos.new_zeros(batch_size, dtype=torch.long)
    deg.scatter_add_(0, batch, torch.ones_like(batch))
    ptr = deg.new_zeros(batch_size + 1)
    torch.cumsum(deg, 0, out=ptr[1:])

    batch_fps_idx = None
    for i in range(batch_size):
        pos_new = pos[ptr[i]: ptr[i + 1]]
        dists = torch.cdist(pos_new, pos_new) if distance is None else distance(pos_new, pos_new)
        num_nodes = pos_new.shape[0]
        m = int(torch.round(torch.Tensor([ratio * num_nodes])))
        dists += -1e8 * torch.eye(num_nodes)
        fps_idx = torch.zeros(m, dtype=torch.int64)
        fps_idx[0] = torch.randint(low=0, high=num_nodes, size=(1,))

        for k in range(1, m):
            dists[:, fps_idx[k - 1]] = -1e-8
            fps_idx[k] = torch.argmax(dists[fps_idx[k - 1]])

        fps_idx = torch.sort(fps_idx)[0] + ptr[i]
        batch_fps_idx = fps_idx if batch_fps_idx is None else torch.hstack((batch_fps_idx, fps_idx))
    # noinspection PyTypeChecker
    return batch_fps_idx.type(torch.long)


def knn(x, y, k: int, batch_x, batch_y, distance: typing.Optional[typing.Callable] = None):
    """
    :param x:
    :param y:
    :param k:
    :param batch_x:
    :param batch_y:
    :param distance:
    """
    batch_size = int(batch_x.max() + 1)

    deg = x.new_zeros(batch_size, dtype=torch.long)
    deg.scatter_add_(0, batch_x, torch.ones_like(batch_x))
    ptr_x = deg.new_zeros(batch_size + 1)
    torch.cumsum(deg, 0, out=ptr_x[1:])

    deg = y.new_zeros(batch_size, dtype=torch.long)
    deg.scatter_add_(0, batch_y, torch.ones_like(batch_y))
    ptr_y = deg.new_zeros(batch_size + 1)
    torch.cumsum(deg, 0, out=ptr_y[1:])
    coo = None

    for i in range(batch_size):
        x_new = x[ptr_x[i]:ptr_x[i + 1]]
        y_new = y[ptr_y[i]:ptr_y[i + 1]]

        k = int(torch.min(torch.Tensor([k, x_new.shape[0]])))
        dist = torch.cdist(y_new, x_new, p=2.0) if distance is None else distance(y_new, x_new)  # P x R
        _, top_k_idx = torch.topk(dist, k, dim=-1, largest=False, sorted=False)
        integers = (torch.arange(y_new.shape[0]) + ptr_y[i]).view(-1, 1).repeat(1, k).view(1, -1)
        out = torch.vstack((integers, (top_k_idx + ptr_x[i]).view(1, -1)))
        coo = out if coo is None else torch.hstack((coo, out))

    # noinspection PyTypeChecker
    return coo.type(torch.long)
