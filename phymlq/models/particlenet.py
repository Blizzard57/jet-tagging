import torch
import torch_geometric.nn

from phymlq.layers.edgeconv import ParticleDynamicEdgeConv
from phymlq.hyperparams import DEVICE


class ParticleNet(torch.nn.Module):

    def __init__(self, settings=None):
        super().__init__()
        if settings is None:
            settings = {
                "conv_params": [
                    (16, (64, 64, 64)),
                    (16, (128, 128, 128)),
                    (16, (256, 256, 256)),
                ],
                "fc_params": [
                    (0.1, 256)
                ],
                "input_features": 4,
                "output_classes": 2,
            }

        previous_output_shape = settings['input_features']
        self.input_bn = torch_geometric.nn.BatchNorm(settings['input_features'])

        self.conv_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['conv_params']):
            k, channels = layer_param
            self.conv_process.append(ParticleDynamicEdgeConv(previous_output_shape, channels, k=k).to(DEVICE))
            previous_output_shape = channels[-1]

        self.fc_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['fc_params']):
            drop_rate, units = layer_param
            # noinspection PyTypeChecker
            seq = torch.nn.Sequential(
                torch.nn.Linear(previous_output_shape, units),
                torch.nn.Dropout(p=drop_rate),
                torch.nn.ReLU()
            ).to(DEVICE)
            self.fc_process.append(seq)
            previous_output_shape = units

        self.output_mlp_linear = torch.nn.Linear(previous_output_shape, settings['output_classes'])
        self.output_activation = torch.nn.Softmax(dim=1)

    def forward(self, batch):
        fts = self.input_bn(batch.x)
        pts = batch.pos

        for idx, layer in enumerate(self.conv_process):
            fts = layer(pts, fts, batch.batch)
            pts = fts

        x = torch_geometric.nn.global_mean_pool(fts, batch.batch)
        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)
        x = self.output_activation(x)
        return x
