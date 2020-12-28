import logging
import os

import numpy as np
import pandas as pd
import uproot_methods
import torch
import torch_geometric

from phymlq import hyperparams
from phymlq.utils.download import download


class TopTaggingDataset(object):

    directory = os.path.join(hyperparams.PROJECT_DIR, 'data', 'top_tagging')

    def __init__(self, filename, n_particles=100):
        self.n_particles = n_particles
        self.filename = os.path.join(self.directory, filename)
        self.data = np.load(self.filename)

    @classmethod
    def download_files(cls):
        """
        Downloads the Original Data Files from the Zenodo website
        """
        os.makedirs(cls.directory, exist_ok=True)
        download('https://zenodo.org/record/2603256/files/train.h5?download=1',
                 os.path.join(cls.directory, '_original_train.h5'))
        download('https://zenodo.org/record/2603256/files/val.h5?download=1',
                 os.path.join(cls.directory, '_original_val.h5'))
        download('https://zenodo.org/record/2603256/files/test.h5?download=1',
                 os.path.join(cls.directory, '_original_test.h5'))
        cls._transform_datafiles('train')
        cls._transform_datafiles('val')
        cls._transform_datafiles('test')

    @classmethod
    def _transform_datafiles(cls, file, chunk_size=300, restrict_files=3):
        """
        Converts DataFrame into Awkward array.
        Batches into smaller Awkward files.
        :param file: str, train, val or test; _original_file.h5 should be present in the directory
        :param chunk_size: int, Number of rows per awkward file, None for all rows in 1 file
        :param restrict_files: int, max number of files to create from a single dataframe iterator
        """
        def generate_col_list_with_prefix(prefix, max_particles=200):
            return ['%s_%d' % (prefix, i) for i in range(max_particles)]

        input_file = os.path.join(cls.directory, '_original_%s.h5' % file)
        output_basename = os.path.join(cls.directory, file)

        frames = pd.read_hdf(input_file, key='table', iterator=True, chunksize=chunk_size)
        # noinspection PyTypeChecker
        for idx, df in enumerate(frames):
            if restrict_files == idx:
                del frames
                break
            output_file = '%s_%d.npz' % (output_basename, idx + 1)
            if os.path.exists(output_file):
                logging.info('File %s already exists' % output_file)
                continue

            px = df[generate_col_list_with_prefix('PX')].values
            py = df[generate_col_list_with_prefix('PY')].values
            pz = df[generate_col_list_with_prefix('PZ')].values
            en = df[generate_col_list_with_prefix('E')].values
            n_particles = np.sum(en > 0, axis=1)
            p4 = uproot_methods.TLorentzVectorArray(px, py, pz, en)
            jet_p4 = p4.sum()
            mask = en > 0

            res = {
                'label_top': df['is_signal_new'].values,
                'label_qcd': 1 - df['is_signal_new'].values,
                'part_pt_log': np.log(p4.pt + mask),  # TODO: take log here
                'part_e_log': np.log(en + mask),  # TODO: take log here
                'part_eta_rel': (p4.eta - jet_p4.eta) * ((np.sign(jet_p4.eta) >= 0) * 2 - 1),
                'part_phi_rel': p4.delta_phi(jet_p4),
                'n_particles': n_particles,
            }
            np.savez(output_file, **res)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, key):
        return self.data[key]

    @property
    def points(self):
        return np.stack([self.data['part_eta_rel'], self.data['part_phi_rel']], axis=-1)

    @property
    def features(self):
        return np.stack([self.data['part_pt_log'], self.data['part_e_log'],
                         self.data['part_eta_rel'], self.data['part_phi_rel']], axis=-1)

    @property
    def mask(self):
        return np.expand_dims(self.data['part_e_log'] > 0, axis=-1)

    @property
    def label_top(self):
        return self.data['label_top']

    @property
    def label_qcd(self):
        return self.data['label_qcd']

    @property
    def x(self):
        return self.points, self.features, self.mask

    @property
    def y(self):
        return np.stack([self.label_top, self.label_qcd], axis=-1)


class TopTaggingGraphDataset(torch_geometric.data.InMemoryDataset):

    directory = os.path.join(hyperparams.PROJECT_DIR, "scratch", "data", "top_tagging")

    def __init__(self, raw_filename):
        self.filename, self.dataset = raw_filename, None
        os.makedirs(os.path.join(self.directory, "graphs"), exist_ok=True)
        self.dataset = TopTaggingDataset(self.filename)
        super(TopTaggingGraphDataset, self).__init__(self.directory, None, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        data_list = []
        ys = np.argmax(self.dataset.y, axis=1)
        for i in range(self.dataset.points.shape[0]):
            if np.where(self.dataset.mask[i] == 0)[0].shape[0] != 0:
                feat = torch.tensor(
                    self.dataset.features[i][:np.where(self.dataset.mask[i] == 0)[0][0]],
                    dtype=torch.float)
                pos = torch.tensor(
                    self.dataset.points[i][:np.where(self.dataset.mask[i] == 0)[0][0]],
                    dtype=torch.float)
            else:
                feat = torch.tensor(self.dataset.features[i], dtype=torch.float)
                pos = torch.tensor(self.dataset.points[i], dtype=torch.float)

            data_list.append(torch_geometric.data.Data(x=feat, pos=pos, y=ys[i]))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        raise NotImplementedError

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [os.path.join(self.directory, "graphs", self.filename)]

    def get_loader(self):
        return torch_geometric.data.DataLoader(self, batch_size=128, shuffle=True)
