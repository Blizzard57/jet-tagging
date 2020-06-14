import logging
import os
import urllib

import numpy as np
import pandas as pd
import awkward
import uproot_methods


class TopTaggingPreprocessor(object):

    @staticmethod
    def download_dataset(store_directory):
        """
        Downloads the Original Data Files from the Zenodo website

        Parameters
        ----------
        store_directory: str
            The directory to store all the downloaded files into
        """
        urllib.request.urlretrieve(
            'https://zenodo.org/record/2603256/files/train.h5?download=1',
            os.path.join(store_directory, 'original_train.h5')
        )
        print('Downloaded train.h5')
        urllib.request.urlretrieve(
            'https://zenodo.org/record/2603256/files/val.h5?download=1',
            os.path.join(store_directory, 'original_val.h5')
        )
        print('Downloaded val.h5')
        urllib.request.urlretrieve(
            'https://zenodo.org/record/2603256/files/test.h5?download=1',
            os.path.join(store_directory, 'original_test.h5')
        )
        print('Downloaded test.h5')

    @staticmethod
    def _col_list(prefix, max_particles=200):
        return ['%s_%d'%(prefix, i) for i in range(max_particles)]

    @classmethod
    def _transform_dataframe(cls, df):
        """
        Takes a DataFrame and converts it into a Awkward array representation
        with features relevant to our model.

        Parameters
        ----------
        df: Pandas DataFrame
            The DataFrame with all the momenta-energy coordinates for all the particles
        
        Returns
        -------
        v: OrderedDict
            A Ordered Dictionary with all properties of interest

        Notes
        -----
        Here the function is just computing 4 quantities of interest:
        * Eta value relative to the jet
        * Phi value relative to the jet
        * Transverse Momentum of the Particle (log of it) 
        * Energy of the Particle (log of it) 
        """
        from collections import OrderedDict
        v = OrderedDict()

        # We take the values in the dataframe for all particles of a single event in each row
        # px, py, pz, e are in separate arrays
        _px = df[cls._col_list('PX')].values
        _py = df[cls._col_list('PY')].values
        _pz = df[cls._col_list('PZ')].values
        _en = df[cls._col_list('E')].values

        mask = _en > 0
        n_particles = np.sum(mask, axis=1)

        px = awkward.JaggedArray.fromcounts(n_particles, _px[mask])
        py = awkward.JaggedArray.fromcounts(n_particles, _py[mask])
        pz = awkward.JaggedArray.fromcounts(n_particles, _pz[mask])
        energy = awkward.JaggedArray.fromcounts(n_particles, _en[mask])

        p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, energy)
        jet_p4 = p4.sum()

        # Getting the Labels
        _label = df['is_signal_new'].values # the target labels, QCD or Top
        v['label'] = np.stack((_label, 1-_label), axis=-1) # Making it categorical [Top, QCD]
        # Transformed features relative to the Jet and log features
        v['part_pt_log'] = np.log(p4.pt)
        v['part_e_log'] = np.log(energy)
        # Flip particle ETA if Jet Eta is negative
        # Particle's phi relative to the Jet
        _jet_etasign = np.sign(jet_p4.eta)
        _jet_etasign[_jet_etasign==0] = 1
        v['part_etarel'] = (p4.eta - jet_p4.eta) * _jet_etasign
        v['part_phirel'] = p4.delta_phi(jet_p4)
        
    @classmethod
    def _convert_datafiles(cls, source_dir, dest_dir, basename, chunksize=1000000):
        """
        Converts the DataFrame into an Awkward array and performs the read-write
        operations for the same. Also performs Batching of the file into smaller
        Awkward files.

        Parameters
        ----------
        source_dir: str
            The location to the H5 file with the dataframe
        destdir_dir: str
            The location we need to write to
        basename: str
            Prefix for all the output file names
        chunksize: int
            Number of rows per awkward file, None for all rows in 1 file
        """
        frames = pd.read_hdf(source_dir, key='table', iterator=True, chunksize=100000)
        for idx, frame in enumerate(frames):
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            output = os.path.join(dest_dir, '%s_%d.awkd'%(basename, idx))
            logging.info(output)
            if os.path.exists(output):
                logging.warning('File already exist: Continuing.')
                continue
            awkward.save(output, cls._transform_dataframe(frame), mode='x')

    @classmethod
    def prepare_dataset(cls, directory, chunksize=1000000):
        cls.convert(os.path.join(directory, 'original_train.h5'), 
                    dest_dir=directory,
                    basename='train_file',
                    chunksize=chunksize)
        cls.convert(os.path.join(directory, 'original_val.h5'), 
                    dest_dir=os.path.join(directory, 'converted'), 
                    basename='val_file',
                    chunksize=chunksize)
        cls.convert(os.path.join(directory, 'original_test.h5'), 
                    dest_dir=os.path.join(directory, 'converted'), 
                    basename='test_file',
                    chunksize=chunksize)


class TopTaggingDataset(object):

    def __init__(self, filepath, value_cols = None, label_cols='label', pad_len=100, data_format='channel_first'):
        self.filepath = filepath
        self.value_cols = value_cols if value_cols is not None else {
            'points': ['part_etarel', 'part_phirel'],
            'features': ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel'],
            'mask': ['part_pt_log']
        }
        self.label_cols, self.pad_len = label_cols, pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values, self._label = {}, None
        self._load()

    @staticmethod
    def _pad_array(jagged_array, max_len, value=0., dtype='float32'):
        """
        Pads a Jagged array with value to make them equal to max_len along each column

        Parameters
        ----------
        jagged_array: awkward array
            The array to be padded
        max_len: int
            Number of columns in each row (truncate or extend to)
        value: int or float
            The value to pad with
        dtype: string or type
            Final Data type of array

        Returns
        -------
        rectangular_array: np.array
            Padded version of the input Jagged Array
        """
        rectangluar_array = np.full(shape=(len(jagged_array), max_len), fill_value=value, dtype=dtype)
        for idx, jagged_element in enumerate(jagged_array):
            if len(jagged_element) != 0:
                trunc = jagged_element[:max_len].astype(dtype)
                rectangluar_array[idx, :len(trunc)] = trunc
        return rectangluar_array

    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        counts = None
        with awkward.load(self.filepath) as a:
            # Load output labels from the awkward array
            self._label = a[self.label_cols]

            for k in self.value_cols:
                cols = self.value_cols[k]
                assert isinstance(cols, (list, tuple))
                arrs = []
                for col in cols:
                    if counts is None:
                        counts = a[col].counts
                    else:
                        assert np.array_equal(counts, a[col].counts)
                    arrs.append(self._pad_array(a[col], self.pad_len))
                self._values[k] = np.stack(arrs, axis=self.stack_axis)
        logging.info('Finished loading file %s' % self.filepath)


    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        return self._label if key == self.label_cols else self._values[key]
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        # Get a random permutation
        if seed is not None: np.random.seed(seed)
        shuffle_indices = np.random.permutation(self.__len__())
        # Reorder the table
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]