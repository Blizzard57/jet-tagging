import tensorflow as tf

from mlphile.models.particle_net.layers import EdgeConvolution


class ParticleNet():

    def __init__(self, input_shapes, num_classes, model_type='particlenet-lite'):
        """
        Make the ParticleNet model

        Parameters
        ----------
        num_classes: int
            Number of output classes
        input_shapes: dict
            The shapes of each input (`points`, `features`, `mask`)
        model_type: str
            The model to build, lite or full

        Example
        -------
        >>> ParticleNet(2, {'points': (100, 2), 'features': (100, 4), 'mask': (100, 1)})
        """
        assert model_type in ['particlenet-lite', 'particlenet-full'], 'Model Name not in list.'

        self.settings = {
            'num_class': num_classes,
            'conv_params':
                [
                    (12, (32, 32, 32)),
                    (12, (64, 64, 64)),
                ] if model_type == 'particlenet-lite' else
                [
                    (16, (64, 64, 64)),
                    (16, (128, 128, 128)),
                    (16, (256, 256, 256)),
                ] if model_type == 'particlenet-full' else
                [],
            'conv_pooling': 'average',
            'fc_params':
                [(128, 0.1)] if model_type == 'particlenet-lite' else
                [(256, 0.1)] if model_type == 'particlenet-full' else
                [],            
            'num_points': input_shapes['points'][0]
        }

        self.layer_points = tf.keras.Input(
            name='points', shape=input_shapes['points'])
        self.layer_features = tf.keras.Input(
            name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
        self.layer_mask = tf.keras.Input(
            name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None

        self.layer_output = self._make_model()

        self.model = tf.keras.models.Model(
            inputs = [self.layer_points, self.layer_features, self.layer_mask], 
            outputs = [self.layer_output], 
            name = 'ParticleNet'
        )

    def _make_model(self):
        """
        Parameters
        ----------
        points: tf.keras.layers.Input, (N, P, C_coord)
            keras input layer for the point
        features: tf.keras.layers.Input, (N, P, C_features), optional
            keras input layer for the features
        mask: tf.keras.layers.Input, (N, P, 1), optinal:
            keras input layer for the mask
        setting: dict
            stores some Hyper parameters in a dictionary
        name: str
            name of the model to be returned (prefix of all parts of this model)

        Notes
        -----
        These are the features included in each of the input facets:
        'points': ['part_etarel', 'part_phirel'],
        'features': ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel'],
        'mask': ['part_pt_log']

        Points exist on the Eta-phi plane, we can choose to augment the information
        with the log of energy and transverse momentum
        """

        with tf.name_scope('particle_net'):

            # If explicit features are not passed, then the features are the points
            if self.layer_features is None:
                self.layer_features = self.layer_points

            # Makes a boolean mask out of the input mask, 1 if it's valid (in the mask), 999 if it's not.
            # Represents a padding particle or Transverse momentum is 1
            # TODO: Find out if an why Transverse Momentum = 1 needs to be checked.
            if self.layer_mask is not None:
                # make valid positions to 1
                mask = tf.cast(tf.not_equal(self.layer_mask, 0), dtype='float32')
                # make non-valid positions to 999
                coord_shift = tf.multiply(999., tf.cast(tf.equal(self.layer_mask, 0), dtype='float32'))
            else:
                raise NotImplementedError('Implement an allow-all mask')

            fts = tf.keras.layers.BatchNormalization(name='fts_bn')(self.layer_features)

            # Extracting CONV Parameters (K, (C1, C2, C3)) from the settings we passed in
            for layer_idx, layer_param in enumerate(self.settings['conv_params']):
                K, channels = layer_param
                # Masked out points get a (+999, +999) - coordinate shift on the 
                # eta-phi plane, after each layer
                pts = tf.add(coord_shift, self.layer_points) if layer_idx == 0 else tf.add(coord_shift, fts)
                # Now we repeat edge-conv over all layers starting with (Points, Features) and
                # then the next layers are (Previous Layer masked, Previous Layer)
                fts = EdgeConvolution(
                    self.settings['num_points'], 
                    K,
                    channels,
                    with_bn=True,
                    activation='relu',
                    pooling=self.settings['conv_pooling'],
                    name='%s_%d' % ('edge_conv', layer_idx)
                )([pts, fts])

            # Filter out the masked out particles to 0
            if mask is not None:
                fts = tf.multiply(fts, mask)

            # Output shape: (N, C) - Takes the average of all the particles in the jet
            pool = tf.reduce_mean(fts, axis=1)  

            # Extracting CONV Parameters (Units, Dropout) from the settings we passed in
            if self.settings['fc_params'] is not None:
                x = pool
                for layer_idx, layer_param in enumerate(self.settings['fc_params']):
                    units, drop_rate = layer_param
                    # Add the Dense and Dropout layers
                    x = tf.keras.layers.Dense(units, activation='relu')(x)
                    if drop_rate is not None and drop_rate > 0:
                        x = tf.keras.layers.Dropout(drop_rate)(x)
                # Final Classification layer
                out = tf.keras.layers.Dense(self.settings['num_class'], activation='softmax')(x)
                return out  # (N, num_classes)
            else:
                return pool