import tensorflow as tf

from phymlq.ml.particle_net.layers import EdgeConvolution, PoolingLayer, MaskOut


class ParticleNet(tf.keras.models.Model):

    def __init__(self, 
                 input_shapes={'points': (100, 2), 'features': (100, 4), 'mask': (100, 1)}, 
                 num_classes=2, 
                 model_type='particlenet-lite'):
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
        >>> ParticleNet({'points': (100, 2), 'features': (100, 4), 'mask': (100, 1)}, 2)
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

        super(ParticleNet, self).__init__(
            inputs = [self.layer_points, self.layer_features, self.layer_mask], 
            outputs = [self.layer_output], 
            name = model_type
        )
        self.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                     metrics=['accuracy'])


    def fit_dataset(self, dataset, val_dataset=None, batch_size=None,
                    checkpoint_file='checkpoints/best.h5', tensorboard_dir='logs/'):
        batch_size = 1024 if 'lite' in self.name else 384
        epochs = 30

        def lr_schedule(epoch):
            return 1e-3 if epoch <= 10 else 1e-4 if epoch <= 20 else 1e-5

        dataset.shuffle()
        self.fit(
            dataset.X, dataset.y,
            batch_size=batch_size,
            epochs=10,
            validation_data=(val_dataset.X, val_dataset.y),
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_file,
                    monitor='val_loss', verbose=True, save_best_only=True,
                    save_weights_only=False, mode='auto', save_freq='epoch'
                ),
                tf.keras.callbacks.LearningRateScheduler(lr_schedule),
                tf.keras.callbacks.ProgbarLogger('steps'),
                tf.keras.callbacks.TensorBoard(tensorboard_dir, histogram_freq=1)
            ]
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

            fts = tf.keras.layers.BatchNormalization(name='bn_features')(self.layer_features)

            for layer_idx, layer_param in enumerate(self.settings['conv_params']):
                K, channels = layer_param
                pts = MaskOut(True, name='mask_%d'%(layer_idx))(
                    [self.layer_points if layer_idx == 0 else fts, self.layer_mask])
                fts = EdgeConvolution(
                    self.settings['num_points'], 
                    K,
                    channels,
                    with_bn=True,
                    activation='relu',
                    pooling=self.settings['conv_pooling'],
                    name='%s_%d' % ('edge_conv', layer_idx)
                )([pts, fts])

            if self.layer_mask is not None:
                fts = MaskOut(False, name='mask_f')([fts, self.layer_mask])
            x = PoolingLayer(axis=1, pool_type='mean', name='average_pool')(fts)

            if self.settings['fc_params'] is not None:
                for layer_idx, layer_param in enumerate(self.settings['fc_params']):
                    units, drop_rate = layer_param
                    x = tf.keras.layers.Dense(units, activation='relu')(x)
                    if drop_rate is not None and drop_rate > 0:
                        x = tf.keras.layers.Dropout(drop_rate)(x)
                x = tf.keras.layers.Dense(self.settings['num_class'], activation='softmax')(x)
                return x  # (N, num_classes)
            else:
                return x