"""
Contains definitions of custom layers for the particle-net model
"""


import tensorflow as tf


class KNeighborSelect(tf.keras.layers.Layer):
    """
    Layer to select the k-nearest neighbors
    """

    def __init__(self, k, num_points, **kwargs):
        """
        Initialize the parameters for this layer
        """
        super(KNeighborSelect, self).__init__(**kwargs)
        self.k = k
        self.n = num_points

    def get_config(self):
        """
        Get saved config to save the model
        """
        return {'k': self.k, 'n': self.n}

    def _distance(self, A, B):
        with tf.name_scope('_distance'):
            # For all N rows: [(P_A, C) -> (P_A)], get A**2
            r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
            # For all N rows: [(P_B, C) -> (P_B)], get B**2
            r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
            # For all N rows: [(P_A, C) * (C, P_B) -> (P_A, P_B)]
            m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
            D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
            return D

    def _gather(self, topk_indices, features):
        with tf.name_scope('_gather'):
            queries_shape = tf.shape(
                features)  # Stores the literal shape tuple (N, P, C)
            batch_size = queries_shape[0]  # Stores the literal values N

            # N array stacked of size (P, K, 1) with all values equal to
            # the Batch ID, overall shape (N, P, K, 1)
            batch_indices = tf.tile(
                tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)),
                (1, self.n, self.k, 1))
            # Top Indices get expanded to the shape (N, P, K, 1)
            # The concatenated results is (N, P, K) pairs with
            # (jet_index (batch_index), closest_neighbor), net shape (N, P, K, 2)
            indices = tf.concat(
                [batch_indices,
                 tf.expand_dims(topk_indices, axis=3)], axis=3)
            # We chose all the C features of the top K neigbors for each particle,
            # done for the entire batch of N Output shape = (N, P, K, C)
            return tf.gather_nd(features, indices)

    def call(self, inputs):
        """
        Apply the layer on the inputs
        """
        points, features = inputs
        # Get the distance of all particle pairs: (N, P, P)
        D = self._distance(points, points)
        # Take the particle indices which are closest to us: (N, P, K+1)
        _, indices = tf.nn.top_k(-D, k=self.k + 1, name='topk')
        # Remove self from the list of nearest neigbors: (N, P, K)
        indices = indices[:, :, 1:]

        # Gives the filtered Feature Vectors: (N, P, K, C)
        knn_fts = self._gather(indices, features)
        # My Features repeated K times: (N, P, K, C)
        knn_fts_center = tf.tile(tf.expand_dims(features, axis=2),
                                 (1, 1, self.k, 1))

        # (My Feature, Neighbor - My Feature) for all (N, P, K): (N, P, K, 2*C)
        # We shall call this our net (input feature vector for the Edge Conv model)
        return tf.concat(
            [knn_fts, tf.subtract(knn_fts, knn_fts_center)], axis=-1)


class MaskOut(tf.keras.layers.Layer):
    def __init__(self, outside, **kwargs):
        """
        Initialize the parameters for this layer
        """
        super(MaskOut, self).__init__(**kwargs)
        self.outside = outside

    def get_config(self):
        """
        Get saved config to save the model
        """
        return {'outside': self.outside}

    def call(self, inputs):
        """
        Apply the layer on the inputs
        """
        data, mask = inputs
        mask = tf.cast(tf.not_equal(mask, 0),
                       dtype='float32')  # make valid positions to 1

        if self.outside:
            shift = tf.multiply(
                999.,
                tf.cast(tf.equal(mask, 0),
                        dtype='float32'))  # make non-valid positions to 999
            return tf.add(shift, data)
        else:
            return tf.multiply(data, mask)


class PoolingLayer(tf.keras.layers.Layer):
    def __init__(self, axis, pool_type='mean', **kwargs):
        """
        Initialize the parameters for this layer
        """
        super(PoolingLayer, self).__init__(**kwargs)
        self.axis = axis
        self.type = pool_type

    def get_config(self):
        """
        Get saved config to save the model
        """
        return {'axis': self.axis}

    def call(self, inputs):
        """
        Apply the layer on the inputs
        """
        if self.type == 'mean':
            return tf.reduce_mean(inputs, axis=self.axis)
        else:
            return tf.reduce_max(inputs, axis=self.axis)


class EdgeConvolution(tf.keras.layers.Layer):
    def __init__(self,
                 num_points,
                 k,
                 channels,
                 with_bn: bool = True,
                 activation='relu',
                 pooling='average',
                 **kwargs):
        """
        Initialize the parameters for this layer
        """
        super(EdgeConvolution, self).__init__(**kwargs)
        self.num_points = num_points
        self.k = k
        self.channels = channels
        self.with_bn = with_bn
        self.activation = activation
        self.pooling = pooling

        # Inisitalize the trainable layers
        self.layers_conv = []
        self.layers_bn = []
        for idx, channel in enumerate(self.channels):
            self.layers_conv.append(
                tf.keras.layers.Conv2D(
                    channel,
                    kernel_size=(1, 1),
                    strides=1,
                    use_bias=False if self.with_bn else True,
                    kernel_initializer='glorot_normal',
                    name='%s/conv_%d' % (self.name, idx)))
            self.layers_bn.append(
                tf.keras.layers.BatchNormalization(name='%s/bn_%d' %
                                                   (self.name, idx)))
        self.layer_shortcut = tf.keras.layers.Conv2D(
            self.channels[-1],
            kernel_size=(1, 1),
            strides=1,
            use_bias=False if self.with_bn else True,
            kernel_initializer='glorot_normal',
            name='%s/sc_conv' % self.name)
        self.layers_bn.append(
            tf.keras.layers.BatchNormalization(name='%s/sc_bn' % self.name))

    def get_config(self):
        """
        Get saved config to save the model
        """
        return {
            'num_points': self.num_points,
            'k': self.k,
            'channels': self.channels,
            'with_bn': self.with_bn,
            'activation': self.activation,
            'pooling': self.pooling
        }

    def call(self, inputs):
        """
        Apply the layer on the inputs
        """
        points, features = inputs
        net = KNeighborSelect(self.k, self.num_points)([points, features])

        for idx, channel in enumerate(self.channels):
            net = self.layers_conv[idx](net)
            if self.with_bn:
                net = self.layers_bn[idx](net)
            if self.activation:
                net = tf.keras.layers.Activation(self.activation,
                                                 name='%s/act_%d' %
                                                 (self.name, idx))(net)
            features = PoolingLayer(
                axis=2)(net) if self.pooling == 'max' else tf.reduce_mean(
                    net, axis=2)  # (N, P, C')

        shortcut = self.layer_shortcut(tf.expand_dims(features, axis=2))
        if self.with_bn:
            shortcut = self.layers_bn[-1](shortcut)
        shortcut = tf.squeeze(shortcut, axis=2)
        return tf.keras.layers.Activation(
            self.activation, name='%s/sc_act' %
            self.name)(shortcut + features) if self.activation else (
                shortcut + features)  # (N, P, C')
