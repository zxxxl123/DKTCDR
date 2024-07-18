import tensorflow as tf
from abc import abstractmethod
'''
2019 by Hongwei Wang.
https://github.com/hwwang55/KGCN
'''
LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, size):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, size)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, size):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        # neighbor_vectors[1]:(65536, 4, 32)===>(65536, 1,4, 32)
        # neighbor_vectors[2]:(65536, 16, 32)===>(65536, 4,4, 32)

        # relation_vectors[0]:(65536, 4, 32)==>(65536,1,4,32)
        # relation_vectors[1]: (65536, 16, 32)==>(65536,4,4,32)
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            #user_embeddings:[65536,32]==>[65536,1,1,32]
            # user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            #[65536,1,1,32]*[65536,1,4,32]===>[65536,1,4]
            #[65536,1,1,32]*[65536,4,4,32]===>[65536,4,4]
            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)

            #[65536,1,4]
            #[65536,4,4]
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1]===>[65536,1,4,1]
            # [batch_size, -1, n_neighbor, 1]===>[65536,4,4,1]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            #[65536,1,4,1]*[65536,1,4,32]===>[65536,1,32]
            #[65536,4,4,1]*[65536,4,4,32]===>[65536,4,32]
            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=3)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=3)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.keras.initializers.glorot_normal(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, size):
        # [batch_size, -1, dim]
        # [65536,1,32]
        # [65536,4,32]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        #(65536, 1, 32)+[65536,1,32]===>[65536,1,32]===>[65536,32]
        #(65536, 4, 32)+[65536,4,32]===>[65536,4,32]===>[262144,32]
        output = tf.reshape(tf.squeeze(tf.reshape(self_vectors, [self.batch_size, size, -1, self.dim])) + tf.squeeze(tf.reshape(neighbors_agg, [self.batch_size, size, -1, self.dim])), [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        #[65536,1,32]
        #[65536,4,32]
        output = tf.reshape(output, [self.batch_size, size, -1, self.dim])

        return self.act(output)
    
class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(ConcatAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.keras.initializers.glorot_normal(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, size):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [batch_size, -1, dim * 2]
        output = tf.concat([tf.squeeze(tf.reshape(self_vectors, [self.batch_size, size, -1, self.dim])), tf.squeeze(tf.reshape(neighbors_agg, [self.batch_size, size, -1, self.dim]))], axis=-1)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, size, -1, self.dim])

        return self.act(output)

class NeighborAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(NeighborAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.keras.initializers.glorot_normal(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, size):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, size, -1, self.dim])

        return self.act(output)