from abc import abstractmethod
import tensorflow as tf
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
import random
from transformer import *

'''
Modification of code based on https://github.com/lpworld/DASL
'''

class Model(object):
    def __init__(self, user_count, item_count, batch_size, pars, window):
        memory_window = window
        self.batch_size = batch_size
        self.memory_window = memory_window

        self.is_domain_a = True
        self.is_training = True
        self.l2_weight = 1e-7
        self.kge_dim = pars['kge_dim']
        self.user_dim = pars['kge_dim']
        self.maxlen = pars['maxlen']
        self.mask_id = pars['mask_id']
        self.sc_dim = 32
        self.source_head_num = pars['source_head']
        self.source_block_num = pars['source_block']
        self.target_head_num = pars['target_head']
        self.target_block_num = pars['target_block']
        self.sc = tf.convert_to_tensor(pars['sc'])
        self.gru_dim = self.user_dim+self.kge_dim

        self.n_users = user_count
        self.regs = 1e-5
        if pars['aggregator'] == 'sum':
            self.aggregator_class = SumAggregator
        elif pars['aggregator'] == 'concat':
            self.aggregator_class = ConcatAggregator
        elif pars['aggregator'] == 'neighbor':
            self.aggregator_class = NeighborAggregator
        self.adj_entity = tf.convert_to_tensor(pars['adj_entity'])
        self.adj_relation = tf.convert_to_tensor(pars['adj_relation'])
        self.adj_1_entity = tf.convert_to_tensor(pars['adj_1_entity'])
        self.adj_1_relation = tf.convert_to_tensor(pars['adj_1_relation'])
        self.adj_2_entity = tf.convert_to_tensor(pars['adj_2_entity'])
        self.adj_2_relation = tf.convert_to_tensor(pars['adj_2_relation'])
        self.n_entities1 = pars['n_entities']
        self.n_relations = pars['n_relations']
        self.neighbor_sample_size = pars['neighbor_sample_size']
        self.n_iter = pars['n_iter']
        self.kg_weight1 = dict()

        initializer = tf.keras.initializers.glorot_normal()
        
        self.u_list = []
        self.i_list = []
        self.len_list = []
        
        self.u_1 = tf.placeholder(tf.int32, [batch_size,]) # [Batch]
        self.i_1 = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.y_1 = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.hist_1 = tf.placeholder(tf.int32, [batch_size, None])
        self.len1 = tf.placeholder(tf.int32, [batch_size,])
        
        self.kg_weight1['entity_embed'] = tf.Variable(initializer([self.n_entities1, self.kge_dim]), trainable=True, name='entity_embed1')
        self.kg_weight1['relation_embed'] = tf.Variable(initializer([self.n_relations, self.kge_dim]), trainable=True, name='relation_embed1')

        self.u_2 = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.i_2 = tf.placeholder(tf.int32, [batch_size,])
        self.y_2 = tf.placeholder(tf.float32, [batch_size,]) # [B]

        self.lr = tf.placeholder(tf.float64, [])
        
        self.user_emb_w_1 = tf.get_variable("user_emb_w_1", [user_count, self.user_dim])
        self.user_b_1 = tf.get_variable("user_b_1", [user_count], initializer=tf.constant_initializer(0.0))
        self.item_b_1 = tf.get_variable("item_b_1", [item_count], initializer=tf.constant_initializer(0.0))
        self.user_b_2 = tf.get_variable("user_b_2", [user_count], initializer=tf.constant_initializer(0.0))
        self.item_b_2 = tf.get_variable("item_b_2", [item_count], initializer=tf.constant_initializer(0.0))
        
        self.doc_emb = self.reduction(self.sc)
        
        user_emb = tf.nn.embedding_lookup(self.user_emb_w_1, self.u_1)
        user_emb_21 = tf.nn.embedding_lookup(self.user_emb_w_1, self.u_2) # [B, H/2]
        with tf.variable_scope('user_transfer',reuse=tf.AUTO_REUSE):
            user_emb_2 = tf.layers.dense(user_emb_21, self.user_dim, name="ut1")
            user_emb_2 = tf.layers.dense(user_emb_2, self.user_dim, name="ut2")
            
        with tf.variable_scope('item_transfer',reuse=tf.AUTO_REUSE):
            tmp = tf.concat([tf.nn.embedding_lookup(self.kg_weight1['entity_embed'], tf.constant(list(range(item_count)))), self.doc_emb], axis=-1)
            others = tf.nn.embedding_lookup(self.kg_weight1['entity_embed'], tf.constant(list(range(item_count,self.n_entities1))))
            self.entity_embed = tf.layers.dense(tmp, self.kge_dim+self.sc_dim//2, name="it3")
            self.entity_embed = tf.layers.dense(self.entity_embed, self.kge_dim, name="it4")
            self.entity_embed = tf.concat([self.entity_embed, others], axis=0)
            
        train_entity_vectors1, train_relation_vectors1 = self.get_neighbors(tf.expand_dims(self.i_1, 1), 1, self.adj_entity, self.adj_relation, self.adj_entity, self.adj_relation, self.adj_entity, self.adj_relation, self.adj_entity, self.adj_relation)
        train_ent1, aggregators1 = self.aggregate(train_entity_vectors1, train_relation_vectors1, user_emb, 1)
        train_entity_vectors2, train_relation_vectors2 = self.get_neighbors(tf.expand_dims(self.i_2, 1), 1, self.adj_2_entity, self.adj_2_relation, self.adj_2_entity, self.adj_2_relation, self.adj_2_entity, self.adj_2_relation, self.adj_2_entity, self.adj_2_relation)
        train_ent2, aggregators2 = self.aggregate(train_entity_vectors2, train_relation_vectors2, user_emb_2, 1)

        hist_entity_vectors1, hist_relation_vectors1 = self.get_neighbors(self.hist_1, self.memory_window, self.adj_entity, self.adj_relation, self.adj_entity, self.adj_relation, self.adj_entity, self.adj_relation, self.adj_entity, self.adj_relation)
        hist_ent1, hist_aggregators1 = self.aggregate(hist_entity_vectors1, hist_relation_vectors1, user_emb, self.memory_window)
        hist_entity_vectors2, hist_relation_vectors2 = self.get_neighbors(self.hist_1, self.maxlen, self.adj_1_entity, self.adj_1_relation, self.adj_1_entity, self.adj_1_relation, self.adj_1_entity, self.adj_1_relation, self.adj_1_entity, self.adj_1_relation)
        hist_ent2, hist_aggregators2 = self.aggregate(hist_entity_vectors2, hist_relation_vectors2, user_emb_2, self.maxlen)
        
        h_emb_1 = tf.concat([hist_ent1,  tf.tile(tf.expand_dims(user_emb, 1), [1, self.memory_window, 1])], axis=2)
        h_emb_1 = tf.layers.dense(h_emb_1, self.kge_dim, name="he1")
        h_emb_2 = tf.concat([hist_ent2, tf.tile(tf.expand_dims(user_emb_2, 1), [1, self.maxlen, 1])], axis=2)
        h_emb_2 = tf.layers.dense(h_emb_2, self.kge_dim, name="he2")
        self.e1 = tf.squeeze(train_ent1)
        self.e2 = tf.squeeze(train_ent2)
        u_b_1 = tf.gather(self.user_b_1, self.u_1)
        i_b_1 = tf.gather(self.item_b_1, self.i_1)
        u_b_2 = tf.gather(self.user_b_2, self.u_2)
        i_b_2 = tf.gather(self.item_b_2, self.i_2)
        
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.hist_1, self.mask_id)), -1)
        with tf.variable_scope('score_1',reuse=tf.AUTO_REUSE):
            preference1 = self.transformer_encoder("domain_1", self.hist_1, h_emb_1, mask, self.memory_window, hidden_units=h_emb_1.shape[-1], l2_emb=0.0, dropout_rate=0.2, num_blocks=self.source_block_num, num_heads=self.source_head_num)
            preference1 = tf.reshape(preference1, [self.batch_size, -1])
            self.weights_1 = tf.get_variable("weights_1", [preference1.shape[-1], self.e1.shape[-1]],
                                           initializer=tf.truncated_normal_initializer(stddev=0.02))
            self.bias_1 = tf.get_variable("bias_1", [self.batch_size, self.e1.shape[-1]], initializer=tf.zeros_initializer())
            preference1 = tf.matmul(preference1, self.weights_1)+self.bias_1
            self.logits_1 = tf.squeeze(tf.reduce_sum(tf.layers.batch_normalization(self.e1) * tf.layers.batch_normalization(preference1), axis=-1))+i_b_1+u_b_1
            self.score_1 = tf.sigmoid(self.logits_1)
            
        with tf.variable_scope('score_2',reuse=tf.AUTO_REUSE): 
            preference2 = self.transformer_encoder("domain_2", self.hist_1, h_emb_2, mask, self.maxlen, hidden_units=h_emb_2.shape[-1], l2_emb=0.0, dropout_rate=0.2, num_blocks=self.target_block_num, num_heads=self.target_head_num)
            preference2 = tf.reshape(preference2, [self.batch_size, -1])
            self.weights_2 = tf.get_variable("weights_2", [preference2.shape[-1], self.e2.shape[-1]],
                                           initializer=tf.truncated_normal_initializer(stddev=0.02))
            self.bias_2 = tf.get_variable("bias_2", [self.batch_size, self.e2.shape[-1]], initializer=tf.zeros_initializer())
            preference2 = tf.matmul(preference2, self.weights_2)+self.bias_2
            self.logits_2 = tf.squeeze(tf.reduce_sum(tf.layers.batch_normalization(self.e2) * tf.layers.batch_normalization(preference2), axis=-1))+i_b_2+u_b_2
            self.score_2 = tf.sigmoid(self.logits_2)
     
        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        self.global_epoch_step_zero = tf.assign(self.global_epoch_step, 0)
        
        self.aggregator1, self.aggregator2 = [], []
        self.aggregator1.extend(aggregators1)
        self.aggregator1.extend(hist_aggregators1)
        self.aggregator2.extend(aggregators2)
        self.aggregator2.extend(hist_aggregators2)

        self.loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_1,labels=self.y_1))
        self.l2_loss1 = tf.nn.l2_loss(self.user_emb_w_1) + tf.nn.l2_loss(self.kg_weight1['entity_embed']) + tf.nn.l2_loss(self.kg_weight1['relation_embed'])
        for ag in self.aggregator1:
            self.l2_loss1 = self.l2_loss1 + tf.nn.l2_loss(ag.weights)
        self.loss_all_1 = self.loss1 + self.l2_weight*self.l2_loss1
        self.loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_2,labels=self.y_2))
        self.l2_loss2 = tf.nn.l2_loss(self.user_emb_w_1) + tf.nn.l2_loss(self.kg_weight1['entity_embed']) + tf.nn.l2_loss(self.kg_weight1['relation_embed'])
        for ag in self.aggregator2:
            self.l2_loss2 = self.l2_loss2 + tf.nn.l2_loss(ag.weights)
        self.loss_all_2 = self.loss2 + self.l2_weight*self.l2_loss2

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        
        gradients1 = tf.gradients(self.loss_all_1, trainable_params)
        clip_gradients1, _ = tf.clip_by_global_norm(gradients1, 1)
        self.train_op1 = self.opt.apply_gradients(zip(clip_gradients1, trainable_params), global_step=self.global_step)
        
        gradients2 = tf.gradients(self.loss_all_2, trainable_params)
        clip_gradients2, _ = tf.clip_by_global_norm(gradients2, 1)
        self.train_op2 = self.opt.apply_gradients(zip(clip_gradients2, trainable_params), global_step=self.global_step)  
    
    def transformer_encoder(self, name, hist, seq_emb, mask, maxlen, hidden_units, l2_emb, dropout_rate, num_blocks, num_heads):
        '''
        June 2017 by kyubyong park. 
        kbpark.linguist@gmail.com.
        https://www.github.com/kyubyong/transformer
        '''
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(hist)[1]), 0), [tf.shape(hist)[0], 1]),
                vocab_size=maxlen,
                num_units=hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=l2_emb,
                scope="dec_pos",
                reuse=tf.AUTO_REUSE,
                with_t=True
            )
            seq_emb += t
            seq_emb = tf.layers.dropout(seq_emb,
                                         rate=dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            seq_emb *= mask
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    seq_emb = multihead_attention(queries=normalize(seq_emb),
                                                   keys=seq_emb,
                                                   num_units=hidden_units,
                                                   num_heads=num_heads,
                                                   dropout_rate=dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    seq_emb = feedforward(normalize(seq_emb), num_units=[hidden_units, hidden_units],
                                           dropout_rate=dropout_rate, is_training=self.is_training)
                    seq_emb *= mask

            seq_emb = normalize(seq_emb)
        return seq_emb
        
    def reduction(self, vector):
        with tf.variable_scope('reduct',reuse=tf.AUTO_REUSE): 
            vector = tf.layers.dense(vector, 100, activation=tf.nn.tanh, name='s1')
            vector = tf.layers.dense(vector, 50, activation=tf.nn.tanh, name='s2')
            vector = tf.layers.dense(vector, 32, activation=tf.nn.tanh, name='s3')
            return vector
    
    def get_neighbors(self, seeds, size, adj_entity_1, adj_relation_1, adj_entity_2, adj_relation_2, adj_entity_3, adj_relation_3, adj_entity_4, adj_relation_4):#65536
        '''
        Modification of code based on https://github.com/hwwang55/KGCN
        '''
        entities = [seeds]
        relations = []
        entity_vectors, relation_vectors = [], []
            #[B,4*W,N]
        #hop 0
        neighbor_entities = tf.nn.embedding_lookup(adj_entity_1, tf.slice(entities[0], [0,0], [self.batch_size,size]))
        neighbor_relations = tf.nn.embedding_lookup(adj_relation_1, tf.slice(entities[0], [0,0], [self.batch_size,size]))
        entities.append(neighbor_entities)
        relations.append(neighbor_relations)
        entity_vectors.append(tf.nn.embedding_lookup(self.entity_embed, tf.slice(entities[0], [0,0], [self.batch_size,size])))
        relation_vectors.append(tf.nn.embedding_lookup(self.kg_weight1['relation_embed'], tf.slice(relations[0], [0,0,0], [self.batch_size,size,self.neighbor_sample_size])))
        #hop 1
        neighbor_entities = tf.nn.embedding_lookup(adj_entity_2, tf.slice(entities[1], [0,0,0], [self.batch_size,size,self.neighbor_sample_size]))
        neighbor_relations = tf.nn.embedding_lookup(adj_relation_2, tf.slice(entities[1], [0,0,0], [self.batch_size,size,self.neighbor_sample_size]))
        entities.append(neighbor_entities)
        relations.append(neighbor_relations)
        entity_vectors.append(tf.nn.embedding_lookup(self.entity_embed, tf.slice(entities[1], [0,0,0], [self.batch_size,size,self.neighbor_sample_size])))
        relation_vectors.append(tf.nn.embedding_lookup(self.kg_weight1['relation_embed'], tf.slice(relations[1], [0,0,0,0], [self.batch_size,size,self.neighbor_sample_size,self.neighbor_sample_size])))
        #hop 2
        if self.n_iter>1:
            neighbor_entities = tf.nn.embedding_lookup(adj_entity_3, tf.slice(entities[2], [0,0,0,0], [self.batch_size,size,self.neighbor_sample_size,self.neighbor_sample_size]))
            neighbor_relations = tf.nn.embedding_lookup(adj_relation_3, tf.slice(entities[2], [0,0,0,0], [self.batch_size,size,self.neighbor_sample_size,self.neighbor_sample_size]))
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
            entity_vectors.append(tf.nn.embedding_lookup(self.entity_embed, tf.slice(entities[2], [0,0,0,0], [self.batch_size,size,self.neighbor_sample_size,self.neighbor_sample_size])))
            relation_vectors.append(tf.nn.embedding_lookup(self.kg_weight1['relation_embed'], tf.slice(relations[2], [0,0,0,0,0], [self.batch_size,size,self.neighbor_sample_size,self.neighbor_sample_size,self.neighbor_sample_size])))
        #hop 3
        if self.n_iter>2:
            neighbor_entities = tf.nn.embedding_lookup(adj_entity_4, tf.slice(entities[3], [0,0,0,0,0], [self.batch_size,size,self.neighbor_sample_size,self.neighbor_sample_size,self.neighbor_sample_size]))
            neighbor_relations = tf.nn.embedding_lookup(adj_relation_4, tf.slice(entities[3], [0,0,0,0,0], [self.batch_size,size,self.neighbor_sample_size,self.neighbor_sample_size,self.neighbor_sample_size]))
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
            entity_vectors.append(tf.nn.embedding_lookup(self.entity_embed, tf.slice(entities[3], [0,0,0,0,0], [self.batch_size,size,self.neighbor_sample_size,self.neighbor_sample_size,self.neighbor_sample_size])))
            relation_vectors.append(tf.nn.embedding_lookup(self.kg_weight1['relation_embed'], tf.slice(relations[3], [0,0,0,0,0,0], [self.batch_size,size,self.neighbor_sample_size,self.neighbor_sample_size,self.neighbor_sample_size,self.neighbor_sample_size])))
        #hop 4
        if self.n_iter>3:
            entity_vectors.append(tf.nn.embedding_lookup(self.entity_embed, tf.slice(entities[4], [0,0,0,0,0,0], [self.batch_size,size,self.neighbor_sample_size,self.neighbor_sample_size,self.neighbor_sample_size,self.neighbor_sample_size])))
        return entity_vectors, relation_vectors
    
    def aggregate(self, entity_vectors, relation_vectors, user_emb, size):
        '''
        2019 by Hongwei Wang.
        https://github.com/hwwang55/KGCN
        '''
        aggregators = []  # store all aggregators
        user_emb = tf.tile(tf.expand_dims(user_emb, 1), [1, size, 1])
        user_emb = tf.reshape(user_emb, [self.batch_size, size, 1, 1, -1])
        #i∈[0,1]
        for i in range(self.n_iter):
            if i == self.n_iter - 1:# i==1
                aggregator = self.aggregator_class(self.batch_size, self.kge_dim, act=tf.nn.tanh)
            else:# i==0
                aggregator = self.aggregator_class(self.batch_size, self.kge_dim)
            aggregators.append(aggregator)
            entity_vectors_next_iter = []

            for hop in range(self.n_iter - i):
                shape = [self.batch_size, size, -1, self.neighbor_sample_size, self.kge_dim]
                vector = aggregator(
                                    self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=user_emb,
                                    size=size)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        res = tf.reshape(entity_vectors[0], [self.batch_size, size, self.kge_dim])

        return res, aggregators    
    
    def train_1(self, sess, uij, lr, dropout):
        self.is_domain_a = True
        self.u_list = uij[0]
        self.i_list = []
        for i in uij[1]:
            self.i_list.extend(i)
        self.i_list.extend(uij[2])
        self.i_list = list(set(self.i_list))
        self.len_list = uij[3]
        loss_all, base_loss, kg_loss, _ = sess.run([self.loss_all_1, self.loss1, self.l2_loss1, self.train_op1], feed_dict={
                self.u_1: uij[0],
                self.hist_1: uij[1],
                self.i_1: uij[2],
                self.len1: uij[3],
                self.y_1: uij[4],
                self.lr: lr
                })
        return loss_all, base_loss, kg_loss, 0

    def train_2(self, sess, uij, lr, dropout):
        self.is_domain_a = False
        self.u_list = uij[0]
        self.i_list = []
        for i in uij[1]:
            self.i_list.extend(i)
        self.i_list.extend(uij[2])
        self.i_list = list(set(self.i_list))
        self.len_list = uij[3]
        loss_all, base_loss, kg_loss, _= sess.run([self.loss_all_2, self.loss2, self.l2_loss2, self.train_op2], feed_dict={
                self.u_2: uij[0],
                self.hist_1: uij[1],
                self.i_2: uij[2],
                self.len1: uij[3],
                self.y_2: uij[4],
                self.lr: lr
                })
        return loss_all, base_loss, kg_loss, 0, 0
    
    def test_auc1(self, sess, uij):
        self.is_domain_a = True
        self.is_training = False
        self.u_list = uij[0]
        self.i_list = []
        for i in uij[1]:
            self.i_list.extend(i)
        self.i_list.extend(uij[2])
        self.i_list = list(set(self.i_list))
        self.len_list = uij[3]
        score_1 = sess.run([self.score_1], feed_dict={
                self.u_1: uij[0],
                self.hist_1: uij[1],
                self.i_1: uij[2],
                self.len1: uij[3],
                self.y_1: uij[4]
                })
        return (score_1[0], uij[4], uij[0])
    
    def test_auc2(self, sess, uij):
        self.is_domain_a = False
        self.is_training = False
        self.u_list = uij[0]
        self.i_list = []
        for i in uij[1]:
            self.i_list.extend(i)
        self.i_list.extend(uij[2])
        self.i_list = list(set(self.i_list))
        self.len_list = uij[3]
        score_2 = sess.run([self.score_2], feed_dict={
                self.u_2: uij[0],
                self.hist_1: uij[1],
                self.i_2: uij[2],
                self.len1: uij[3],
                self.y_2: uij[4]
                })
        return (score_2[0], uij[4], uij[0])
    
    def update_adj(self, pars):
        self.adj_entity = tf.convert_to_tensor(pars['adj_entity'])
        self.adj_relation = tf.convert_to_tensor(pars['adj_relation'])
        self.adj_1_entity = tf.convert_to_tensor(pars['adj_1_entity'])
        self.adj_1_relation = tf.convert_to_tensor(pars['adj_1_relation'])
        self.adj_2_entity = tf.convert_to_tensor(pars['adj_2_entity'])
        self.adj_2_relation = tf.convert_to_tensor(pars['adj_2_relation'])
        return