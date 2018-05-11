# coding: utf-8

import numpy as np
import os, time , sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode

from data import pad_sequences, batch_yield
from utils import get_logger
from eval import conlleval
printDebug=1
class BiLSTM_CRF(object):
    def __init__(self, batch_size, epoch_num, hidden_dim, embeddings,
                dropout_keep, optimizer, lr, clip_grad,
                tag2label, vocab, shuffle,
                model_path, summary_path, log_path, result_path,
                CRF=True, update_embedding=True):
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        self.dropout_keep_prob = dropout_keep
        self.optimizer = optimizer
        self.lr = lr
        print('self.lr=', self.lr)
        self.clip_grad = clip_grad
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = shuffle
        self.model_path  = model_path
        self.summary_path = summary_path
        self.logger = get_logger(log_path)
        self.result_path = result_path
        self.CRF = CRF
        self.update_embedding = update_embedding
    #构建biLstm+crf的网络结构
    def build_graph(self):
        self.add_placeholders_op() #待输入变量初始化
        self.add_init_op()       #初始化embedding等
        self.biLSTM_layer_op()     #构建biLstm层
        self.loss_op()          #计算损失函数
        self.trainstep_op()       #参数迭代更新
        self.init_op()          #初始化
        
    def add_placeholders_op(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="label")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(tf.float32, shape=[], name="lr")
    #初始化词嵌入
    def add_init_op(self):
        with tf.device('/cpu:0'):
            word_embeddings = tf.Variable(
                self.embeddings,
                dtype = tf.float32,
                trainable = self.update_embedding,
                name = "word_embeddings")

            embedded_words = tf.nn.embedding_lookup(
                params = word_embeddings,
                ids = self.word_ids,
                name = "word_embeddings")
            self.embedded_words = tf.nn.dropout(embedded_words, self.dropout_pl)
            if printDebug:
                print("embeded_words=",self.embedded_words.get_shape())
    #BiLSTM层,前向和后向
    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            #定义前向网络和后向网络
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            #outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组
            (output_fw_seq , output_bw_seq) , _= tf.nn.bidirectional_dynamic_rnn(
                cell_fw = cell_fw,
                cell_bw = cell_bw,
                inputs = self.embedded_words,
                sequence_length= self.sequence_lengths,
                dtype = tf.float32
            )
            #前向结果和后向结果拼接
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)
            if printDebug:
                print("biLSTM output=",output.get_shape())
        with tf.variable_scope("proj"):
            #概率矩阵，列数为标签数量
            W = tf.get_variable(name='W',
                shape=[2*self.hidden_dim, self.num_tags],
                initializer = tf.contrib.layers.xavier_initializer(),
                dtype = tf.float32
                               )
            b = tf.get_variable(name="b",
                shape=[self.num_tags],
                initializer = tf.zeros_initializer(),
                dtype = tf.float32
                               )
            s = tf.shape(output)

            output = tf.reshape(output,[-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W)+b

            self.logits = tf.reshape(pred, [-1,s[1], self.num_tags])
            if printDebug:
                print("logits=", self.logits.get_shape())
            
    def loss_op(self):
        if self.CRF:
            '''
            inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,一般使用BILSTM处理之后输出转换为他要求的形状作为CRF层的输入. 
            tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签. 
            sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度. 
            transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            '''
            log_likelihood, self.transition_params = crf_log_likelihood(
                inputs=self.logits,
                tag_indices=self.labels,
                sequence_lengths=self.sequence_lengths
            )
            self.loss = -tf.reduce_mean(log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = self.logits,
                labels=self.labels
            )
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
            #softmax 
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)
        tf.summary.scalar("loss", self.loss)
    #训练过程参数优化
    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step",trainable=False)

            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer=='Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer=='RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer=='Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer=='SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip,  global_step=self.global_step)
    def init_op(self):
        self.init_op = tf.global_variables_initializer()
    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()

        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)