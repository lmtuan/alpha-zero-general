import sys
sys.path.append('..')
from utils import *

import tensorflow as tf

class Score4NNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Renaming functions 
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense

        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y, self.board_z, 2])    # s: batch_size x board_x x board_y x board_z x num_players
            self.dropout = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool, name="is_training")

            x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, self.board_z, 2])
            h_conv1 = Relu(
                        BatchNormalization(
                            self.conv3d(x_image, args.num_channels, 'same'),
                            axis = -1,
                            training = self.isTraining
                        )
                    )
            h_conv2 = Relu(
                        BatchNormalization(
                            self.conv3d(h_conv1, args.num_channels, 'same'),
                            axis = -1,
                            training = self.isTraining
                        )
                    )
            h_conv3 = Relu(
                        BatchNormalization(
                            self.conv3d(h_conv2, args.num_channels, 'valid'),
                            axis = -1,
                            training = self.isTraining
                        )
                    )
            h_conv_flat = tf.reshape(h_conv3, [-1, args.num_channels*(self.board_x-1)*(self.board_y-1)*(self.board_z-1)])
            s_fc1 = Dropout(
                        Relu(
                            BatchNormalization(
                                Dense(h_conv_flat, 1024), 
                                axis=-1, 
                                training=self.isTraining
                            )
                        ), 
                        rate=self.dropout
                    ) # batch_size x 1024
            s_fc2 = Dropout(Relu(BatchNormalization(Dense(s_fc1, 512), axis=-1, training=self.isTraining)), rate=self.dropout)         # batch_size x 512
            self.pi = Dense(s_fc2, self.action_size)                                                        # batch_size x self.action_size
            self.prob = tf.nn.softmax(self.pi)
            self.v = Tanh(Dense(s_fc2, 1))                                                               # batch_size x 1

            self.calculate_loss()

    def conv2d(self, x, out_channels, padding):
      return tf.layers.conv2d(x, out_channels, kernel_size=[3,3], padding=padding)

    def conv3d(self, x, out_channels, padding):
        return tf.layers.conv3d(x, out_channels, kernel_size=[2,2,2], padding=padding)

    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)