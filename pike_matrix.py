import tensorflow as tf
from tensorflow.keras import layers, Model

class Net(tf.keras.Model):
    def __init__(self, input_dim, output_dim, device, coef_m=None):
        super(Net, self).__init__()
        self.output_dim = output_dim
        self.device = device

        # Define layers
        self.fc_input = layers.Dense(1024, activation='relu', input_shape=(input_dim,))
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)

        # Output layers
        self.output_list = [layers.Dense(64, activation='relu') for _ in range(output_dim)]
        self.output_list2 = [layers.Dense(1, activation='elu') for _ in range(output_dim)]

        # GCN-related layers
        self.trans_gcn = layers.Dense(64, activation='sigmoid')
        self.nodes_trans_nn = layers.Dense(output_dim, use_bias=False)
        
        # Coefficient matrix
        self.nodes_trans = tf.Variable(tf.convert_to_tensor(coef_m, dtype=tf.float32), trainable=True)
        coef_w = tf.convert_to_tensor(coef_m * 0.25, dtype=tf.float32)
        self.nodes_trans_nn.build((None, output_dim))
        self.nodes_trans_nn.set_weights([coef_w])

        # Final output mapping
        self.output_mapping = layers.Dense(1, activation=None)
        self.prelu = layers.PReLU()

    def call(self, x):
        x = tf.convert_to_tensor(x)
        x = self.fc_input(x)
        x = self.fc1(x)
        
        hidden_list = [layer(x) for layer in self.output_list]
        hidden_list = [tf.expand_dims(hidden, axis=1) for hidden in hidden_list]
        
        hiddens = tf.concat(hidden_list, axis=1)
        nodes_hidden = tf.transpose(hiddens, perm=[0, 2, 1])
        nodes_trans = self.nodes_trans_nn(nodes_hidden)
        nodes_trans = tf.transpose(nodes_trans, perm=[0, 2, 1])
        trans = self.trans_gcn(nodes_trans)
        
        output_list = [layer(tf.squeeze(trans[:, i:i+1, :], axis=1)) for i, layer in enumerate(self.output_list2)]
        outputs = self.prelu(self.output_mapping(trans))
        return tf.squeeze(outputs, axis=-1)


class Net2(tf.keras.Model):
    def __init__(self, input_dim, output_dim, device, coef_m=None):
        super(Net2, self).__init__()
        self.output_dim = output_dim
        self.device = device
        
        # Define layers
        self.fc_input = layers.Dense(1024, activation='relu', input_shape=(input_dim,))
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)

        self.coef_m = tf.Variable(tf.convert_to_tensor(coef_m, dtype=tf.float32), trainable=True)
        self.coef_m = tf.transpose(self.coef_m, perm=[1, 2, 0])

        # Coefficient transformation layers
        self.feature_hidden = 32
        self.coef_nn_list = [layers.Dense(self.feature_hidden, activation='elu') for _ in range(self.coef_m.shape[2])]
        self.coef_nn_list_att = [layers.Dense(self.feature_hidden, use_bias=False) for _ in range(self.coef_m.shape[2])]

        # Output attention layers
        self.output_list_att = [layers.Dense(self.feature_hidden, use_bias=False) for _ in range(output_dim)]

        # GCN-related layers
        self.trans_gcn = layers.Dense(64 + self.feature_hidden * output_dim, use_bias=False, activation='sigmoid')
        self.nodes_trans_nn = layers.Dense(output_dim, use_bias=False)

        # Final output mapping
        self.output_mapping = layers.Dense(1, activation=None)
        self.prelu = layers.PReLU()

        # Output layers
        self.output_list = [layers.Dense(64, activation='relu') for _ in range(output_dim)]
        self.output_list2 = [layers.Dense(1) for _ in range(output_dim)]

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.convert_to_tensor(x)
        x = self.fc_input(x)
        x = self.fc1(x)
        
        hidden_list = [layer(x) for layer in self.output_list]
        hidden_list = [tf.expand_dims(hidden, axis=1) for hidden in hidden_list]
        
        coef_feature_list = [layer(tf.expand_dims(self.coef_m[:, :, i], axis=-1)) for i, layer in enumerate(self.coef_nn_list)]
        coef_feature = tf.concat(coef_feature_list, axis=2)
        coef_feature_sums = tf.reduce_sum(tf.exp(coef_feature), axis=2, keepdims=True)

        hiddens = tf.concat(hidden_list, axis=1)

        coef_feature_list_att = []
        coef_feature_s = tf.zeros([self.output_dim, self.output_dim, self.feature_hidden], dtype=tf.float32)
        for idx, f in enumerate(coef_feature_list):
            att = tf.exp(f)
            p = att * self.coef_nn_list_att[idx](f) / coef_feature_sums
            coef_feature_list_att.append(p)
            coef_feature_s += p
        coef_feature_s /= len(self.coef_nn_list)

        coef_sum = tf.reduce_sum(tf.exp(coef_feature_s), axis=1)
        coef_feature_att = []
        for i in range(self.output_dim):
            nodes_feat = coef_feature_s[:, i, :]
            p = tf.exp(nodes_feat) / coef_sum
            att = p * self.output_list_att[i](nodes_feat)
            coef_feature_att.append(att)
        coef_feature_att = tf.concat(coef_feature_att, axis=1)
        coef_feature_att = tf.expand_dims(coef_feature_att, axis=0)
        coef_feature_att = tf.tile(coef_feature_att, [batch_size, 1, 1])
        hiddens = tf.concat([hiddens, coef_feature_att], axis=2)

        nodes_hidden = tf.transpose(hiddens, perm=[0, 2, 1])
        nodes_trans = self.nodes_trans_nn(nodes_hidden)
        nodes_trans = tf.transpose(nodes_trans, perm=[0, 2, 1])
        trans = self.trans_gcn(nodes_trans)
        
        output_list = [self.prelu(layer(tf.squeeze(trans[:, i:i + 1, :], axis=1))) for i, layer in enumerate(self.output_list2)]
        outputs = self.prelu(self.output_mapping(trans))
        return tf.squeeze(outputs, axis=-1)
