#import os
import tarfile
import pandas
#import protein DB data
from io import StringIO
import pytorch
#import tensorflow 
import torch
import torch.nn.functional as F
from torch_geometric.data import data
from torch_geometric.nn import GCNConv
import torch.optim as optim
import sys
import time
import scipy
import Pylance
#for plotting GNN output in 2D
from matplotlib import pyplot as plt
from random import randint


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
        if coef_m is not None:
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

        if coef_m is not None:
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

#randomise graph date for testing purposes
graph_date = []
# Extract DNA sequencing data from tar file
def extract_dna_data(tar_file_path, extract_path):
    with tarfile.open(tar_file_path, "r:*") as tar:
        tar.extractall(path=extract_path)

#Tar example
tar_file_path = "path/to/your/dna_data.tar"
extract_path = "path/to/extracted_files"
extract_dna_data(tar_file_path, extract_path)

#Import protein data from online database
def import_protein_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        protein_data = pd.read_csv(StringIO(response.text), sep="\t")  # Adjust separator based on the data format
        return protein_data
    else:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

#Example for protein data import
protein_data_url = "http://example.com/protein_data.tsv"
protein_data = import_protein_data(protein_data_url)

#Load and preprocess dna data (must be .csv format)
def load_dna_data(extract_path):
    dna_data_files = [f for f in os.listdir(extract_path) if f.endswith(".csv")]
    dna_data_list = [pd.read_csv(os.path.join(extract_path, f)) for f in dna_data_files]
    
#Combining dna/protein data into a single dataset
    dna_data = pd.concat(dna_data_list, ignore_index=True)
    return dna_data

#Example 
dna_data = load_dna_data(extract_path)

def matrix_to_graph(matrix):

    matrix = torch.tensor(matrix, dtype=torch.float32)
    edge_index = (matrix > 0).nonzero(as_tuple=False).t().contiguous()
    
    edge_attr = matrix[edge_index[0], edge_index[1]]
    
    num_nodes = matrix.size(0)
    node_features = torch.eye(num_nodes)  
#Create a Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return data

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
#1st conv
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
#2nd conv
        x = self.conv2(x, edge_index)
        
        return x

#Example
input_dim = graph_data.num_features
hidden_dim = 16
output_dim = 2  

model = GNN(input_dim, hidden_dim, output_dim)

# Forward pass with the graph data
output = model(graph_data)
print(output)

# Example training loop (ensure nodes are labelled)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    out = model(graph_data)
    
    
    labels = torch.tensor([0, 1, 0, 1, 0])  
    loss = criterion(out, labels)
    
#Optimisation
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

#Layer wise relevance backpropagation

def LRP(model, data, target):
    model.eval()
    output = model(data)
    loss = F.cross_entropy(output.view(1, -1), target)
    loss.backward()
    
    relevance = data.x.grad
    return relevance

#visualise the output of the GNN in 2D
def plot_output(data, output):
    output = output.detach().numpy()
    plt.scatter(output[:, 0], output[:, 1])
    
    for i in range(data.num_nodes):
        x = output[i, 0]
        y = output[i, 1]
        plt.text(x, y, str(i), color="red")
        
    plt.show()

#Example
LRP(model, graph_data, torch.tensor([0]))
plot_output(graph_data, output)

if(length(outputs) == 1){
  write.tsv(settings, file = paste0(output.folder, "/", "NodeWeights_MetaData.tsv"))
  write.csv(toMT(nodeWeights, row = "Node", col="replicate", val = "value"), file = paste0(output.folder, "/", "NodeWeights.csv"), quote=F)
} else {
  nodeWeights[,id := paste0(replicate, "_", variable)]
  write.csv(toMT(nodeWeights, row = "Node", col="id", val = "value"), file = paste0(output.folder, "/", "NodeWeights.csv"), quote=F)
  
  settings2 <- data.table()
  for(i in 1:nrow(settings)){
    xx <- unique(nodeWeights[replicate == settings[i]$replicate][,c("replicate", "variable", "id")])
    settings2 <- rbind(settings2, data.table(settings[i][,-"replicate"], replicate = xx$id, outputNode=xx$variable))
  }
  write.tsv(settings2, file = paste0(output.folder, "/", "NodeWeights_MetaData.tsv"))
}

def __init__(self, model, samples, labels):
        """
        Initialization of internals for relevance computations.
        :param model: gcnn model to LRP procedure is applied on
        :param samples: samples to calculate relevance on, num of samples <= models batch size
        :param labels: used as values for the gcnn output nodes to propagate relevances, num of labels == models batch size
        """
        self.epsilon = 1e-10  # for numerical stability

        start = time.time()
        print("\n\tCalculating Polynomials of Laplace Matrices...", end=" ")
        self.polynomials = [self.calc_Laplace_Polynom(lap, K=model.K[i]) for i, lap in enumerate(model.L)]
        end = time.time()
        print("Time: ", end - start, "\n")

        weights = model.get_weights()
        self.activations = model.activations

        self.model = model
        self.model.graph._unsafe_unfinalize()  # the computational graph of the model will be modified
        self.labels = labels
        self.samples = samples
        self.X = self.activations[0]  # getting the first

        # self.y = self.activations.pop(0)
        # I am getting the activation of the first, but not useful here, self.y will be assigned a placeholder

        self.ph_dropout = model.ph_dropout
        self.batch_size = self.X.shape[0]

        with self.model.graph.as_default():
            self.y = tf.placeholder(tf.float32, (self.batch_size, labels.shape[1]), 'labels_hot_encoded')

        self.act_weights = {}  # example in this dictionary "conv1": [weights, bias]

        for act in self.activations[1:]:
            w_and_b = []  # 2 element list of weight and bias of one layer.
            name = act.name.split('/')
            # print(name)
            for wt in weights:
                # print(wt.name)
                if name[0] == wt.name.split('/')[0]:
                    w_and_b.append(wt)
            if w_and_b and (name[0] not in self.act_weights):
                self.act_weights[name[0]] = w_and_b

        # !!!
        # first convolutional layer filters
        self.filters_gc1 = []
        self.filtered_signal_gc1 = []

    def get_relevances(self):
        """
        Computes relevances based on input samples.
        :param rule: the propagation rule of the first layer
        :return: the list of relevances, corresponding to different layers of the gcnn.
        The last element of this list contains input relevances and has shape (batch_size, num_of_input_features)
        """

        # Backpropagate softmax value
        # relevances = [tf.nn.softmax(self.activations[0])*tf.cast(self.y, tf.float32)]

        # Backpropagate a value from given labels y
        relevances = [tf.cast(self.y, tf.float32)]

        loc_poly = [pol for pol in self.polynomials]
        loc_pooling = [p for p in self.model.p]
        print("\n    Relevance calculation:")

        for i in range(len(self.activations)-2, -1, -1):  # reverse order of the activations and excluding the very
            # first activation 'ExpandDims:0'
            name = self.activations[i+1].name.split('/')

            # print("\n\t name of the activation", name)

            if 'logits' in name[0] or 'fc' in name[0]:
                print("\tFully connected:", name[0])
                relevances.append(self.prop_fc(name[0], self.activations[i], relevances[-1]))
            elif 'flatten' in name[0]:
                print("\tFlatten layer:", name[0])
                relevances.append(self.prop_flatten(self.activations[i], relevances[-1]))
                # print("\n")
            elif 'pooling' in name[1]:
                # TODO: incorporate pooling type and value into name
                print("\tPooling:", name[0] + " " + name[1])
                print("\t\tname of pooling:", self.model.pool.__name__)
                if self.model.pool.__name__ == 'apool1':
                    p = loc_pooling.pop()
                    relevances.append(self.prop_avg_pool(self.activations[i], relevances[-1], ksize=[1, p, 1, 1],
                                                         strides=[1, p, 1, 1]))
                elif self.model.pool.__name__ == 'mpool1':
                    p = loc_pooling.pop()
                    relevances.append(self.prop_max_pool(self.activations[i], relevances[-1], ksize=[1, p, 1, 1],
                                                         strides=[1, p, 1, 1]))
                else:
                    raise Exception('Error parsing the pooling type')

            elif 'conv' in name[0]:
                if len(loc_poly) > 1:
                    print("\tConvolution: ", name[0], "\n")
                    relevances.append(self.prop_gconv(name[0], self.activations[i], relevances[-1],
                                                      polynomials=loc_poly.pop()))
                else:
                    print("\tConvolution, the first layer:", name[0], "\n")
                    relevances.append(self.prop_gconv_first_conv_layer(name[0], self.activations[i], relevances[-1],
                                                                       polynomials=loc_poly.pop()))
            else:
                raise Exception('Error parsing layer')

        return relevances

    def prop_fc(self, name, activation, relevance):
        """Propagates relevances through fully connected layers."""
        w = self.act_weights[name][0]
        # b = self.act_weights[name][1]  # bias
        w_pos = tf.maximum(0.0, w)
        z = tf.matmul(activation, w_pos) + self.epsilon
        s = relevance / z
        c = tf.matmul(s, tf.transpose(w_pos))
        return c * activation

    def prop_flatten(self, activation, relevance):
        """Propagates relevances from the fully connected part to convolutional part."""
        shape = activation.get_shape().as_list()
        return tf.reshape(relevance, shape)

    def prop_max_pool(self, activation, relevance, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1]):
        """Propagates relevances through max pooling."""
        act = tf.expand_dims(activation, 3)  # N x M x F x 1
        z = tf.nn.max_pool(act, ksize, strides, padding='SAME') + self.epsilon
        with self.model.graph.as_default():
            rel = tf.expand_dims(relevance, 3)
        s = rel / z
        c = gen_nn_ops.max_pool_grad_v2(act, z, s, ksize, strides, padding='SAME')
        tmp = c * act
        return tf.squeeze(tmp, [3])

    def prop_avg_pool(self, activation, relevance, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1]):
        """Propagates relevances through avg pooling."""
        act = tf.expand_dims(activation, 3)  # N x M x F x 1
        z = tf.nn.avg_pool(act, ksize, strides, padding='SAME') + self.epsilon
        with self.model.graph.as_default():
            rel = tf.expand_dims(relevance, 3)
        s = rel / z
        c = gen_nn_ops.avg_pool_grad(tf.shape(act), s, ksize, strides, padding='SAME')
        tmp = c * act
        return tf.squeeze(tmp, [3])

    def prop_gconv(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through Graph Convolutional Layers.
        All essential operations are in SCIPY.
        """
        start = time.time()
        w = self.act_weights[name][0]  # weight
        b = self.act_weights[name][1]  # bias
        # print("\nInside gconv")
        # print("weights of current gconv, w:", w)
        # activation
        N, M, Fin = activation.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(w.get_shape().as_list()[-1])

        K = int(w.get_shape().as_list()[0] / Fin)
        W = self.model._get_session().run(w)
       
        activation = self.run_tf_tensor(activation, samples=self.samples)

        
        if tf.is_tensor(relevance):
            relevance = self.run_tf_tensor(relevance, self.samples)

        W = np.reshape(W, (int(W.shape[0] / K), K, Fout))
        W = np.transpose(W, (1, 0, 2))  # K x Fin x Fout

        rel = np.zeros(shape=[N, M * Fin], dtype=np.float32)
        for i in range(0, Fout):
            w_pos = polynomials.dot(W[:, :, i])
            w_pos = np.maximum(0.0, w_pos)
            w_pos = np.reshape(w_pos, [M, M, Fin])
            w_pos = np.transpose(w_pos, axes=[0, 2, 1])  # M x Fin x M
            w_pos = np.reshape(w_pos, [M * Fin, M])
            activation = np.reshape(activation, [N, Fin * M])  # N x Fin*M
            z = np.matmul(activation, w_pos) + self.epsilon  # N x M
            s = relevance[:, :, i] / z  # N x M
            c = np.matmul(s, np.transpose(w_pos))  # N x M by transpose(M * Fin, M) = N x M * Fin
            rel += c * activation
        end = time.time()
        # #
        rel = np.reshape(rel, [N, M, Fin])
        print("\n\t" + name + ",", "relevance propagation time is: ", end - start)

        return rel
#Example of relevance propagation

