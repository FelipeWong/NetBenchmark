import scipy.sparse as sp
import torch
import torch.nn as nn
from GraphCL_pacakage import (GCN, AvgReadout, Discriminator, Discriminator2,
                              graphcl_process, aug)
from models.dgi_package import process
from models.model import *
from hyperparameters.public_hyper import SPACE_TREE
import pdb

"""
================================================================
The model base is from the repository of Shen-Lab/GraphCL folders "unsupervised_Cora_Citeseer"
Reference: https://proceedings.neurips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf
================================================================
"""


class GraphCL_test(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(GraphCL_test, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.disc2 = Discriminator2(n_h)

    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2, aug_type):

        h_0 = self.gcn(seq1, adj, sparse)
        if aug_type == 'edge':

            h_1 = self.gcn(seq1, aug_adj1, sparse)
            h_3 = self.gcn(seq1, aug_adj2, sparse)

        elif aug_type == 'mask':

            h_1 = self.gcn(seq3, adj, sparse)
            h_3 = self.gcn(seq4, adj, sparse)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = self.gcn(seq3, aug_adj1, sparse)
            h_3 = self.gcn(seq4, aug_adj2, sparse)

        else:
            assert False

        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_3, msk)
        c_3 = self.sigm(c_3)

        h_2 = self.gcn(seq2, adj, sparse)

        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)

        ret = ret1 + ret2
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


"""
================================================================
According to the readme of the DEEP-PolyU/NetBenchmark repository, I need to add 
some subclasses, like check_train_parameters and train_model, to import the new 
algorithm easily. Besides, the new algorithm should be able to inherit from the base 
class from ./models/model.py so that we can do tasks 1, 2 & 3 for benchmarking
================================================================
"""


class GraphCL(Models):
    @classmethod
    def is_preprocessing(cls):
        return False

    @classmethod
    def is_deep_model(cls):
        return True

    @classmethod
    def is_end2end(cls):
        return False

    def check_train_parameters(self):
        space_dtree = {
            "aug_type": hp.choice("aug_type", ["edge", "mask", "node", "subgraph"]),
            "drop_percent": hp.choice("drop_percent", [0.0, 0.1, 0.2, 0.3]),
            "lr": SPACE_TREE["lr"],
            "nb_epochs": SPACE_TREE["nb_epochs"]
        }
        return space_dtree

    def train_model(self, **kwargs):
        aug_type = kwargs["aug_type"]
        drop_percent = int(kwargs["drop_percent"])
        np.random.seed(42)
        torch.manual_seed(42)
        if self.use_gpu:
            device = self.device
            torch.cuda.manual_seed(42)
        else:
            device = self.device
            print("--> No GPU")

        # training params
        # print(int(kwargs["batch_size"]))
        batch_size = 1
        nb_epochs = int(kwargs["nb_epochs"])
        patience = 20
        lr = kwargs["lr"]
        l2_coef = 0.0
        drop_prob = 0.0
        hid_units = 512
        sparse = True

        nonlinearity = 'prelu'  # special name to separate parameters
        adj, features, labels, idx_train, idx_val, idx_test = process.load_citationmat(self.mat_content)
        features, _ = graphcl_process.preprocess_features(features)

        nb_nodes = features.shape[0]  # node number
        ft_size = features.shape[1]  # node features dim
        nb_classes = labels.shape[1]  # classes = 6

        features = torch.FloatTensor(features[np.newaxis])

        '''
        ------------------------------------------------------------
        edge node mask subgraph
        ------------------------------------------------------------
        '''
        # print("Begin Aug:[{}]".format(aug_type))
        if aug_type == 'edge':

            aug_features1 = features
            aug_features2 = features

            aug_adj1 = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges
            aug_adj2 = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges

        elif aug_type == 'node':

            aug_features1, aug_adj1 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)
            aug_features2, aug_adj2 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)

        elif aug_type == 'subgraph':

            aug_features1, aug_adj1 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)
            aug_features2, aug_adj2 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)

        elif aug_type == 'mask':

            aug_features1 = aug.aug_random_mask(features, drop_percent=drop_percent)
            aug_features2 = aug.aug_random_mask(features, drop_percent=drop_percent)

            aug_adj1 = adj
            aug_adj2 = adj

        else:
            assert False

        '''
        ------------------------------------------------------------
        '''

        adj = graphcl_process.normalize_adj(adj + sp.eye(adj.shape[0]))
        aug_adj1 = graphcl_process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
        aug_adj2 = graphcl_process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

        if sparse:
            sp_adj = graphcl_process.sparse_mx_to_torch_sparse_tensor(adj)
            sp_aug_adj1 = graphcl_process.sparse_mx_to_torch_sparse_tensor(aug_adj1)
            sp_aug_adj2 = graphcl_process.sparse_mx_to_torch_sparse_tensor(aug_adj2)

        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()
            aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
            aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()

        '''
        ------------------------------------------------------------
        mask
        ------------------------------------------------------------
        '''

        '''
        ------------------------------------------------------------
        '''
        if not sparse:
            adj = torch.FloatTensor(adj[np.newaxis])
            aug_adj1 = torch.FloatTensor(aug_adj1[np.newaxis])
            aug_adj2 = torch.FloatTensor(aug_adj2[np.newaxis])

        labels = torch.FloatTensor(labels[np.newaxis])
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        model = GraphCL_test(ft_size, hid_units, nonlinearity)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

        if torch.cuda.is_available():
            # print('Using CUDA')
            model.cuda(device)
            features = features.cuda(device)
            aug_features1 = aug_features1.cuda(device)
            aug_features2 = aug_features2.cuda(device)
            if sparse:
                sp_adj = sp_adj.cuda(device)
                sp_aug_adj1 = sp_aug_adj1.cuda(device)
                sp_aug_adj2 = sp_aug_adj2.cuda(device)
            else:
                adj = adj.cuda(device)
                aug_adj1 = aug_adj1.cuda(device)
                aug_adj2 = aug_adj2.cuda(device)

            labels = labels.cuda(device)
            idx_train = idx_train.cuda(device)
            idx_val = idx_val.cuda(device)
            idx_test = idx_test.cuda(device)

        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0
        best_model = None
        start_time = time.time()

        for epoch in range(nb_epochs):

            model.train()
            optimiser.zero_grad()

            idx = np.random.permutation(nb_nodes)
            shuf_fts = features[:, idx, :]

            lbl_1 = torch.ones(batch_size, nb_nodes)
            lbl_2 = torch.zeros(batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda(device)
                lbl = lbl.cuda(device)

            logits = model(features, shuf_fts, aug_features1, aug_features2,
                           sp_adj if sparse else adj,
                           sp_aug_adj1 if sparse else aug_adj1,
                           sp_aug_adj2 if sparse else aug_adj2,
                           sparse, None, None, None, aug_type=aug_type)

            loss = b_xent(logits, lbl)
            # print('Loss:[{:.4f}]'.format(loss.item()))

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                # torch.save(model.state_dict(), args.save_name)
                best_model = model.state_dict()
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                # print('Early stopping!')
                break

            loss.backward()
            optimiser.step()

        # print('Loading {}th epoch'.format(best_t))
        # model.load_state_dict(torch.load(args.save_name))
        model.load_state_dict(best_model)

        embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)

        # train_embs = embeds[0, idx_train]
        # val_embs = embeds[0, idx_val]
        # test_embs = embeds[0, idx_test]

        node_emb = embeds.data.cpu().numpy()
        # print('node_shape ', node_emb.shape)
        node_emb = node_emb.reshape(node_emb.shape[1:])
        # print('node_shape_new ', node_emb.shape)

        return node_emb
