import torch
import torch.nn as nn
import torch.nn.functional as F

from CoANE_package.Data_Processing import *
from CoANE_package.Random_Walk import *
from CoANE_package.Contexts_Generator import *

from models.model import *
from hyperparameters.public_hyper import SPACE_TREE

from tqdm import tqdm
from time import time
import scipy.io
import pickle

"""
================================================================
The model base is from the repository of ICHproject/CoANE folders "CoANE with attributes preservation"
Reference: https://ieeexplore.ieee.org/document/9835442
================================================================
"""


class CoANE_test(torch.nn.Module):
    def __init__(self, feat_dim, nb_filter, win_len, t_feat, drop=0.5):
        super(CoANE_test, self).__init__()

        self.win_cnnEncoder1 = nn.Conv1d(in_channels=feat_dim, out_channels=nb_filter, kernel_size=win_len,
                                         stride=win_len)
        self.feat_emb = nn.Embedding.from_pretrained(t_feat)

        self.r1 = nn.Linear(nb_filter, int(nb_filter * 2))
        self.r2 = nn.Linear(int(nb_filter * 2), feat_dim)
        self.MSE = nn.MSELoss()
        self.drop = drop

    def forward(self, x):
        gather_feat = self.feat_emb(x[0])
        gather_feat_flat = torch.transpose(gather_feat, 1, 2)
        x_average_no = x[1]

        win_Encoder_feat = self.win_cnnEncoder1(gather_feat_flat)

        feat_pool = torch.sum(win_Encoder_feat, dim=2)
        feat_avg = torch.transpose(x_average_no * torch.transpose(feat_pool, 0, 1), 0, 1)

        return win_Encoder_feat, feat_avg

    def forward_f(self, x):
        x = torch.relu(self.r1(x))
        x = F.dropout(x, self.drop, training=self.training)
        x = torch.relu(self.r2(x))
        return x


def train_batch_processing(id_node, V_BATCH_SIZE):
    np.random.shuffle(id_node)
    mbatch = []
    num_mbatch = max(int(len(id_node) / V_BATCH_SIZE), 1)
    mbatch.append(id_node[:V_BATCH_SIZE])
    i = 0
    if (V_BATCH_SIZE * (i + 1)) * 2 < len(id_node):
        for i in range(1, num_mbatch):
            mbatch.append(id_node[(i * (V_BATCH_SIZE)):(V_BATCH_SIZE * (i + 1))])
    if (V_BATCH_SIZE * (i + 1)) < len(id_node):
        num_mbatch += 1
        mbatch.append(id_node[(V_BATCH_SIZE * (i + 1))::])
    return mbatch, num_mbatch


def train(PAR, Context, args):
    # verbose = args.verbose
    verbose = args["verbose"]
    # seed = args.seed
    seed = args["seed"]
    print("seed number: " + str(seed))
    nb_node = PAR['nb_id']  # Scan["mulreformfn"][0].shape[0]
    # nb_filter = args.emb_dim  # dimension of embedding
    nb_filter = args["emb_dim"]  # dimension of embedding

    # win_len = 2 * args.window_hsize + 1
    win_len = 2 * args["window_hsize"] + 1
    max_win_count = Context["Max_win_count"]

    # num_epochs = args.num_epochs
    num_epochs = args["num_epochs"]
    # V_BATCH_SIZE = min(args.num_Vbatch, nb_node)
    V_BATCH_SIZE = min(args["num_Vbatch"], nb_node)
    R_SIZE = len(Context["mulreformfn"])
    # device = args.device
    device = args["device"]

    feat_dim = PAR['feat'].shape[1]

    # INPUT initial
    id_node = list(range(1, nb_node))
    num_mbatch = max(int(nb_node / V_BATCH_SIZE), 1)
    # Input initial
    X, Y = [], []

    # controller
    r2 = torch.from_numpy(np.array(Context["mulnegFrea"][0])).type(torch.FloatTensor).to(device)

    t_feat = torch.from_numpy(PAR['feat'].todense()).type(torch.FloatTensor).to(device)

    x_callfeat = torch.from_numpy(np.array(list(range(nb_node))).astype(int)).type(torch.LongTensor).to(device)
    for i in range(R_SIZE):
        # contexts
        # print(len(Context["mulreformfn"][i]), len(Context["mulreformfn"][i][0]))
        # n_c = (args.window_hsize * 2 + 1)
        n_c = (args["window_hsize"] * 2 + 1)
        # print(Context["mulreformfn"][i][:10])
        n_c_max = max([len(c) for c in Context["mulreformfn"][i]])

        Context_m = [sum(c + (n_c_max - len(c)) * [[0] * n_c], []) for c in Context["mulreformfn"][i]]
        # assert 0
        x_reformfn = torch.LongTensor(Context_m).to(device)
        # torch.from_numpy(Context["mulreformfn"][i].todense().astype(int)).type(torch.LongTensor).to(device)
        Context["mulcount_list"][i][0] += 1
        # avg. par
        x_average_no = torch.from_numpy(np.array([1. / (i if i else 1) for i in Context["mulcount_list"][i]])).type(
            torch.FloatTensor).to(device)
        # neg  samples
        x_negSample = torch.from_numpy(np.array(Context["mulnegFre"][i].todense()).astype(int)).type(
            torch.LongTensor).to(device)
        # co-occurance matrix
        y_D = torch.from_numpy(Context["mulDmatrix"][i].todense() + Context["mulComatrix_1hop"][i].todense()).type(
            torch.FloatTensor).to(device)

        #
        x = [x_reformfn, x_average_no, x_negSample]
        y = [y_D]
        #
        X.append(x)
        Y.append(y)

    Context = None

    # MODEL initial
    model = CoANE_test(feat_dim, nb_filter, win_len, t_feat).to(device)
    mem_emb = nn.Embedding(nb_node, nb_filter).to(device)
    mem_emb.weight.requires_grad = False

    # Optimizer
    # optimizer = torch.optim.Adam(list(model.parameters()), weight_decay=args.decay)
    optimizer = torch.optim.Adam(list(model.parameters()), weight_decay=args["decay"])
    # print("weight_decay = ", args.decay)
    print("weight_decay = ", args["decay"])
    # Train model
    print('Training...')
    # if controller is not given
    if 'contoller' not in args:
        r2 = r2[0].detach().cpu().item()
    else:
        # r2 = args.contoller
        r2 = args["contoller"]

    for epoch in range(num_epochs):
        idxy = list(range(R_SIZE))
        np.random.shuffle(idxy)
        sentence_step = 0
        ##for each sentence
        for i in idxy:
            x = X[i]
            y = Y[i]

            ###Split V_Batch
            np.random.seed(seed)
            np.random.shuffle(id_node)

            mbatch, num_mbatch = train_batch_processing(id_node, V_BATCH_SIZE)

            step = 0
            ###
            loss_r = []
            loss_r_p = []
            loss_r_n = []
            for j in mbatch:
                model.train()
                # Part 1--------------------------------------
                # Generate emb
                win_Encoder_feat, feat_avg = model([x[0][j], x[1][j], t_feat])

                # Update memory emb
                mem_feat_avg = mem_emb(x_callfeat)
                mem_feat_avg[j] = feat_avg
                mem_emb = nn.Embedding.from_pretrained(mem_feat_avg)
                gather_neg_emb = mem_emb(x[2][j])

                # Part 2--------------------------------------
                # Loss
                loss_pos = (((torch.mm(feat_avg[:, :int(nb_filter / 2)],
                                       torch.t(mem_feat_avg[1:, int(nb_filter / 2):])).sigmoid() + 1e-10).log() * -y[0][
                                                                                                                   j,
                                                                                                                   1:]).sum())
                loss_neg = (((torch.bmm(-feat_avg.view(-1, 1, nb_filter),
                                        torch.transpose(gather_neg_emb, 1, 2)).squeeze() ** 2) * r2).sum())

                # if args.c_f:
                if args["c_f"]:
                    # loss_mse = model.MSE(model.forward_f(feat_avg), t_feat[j]) * args.c_f  # original
                    loss_mse = model.MSE(model.forward_f(feat_avg), t_feat[j]) * args["c_f"]  # original
                else:
                    loss_mse = loss_neg * 0
                loss = loss_pos + loss_neg + loss_mse
                loss_r.append([loss.item(), loss_pos.item(), loss_neg.item(), loss_mse.item()])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
            ###
            if verbose:
                loss_r = np.array(loss_r).sum(0)
                print(" Epoch [%d/%d], \n Avg. Loss:  {:%.2f} -p {:%.2f} -n {:%.2f} -m {:%.2f}" % (epoch + 1, \
                                                                                                   num_epochs,
                                                                                                   loss_r[0] / len(
                                                                                                       id_node), \
                                                                                                   loss_r[1] / len(
                                                                                                       id_node),
                                                                                                   loss_r[2] / len(
                                                                                                       id_node),
                                                                                                   loss_r[3] / len(
                                                                                                       id_node)))
                if epoch and (epoch + 1) % 5 == 0:
                    embeddings = np.zeros((nb_node, nb_filter))
                    m = V_BATCH_SIZE
                    with torch.no_grad():
                        for k in range(R_SIZE):
                            for l in tqdm(
                                    range(1, int(np.ceil(embeddings.shape[0] / m)) + 1), mininterval=2,
                                    desc='  - (renew-embedding)   ', leave=False):
                                embeddings[((l - 1) * m + 1):(l * m + 1), :] += model(
                                    [X[k][0][((l - 1) * m + 1):(l * m + 1)], X[k][1][((l - 1) * m + 1):(l * m + 1)],
                                     t_feat])[-1].detach().cpu().squeeze().numpy()
                        embeddings /= R_SIZE

                    id_list = [PAR['word2id'][str(i)] for i in PAR['gl']]
                    embeddings = embeddings[id_list]
                    PAR['embeddings'] = embeddings
                    # accuracy = evaluation(PAR)

            sentence_step += 1
            torch.cuda.empty_cache()

    # Renew
    print("Renewing embeddings...")
    embeddings = np.zeros((nb_node, nb_filter))
    m = V_BATCH_SIZE
    with torch.no_grad():
        for k in range(R_SIZE):
            for l in tqdm(
                    range(1, int(np.ceil(embeddings.shape[0] / m)) + 1), mininterval=2,
                    desc='  - (renew-embedding)   ', leave=False):
                embeddings[((l - 1) * m + 1):(l * m + 1), :] += \
                    model([X[k][0][((l - 1) * m + 1):(l * m + 1)], X[k][1][((l - 1) * m + 1):(l * m + 1)], t_feat])[
                        -1].detach().cpu().squeeze().numpy()
        embeddings /= R_SIZE

    id_list = [PAR['word2id'][str(i)] for i in PAR['gl']]
    embeddings = embeddings[id_list]
    PAR['embeddings'] = embeddings

    if verbose == True:
        print('Training Done!')
    return PAR, model


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')


"""
================================================================
According to the readme of the DEEP-PolyU/NetBenchmark repository, I need to add 
some subclasses, like check_train_parameters and train_model, to import the new 
algorithm easily. Besides, the new algorithm should be able to inherit from the base 
class from ./models/model.py so that we can do tasks 1, 2 & 3 for benchmarking
================================================================
"""


class CoANE(Models):
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
            "window_hsize": hp.uniformint("window_size", 5, 15),
            "num_epochs": SPACE_TREE["nb_epochs"]
        }
        return space_dtree

    def train_model(self, **kwargs):
        args = kwargs
        args["data_dir"] = "./data"
        args["seed"] = 42

        if self.use_gpu:
            args["device"] = self.device
            torch.cuda.manual_seed(args["seed"])
        else:
            args["device"] = self.device
            print("--> No GPU")

        # training params
        args["verbose"] = True
        args["num_Vbatch"] = 64
        args["decay"] = 0
        args["contoller"] = 1e-3
        args["c_f"] = 1e5

        args["dataset"] = self.mat_content
        print("What is data? " + str(args["dataset"]) + " here it is")

        # Data Processing
        try:
            # network_tuple = load_obj(args.data_dir + "/{}/adj-feat-label".format(args.dataset))
            network_tuple = load_obj(args["data_dir"] + "/{}/adj-feat-label".format(args["dataset"]))
        except:
            network_tuple = read_dataset(args)
        # Random Walk
        try:
            # PAR = load_obj(args.data_dir + "/{0}/PAR/PAR_{1}".format(args.dataset))
            PAR = load_obj(args["data_dir"] + "/{0}/PAR/PAR_{1}".format(args["dataset"]))
            # print("loading PAR: ", args.data_dir + "/{0}/PAR/PAR_{1}".format(args.dataset))
            print("loading PAR: ", args["data_dir"] + "/{0}/PAR/PAR_{1}".format(args["dataset"]))
        except:
            PAR = random_walk(network_tuple, args)
        # Generating Contexts
        try:
            # Context = load_obj("./Context/Context_{0}_{1}".format(args.dataset, args.window_hsize))
            Context = load_obj("./Context/Context_{0}_{1}".format(args["dataset"], args["window_hsize"]))
            # print('Load.....Context ', "Context_{0}_{1}".format(args.dataset, args.window_hsize))
            print('Load.....Context ', "Context_{0}_{1}".format(args["dataset"], args["window_hsize"]))
        except:
            Context = Scan_Context_process(PAR, args)
            print('Save.....')

        # Training
        T1 = time()
        PAR_emb, CoANE_test = train(PAR, Context, args)
        T2 = time()
        print("Time Cost: ", T2 - T1)

        # Evaluating Link Prediction
        # accuracy = evaluation(PAR_emb)

        # save_obj(PAR_emb, "CoANE_{0}_emb".format(args.dataset))

        return PAR_emb
