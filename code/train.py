from __future__ import division
from __future__ import print_function

import argparse
import time

import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

from model import GCNModelVAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, divide_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, \
    roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer wLabel.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer Label_-half.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (wLabel - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.04, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--alpha', type=float, default=0.0, help='Alpha for the leaky_relu.')
parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
parser.add_argument('--dataset-str', type=str, default='precalculus', help='type of dataset.')

args = parser.parse_args()


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def gae_for(args, seed_num):
    np.random.seed(seed_num)
    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_data(args.dataset_str)

    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    # Dataset divided by weak labels
    df = pd.read_csv(
        r'D:\Code\PycharmProjects\LK_project\MHAVGAE-main\semantic_similarity\wLabel(PMI+BERT)\测试一个领域(bert前300词+PMI弱标签)\Label_30%\ind.precalculus.up.graph',
        header=None)
    df1 = pd.read_csv(
        r'D:\Code\PycharmProjects\LK_project\MHAVGAE-main\semantic_similarity\wLabel(PMI+BERT)\测试一个领域(bert前300词+PMI弱标签)\Label_30%\ind.precalculus02.up.graph',
        header=None)
    df2 = pd.read_excel(
        r'D:\Code\PycharmProjects\LK_project\MHAVGAE-main\semantic_similarity\wLabel(PMI+BERT)\测试一个领域(bert前300词+PMI弱标签)\Label_30%\pre_test.xlsx')

    df3 = pd.read_csv(
        r'D:\Code\PycharmProjects\LK_project\MHAVGAE-main\semantic_similarity\wLabel(PMI+BERT)\测试一个领域(bert前300词+PMI弱标签)\Label_30%\ind.precalculus.down.graph',
        header=None)
    df4 = pd.read_csv(
        r'D:\Code\PycharmProjects\LK_project\MHAVGAE-main\semantic_similarity\wLabel(PMI+BERT)\测试一个领域(bert前300词+PMI弱标签)\Label_30%\ind.precalculus02.down.graph',
        header=None)

    adj_train_up, adj_postrain_up, test_edges, test_edges_false = divide_dataset(df, df1, df2)
    adj_train_down, adj_postrain_down, test_edges2, test_edges_false2 = divide_dataset(df3, df4, df2)

    adj_up = adj_train_up
    adj_down = adj_train_down

    adj_norm_up = preprocess_graph(adj_up)  # perform normalization processing
    adj_norm_down = preprocess_graph(adj_down)
    adj_label_up = adj_postrain_up + sp.eye(adj_postrain_up.shape[0])
    adj_label_down = adj_postrain_down + sp.eye(adj_postrain_down.shape[0])

    adj_label_up = torch.FloatTensor(adj_label_up.toarray())
    adj_label_down = torch.FloatTensor(adj_label_down.toarray())

    pos_weight_up = torch.DoubleTensor(np.array(float(adj_up.shape[0] * adj_up.shape[0] - adj_up.sum()) / adj_up.sum()))
    norm_up = adj_up.shape[0] * adj_up.shape[0] / float((adj_up.shape[0] * adj_up.shape[0] - adj_up.sum()) * 2)

    pos_weight_down = torch.DoubleTensor(np.array(float(adj_down.shape[0] * adj_down.shape[0] - adj_down.sum()) / adj_down.sum()))
    norm_down = adj.shape[0] * adj.shape[0] / float((adj_down.shape[0] * adj_down.shape[0] - adj_down.sum()) * 2)



    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout, args.alpha, args.nb_heads)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    hidden_emb1 = None
    hidden_emb2 = None
    for epoch in range(args.epochs):
        t1 = time.time()
        model.train()
        optimizer.zero_grad()

        # Forward pass and backpropagation for the first call
        recovered1, mu1, logvar1 = model(features, adj_norm_up)
        loss1 = loss_function(preds=recovered1, labels=adj_label_up,
                              mu=mu1, logvar=logvar1, n_nodes=n_nodes,
                              norm=norm_up, pos_weight=pos_weight_up)
        loss1.backward()
        cur_loss1 = loss1.item()
        optimizer.step()

        # Clear gradients and forward pass for the second call
        optimizer.zero_grad()
        recovered2, mu2, logvar2 = model(features, adj_norm_down)
        loss2 = loss_function(preds=recovered2, labels=adj_label_down,
                              mu=mu2, logvar=logvar2, n_nodes=n_nodes,
                              norm=norm_down, pos_weight=pos_weight_down)
        loss2.backward()
        cur_loss2 = loss2.item()
        optimizer.step()

        # Calculate hidden embeddings
        hidden_emb1 = mu1.data.numpy()
        hidden_emb2 = mu2.data.numpy()

        # Compute upper and lower triangular matrices and concatenate them
        hidden_emb1 = np.dot(hidden_emb1, hidden_emb1.T)
        hidden_emb2 = np.dot(hidden_emb2, hidden_emb2.T)
        hidden_emb_up = np.triu(hidden_emb1, 1)
        hidden_emb_down = np.tril(hidden_emb2, -1)
        hidden_emb = hidden_emb_up + hidden_emb_down

        # Calculate training accuracy for both calls
        train_acc1 = get_acc(recovered1, adj_label_up)
        train_acc2 = get_acc(recovered2, adj_label_down)

        # roc_curr, ap_curr, _, _ = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        # print("Epoch:", '%04d' % (epoch + wLabel), "train_loss=", "{:.5f}".format(cur_loss),
        #       "train_acc=", "{:.5f}".format(train_acc),
        #       "val_auc=", "{:.5f}".format(roc_curr),
        #       "val_ap=", "{:.5f}".format(ap_curr),
        #       "time=", "{:.5f}".format(time.time() - t)
        #       )

        print("Epoch:", '%04d' % (epoch + 1), "train_loss1=", "{:.5f}".format(cur_loss1),
              "train_acc1=", "{:.5f}".format(train_acc1), "train_loss2=", "{:.5f}".format(cur_loss2),
              "train_acc2=", "{:.5f}".format(train_acc2),
              "time=", "{:.5f}".format(time.time() - t1)
              )

    print("Optimization Finished!")

    roc_score, ap_score, preds_all, labels_all = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    return roc_score, ap_score, preds_all, labels_all, hidden_emb


if __name__ == '__main__':
    # roc_score, ap_score, preds_all, labels_all = gae_for(args, args.seed)
    roc_score, ap_score, preds_all, labels_all, hidden_emb = gae_for(args, args.seed)

    preds_all_temp = preds_all
    preds_all_temp[preds_all_temp >= 0.6] = 1
    preds_all_temp[preds_all_temp < 0.6] = 0

    ACC = accuracy_score(labels_all, preds_all_temp)
    F1 = f1_score(labels_all, preds_all_temp)
    pre = precision_score(labels_all, preds_all_temp)
    re = recall_score(labels_all, preds_all_temp)

    print('Test acc score: ', float('%.4f' %ACC))
    print('Test precision score: ', float('%.4f' %pre))
    print('recall_score: ', float('%.4f' % re))
    print('Test f1 score: ', float('%.4f' %F1))  
    print('Test auc score: ', float('%.4f' %roc_score))

