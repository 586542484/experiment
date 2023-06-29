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
    # t_adj, t_features, tfidf_feature = read_text_data(args.dataset_str)  # 文本特征与矩阵
    n_nodes, feat_dim = features.shape
    # t_n_nodes, t_feat_dim = t_features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    # lk按照弱标签切分的数据集
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

    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj_up = adj_train_up
    adj_down = adj_train_down

    # Some preprocessing
    # t_adj_norm = preprocess_graph(t_adj) #进行归一化处理
    adj_norm_up = preprocess_graph(adj_up)  # 进行归一化处理
    adj_norm_down = preprocess_graph(adj_down)
    adj_label_up = adj_postrain_up + sp.eye(adj_postrain_up.shape[0])
    adj_label_down = adj_postrain_down + sp.eye(adj_postrain_down.shape[0])
    # adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
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

    # dfx = pd.DataFrame(preds_all_temp)
    # dfx.to_excel('预测标签.xlsx', header=None, index=None)

    # print(float('%.4f' % ACC))
    # print(float('%.4f' % pre))
    # print(float('%.4f' % re))
    # print(float('%.4f' % F1))
    # print('Test map_score score: ', float('%.4f' %ap_score))
    # print(float('%.4f' % roc_score))
    # print(hidden_emb)
    # adj_rec = np.dot(hidden_emb, hidden_emb.T)
    # df = pd.DataFrame(adj_rec)
    # df.to_excel('bbb.xlsx')

    # reconstrution_thr = np.arange(0.wLabel, wLabel, 0.wLabel)
    # for thr in reconstrution_thr:
    #     preds_all_temp[preds_all_temp >= thr] = wLabel
    #     preds_all_temp[preds_all_temp < thr] = 0
    #
    #     ACC = accuracy_score(labels_all, preds_all_temp)
    #     F1 = f1_score(labels_all, preds_all_temp)
    #     pre = precision_score(labels_all, preds_all_temp)
    #     re = recall_score(labels_all, preds_all_temp)
    #
    #     print('分类阈值：', thr)
    #     print('Test acc score: ', float('%.4f' %ACC))
    #     print('Test f1 score: ', float('%.4f' %F1))
    #     print('Test precision score: ', float('%.4f' %pre))
    #     print('recall_score: ', float('%.4f' % re))
    #     # print('Test map_score score: ', float('%.4f' %ap_score))
    #     print('Test auc score: ', float('%.4f' %roc_score))

    # scaler = MinMaxScaler()  # 实例化
    # scaler = scaler.fit(adj_rec)  # fit，在这里本质是生成min(x)和max(x)
    # result = scaler.transform(adj_rec)  # 通过接口导出结果

    # scaler = StandardScaler()
    # result = scaler.fit_transform(adj_rec)
    #
    # print(result.shape)
    # df = pd.DataFrame(adj_rec)
    # df.to_excel('重构矩阵.xlsx', header=None, index=None)


