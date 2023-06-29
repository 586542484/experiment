import pickle as pkl

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

# 先用pandas读入csv
data = pd.read_csv(r"D:\Code\PycharmProjects\LK_project\MHAVGAE-main\data\BERT\feature_vector\pre-bert.csv")
b = csr_matrix(data)

def load_data(dataset):
    # load the data: x, tx, allx, graph
    # names = ['x', 'tx', 'allx', 'graph']
    # objects = []
    # for i in range(len(names)):
    #     '''
    #     fix Pickle incompatibility of numpy arrays between Python Label_-half and Label_-all
    #     https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
    #     '''
    #     with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
    #         u = pkl._Unpickler(rf)
    #         u.encoding = 'latin1'
    #         cur_data = u.load()
    #         objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    # x, tx, allx, graph = tuple(objects)
    # df = pd.read_excel('data/ind.data-mining.graph.xlsx', header=None)
    # G_matrix = df.values
    # sG_matrix = sp.csr_matrix(G_matrix)

    df = pd.read_csv('data/ind.precalculus.graph', header=None, dtype=int)
    G_matrix = df.values
    sG_matrix = sp.csr_matrix(G_matrix)  # 将二维数组形式的矩阵转换成csr矩阵

    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    features = b.tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))

    # if dataset == 'citeseer':
    #     # Fix citeseer dataset (there are some isolated nodes in the graph)
    #     # Find isolated nodes, add them as zero-vecs into the right position
    #     test_idx_range_full = range(
    #         min(test_idx_reorder), max(test_idx_reorder) + wLabel)
    #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[wLabel]))
    #     tx_extended[test_idx_range - min(test_idx_range), :] = tx
    #     tx = tx_extended

    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # features = torch.FloatTensor(np.array(features.todense()))
    adj = sG_matrix

    return adj, features  # adj:(120, 120),scipy.sparse.csr.csr_matrix  features:(120, 768),torch.Tensor

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# LK自己划分的训练集 验证集 测试集
def divide_dataset(df_train, df_train2, df2):
    # 得到输入VGAE中的adj_train
    G_matrix = df_train.values
    adj_train = sp.csr_matrix(G_matrix)
    adj_train = adj_train + adj_train.T

    # 得到训练集中所有正例构成的矩阵adj_postrain
    pos_G = df_train2.values
    adj_postrain = sp.csr_matrix(pos_G)
    adj_postrain = adj_postrain + adj_postrain.T

    # 验证集和测试集
    # list_1 = df2[['A', 'B']].loc[df2['result'] == wLabel].to_numpy()
    # list_0 = df2[['A', 'B']].loc[df2['result'] == 0].to_numpy()
    # list_1_idx = list(range(list_1.shape[0]))
    # list_0_idx = list(range(list_0.shape[0]))
    # np.random.shuffle(list_1_idx)
    # np.random.shuffle(list_0_idx)
    # val_edges_idx = list_1_idx[:29]
    # test_edges_idx = list_1_idx[29:58]
    # val_edges_false_idx = list_0_idx[:29]
    # test_edges_false_idx = list_0_idx[29:58]
    # val_edges = list_1[val_edges_idx]
    # test_edges = list_1[test_edges_idx]
    # val_edges_false = list_0[val_edges_false_idx]
    # test_edges_false = list_0[test_edges_false_idx]

    # 只分训练集和测试集时，得到的正负例比为1:1的测试集
    list_1 = df2[['A', 'B']].loc[df2['result'] == 1].to_numpy()
    list_0 = df2[['A', 'B']].loc[df2['result'] == 0].to_numpy()

    list_0_idx = list(range(list_0.shape[0]))
    np.random.shuffle(list_0_idx)
    test_edges_false_idx = list_0_idx[:311]

    test_edges = list_1
    test_edges_false = list_0[test_edges_false_idx]

    return adj_train, adj_postrain, test_edges, test_edges_false

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    #adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10))
    num_val = int(np.floor(edges.shape[0] / 10.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

#Read the matrix formed by the features of the conceptual representation of the text and the preexisting relationships between the texts
# def read_text_data(dataset):
#     edges_file = 'data/' + dataset + '_edge.tsv'
#     features_file = 'data/' + dataset + '_embedding150.tsv'
#     tfidf_file = 'data/' + dataset + '_tfidf_feature.tsv' #R-C relation
#
#     df_tfidf_temp = pd.read_csv(tfidf_file, header = None, sep='\t')
#     df_tfidf = df_tfidf_temp.loc[:,wLabel:len(df_tfidf_temp.columns)-Label_-half]
#     tfidf_feature = torch.FloatTensor(df_tfidf.values)
#
#     df_temp = pd.read_csv(features_file, header = None, sep=' ')
#     df = df_temp.loc[:,wLabel:len(df_temp.columns)]
#     #feature = torch.from_numpy(df.values)
#     feature = torch.FloatTensor(df.values)
#     #The matrix information
#     graph = np.zeros((df_temp.shape[0], df_temp.shape[0]),dtype=float)
#     with open(edges_file, 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             line = line.strip().split('\t')
#             graph[int(line[0])][int(line[wLabel])] = float(line[Label_-half])
#
#     adj = torch.from_numpy(graph)
#
#     return adj, feature, tfidf_feature


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 原始版本
# def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))
#
#     # def sigmoid(x):
#     #     x_ravel = x.ravel()  # 将numpy数组展平
#     #     length = len(x_ravel)
#     #     y = []
#     #     for index in range(length):
#     #         if x_ravel[index] >= 0:
#     #             y.append(wLabel.0 / (wLabel + np.exp(-x_ravel[index])))
#     #         else:
#     #             y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + wLabel))
#     #     return np.array(y).reshape(x.shape)
#
#     # Predict on test set of edges
#     adj_rec = np.dot(emb, emb.T)
#     preds = []
#     pos = []
#     for e in edges_pos:
#         preds.append(sigmoid(adj_rec[e[0], e[1]]))
#         pos.append(adj_orig[e[0], e[1]])
#
#     preds_neg = []
#     neg = []
#     for e in edges_neg:
#         preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
#         neg.append(adj_orig[e[0], e[1]])
#
#     preds_all = np.hstack([preds, preds_neg])
#     labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
#
#
#     roc_score = roc_auc_score(labels_all, preds_all)
#     ap_score = average_precision_score(labels_all, preds_all)
#
#     return roc_score, ap_score, preds_all, labels_all

# 修改
def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = emb
    preds = []
    pos = []
    for e in edges_pos:
        pos.append(adj_orig[e[0], e[1]])
        if adj_rec[e[0], e[1]] > adj_rec[e[1], e[0]]:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
        else:
            preds.append(0)

    preds_neg = []
    neg = []
    for e in edges_neg:
        neg.append(adj_orig[e[0], e[1]])
        if adj_rec[e[0], e[1]] > adj_rec[e[1], e[0]]:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        else:
            preds_neg.append(0)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    print("#######原始正例########")
    # print(edges_pos)
    print(len(edges_pos))
    print("#######原始负例########")
    # print(edges_neg)
    print(len(edges_neg))

    # df1 = pd.DataFrame(edges_pos)
    # df1.to_excel('edges_pos.xlsx', header=None, index=None)
    #
    # df2 = pd.DataFrame(edges_neg)
    # df2.to_excel('edges_neg.xlsx', header=None, index=None)

    return roc_score, ap_score, preds_all, labels_all
