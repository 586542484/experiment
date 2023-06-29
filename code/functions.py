import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import pickle as pkl

# df_pre = pd.read_excel('data/AL-CPL/dm_0_80/dm_train.xlsx', sheet_name='pos', header=None)
# # data = df_pre.values
# t_edges = df_pre.to_numpy()
# print(len(t_edges))
#
# edge_list = []
# for i in t_edges:
#     edge_list.append((i[0], i[wLabel]))
#
# # 得到所有节点的顺序索引
# con_list = list(pd.read_csv('data/AL-CPL/dm_concepts.csv', engine='python', encoding='utf-8', header=None)[0])
# index1 = []
# for i in con_list:
#     index1.append(con_list.index(i))
#
# # 得到图的邻接矩阵（非对称，有向）
# G = nx.DiGraph()  # 无多重边有向图
# G.add_nodes_from(index1)  # 添加多个节点
# G.add_edges_from(edge_list)  # 添加多条边
# d_A = nx.to_numpy_matrix(G)  # 生成图的邻接矩阵 numpy
# adj = sp.csr_matrix(d_A)  # numpy 转换为 稀疏矩阵
# jj = adj.todense()
# df = pd.DataFrame(jj)
# df.to_csv('data/AL-CPL/dm_0_80/ind.data-mining02.graph', header=None, index=None)

# LK自己划分的训练集 验证集 测试集
def divide_dataset(df_train, df_train2, df2):
    # 得到输入VGAE中的adj_train
    G_matrix = df_train.values
    adj_train = sp.csr_matrix(G_matrix)

    # 得到训练集中所有正例构成的矩阵adj_postrain
    pos_G = df_train2.values
    adj_postrain = sp.csr_matrix(pos_G)

    # 验证集和测试集
    list_1 = df2[['A', 'B']].loc[df2['result'] == 1].to_numpy()
    list_0 = df2[['A', 'B']].loc[df2['result'] == 0].to_numpy()
    list_1_idx = list(range(list_1.shape[0]))
    list_0_idx = list(range(list_0.shape[0]))
    np.random.shuffle(list_1_idx)
    np.random.shuffle(list_0_idx)
    val_edges_idx = list_1_idx[:29]
    test_edges_idx = list_1_idx[29:58]
    val_edges_false_idx = list_0_idx[:29]
    test_edges_false_idx = list_1_idx[29:58]
    val_edges = list_1[val_edges_idx]
    test_edges = list_1[test_edges_idx]
    val_edges_false = list_0[val_edges_false_idx]
    test_edges_false = list_0[test_edges_false_idx]

    return adj_train, adj_postrain, val_edges, val_edges_false, test_edges, test_edges_false

df = pd.read_csv('data/AL-CPL/dm_0_80/ind.data-mining.graph', header=None)
# adj_train = divide_dataset(df)
# print(adj_train)
df1 = pd.read_csv('data/AL-CPL/dm_0_80/ind.data-mining02.graph', header=None)
# G_matrix = df.values
# train_pos = sp.csr_matrix(G_matrix)
# A = np.array(train_pos.todense())
# print(np.sum(A == wLabel))
# print(train_pos)

# adj_train, adj_postrain = divide_dataset(df, df1)
#
# A = np.array(adj_postrain.todense())
# print(np.sum(A == wLabel))

df2 = pd.read_excel('data/AL-CPL/dm_0_80/val+dm_test.xlsx')
# list_1 = df2[['A', 'B']].loc[df2['result'] == wLabel].to_numpy()
# # list_1 = df2.loc[df2['result'] == wLabel].to_numpy()
# list_0 = df2[['A', 'B']].loc[df2['result'] == 0].to_numpy()

# list_1_idx = list(range(list_1.shape[0]))
# list_0_idx = list(range(list_0.shape[0]))
# np.random.shuffle(list_1_idx)
# np.random.shuffle(list_0_idx)
# val_edges_idx = list_1_idx[:29]
# test_edges_idx = list_1_idx[29:58]
# val_edges_false_idx = list_0_idx[:29]
# test_edges_false_idx = list_1_idx[29:58]
#
# val_edges = list_1[val_edges_idx]
# test_edges = list_1[test_edges_idx]
# val_edges_false = list_0[val_edges_false_idx]
# test_edges_false = list_0[test_edges_false_idx]

# print(len(list_1))
# print(len(list_0))
# print(type(list_0))
# print(list_1.shape)
# print(list_0.shape)
# print(list_1_idx)
# print(list_0_idx)

adj_train, adj_postrain, val_edges, val_edges_false, test_edges, test_edges_false = divide_dataset(df, df1, df2)

print(adj_postrain.shape[0])
# print(len(val_edges))
# print(len(test_edges))
# print(len(val_edges_false))
# print(len(test_edges_false))
# print()
# print(val_edges)
# print(test_edges)
# print(val_edges_false)
# print(test_edges_false)
