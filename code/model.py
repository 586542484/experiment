import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttentionLayer, fully_connectedLyer

#Define Multi-Head Attention
class MHGAT(nn.Module):
    def __init__(self, input_feat_dim, output_feat_dim, dropout, alpha, nheads):
        super(MHGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(input_feat_dim, output_feat_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.fc = fully_connectedLyer(output_feat_dim * nheads, input_feat_dim)

    def forward(self, x, adj):
        adj = adj.to_dense()
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        return self.fc(x)
        
class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, alpha, nheads):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        #self.input_feat_dim = input_feat_dim
        #self.output_feat_dim = input_feat_dim

        # Multi-Head Attention
        self.input_feat_dim = input_feat_dim
        self.output_feat_dim = input_feat_dim
        self.gat = MHGAT(self.input_feat_dim, self.output_feat_dim, dropout, alpha, nheads)

        #fusion
        # self.gatefusion = gatedFusion(input_feat_dim, dropout, alpha, nheads)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        x = self.gat(x, adj)
        mu, logvar = self.encode(x, adj)
        
        z = self.reparameterize(mu, logvar)
        
        return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)   #dropout的training = True
        adj = self.act(torch.mm(z, z.t()))
        return adj
