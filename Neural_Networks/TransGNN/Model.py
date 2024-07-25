import torch
from torch import nn
import torch.nn.functional as F
#from Params import args
from .Utils.Utils import pairPredict
from .Transformer import Encoder_Layer, TransformerEncoderLayer

class TransGNN(nn.Module):
    def __init__(self, user=9, item=10, latdim=32, num_head=4, dropout=0.5, block_num=2, edgeSampRate=0.1):
        super(TransGNN, self).__init__()
        self.recurrent = False

        self.user = user
        self.item = item
        self.block_num = block_num
        self.edgeSampRate = edgeSampRate

        self.user_embeding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(user, latdim)))
        self.item_embeding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(item, latdim)))
        self.user_transformer_encoder = TransformerEncoderLayer(d_model=latdim, num_heads=num_head, dropout=dropout)
        self.item_transformer_encoder = TransformerEncoderLayer(d_model=latdim, num_heads=num_head, dropout=dropout)
    

    def user_transformer_layer(self, embeds, mask=None):
        assert len(embeds.shape) <= 3, "Shape Error, embed shape is {}, out of size!".format(embeds.shape)
        if len(embeds.shape) == 2:
            embeds = embeds.unsqueeze(dim=0)
            embeds = self.user_transformer_encoder(embeds, mask)
            embeds = embeds.squeeze()
        else:
            embeds = self.user_transformer_encoder(embeds, mask)
        
        return embeds
    
    
    def item_transformer_layer(self, embeds, mask=None):
        assert len(embeds.shape) <= 3, "Shape Error, embed shape is {}, out of size!".format(embeds.shape)
        if len(embeds.shape) == 2:
            embeds = embeds.unsqueeze(dim=0)
            embeds = self.item_transformer_encoder(embeds, mask)
            embeds = embeds.squeeze()
        else:
            embeds = self.item_transformer_encoder(embeds, mask)
        
        return embeds
    

    def gnn_message_passing(self, adj, embeds):
        return torch.spmm(adj, embeds)
    
    def forward(self, adj):
        print("in TransGNN, adj:\n", adj, "\n")
        
        embeds = [torch.concat([self.user_embeding, self.item_embeding], dim=0)]     
        for i in range(self.block_num):
            tmp_embeds = self.gnn_message_passing(adj, embeds[-1])
            embeds.append(tmp_embeds)

        embeds = sum(embeds)
        user_embeds = embeds[:self.user]
        item_embeds = embeds[self.user:]
        return embeds, user_embeds, item_embeds

    def pickEdges(self, adj):
        idx = adj._indices()
        rows, cols = idx[0, :], idx[1, :]
        mask = torch.logical_and(rows <= self.user, cols > self.user)
        rows, cols = rows[mask], cols[mask]
        edgeSampNum = int(self.edgeSampRate * rows.shape[0])
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        edgeids = torch.randint(rows.shape[0], [edgeSampNum])
        pckUsrs, pckItms = rows[edgeids], cols[edgeids] - self.user
        return pckUsrs, pckItms
    
    def pickRandomEdges(self, adj):
        edgeNum = adj._indices().shape[1]
        edgeSampNum = int(self.edgeSampRate * edgeNum)
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        rows = torch.randint(self.user, [edgeSampNum])
        cols = torch.randint(self.item, [edgeSampNum])
        return rows, cols
    
    def bprLoss(self, user_embeding, item_embeding, ancs, poss, negs):
        ancEmbeds = user_embeding[ancs]
        posEmbeds = item_embeding[poss]
        negEmbeds = item_embeding[negs]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - ((scoreDiff).sigmoid() + 1e-6).log().mean()
        return bprLoss
    
    def calcLosses(self, ancs, poss, negs, adj):
        embeds, user_embeds, item_embeds = self.forward(adj)
        user_embeding, item_embeding = embeds[:self.user], embeds[self.user:]

        bprLoss = self.bprLoss(user_embeding, item_embeding, ancs, poss, negs) + self.bprLoss(user_embeds, item_embeds, ancs, poss, negs)
        return bprLoss
    
    def predict(self, adj):
        embeds, user_embeds, item_embeds = self.forward(adj)
        return user_embeds, item_embeds