import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from tqdm import tqdm
from dgl.nn.pytorch import GATConv, HeteroGraphConv, GraphConv
import numpy as np
import math

class RelationAgg(nn.Module):
    def __init__(self, n_inp: int, n_hid: int):
        super(RelationAgg, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(n_inp, n_hid),
            nn.Tanh(),
            nn.Linear(n_hid, 1, bias=False)
        )
        self.norm = nn.LayerNorm(n_hid)
        self.rel_weight = {'issue': {'report': 0.5, 'comment': 0.3, 'similar_r': 1},
                           'developer': {'report_r': 1, 'comment_r': 1, 'related': 0.3, 'created_r': 1, 'removed_r': 1},
                           'file': {'created': 1, 'removed': 1, 'similar': 1, 'depend': 0.5, 'depend_r': 0.5}}

class HTGNNLayer(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, n_heads: int, timeframe: list, norm: bool,
                 device: torch.device, dropout: float):

        super(HTGNNLayer, self).__init__()

        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_heads = n_heads
        self.timeframe = timeframe
        self.norm = norm
        self.dropout = dropout
        self.device = device

        # intra reltion aggregation modules
        self.intra_rel_agg = nn.ModuleDict({
            etype: GATConv(n_inp, n_hid, n_heads, feat_drop=dropout, allow_zero_in_degree=True)
            for srctype, etype, dsttype in graph.canonical_etypes
        })

        # inter relation aggregation modules
        self.inter_rel_agg = nn.ModuleDict({
            ttype: RelationAgg(n_hid, n_hid)
            for ttype in timeframe
        })

        self.res_fc = nn.ModuleDict()
        self.res_weight = nn.ParameterDict()
        for ntype in graph.ntypes:
            self.res_fc[ntype] = nn.Linear(n_inp, n_heads * n_hid)
            self.res_weight[ntype] = nn.Parameter(torch.randn(1))
        self.leakyrelu = nn.LeakyReLU()

        # LayerNorm
        if norm:
            self.norm_layer = nn.ModuleDict({ntype: nn.LayerNorm(n_hid) for ntype in graph.ntypes})

        self.dense_biinter = nn.ModuleDict({ntype: nn.Linear(n_hid, n_hid) for ntype in graph.ntypes})
        self.dense_siinter = nn.ModuleDict({ntype: nn.Linear(n_hid, n_hid) for ntype in graph.ntypes})
        self.reset_parameters()

    def reset_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        """Reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        for ntype in self.res_fc:
            nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)
        for ntype in self.dense_biinter:
            init_weights(self.dense_biinter[ntype])
        for ntype in self.dense_siinter:
            init_weights(self.dense_siinter[ntype])


    def feat_interaction(self, feature_embedding, fun_bi, fun_si, dimension):
        summed_features_emb_square = (torch.sum(feature_embedding, dim=dimension)).pow(2)  
        squared_sum_features_emb = torch.sum(feature_embedding.pow(2), dim=dimension)  
        deep_fm = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        deep_fm = self.leakyrelu(fun_bi(deep_fm))
        bias_fm = self.leakyrelu(fun_si(feature_embedding.sum(dim=dimension)))
        nfm = deep_fm + bias_fm
        return nfm


    def forward(self, graph: dgl.DGLGraph, node_features: dict):
        # same type neighbors aggregation
        intra_features = dict({ttype: {} for ttype in self.timeframe})
        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            ttype = etype.split('_')[-1]
            dst_feat = self.intra_rel_agg[etype](rel_graph, (node_features[stype][ttype], node_features[dtype][ttype]), rel_graph[(stype, etype, dtype)].edata['w'])
            intra_features[ttype][(stype, etype, dtype)] = dst_feat.squeeze()  

        # different types aggregation 
        inter_features = dict({ntype: {} for ntype in graph.ntypes})
        for ttype in intra_features.keys():
            for ntype in graph.ntypes:
                types_features = []
                for stype, etype, dtype in intra_features[ttype]:
                    reltype = etype.split('_t')[0]
                    if ntype == dtype:
                        types_features.append(intra_features[ttype][(stype, etype, dtype)])
                try:
                    types_features = torch.stack(types_features, dim=1)
                    out_feat = self.feat_interaction(types_features, self.dense_biinter[ntype], self.dense_siinter[ntype], dimension=1)
                    inter_features[ntype][ttype] = out_feat
                except Exception as e:
                    inter_features[ntype][ttype] = []


        new_features = {}  
        for ntype in inter_features:
            new_features[ntype] = {}
            alpha = torch.sigmoid(self.res_weight[ntype])
            for ttype in self.timeframe:
                try:
                    new_features[ntype][ttype] = inter_features[ntype][ttype] + node_features[ntype][ttype] * alpha
                    if self.norm:
                        new_features[ntype][ttype] = self.norm_layer[ntype](new_features[ntype][ttype]) 
                except Exception as e:
                    new_features[ntype][ttype] = []

        return new_features



class HTGNN(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, issue_num, dev_num, file_num, n_inp: int, n_hid: int, n_layers: int, n_heads: int, time_window: int, norm: bool, device: torch.device, dropout: float = 0.2):
        super(HTGNN, self).__init__()

        self.n_inp     = n_inp
        self.n_hid     = n_hid
        self.n_layers  = n_layers
        self.n_heads   = n_heads
        self.timeframe = [f't{_}' for _ in range(time_window + 1)]
        self.node_num = {'issue': issue_num, 'developer': dev_num, 'file': file_num}

        self.gnn_layers     = nn.ModuleList([HTGNNLayer(graph, n_hid, n_hid, n_heads, self.timeframe, norm, device, dropout) for _ in range(n_layers)])

        self.embed = nn.ModuleDict({ntype: nn.Embedding(self.node_num[ntype], n_hid) for ntype in graph.ntypes})

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embed.values():
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, graph: dgl.DGLGraph, ID_dict: dict, ID_dict_with_time: dict):
        inp_feat = {}
        for ntype in graph.ntypes:
            inp_feat[ntype] = {}
            for ttype in self.timeframe:
                inp_feat[ntype][ttype] = torch.zeros(graph.num_nodes(ntype), self.n_hid)
                node_id = ID_dict_with_time[ntype][ttype]
                for id in node_id:
                    inp_feat[ntype][ttype][ID_dict[ntype][id.item()]] = self.embed[ntype](id)

        # gnn
        for i in range(self.n_layers):
            inp_feat = self.gnn_layers[i](graph, inp_feat)

        out_feat = dict()
        for ntype in graph.ntypes:
            try:
                out_feat[ntype] = sum(inp_feat[ntype][ttype] for ttype in inp_feat[ntype].keys())
            except Exception as e:
                out_feat[ntype] = []

        return out_feat


class LinkPredictor(nn.Module):
    def __init__(self, n_inp: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(n_inp * 2, n_inp)
        self.fc2 = nn.Linear(n_inp, n_classes)

    def apply_edges(self, edges):
        x = torch.cat([edges.src['h'], edges.dst['h']], 1)  
        y = self.fc2(F.relu(self.fc1(x)))
        return {'score': y}

    def forward(self, graph: dgl.DGLGraph, dev_node_feat: torch.tensor, issue_node_feat: torch.tensor):
        with graph.local_scope():
            graph.nodes['developer'].data['h'] = dev_node_feat
            graph.nodes['issue'].data['h'] = issue_node_feat
            graph.apply_edges(self.apply_edges)

            reshape_size = (graph.number_of_nodes('issue'), graph.number_of_nodes('developer'))
            scores = graph.edata['score'].view(*reshape_size)

            return scores

