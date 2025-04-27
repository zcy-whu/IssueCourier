import dgl
import torch
from utils.utils import mp2vec_feat
import pickle
import random
import numpy as np
import torch.utils.data as data

dgl.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

def construct_htg(glist, idx, time_window):
    sub_glist = glist[(idx - time_window):(idx + 1)]
    ID_dict = {}
    ID_dict_with_time = {}
    # ['developer', 'file', 'issue']
    for ntype in glist[0].ntypes:
        ID_set = set()  
        ID_dict_with_time[ntype] = {}
        for (t, g_s) in enumerate(sub_glist):
            if ntype in g_s.ndata['_ID'].keys():
                tmp_set = set(g_s.ndata['_ID'][ntype].tolist())
                ID_set.update(tmp_set)  
                ID_dict_with_time[ntype][f't{t}'] = g_s.ndata['_ID'][ntype]
        ID_dict[ntype] = {ID: idx for idx, ID in enumerate(sorted(list(ID_set)))} 

    hetero_dict = {}
    for (t, g_s) in enumerate(sub_glist):
        # [('developer', 'comment', 'issue'), ('developer', 'created', 'file'), ('developer', 'modified', 'file'), ('developer', 'report', 'issue'), ('issue', 'similar', 'file')]
        for srctype, etype, dsttype in g_s.canonical_etypes:
            src, dst = g_s.in_edges(g_s.nodes(dsttype), etype=etype)
            ID_src = g_s.ndata['_ID'][srctype]
            ID_dst = g_s.ndata['_ID'][dsttype]
            new_src = ID_src[src]
            new_dst = ID_dst[dst]
            new_new_src = [ID_dict[srctype][e.item()] for e in new_src]
            new_new_dst = [ID_dict[dsttype][e.item()] for e in new_dst]
            if etype in ['related', 'depend']: 
                continue
            else:
                hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (new_new_src, new_new_dst)
                hetero_dict[(dsttype, f'{etype}_r_t{t}', srctype)] = (new_new_dst, new_new_src)

    G_feat = dgl.heterograph(hetero_dict, num_nodes_dict={'issue': len(ID_dict['issue']), 'developer': len(ID_dict['developer']), 'file': len(ID_dict['file'])}) 

    for (t, g_s) in enumerate(sub_glist):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            weight = g_s[(srctype, etype, dsttype)].edata['w']
            if etype in ['related', 'depend']:
                continue
            else:
                G_feat[(srctype, f'{etype}_t{t}', dsttype)].edata['w'] = weight
                G_feat[(dsttype, f'{etype}_r_t{t}', srctype)].edata['w'] = weight

    return G_feat, ID_dict, ID_dict_with_time


def construct_htg_label(glist, ID_dict, issue_fixer, issue_related_developers, issue_candidate_simi, report_relations_dict, idx):
    cur_dev_dict = ID_dict['developer']
    cur_issue_dict = ID_dict['issue']
    cur_file_dict = ID_dict['file']
    cur_dev = list(ID_dict['developer'].keys())
    cur_issue = list(ID_dict['issue'].keys())
    cur_file = list(ID_dict['file'].keys())

    new_issue = glist[idx].ndata['_ID']['issue'].tolist()  
    new_issue_dict = {value: (len(cur_issue_dict) + index - len(new_issue)) for index, value in enumerate(new_issue)}  
    new_issue_ids = torch.tensor(list(new_issue_dict.values()))

    new_issue_has_relations_in_pre_graph = []
    new_edges = {}
    # report relation
    report_relations = report_relations_dict[idx]
    new_new_src = []
    new_new_dst = []
    report_weight = []
    for dev_src, issue_dst in report_relations:
        if dev_src not in cur_dev:
            continue
        new_new_src.append(cur_dev_dict[dev_src])
        new_new_dst.append(new_issue_dict[issue_dst])
    new_edges[('developer', 'report', 'issue')] = (new_new_src, new_new_dst)
    new_edges[('issue', 'report_r', 'developer')] = (new_new_dst, new_new_src)
    report_weight = torch.ones(len(new_new_dst))

    new_issue_has_relations_in_pre_graph += new_new_dst

    # comment relation
    comment_weight = []
    for srctype, etype, dsttype in glist[idx].canonical_etypes:
        if etype in ['comment']:
            src, dst = glist[idx].in_edges(glist[idx].nodes(dsttype), etype=etype)
            ID_src = glist[idx].ndata['_ID'][srctype]  
            ID_dst = glist[idx].ndata['_ID'][dsttype]  
            new_src = ID_src[src]  
            new_dst = ID_dst[dst]  
            new_new_src = []
            new_new_dst = []
            for ns, nd in zip(new_src, new_dst):
                if ns.item() not in cur_dev: 
                    continue
                new_new_src.append(cur_dev_dict[ns.item()])
                new_new_dst.append(new_issue_dict[nd.item()])
            new_edges[(srctype, f'{etype}', dsttype)] = (new_new_src, new_new_dst)
            new_edges[(dsttype, f'{etype}_r', srctype)] = (new_new_dst, new_new_src)
            comment_weight = torch.ones(len(new_new_dst))

            new_issue_has_relations_in_pre_graph += new_new_dst
    new_issue_has_relations_in_pre_graph = set(new_issue_has_relations_in_pre_graph)  

    # similar relation
    similar_relations = issue_candidate_simi[idx][1]
    new_new_src = []
    new_new_dst = []
    similar_weight = []
    for issue_src, candidates in similar_relations.items():
        file_relation_count = 0
        for file_dst, score in candidates.items():
            if file_relation_count >= 2:
                break
            new_new_src.append(new_issue_dict[issue_src])
            new_new_dst.append(cur_file_dict[file_dst])
            similar_weight.append(score)
            file_relation_count += 1
    new_edges[('issue', 'similar', 'file')] = (new_new_src, new_new_dst)
    new_edges[('file', 'similar_r', 'issue')] = (new_new_dst, new_new_src)

    G_new = dgl.heterograph(new_edges, num_nodes_dict={'issue': len(cur_issue) + len(new_issue), 'developer': len(cur_dev), 'file': len(cur_file)})

    for srctype, etype, dsttype in G_new.canonical_etypes:
        if etype in ['report', 'report_r']:
            G_new[(srctype, etype, dsttype)].edata['w'] = report_weight
        elif etype in ['comment', 'comment_r']:
            G_new[(srctype, etype, dsttype)].edata['w'] = comment_weight
        elif etype in ['similar', 'similar_r']:
            G_new[(srctype, etype, dsttype)].edata['w'] = torch.tensor(similar_weight)

    related_developers = {}
    issue_ground_truth = {}
    for ni in new_issue:
        related_devs = issue_related_developers[ni]
        related_developers[new_issue_dict[ni]] = []
        for rd in related_devs:
            if rd in cur_dev:
                related_developers[new_issue_dict[ni]] += [cur_dev_dict[rd]]

        issue_ground_truth[new_issue_dict[ni]] = []
        ground_truth = issue_fixer[ni]
        for gd in ground_truth:
            if gd in cur_dev:
                issue_ground_truth[new_issue_dict[ni]] += [cur_dev_dict[gd]]

        if len(issue_ground_truth[new_issue_dict[ni]]) == 0:
            for new_assignee in related_developers[new_issue_dict[ni]]:
                issue_ground_truth[new_issue_dict[ni]] += [new_assignee]
                break

    test_src = []
    test_dst = []
    for new_issue_index, new_issue_id in enumerate(new_issue):
        for can_dev in cur_dev:
            test_src.append(new_issue_index)
            test_dst.append(cur_dev_dict[can_dev])
    test_src_tensor = torch.tensor(test_src)
    test_dst_tensor = torch.tensor(test_dst)
    test_g = dgl.heterograph({
        ('issue', 'candidate', 'developer'): (test_src_tensor, test_dst_tensor)
    }, num_nodes_dict={'issue': len(new_issue), 'developer': len(cur_dev)})
    test_g.nodes['issue'].data['_ID'] = torch.tensor(new_issue)

    return (new_issue_ids, new_edges), issue_ground_truth, related_developers, (G_new, test_g)



def load_data(glist, issue_fixer, issue_related_developers, issue_candidate_simi, time_window, report_relations_dict, test_i):
    train_feats, train_labels, train_label_graph = [], [], []
    val_feats, val_labels, val_label_graph = [], [], []
    test_feats, test_labels, test_label_graph = [], [], []
    issue_nodes, dev_nodes, file_nodes = [], [], []

    print(f'generating datasets ')
    for i in range(len(glist[:test_i])):
        if i >= time_window:
            G_feat, ID_dict, ID_dict_with_time = construct_htg(glist[:test_i], i, time_window)
            new_issue_info, labels, related_developers, label_graph = construct_htg_label(glist[:test_i], ID_dict, issue_fixer, issue_related_developers, issue_candidate_simi, report_relations_dict, i)

            if i == len(glist[:test_i]) - 1:
                test_feats.append((G_feat, ID_dict, ID_dict_with_time))
                test_labels.append((new_issue_info, labels, related_developers))
                test_label_graph.append(label_graph)
            elif i == len(glist) - 2:
                val_feats.append((G_feat, ID_dict, ID_dict_with_time))
                val_labels.append((new_issue_info, labels, related_developers))
                val_label_graph.append(label_graph)
            else:
                train_feats.append((G_feat, ID_dict, ID_dict_with_time))
                train_labels.append((new_issue_info, labels, related_developers))
                train_label_graph.append(label_graph)

            issue_nodes += list(ID_dict['issue'].keys())
            dev_nodes += list(ID_dict['developer'].keys())
            file_nodes += list(ID_dict['file'].keys())

        issue_nodes += glist[i].ndata['_ID']['issue'].tolist()
        dev_nodes += glist[i].ndata['_ID']['developer'].tolist()
        file_nodes += glist[i].ndata['_ID']['file'].tolist()

    issue_num = max(issue_nodes) + 1
    dev_num = max(dev_nodes) + 1
    file_num = max(file_nodes) + 1

    return train_feats, train_labels, train_label_graph, val_feats, val_labels, val_label_graph, test_feats, test_labels, test_label_graph, issue_num, dev_num, file_num

