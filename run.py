import pickle
import dgl
from dgl.data.utils import load_graphs
from model import *
from utils.pytorchtools import EarlyStopping
from utils.utils import *
from utils.data import *
import time

dgl.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


def evaluate(model, val_feats, val_labels, val_label_graph):
    model.eval()
    with torch.no_grad():
        for ((G_feat, ID_dict, ID_dict_with_time), (new_issue_info, labels, related_developers), (G_new, no_label)) in zip(val_feats, val_labels, val_label_graph):
            h = model[0](G_feat, ID_dict, ID_dict_with_time)
            new_issue_ids, _ = new_issue_info
            all_pred = model[1](no_label, h['developer'], h['issue'][-len(new_issue_ids):])

            positive_indices_list = prepare_labels(labels, G_new.number_of_nodes('developer'))
            related_indices_list = prepare_labels(related_developers, G_new.number_of_nodes('developer'))

            loss_function = MultiLabelRankingLoss()
            loss = loss_function(all_pred, positive_indices_list, G_new.number_of_nodes('developer'))
            # accuracy, mrr, hit_k = accuracy_at_one_multi_label(all_pred, related_indices_list)
            accuracy, mrr, hit_k = accuracy_at_one_multi_label(all_pred, positive_indices_list)

    return loss, accuracy, mrr, hit_k, all_pred, positive_indices_list, related_indices_list, h['developer'], h['issue'][-len(new_issue_ids):], (new_issue_info, labels, related_developers)


# dotCMS_core, hazelcast_hazelcast, eclipse_che, prestodb_presto, wildfly_wildfly
repo_name = 'dotCMS_core'
# core, hazelcast, che, presto, wildfly
graph_name = 'core'
glist, _ = load_graphs(f'dataset/{repo_name}/{graph_name}.bin')
with open(f'dataset/{repo_name}/issue_fixer', 'rb') as f:
    issue_fixer = pickle.load(f)
with open(f'dataset/{repo_name}/issue_related_developers', 'rb') as f:
    issue_related_developers = pickle.load(f)
with open(f'dataset/{repo_name}/report_relations_dict', 'rb') as f:
    report_relations_dict = pickle.load(f)
device = torch.device('cuda:0')


model_out_path = f'output/{repo_name}/'

test_i = 9
for time_window in range(2,3):
    with open(f'dataset/{repo_name}/issue_chunk_candidates_score_{time_window}', 'rb') as f:
        issue_candidate_simi = pickle.load(f)

    (train_feats, train_labels, train_label_graph, val_feats, val_labels, val_label_graph, test_feats, test_labels, test_label_graph,
     issue_num, dev_num, file_num) \
        = load_data(glist, issue_fixer, issue_related_developers, issue_candidate_simi, time_window,
                         report_relations_dict, test_i+1)
    graph_atom = train_feats[0][0]

    htgnn = HTGNN(graph=graph_atom, issue_num=issue_num, dev_num=dev_num, file_num=file_num, n_inp=128, n_hid=32, n_layers=2, n_heads=1, time_window=time_window, norm=True, device=device)
    predictor = LinkPredictor(n_inp=32, n_classes=1)
    model = nn.Sequential(htgnn, predictor)

    print(f'---------------G{test_i} Time Window: {time_window}---------------------')
    print(f'# params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    optim = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4) 

    early_stopping = EarlyStopping(patience=10, verbose=True, path=f'{model_out_path}/checkpoint_HTGNN_{time_window}.pt')

    for epoch in range(100):
        model.train()
        for train_index, ((G_feat, ID_dict, ID_dict_with_time), (new_issue_info, labels, related_developers), (G_new, no_label)) in enumerate(zip(train_feats, train_labels, train_label_graph)):
            h = model[0](G_feat, ID_dict, ID_dict_with_time)
            new_issue_ids, _ = new_issue_info
            all_pred = model[1](no_label, h['developer'], h['issue'][-len(new_issue_ids):])

            positive_indices_list = prepare_labels(labels, G_new.number_of_nodes('developer'))
            related_indices_list = prepare_labels(related_developers, G_new.number_of_nodes('developer'))

            loss_function = MultiLabelRankingLoss()
            loss = loss_function(all_pred, positive_indices_list, G_new.number_of_nodes('developer'))
            
            accuracy, mrr, hit_k = accuracy_at_one_multi_label(all_pred, positive_indices_list)

            optim.zero_grad()
            loss.backward()  
            optim.step()
            print(f'loss: {loss}, accuracy: {accuracy}, mrr: {mrr}, hit_k: {hit_k}')

        val_loss, val_accuracy, val_mrr, val_hit_k, val_all_pred, val_positive_indices_list, val_related_indices_list, val_dev_h, val_new_issue_h, val_new_issue_info  = evaluate(model, val_feats, val_labels, val_label_graph)
        print(f'epoch: {epoch}, val loss: {val_loss}, val accuracy: {val_accuracy}, val mrr: {val_mrr}, val hit_k: {val_hit_k}')

        test_loss, test_accuracy, test_mrr, test_hit_k, test_all_pred, test_positive_indices_list, test_related_indices_list, test_dev_h, test_new_issue_h, test_new_issue_info = evaluate(model, test_feats, test_labels, test_label_graph)
        print(f'epoch: {epoch}, test loss: {test_loss}, test accuracy: {test_accuracy}, test mrr: {test_mrr}, test hit_k: {test_hit_k}') 

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(f'{model_out_path}/checkpoint_HTGNN_{time_window}.pt'))
    test_loss, test_accuracy, test_mrr, test_hit_k, test_all_pred, test_positive_indices_list, test_related_indices_list, test_dev_h, test_new_issue_h, test_new_issue_info = evaluate(model, test_feats, test_labels, test_label_graph)
    print(f'test accuracy: {test_accuracy}, test mrr: {test_mrr}, test hit_k: {test_hit_k}')

