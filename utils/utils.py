import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def prepare_labels(label_dict, num_developers):
    positive_indices_list = []
    for issue_id, pos_devs in label_dict.items():
        pos_idxs = [dev_id for dev_id in pos_devs]
        pos_idxs = [min(dev_id, num_developers - 1) for dev_id in pos_idxs]
        positive_indices_list.append(torch.tensor(pos_idxs, dtype=torch.long))

    return positive_indices_list


def accuracy_at_one_multi_label(scores, labels, K=10):
    _, predicted_indices = torch.max(scores, dim=1)

    correct_predictions = 0
    for predicted_idx, positive_indices in zip(predicted_indices, labels):
        if predicted_idx.item() in positive_indices:
            correct_predictions += 1

    accuracy = correct_predictions / scores.shape[0]

    reciprocal_rank = []
    at_k = [0] * K 
    num_samples = scores.shape[0]
    for i in range(num_samples):
        candidate_score = {dev_id: score for dev_id, score in enumerate(scores[i])}
        sorted_candidate_score = dict(sorted(candidate_score.items(), key=lambda item: item[1], reverse=True))
        ranked_candidates = list(sorted_candidate_score.keys())
        ranked_candidates_index = {dev_id: rank_index for rank_index, dev_id in enumerate(ranked_candidates)}
        relevant_indices = labels[i].tolist()
        if len(relevant_indices) > 0: 
            top_rank = min([ranked_candidates_index[dev_id] for dev_id in relevant_indices])
            reciprocal_rank.append(1 / (top_rank+1))
            if (top_rank + 1) <= K:
                at_k[top_rank] += 1

    mrr = sum(reciprocal_rank) / num_samples
    hit_k = [sum(at_k[:k + 1]) / num_samples for k in range(K)]

    return accuracy, mrr, hit_k  

class MultiLabelRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MultiLabelRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, scores, positive_indices, num_developers):
        assert scores.size(0) == len(positive_indices)

        loss = 0.0
        for i, pos_idxs in enumerate(positive_indices):
            neg_idxs = torch.tensor([j for j in range(num_developers) if j not in pos_idxs], dtype=torch.long,
                                    device=scores.device)

            pos_scores = scores[i, pos_idxs]
            neg_scores = scores[i, neg_idxs]

            hinge_losses = F.relu(neg_scores.unsqueeze(1) - pos_scores.unsqueeze(0) + self.margin)
            hinge_losses = hinge_losses.view(-1)  
            loss += hinge_losses.sum()

        num_pos_neg_pairs = sum(len(pos_idxs) * (num_developers - len(pos_idxs)) for pos_idxs in positive_indices)
        if num_pos_neg_pairs == 0:  
            return torch.tensor(0.0, device=scores.device)
        loss /= num_pos_neg_pairs
        return loss
