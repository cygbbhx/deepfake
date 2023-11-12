import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_loss_function(name='crossentropy'):
    loss_dict = {'crossentropy': nn.CrossEntropyLoss(),
                 'SimCLR': SimCLR_Loss(),
                 'SupCon': SupConLoss(),
                 'triplet': TripletLoss()}
    
    is_contrastive_learning = name != 'crossentropy'
    
    return loss_dict[name], is_contrastive_learning

# 출처 : https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7

class SimCLR_Loss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z, labels):

        N = z.shape[0]
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # Create a mask for positive samples (similar pairs)
        positive_mask = labels == 1
        positive_samples = sim[positive_mask]

        # Create a mask for negative samples (dissimilar pairs)W
        negative_mask = labels == 0
        negative_samples = sim[negative_mask]

        # Combine positive and negative samples
        logits = torch.cat((positive_samples, negative_samples), dim=0)
        
        # Create labels for positive and negative samples
        positive_labels = torch.ones(positive_samples.shape[0], dtype=torch.long)
        negative_labels = torch.zeros(negative_samples.shape[0], dtype=torch.long)
        labels = torch.cat((positive_labels, negative_labels), dim=0).to(self.device)

        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss
    
"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):
        anchor, positive, negative = features
        # Calculate Euclidean distances
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)

       # Find the hardest negative sample for each anchor
        hardest_pos_distance = pos_distance.max(dim=0)[0]    
        hardest_neg_distance = neg_distance.max(dim=0)[0]

        # Compute the triplet loss
        loss = max(hardest_pos_distance - hardest_neg_distance + self.margin, 0)
        #loss = torch.mean(torch.relu(pos_distance - neg_distance + self.margin))

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # print("\nlogits---")
        # print(logits)

        # print("\nexp_logits--->")
        # print(exp_logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss