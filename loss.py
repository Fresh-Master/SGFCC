import torch
import torch.nn as nn
import math


class Contrastive_loss(nn.Module):
    def __init__(self, args):
        super(Contrastive_loss, self).__init__()
        self.batch_size = args.batch_size
        self.class_num = args.num_cluster
        self.temperature_f = args.temperature_f
        self.temperature_c = args.temperature_c
        self.con_lambda = args.con_lambda
        self.device = args.device

        self.mask = self.mask_correlated_samples(self.batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return self.con_lambda * loss

    def forward_class(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_c
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy


# class ClaCon_loss(nn.Module):
#     def __init__(self, args):
#         super(ClaCon_loss, self).__init__()
#         self.class_num = args.num_cluster
#         self.temperature_c = args.temperature_c
#         self.batch_size = args.batch_size
#
#         self.mask = self.mask_correlated_samples(self.batch_size)
#         self.similarity = nn.CosineSimilarity(dim=2)
#         self.criterion = nn.CrossEntropyLoss(reduction="sum")
#
#     def mask_correlated_samples(self, N):
#         mask = torch.ones((N, N))
#         mask = mask.fill_diagonal_(0)
#         for i in range(N//2):
#             mask[i, N//2 + i] = 0
#             mask[N//2 + i, i] = 0
#         mask = mask.bool()
#         return mask
#
#     def forward_class(self, q_i, q_j):
#         p_i = q_i.sum(0).view(-1)
#         p_i /= p_i.sum()
#         ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
#         p_j = q_j.sum(0).view(-1)
#         p_j /= p_j.sum()
#         ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
#         entropy = ne_i + ne_j
#
#         q_i = q_i.t()
#         q_j = q_j.t()
#         N = 2 * self.class_num
#         q = torch.cat((q_i, q_j), dim=0)
#
#         sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_c
#         sim_i_j = torch.diag(sim, self.class_num)
#         sim_j_i = torch.diag(sim, -self.class_num)
#
#         positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
#         mask = self.mask_correlated_samples(N)
#         negative_clusters = sim[mask].reshape(N, -1)
#
#         labels = torch.zeros(N).to(positive_clusters.device).long()
#         logits = torch.cat((positive_clusters, negative_clusters), dim=1)
#         loss = self.criterion(logits, labels)
#         loss /= N
#         return loss + entropy


class DECLoss(nn.Module):
    """
    Deep embedding clustering.
    Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis."
    International conference on machine learning. PMLR, 2016.
    """

    def __init__(self):
        super(DECLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward_clu(self, logist):
        Q = self.target_distribution(logist).detach()
        loss = self.criterion(logist.log(), Q) / logist.shape[0]
        return loss

    def target_distribution(self, logist) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (logist ** 2) / torch.sum(logist, 0)
        return (weight.t() / torch.sum(weight, 1)).t()


