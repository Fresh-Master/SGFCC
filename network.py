import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.nn import Parameter
from sklearn.cluster import KMeans
import torch
from typing import Optional
from loss import DECLoss, Contrastive_loss


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()  ##super用来调用父类
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Clustering(nn.Module):
    def __init__(self, args):
        super(Clustering, self).__init__()
        self.kmeans = KMeans(n_clusters=args.num_cluster, n_init=20)
        self.clustering_layer = DECModule(cluster_number=args.num_cluster,
                                          embedding_dimension=args.cluster_hidden_dim)

    def forward(self, h):
        # self.kmeans.fit(h.cpu().detach().numpy())
        clustering_layer = self.clustering_layer
        # cluster_centers = torch.tensor(
        #     self.kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)
        # # cluster_centers = cluster_centers.to(device)
        # with torch.no_grad():
        #     clustering_layer.cluster_centers.copy_(cluster_centers)
        q = clustering_layer(h)
        return q


# def build_clustering_module(args):
#     clustering_layer = DECModule(cluster_number=args.num_cluster,
#                                  embedding_dimension=args.high_feature_dim)
#     con_loss = SimCLRLoss(args)
#     return clustering_layer


# def build_cf_module(args):
#     contrastive_fusion_module = CfModule(args)
#     con_loss = SimCLRLoss(args)
#     return contrastive_fusion_module


class Network(nn.Module):
    def __init__(self, view, input_size, args, class_num, device):
        super(Network, self).__init__()
        feature_dim = args.feature_dim
        high_feature_dim = args.high_feature_dim
        self.encoders = []
        self.decoders = []
        self.log_y = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )
        self.cluster_contrastive_module = nn.Sequential(
            nn.Linear(class_num, class_num),
        )
        self.con_criterion = Contrastive_loss(args)
        self.clustering = Clustering(args)
        self.clu_loss = DECLoss()
        self.view = view
        self.con_lambda = args.con_lambda

    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        qs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            h_v = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.clustering(h_v)
            # q = self.cluster_contrastive_module(q_v)
            zs.append(z)
            hs.append(h_v)
            xrs.append(xr)
            qs.append(q)
        return hs, qs, xrs, zs

    # @torch.no_grad()
    # def commonZ(self, xs):
    #     zs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         z = self.encoders[v](x)
    #         zs.append(z)
    #     h = self.contrastive_fusion_module(zs)
    #     return h

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h_v = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.clustering(h_v)
            qs.append(q)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds

    def get_loss(self, hs, view):
        sub_loss = 0
        for h in hs:
            q = self.clustering(h)
            sub = self.clu_loss.forward_clu(q)
            sub_loss += sub
        return self.con_lambda * sub_loss / view


class DECModule(nn.Module):
    def __init__(
            self,
            cluster_number: int,
            embedding_dimension: int,
            alpha: float = 1.0,
            cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        param cluster_number: number of clusters
        param embedding_dimension: embedding dimension of feature vectors
        param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(DECModule, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.
        param batch: FloatTensor of [batch size, embedding dimension]
        return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class Network1(nn.Module):
    def __init__(self, view, input_size, args, class_num, device):
        super(Network1, self).__init__()
        feature_dim = args.feature_dim
        high_feature_dim = args.high_feature_dim
        self.encoders = []
        self.decoders = []
        self.log_y = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], high_feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], high_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        # self.feature_contrastive_module = nn.Sequential(
        #     nn.Linear(feature_dim, high_feature_dim),
        # )
        self.cluster_contrastive_module = nn.Sequential(
            nn.Linear(class_num, class_num),
        )
        self.con_criterion = Contrastive_loss(args)
        self.clustering = Clustering(args)
        self.clu_loss = DECLoss()
        self.view = view
        self.con_lambda = args.con_lambda

    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        qs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            # h_v = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.clustering(z)
            # q = self.cluster_contrastive_module(q_v)
            zs.append(z)
            # hs.append(h_v)
            xrs.append(xr)
            qs.append(q)
        return hs, qs, xrs, zs

    # @torch.no_grad()
    # def commonZ(self, xs):
    #     zs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         z = self.encoders[v](x)
    #         zs.append(z)
    #     h = self.contrastive_fusion_module(zs)
    #     return h

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            # h_v = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.clustering(z)
            qs.append(q)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds

    def get_loss(self, hs, view):
        sub_loss = 0
        for h in hs:
            q = self.clustering(h)
            sub = self.clu_loss.forward_clu(q)
            sub_loss += sub
        return self.con_lambda * sub_loss / view