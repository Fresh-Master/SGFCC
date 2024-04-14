# import numpy
import torch
from network import Network, Network1
from metric import valid, eva
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import DECLoss, Contrastive_loss
from dataloader import load_data
# from sklearn import manifold
# import matplotlib.pyplot as plt
import scipy.io as sio
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech
# CIFAR
Dataname = 'CIFAR'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.3)  # 0.5
parser.add_argument("--temperature_c", default=0.01)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=2)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--cluster_hidden_dim", default=128)
parser.add_argument("--num_cluster", default=10)
parser.add_argument("--views", default=3)
parser.add_argument("--con_lambda", default=0.001)
parser.add_argument("--device",
                    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "MNIST-USPS":
    args.con_epochs = 50
    seed = 10
if args.dataset == "BDGP":
    args.con_epochs = 50  # 10
    seed = 10
if args.dataset == "CCV":
    args.con_epochs = 50
    seed = 3
if args.dataset == "CIFAR":
    args.con_epochs = 50
    seed = 3
if args.dataset == "Caltech-5V":
    args.con_epochs = 100
    seed = 5
if args.dataset == "Caltech-4V":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Caltech-3V":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Caltech-2V":
    args.con_epochs = 100
    seed = 10


def setup_seed(seed):
    torch.manual_seed(seed)  # 设置CPU生成随机数的种子，方便下次复现实验结果
    torch.cuda.manual_seed_all(seed)  # 在GPU中设置生成随机数的种子
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)
# 按照batch size封装成Tensor
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()  # 均方损失
    for batch_idx, (xs, _, _) in enumerate(data_loader):  # enumerate：将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        for v in range(view):
            xs[v] = xs[v].to(args.device)
        optimizer.zero_grad()
        hs, _, xrs, zs = model(xs)
        loss_list = []
        # Zs = []
        # Hs = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
            # Zs.append(zs[v].cpu().detach().numpy())
            # Hs.append(hs[v].cpu().detach().numpy())
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    # if epoch % 50 == 0:
    #     Z_path = args.dataset + '_Z_pre' + str(epoch)
    #     Z_path1 = args.dataset + '_H_pre' + str(epoch)
    #     sio.savemat(Z_path + '.mat', {'Z': Zs})
    #     sio.savemat(Z_path1 + '.mat', {'H': Hs})
    return tot_loss / len(data_loader)


def contrastive_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    Z_batch = []
    H_batch = []
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(args.device)
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs)
        loss_list = []
        # Z = sum(zs) / view
        # H = sum(hs) / view
        # Q = sum(qs) / view
        # Z_cat = torch.cat(zs, 0)
        # Z_batch.append(Z.detach().cpu())
        # H_batch.append(H.detach().cpu())
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion1.forward_feature(zs[v], zs[w]))  # 特征对比损失
                loss_list.append(criterion1.forward_class(qs[v], qs[w]))
            loss_list.append(mse(xs[v], xrs[v]))  # 重建损失
        loss_list.append(model.get_loss(hs, view))  # 聚类损失
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    # for i in range(len(Z_batch)):
    #     Z_total = Z_batch[i]
    #
    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    # if epoch % 10 == 0:
    #     Z_path = args.dataset + '_Z_con' + str(epoch)
    #     Z_path1 = args.dataset + '_H_con' + str(epoch)
    #     sio.savemat(Z_path + '.mat', {'Z': Z})
    #     sio.savemat(Z_path1 + '.mat', {'H': H})
    return tot_loss / len(data_loader)



accs = []
nmis = []
aris = []
loss = []
# if not os.path.exists('./models'):
#     os.makedirs('./models')
# if not os.path.exists('./results'):
#     os.makedirs('./results')
T = 1
for i in range(T):
    print("ROUND:{}".format(i + 1))

    # model = Network(view, dims, args, class_num, args.device)
    model = Network(view, dims, args, class_num, args.device)
    print(model)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion1 = Contrastive_loss(args).to(args.device)
    # criterion2 = ClaCon_loss(args).to(args.device)
    criterion3 = DECLoss().to(args.device)

    epoch = 1
    while epoch <= args.mse_epochs:
        loss_epoch = pretrain(epoch)
        # loss.append(loss_epoch)
        acc, nmi, pur, h, _ = eva(model, args.device, dataset, view, data_size, class_num, eval_h=False)
        # accs.append(acc)
        # nmis.append(nmi)
        # aris.append(pur)
        epoch += 1
    while epoch <= args.mse_epochs + args.con_epochs:
        loss_epoch = contrastive_train(epoch)
        # loss.append(loss_epoch)
        acc, nmi, pur, h, q = eva(model, args.device, dataset, view, data_size, class_num, eval_h=False)
        # accs.append(acc)
        # nmis.append(nmi)
        # aris.append(pur)
        # if epoch == args.mse_epochs + args.con_epochs:
        #     fig = plt.figure(figsize=(8, 6), dpi=300)
        #     ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
        #     y = ts.fit_transform(h)
        #     # ax1 = fig.add_subplot(1, 2, 1)
        #     # y_min, y_max = np.min(y, 0), np.max(y, 0)
        #     # y = (y - y_min) / (y_max - y_min)
        #     # color = Q.cpu().detach().numpy()
        #     color = q
        #     # color = [np.argmax(i) for i in color]
        #     # color = np.stack(color, axis=0)
        #     cm = 'viridis'
        #     plt.scatter(y[:, 0], y[:, 1], c=color, cmap=cm)
        #     # plt.colorbar()
        #     # plt.title("Ours", fontsize=14)
        #     plt.show()
        epoch += 1
        # if epoch == args.mse_epochs + args.con_epochs:
        #     acc, nmi, ari = valid(model, args.device, dataset, view, data_size, class_num, eval_h=False)
    #     epoch += 1
    # while epoch <= args.mse_epochs + args.con_epochs + args.tune_epochs:
    #     loss_epoch = fine_tune(epoch)
    #     loss.append(loss_epoch)
    #     acc, nmi, ari = eva(model, args.device, dataset, view, data_size, class_num, eval_h=False)
    #     accs.append(acc)
    #     nmis.append(nmi)
    #     aris.append(ari)
    #     if epoch == args.mse_epochs + args.con_epochs + args.tune_epochs:
    #         acc, nmi, ari = valid(model, args.device, dataset, view, data_size, class_num, eval_h=False)
    #         accs.append(acc)
    #         nmis.append(nmi)
    #         aris.append(ari)
    #         # state = model.state_dict()
    #         # torch.save(state, './models/' + args.dataset + '.pth')
    #         # print('Saving..')
    #     epoch += 1
    # sio.savemat(args.dataset + '_loss' + '.mat', {'loss': loss})
    # sio.savemat(args.dataset + '_acc' + '.mat', {'acc': accs})
    # sio.savemat(args.dataset + '_nmi' + '.mat', {'nmi': nmis})
    # sio.savemat(args.dataset + '_ari' + '.mat', {'ari': aris})
