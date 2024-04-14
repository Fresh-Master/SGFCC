import torch
from network import Network
from metric import valid
import argparse
from dataloader import load_data

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)  # 0.5
parser.add_argument("--temperature_l", default=1.0)  # 1.0
parser.add_argument("--learning_rate", default=0.0008)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=100)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--hidden_dim", default=288)
parser.add_argument("--high_feature_dim", default=256)
parser.add_argument("--fusion_layers", default=2)
parser.add_argument("--cluster_hidden_dim", default=128)
parser.add_argument("--num_cluster", default=5)
parser.add_argument("--views", default=2)
parser.add_argument("--temperature", default=0.1, type=float)  # temperature coef.
parser.add_argument("--contrastive_lambda", default=1.0, type=float)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(args.dataset)
model = Network(view, dims, args, class_num, device)
model = model.to(device)
checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
valid(model, device, dataset, view, data_size, class_num, eval_h=False)
