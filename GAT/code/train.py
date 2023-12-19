from __future__ import division
from __future__ import print_function

import os
import re
import sys
import datetime
import glob
import time
import random
import argparse
import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as  F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


from utils import laod_data, accuracy
from model import GAT

parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action='store_true',default=False,help='Disables CUDA training')
parser.add_argument("--fast_mode",action='store_true',default=False,help='Validate during training pass')
parser.add_argument("--sparse",action='store_true',default=False,help='GAT with sparse version or not')
parser.add_argument("--seed",type=int,default=1234,help='Random seed')
parser.add_argument("--epochs",type=int,default=1000,help='Numbers of epochs to train')
parser.add_argument("--lr",type=float,default=0.003,help='learning rate')
parser.add_argument("--weight_decay",type=float,default=5e-4,help='Weight decay(L2 loss on parameters)')
parser.add_argument("--hidden",type=int,default=8,help='Number of head units')
parser.add_argument("--nb_heads",type=int,default=8,help='Number of head attentions')
parser.add_argument("--dropout",type=float,default=0.5,help='Dropout rate')
parser.add_argument("--alpha",type=float,default=0.2,help='Alpha for the leaky_relu')
parser.add_argument("--patience",type=int,default=100,help='Patience(early stop)')
parser.add_argument("--data_path",default='./data/cora/',help='data_path')
parser.add_argument("--dataset",default='cora',help='dataset')
parser.add_argument("--save_path",default='./checkpoints/',help='the path of checkpoints')
parser.add_argument("--log_path",default='./logs/',help='the result')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


adj,features,labels,idx_train,idx_val,idx_test = laod_data(args.data_path,args.dataset)

if args.sparse:
    raise NotImplementedError('尚未实现')

else:
    model = GAT(n_feature=features.shape[1],
                n_hid=args.hidden,
                n_class=int(labels.max())+1,
                dropout=args.dropout,
                n_head=args.nb_heads,
                alpha=args.alpha)

optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features ,adj,labels = Variable(features),Variable(adj),Variable(labels)

writer = SummaryWriter()
def train(epoch):
    start_time =time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,adj)
    train_loss = F.nll_loss(output[idx_train],labels[idx_train])
    train_acc = accuracy(output[idx_train],labels[idx_train])
    writer.add_scalars("Train",{'loss':train_loss,'acc':train_acc},epoch)
    train_loss.backward()
    optimizer.step()

    if not args.fast_mode:
        model.eval()
        output = model(features,adj)

    val_loss = F.nll_loss(output[idx_val], labels[idx_val])
    val_acc = accuracy(output[idx_train],labels[idx_train])

    writer.add_scalars("Val",{'loss':val_loss,'acc':val_acc},epoch)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(train_loss.data.item()),
          'acc_train: {:.4f}'.format(train_acc.data.item()),
          'loss_val: {:.4f}'.format(val_loss.data.item()),
          'acc_val: {:.4f}'.format(val_acc.data.item()),
          'time: {:.4f}s'.format(time.time() - start_time))
    writer.flush()
    return val_loss.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    test_loss = F.nll_loss(output[idx_test], labels[idx_test])
    test_acc = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(test_loss.data.item()),
          "accuracy= {:.4f}".format(test_acc.data.item()))
    
# train model
def main():    
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs+1
    best_epoch = 0
    validation_history = 0.0
    for epoch in range(args.epochs):
        loss_values.append(train(epoch))
        torch.save(model.state_dict(), '{}{}.pkl'.format(args.save_path,epoch))

        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
             
        else:
            bad_counter += 1
        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl',root_dir=args.save_path)
        for file in files:
            epoch_nb = int(file.split('.')[0])
            # epoch_nb = int(re.findall(r'\d+', file))
            if epoch_nb < best_epoch:
                os.remove(os.path.join(args.save_path,file))
    files = glob.glob('*.pkl',root_dir=args.save_path)
    for file in files:
        epoch_nb = int(file.split('.')[0])
        # epoch_nb = int(re.findall(r'\d+', file))
        if epoch_nb > best_epoch:
            os.remove(os.path.join(args.save_path,file))
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}{}.pkl'.format(args.save_path,best_epoch)))

    writer.close()

    #test
    compute_test()

class DualLogger:
    def __init__(self, filename, stdout):
        self.log_file = open(filename, 'w')
        self.stdout = stdout

    def write(self, message):
        self.log_file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.log_file.flush()
        self.stdout.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None


if __name__ =='__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.log_path}{args.dataset}_train_output_{timestamp}.log"

    # logger
    original_stdout = sys.stdout
    sys.stdout = DualLogger(log_filename, original_stdout)

    #  main code
    main()
    sys.stdout.close()
    sys.stdout = original_stdout







