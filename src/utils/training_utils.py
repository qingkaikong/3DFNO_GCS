import numpy as np
import torch
from timeit import default_timer
import os
from utils import distributed_utils
import torch.distributed as dist

# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(
            num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
            
def train(epochs, model, model_without_ddp, device, train_loader, val_loader, optimizer, lr_scheduler, checkpoint_dir):
    nval = len(val_loader.dataset)
    ntrain = len(train_loader.dataset)
    
    L2 = LpLoss(p=2, size_average=False)
    L1 = LpLoss(p=1, size_average=False)
    # y_normalizer.cuda()
    losstrain = np.zeros(epochs)
    lossval = np.zeros(epochs)
    l2train = np.zeros(epochs)
    l2val = np.zeros(epochs)
    l1train = np.zeros(epochs)
    l1val = np.zeros(epochs)
    best_ep = np.zeros(epochs)
    time_ep = np.zeros(epochs)
    val_min_loss = 1000000
    for ep in range(epochs):
        # break out the last epoch hang
        if ep == epochs-1:
            break
        model.train()
        train_loader.sampler.set_epoch(ep)
        t1 = default_timer()
        train_loss = 0
        train_L2 = 0.0
        train_L1 = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            L2_loss = L2(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
            L1_loss = L1(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
            loss = 0.9 * L1_loss + 0.1 * L2_loss
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                #train_loss += loss.item()
                #train_L2 += L2_loss.item()
                #train_L1 += L1_loss.item()       
                train_loss += loss
                train_L2 += L2_loss
                train_L1 += L1_loss
                
        if lr_scheduler:
            #print('Epoch:', ep,'LR:', lr_scheduler.get_lr())
            lr_scheduler.step()
        
        # reduce all the loss to master
        dist.reduce(train_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(train_L2, dst=0,op=dist.ReduceOp.SUM)
        dist.reduce(train_L1, dst=0,op=dist.ReduceOp.SUM)
        
        # do the evaluation
        model.eval()
        val_loader.sampler.set_epoch(ep)
        val_loss = 0.0
        val_L1 = 0.0
        val_L2 = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)

                L2_loss = L2(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
                L1_loss = L1(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
                loss = 0.9 * L1_loss + 0.1 * L2_loss
                val_loss += loss
                val_L2 += L2_loss
                val_L1 += L1_loss
               
        dist.reduce(val_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(val_L2, dst=0,op=dist.ReduceOp.SUM)
        dist.reduce(val_L1, dst=0,op=dist.ReduceOp.SUM)
        
        if distributed_utils.is_main_process():    
            
            train_loss = train_loss.item()/ntrain
            val_loss = val_loss.item()/nval
            train_L1 = train_L1.item()/ntrain
            train_L2 = train_L2.item()/ntrain
            val_L1 = val_L1.item()/nval
            val_L2 = val_L2.item()/nval

            t2 = default_timer()
            t_diff = t2 - t1
            print(
                ep,
                t_diff,
                train_loss,
                val_loss,
                train_L2,
                val_L2,
                train_L1,
                val_L1)

            losstrain[ep] = train_loss
            lossval[ep] = val_loss
            l2train[ep] = train_L2
            l2val[ep] = val_L2
            l1train[ep] = train_L1
            l1val[ep] = val_L1
            time_ep[ep] = t_diff

            if val_loss < val_min_loss:
                checkpoint = {
                    'epoch': ep + 1,
                    'state_dict': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': val_loss
                    } 
                distributed_utils.save_on_master(checkpoint, os.path.join(checkpoint_dir, f"best_model.pt"))
                val_min_loss = val_loss
                best_ep[ep] = 1
                best_model = model
            out = np.c_[losstrain, lossval, l1train, l1val, l2train, l2val, best_ep, time_ep]
            np.savetxt(checkpoint_dir + 'training_history.csv', out, delimiter=',',
                       header='train_loss,val_loss,l1_train,l1_val,l2_train,l2_val,best_ep,t')