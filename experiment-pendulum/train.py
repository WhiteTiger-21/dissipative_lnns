# Dissipative Lagrangian Neural Networks|2023
# Takemori Masaki,Kamakura Yoshinari
# This software includes the work that is distributed in the Apache License 2.0 

import torch, argparse
import numpy as np
import dadaptation
from timm.scheduler import CosineLRScheduler

from dlnn import DLNN

import os, sys,copy
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP

from data import get_lagrangian_trajectory
from data import get_DLNN_dataset,get_DLNN_dataset_all
import matplotlib.pyplot as plt

import pickle as pkl

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=512, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='swish', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=1000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=1000, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='pend', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--samples', default=100, type=int, help='num of samples traindata')
    parser.add_argument('--test_split', default=0.9, type=int, help='num of samples traindata')
    parser.add_argument('--batch_size', default=5000, type=int, help='batch_size,more than samples')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def get_traindata(args,q=1,qd=10):
    traindata_name = '{}/{}'.format(args.save_dir,args.name)
    if  os.path.exists("{}_{}-{}.pkl".format(traindata_name,q,qd)) :
        print("load train data")
        data = pkl.load(open("{}_{}-{}.pkl".format(traindata_name,q,qd), 'rb'))
    else :
        print("make train data")
        if q ==1 and qd == 10 :
            data = get_DLNN_dataset_all(seed=args.seed,samples=args.samples,test_split=args.test_split,q=q,qd=qd)
        else :
            data = get_DLNN_dataset(seed=args.seed,samples=args.samples,test_split=args.test_split,q=q,qd=qd)
        pkl.dump(data,open("{}_{}-{}.pkl".format(traindata_name,q,qd),'wb'))
    return data

def model_save(model,args,status="",ckpt=False):
    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    if args.baseline :
        label = '-baseline'
    else :
        label =  '-dlnn'
    path = '{}/{}{}{}.pt'.format(args.save_dir,status, args.name, label)
    if ckpt :
        torch.save(model, path)
    else :
        torch.save(model.state_dict(), path)

def model_load(model,args,status=""):
    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    if args.baseline :
        label = '-baseline'
    else :
        label =  '-dlnn'
    path = '{}/{}{}{}.pt'.format(args.save_dir,status, args.name, label)
    if  os.path.exists("{}".format(path)) :
        model = torch.load(path)
        return model,True
    else :
        return model,False

def train(args):
    seed = args.seed
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(args.seed)

    # init model and optimizer
    if args.verbose:
        if args.baseline :
            label = 'baseline'
        else :
            label =  'dlnn'
        print("Training {} model:".format(label))

    output_dim = args.input_dim
    if args.baseline :
        model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    else :
        nn_model = MLP(args.input_dim, args.hidden_dim, 2, args.nonlinearity)
        model = DLNN(model=nn_model)
    
    optim = dadaptation.DAdaptAdam(model.parameters(), 1, weight_decay=1e-4,log_every=1000,d0=1e-6)

    loss_fn = torch.nn.MSELoss()

    batch = args.batch_size
    
    if torch.cuda.is_available() :
        model = model.cuda()

    best_parm_loss = np.inf
    best_parm = copy.deepcopy(model)
    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}

    reload = 0
    permit = 1 + 1e-4
    permit_s = 1+1e-2
    samples = args.samples
    q = 0
    qd = 0
    for cycle in range(16) :
        if cycle == 0: 
            args.samples = samples*10
            q = 1
            qd = 10
        elif cycle == 1 :
            args.samples = samples*5
            q = 1
            qd = 5
        elif cycle == 2 :
            q = 1
            qd = 3

        elif cycle == 3 :
            q = 0.75
            qd = 10
            
        elif cycle == 4 :
            q=0.75
            qd= 5
            
        elif cycle == 5 :
            q=0.75
            qd= 3
            
        elif cycle == 6 :
            q=0.5
            qd=10

        elif cycle == 7:
            q=0.25
            qd=10
            
        elif cycle == 8 :
            q = 1
            qd = 1

        elif cycle == 9 :
            q=0.5
            qd=5
            
        elif cycle == 10 :
            q=0.25
            qd=5
            
        elif cycle == 11 :
            args.samples = samples
            q=0.75
            qd=1
            
        elif cycle == 12 :
            q=0.5
            qd=3
            
        elif cycle == 13 :
            q=0.25
            qd=3
            
        elif cycle == 14 :
            q=0.5
            qd=1
            
        elif cycle == 15 :
            q = 0.25
            qd = 1        
            
        data = get_traindata(args,q=q,qd=qd)
            
        if cycle == 0 :
            # arrange data
            x = torch.tensor(data['x'], dtype=torch.float32)
            test_x = torch.tensor(data['test_x'],  dtype=torch.float32)
            dxdt = torch.Tensor(data['dx'])
            test_dxdt = torch.Tensor(data['test_dx'])
            
        else :
            x = torch.cat([x,torch.tensor(data['x'], dtype=torch.float32)])
            test_x = torch.cat([test_x,torch.tensor(data['test_x'],  dtype=torch.float32)])
            dxdt =  torch.cat([dxdt,torch.Tensor(data['dx'])])
            test_dxdt =  torch.cat([test_dxdt,torch.Tensor(data['test_dx'])])

        print("Num of Samples:{}".format(len(x)))
        if len(x) // batch == 0:
            print("! More Samples !")
            sys.exit()

        test_batch = batch
        if test_batch > 5000 :
            test_batch = batch // 10
        elif test_batch < 2500 :
            test_batch *= 10

        

        if len(test_x) // test_batch == 0:
            test_batch = len(test_x)

        train_ds = torch.utils.data.TensorDataset(x,dxdt)
        test_ds = torch.utils.data.TensorDataset(test_x,test_dxdt)
            
        train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch,shuffle=True,num_workers = os.cpu_count(),drop_last=True,pin_memory = True)
        test_loader = torch.utils.data.DataLoader(test_ds,batch_size=test_batch,shuffle=True,num_workers = os.cpu_count(),drop_last=True,pin_memory = True)



        optim = dadaptation.DAdaptAdam(model.parameters(), 1, weight_decay=1e-4,log_every=5000,d0=1e-6)
        mul = 0.1 * cycle if cycle <= 9 else mul

        steps = args.total_steps - int((args.total_steps)*mul)
        cycle_num = 4
        div_hot_num = 10
        scheduler = CosineLRScheduler(optim, t_initial=steps//cycle_num, cycle_limit=cycle_num,cycle_decay=0.8,lr_min=args.learn_rate,warmup_t = steps//div_hot_num,warmup_lr_init=args.learn_rate*1e3,warmup_prefix=True)
        
        best_parm_loss = np.inf
        best_parm = copy.deepcopy(model)

        
        model,ExisitOrNot = model_load(model,args,"ckpt-{}-{}-{}-".format(cycle, q,qd))

        if ExisitOrNot :
            print('pre trained')
            continue

        if torch.cuda.is_available() :
            model = model.cuda()
        if args.baseline :
            for step in range(steps+1):
                model.train()
                    
                for b,(xb ,dxdtb) in enumerate(train_loader):
                    if torch.cuda.is_available() :
                        xb = xb.cuda()
                        dxdtb = dxdtb.cuda()

                    # train step
                    dxdt_hat = model(xb)
                    loss = loss_fn(dxdtb, dxdt_hat)
                    loss.backward() ; optim.step() ; optim.zero_grad()
                    stats['train_loss'].append(loss.item())
                    


                    if b % args.print_every == 0:
                        print("step {}:{}:{}, train_loss {:.4e}".format(cycle,step,b, loss.item()))
                        print("Learning rate:", optim.param_groups[0]['lr'])

                scheduler.step(step+1)
                print("the worst train_loss in step {} : {:.4e}".format(step,np.max(stats['train_loss'][-len(train_loader):])))
                with torch.no_grad() :
                    model.eval()
                    for b,(test_xb ,test_dxdtb) in enumerate(test_loader):
                        if torch.cuda.is_available() :
                            test_xb = test_xb.cuda()
                            test_dxdtb = test_dxdtb.cuda()

                        # test step
                        dxdt_hat = model(test_xb)
                        test_loss = loss_fn(test_dxdtb, dxdt_hat)
                        stats['test_loss'].append(test_loss.item())

                        if b % args.print_every == 0:
                            print("step {}:{}:{}, test_loss {:.4e}".format(cycle,step, b, test_loss.item()))
                                    
                print("the worst test_loss in step {} : {:.4e}".format(step,np.max(stats['test_loss'][-len(test_loader):])))
                mean_test_loss = np.mean(stats['test_loss'][-len(test_loader):])
                delta_loss = mean_test_loss - best_parm_loss
                print("step {},mean_test_loss {:.8e},delta {:.8e}".format(step,mean_test_loss,delta_loss))
                if mean_test_loss < best_parm_loss/permit_s:
                    reload = 0
                    print("save this step model")
                    best_parm_loss = mean_test_loss
                    best_parm = copy.deepcopy(model)
                    best_parm = best_parm.cpu() if torch.cuda.is_available() else best_parm
                    model_save(best_parm,args,"middle-")
                elif mean_test_loss < best_parm_loss / permit:
                    reload = 0
                    best_parm_loss = mean_test_loss
                    best_parm = copy.deepcopy(model)
                elif np.isnan(mean_test_loss) :
                    print("reload best")
                    model = copy.deepcopy(best_parm)
                    model = model.cuda() if torch.cuda.is_available() else model.cpu()
                elif mean_test_loss > (1+best_parm_loss)**2 and step >= (args.total_steps)*mul//div_hot_num:
                    reload = 0
                    print("reload best")
                    model = copy.deepcopy(best_parm)
                    model = model.cuda() if torch.cuda.is_available() else model.cpu()
                # prevention NaN
                elif np.isnan(mean_test_loss) :
                    print("reload best")
                    reload = 0
                    model = copy.deepcopy(best_parm)
                    model = model.cuda() if torch.cuda.is_available() else model.cpu()
                else :
                    if reload >= args.total_steps//(cycle_num*2) and step > int((args.total_steps)*mul)// div_hot_num + int((args.total_steps)*mul)//cycle_num :
                        model = copy.deepcopy(best_parm)
                        model = model.cuda() if torch.cuda.is_available() else model.cpu()
                        break            
                    reload += 1

            
        else :
            for step in range(steps+1):
                model.train()
                    
                for b,(xb ,dxdtb) in enumerate(train_loader):
                    if torch.cuda.is_available() :
                        xb = xb.cuda()
                        dxdtb = dxdtb.cuda()

                    # train step
                    dxdt_hat = model.time_derivative(xb)
                    loss = loss_fn(dxdtb[:,1:], dxdt_hat[:,1:])
                    loss.backward() ; optim.step() ; optim.zero_grad()
                    stats['train_loss'].append(loss.item())
                    


                    if b % args.print_every == 0:
                        print("step {}:{}:{}, train_loss {:.4e}".format(cycle,step,b, loss.item()))
                        print("Learning rate:", optim.param_groups[0]['lr'])

                scheduler.step(step+1)
                print("the worst train_loss in step {} : {:.4e}".format(step,np.max(stats['train_loss'][-len(train_loader):])))
                model.eval()
                for b,(test_xb ,test_dxdtb) in enumerate(test_loader):
                    if torch.cuda.is_available() :
                        test_xb = test_xb.cuda()
                        test_dxdtb = test_dxdtb.cuda()

                    # test step
                    dxdt_hat = model.time_derivative(test_xb)
                    test_loss = loss_fn(test_dxdtb[:,1:], dxdt_hat[:,1:])
                    stats['test_loss'].append(test_loss.item())

                    if b % args.print_every == 0:
                        print("step {}:{}:{}, test_loss {:.4e}".format(cycle,step, b, test_loss.item()))
                                    
                print("the worst test_loss in step {} : {:.4e}".format(step,np.max(stats['test_loss'][-len(test_loader):])))
                mean_test_loss = np.mean(stats['test_loss'][-len(test_loader):])
                delta_loss = mean_test_loss - best_parm_loss
                print("step {},mean_test_loss {:.8e},delta {:.8e}".format(step,mean_test_loss,delta_loss))
                if mean_test_loss < best_parm_loss/permit_s:
                    reload = 0
                    print("save this step model")
                    best_parm_loss = mean_test_loss
                    best_parm = copy.deepcopy(model)
                    best_parm = best_parm.cpu() if torch.cuda.is_available() else best_parm
                    model_save(best_parm,args,"middle-")
                elif mean_test_loss < best_parm_loss / permit:
                    reload = 0
                    best_parm_loss = mean_test_loss
                    best_parm = copy.deepcopy(model)
                elif np.isnan(mean_test_loss) :
                    print("reload best")
                    model = copy.deepcopy(best_parm)
                    model = model.cuda() if torch.cuda.is_available() else model.cpu()
                elif mean_test_loss > (1+best_parm_loss)**2 and step >= (args.total_steps)*mul//div_hot_num:
                    reload = 0
                    print("reload best")
                    model = copy.deepcopy(best_parm)
                    model = model.cuda() if torch.cuda.is_available() else model.cpu()
                # prevention NaN
                elif np.isnan(mean_test_loss) :
                    print("reload best")
                    reload = 0
                    model = copy.deepcopy(best_parm)
                    model = model.cuda() if torch.cuda.is_available() else model.cpu()
                else :
                    if reload >= args.total_steps//(cycle_num*2) and step > int((args.total_steps)*mul)// div_hot_num + int((args.total_steps)*mul)//cycle_num :
                        model = copy.deepcopy(best_parm)
                        model = model.cuda() if torch.cuda.is_available() else model.cpu()
                        break            
                    reload += 1

        print("save step {} model".format(cycle))
        model_save(best_parm,args,"ckpt-{}-{}-{}-".format(cycle, q,qd),True)    
            
    return best_parm, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    print("save model")
    model_save(model,args)
