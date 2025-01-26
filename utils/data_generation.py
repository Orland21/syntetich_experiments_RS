''' data generations process'''

import itertools
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
from itertools import product
import numpy as np

def causal_sampling(n_size):
#for the moment I consider them i.i.d.
    g1=torch.randn(n_size)
    g2=torch.normal(mean=-1, std=1, size=(n_size,)) #+ 0.5 * g1
    g3=torch.normal(mean=5, std=5, size=(n_size,)) #+ 0.5 * g2
    
    return torch.stack((g1,g2,g3), dim=1)

def true_concepts_1(g,mu=0,std=1):
    concepts = torch.zeros(g.size(0))

    for i in range(g.size(0)):
        if g[i] < -0.8416*std+mu:
            concepts[i] = 0  # 20%
        elif -0.8416*std+mu <= g[i]  < -0.2533*std+mu:
            concepts[i] = 1 
        elif -0.2533*std+mu <= g[i] < 0.2533*std+mu:
            concepts[i] = 2  
        elif 0.2533*std+mu <= g[i] < 0.8416*std+mu:
            concepts[i] = 3 
        else:
            concepts[i] = 4
    
    return concepts

def true_concepts_2(g, mu=-1,std=1):
    concepts = torch.zeros(g.size(0))
   
    for i in range(g.size(0)):#dovrebbe comportarsi male tra 0 e 4, 2 e 3: dovuto al x2 mod 4
        if g[i] < -0.8416*std+mu:
            concepts[i] = 0 # 20%
        elif -0.8416*std+mu <= g[i]  < -0.2533*std+mu:
            concepts[i] = 4
        elif -0.2533*std+mu <= g[i] < 0.2533*std+mu:
            concepts[i] = 2
        elif 0.2533*std+mu <= g[i] < 0.8416*std+mu:
            concepts[i] =  1
        else:
            concepts[i] = 3
    
    return concepts

def true_concepts_3(g,mu=5,std=5):
    concepts = torch.zeros(g.size(0))

    for i in range(g.size(0)): #gravi problemi tra 0 e 4 e non tra 1 e 3
        if g[i] < -0.8416*std+mu:
            concepts[i] = 4  # 20%
        elif -0.8416*std+mu <= g[i]  < -0.2533*std+mu:
            concepts[i] = 0 
        elif -0.2533*std+mu <= g[i] < 0.2533*std+mu:
            concepts[i] = 1  
        elif 0.2533*std+mu <= g[i] < 0.8416*std+mu:
            concepts[i] =  3
        else:
            concepts[i] = 2
    
    return concepts

'''tensor(12.4370) tensor(1.0685)
tensor(8.6492) tensor(3.8447)
tensor(20.9878) tensor(2.0691)

tensor(15.3234) tensor(1.0943)
tensor(8.1049) tensor(3.8524)
tensor(20.4330) tensor(2.0811)

tensor(11.4972) tensor(1.1940)
tensor(8.6706) tensor(3.7409)
tensor(22.9784) tensor(2.1917)'''

def element_wise_diff(g):
# check per vedere se funziona tutto
    g1=2/torch.exp((g[:,0]-2)*0.3)
    g2=5/torch.exp(0.1*g[:,1])
    g3=5*torch.exp(0.06*g[:,2])

    return torch.stack((g1,g2,g3), dim=1)

def label_generation(args,g):
    if args.dataset=='addition':
        Y= torch.sum(g,dim=1)
    elif args.dataset=='inequality':
        Y=(g[:,0]<=2*g[:,1]-g[:,2]).int()
        add=(g[:,0]==2*g[:,1]-g[:,2]).int()
        Y=Y+add
    elif args.dataset=='modulo_asym':
        diff = g[:, 0]+2*g[:, 1] - g[:, 2]
        Y = diff%4
    elif args.dataset=='modulo_sym':
        diff = g[:, 0]+2*g[:, 1] - g[:, 2]
        Y = diff%4
    elif args.dataset=='equation':
        Y = g[:, 0]+2*g[:, 1] - g[:, 2]
    return Y

def permutation(z,p=0):

    permutazioni = list(itertools.permutations([0, 1, 2 ]))
    return z[:,permutazioni[p]]


def data_download(args):

    n_size={'train':42000,'validation':15000,'test': 15000}
    path = args.path

    os.makedirs(path, exist_ok=True)

    for split in n_size.keys():
        causal_data=causal_sampling(n_size[split])
        torch.save(causal_data,path+'/variables_'+split+'.pt')


def data_generation(args,split,p=0):

    path = args.path

    causal_data=torch.load(path+'/variables_'+split+'.pt')
    concepts=torch.stack([true_concepts_1(causal_data[:,0]),true_concepts_2(causal_data[:,1]),true_concepts_3(causal_data[:,2])],dim=1)
    z=element_wise_diff(causal_data)
    z=permutation(z,p)
    #z=causal_data
    # print(torch.max(z[:,0]),torch.min(z[:,0]))
    # print(torch.max(z[:,1]),torch.min(z[:,1]))
    # print(torch.max(z[:,2]),torch.min(z[:,2]))
    label=label_generation(args,concepts)

    return causal_data, label ,concepts, z


class Nesydata (Dataset):
    def __init__(self,split,args,p):

        self.data, self.labels , self.c, self.representation=data_generation(args,split,p)
            
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
            
        latents = self.data[idx,:]
        labels=self.labels[idx]
        concepts=self.c[idx,:]
        representation=self.representation[idx,:]

        return latents, labels, concepts, representation
    

def get_loader(dataset, batch_size, num_workers=0, val_test=False):

    if val_test:
        return torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )
    else:
        #bilanciamo i dati
        labels = dataset.labels
        tot = int(torch.max(labels))
        class_sample_count = np.array([len(torch.nonzero(labels == t)) for t in torch.unique(labels)])
        # print(class_sample_count)


        c = dataset.c
        pairs = list(product(range(5), repeat=3)) 
        concepts=np.array([tuple(row.tolist()) for row in c])
        pair_sample_count = np.array([np.sum(np.all(concepts == t, axis=1)) for t in pairs])
        # print(pair_sample_count)
        

        tot = np.ones(int(torch.max(labels))+1)
        tot_c = np.ones(int(len(pairs)))
        weight = 1. / class_sample_count
        weigth_c=1. / pair_sample_count

        j = 0
        for i in range(int(torch.max(labels)) +1):
            if i in torch.unique(labels):
                tot[i] = (weight[j])
                j += 1
        for i in range(len(pairs)):
            tot_c[i] = (weigth_c[i])

        samples_weight = np.array([tot[int(t)] for t in labels])
        samples_weight_c = np.array([tot_c[pairs.index(tuple(row.tolist()))] for row in c])
        samples_weight = samples_weight * samples_weight_c
        samples_weight = torch.from_numpy(samples_weight)
        
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        return DataLoader(dataset, batch_size=batch_size, 
                                num_workers=num_workers,
                                sampler=sampler)


def get_data_loaders(args,p):

    dataset_train=Nesydata('train',args,p)
    dataset_test=Nesydata('test',args,p)
    dataset_val=Nesydata('validation',args,p)
    
    train_loader = get_loader(dataset_train, args.batch_size,val_test=False)
    val_loader   = get_loader(dataset_val, args.batch_size,val_test=True)
    test_loader  = get_loader(dataset_test, args.batch_size, val_test=True)

    return train_loader, val_loader, test_loader
        
