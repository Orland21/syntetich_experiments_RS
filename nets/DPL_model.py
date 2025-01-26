'''modello'''
import torch
import torch.nn as nn

from nets.utils.utils_problog import build_worlds_queries_matrix
from utils.conf import get_device
from utils.dpl_loss import ADDMNIST_DPL
from utils.losses import *
from nets.MLP import MLP_inequality as MLP
import torch.nn.functional as F

class DPL_synthetic(nn.Module):

    def __init__(self,args,n_facts=5,nr_classes=13, input_size=3):
        super(DPL_synthetic, self).__init__()

        # Worlds-queries matrix
        if args.dataset == 'addition':
            self.n_facts = 5
            self.w_q = build_worlds_queries_matrix(3, self.n_facts, 'addition') # w_q[w, q] = 1
            self.nr_classes = 13
            self.input_size=1
            self.map=MLP(self.input_size,self.n_facts)
        elif args.dataset == 'inequality':
            self.n_facts = 5
            self.w_q = build_worlds_queries_matrix(3, self.n_facts, 'inequality')
            self.nr_classes = 3
            self.input_size=1
            self.map=MLP(self.input_size,self.n_facts)
        elif args.dataset == 'equation':
            self.n_facts = 5
            self.w_q = build_worlds_queries_matrix(3, self.n_facts, 'equation')
            self.nr_classes = 17
            self.input_size=1
            self.map=MLP(self.input_size,self.n_facts)
        elif args.dataset == 'modulo_asym':
            self.n_facts = 5
            self.w_q = build_worlds_queries_matrix(3, self.n_facts, 'modulo_asym')
            self.nr_classes = 4
            self.input_size=1
            self.map=MLP(self.input_size,self.n_facts)
        elif args.dataset == 'modulo_sym':
            self.n_facts = 5
            self.w_q = build_worlds_queries_matrix(3, self.n_facts, 'modulo_sym')
            self.nr_classes = 4
            self.input_size=1
            self.map=MLP(self.input_size,self.n_facts)

        #self.logit_permutations=logits_permutations(10,2)
        # opt and device
        self.opt=None
        self.device=get_device()
        self.w_q=self.w_q.to(self.device)

    def forward(self, z):
        # starting from representation
        
        logits_1, logits_2, logits_3 = self.map(z)
        cs = torch.stack([logits_1, logits_2,logits_3], dim=1)  # nx3x5 create a new dimension, differently from cat
        # logits = self.logit_permutations(cs)
        # logit = torch.mean(logits, dim=0)
        # #print(logit)
        # p = F.gumbel_softmax(logit, tau=0.5, hard=True)
        # #print(p.tolist())
        # cs = self.permutations(cs, p)
        # normalize concept predictions
        pCs = self.normalize_concepts(cs)  # applico la softmax e un eps per evitare underflow poi normalizzo in modo da avere probabilità
        
        # Problog inference to compute worlds and query probability distributions
        py, worlds_prob = self.problog_inference(pCs)  # nxn_labels 
        return {'CS': cs, 'YS': py, 'pCS': pCs}

    def permutations(self, z, p):
        z1 = z[:, :5]
        z2 = z[:, 5:]
        z1 = z1 * p[0] + z2 * (1 - p[1])
        z2 = z1 * (1 - p[0]) + z2 * p[1]
        return torch.stack([z1, z2], dim=1)
                
        
    def problog_inference(self, pCs, query=None):
        """
        Performs ProbLog inference to retrieve the worlds probability distribution P(w).
        Works with two encoded bits.
        """
        
        # Extract first and second digit probability
        prob_digit_1, prob_digit_2, prob_digit_3 = pCs[:, 0, :], pCs[:, 1,:], pCs[:, 2,:] #nx5

        # Compute worlds probability P(w) (the two digits values are independent)
        Z_1 = prob_digit_1[..., None]
        Z_2 = prob_digit_2[:, None, :]
        Z_3 = prob_digit_3[:, None, None, :]

        probs = Z_1.multiply(Z_2) #nx5x5
        probs = probs[..., None]
        probs = probs.multiply(Z_3) #nx5x5x5 

        worlds_prob = probs.reshape(-1, self.n_facts*self.n_facts*self.n_facts)# autox125, auto=n ordine contiguo in memoria, che è tipicamente per righe (row-major order) 
        #per righe poi per slice
        
        # Compute query probability P(q)
        query_prob = torch.zeros(size=(len(probs), self.nr_classes), device=probs.device) #n(numero prima dimensione)x9

        for i in range(self.nr_classes):
            query = i
            query_prob[:,i] = self.compute_query(query, worlds_prob).view(-1) #probabilità delle somme per ogni batch

        # add a small offset
        query_prob += 1e-5
        with torch.no_grad(): #normalizzo le probabilità per le queries
            Z = torch.sum(query_prob, dim=-1, keepdim=True)
        query_prob = query_prob / Z

        return query_prob, worlds_prob

    def compute_query(self, query, worlds_prob):
            """Computes query probability given the worlds probability P(w)."""
            # Select the column of w_q matrix corresponding to the current query
            w_q = self.w_q[:, query] 
            # Compute query probability by summing the probability of all the worlds where the query is true
            query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True) #64x1
            return query_prob

    def normalize_concepts(self, z, split=2):
        """Computes the probability for each ProbLog fact given the latent vector z"""
        # Extract probs for each digit

        prob_digit_1, prob_digit_2,prob_digit_3 = z[:, 0,:], z[:, 1,:], z[:, 2,:] #nx5

        # # Add stochasticity on prediction
        # prob_digit_1 += 0.5 * torch.randn_like(prob_digit_1, device=prob_digit_1.device)
        # prob_digit_2 += 0.5 * torch.randn_like(prob_digit_2, device=prob_digit_1.device)

        # Sotfmax on digits_probs (the 5 digits values are mutually exclusive)
        prob_digit_1 = nn.Softmax(dim=1)(prob_digit_1) #vettore di probabilità nx5
        prob_digit_2 = nn.Softmax(dim=1)(prob_digit_2)
        prob_digit_3 = nn.Softmax(dim=1)(prob_digit_3)

        # Clamp digits_probs to avoid ProbLog underflow
        eps = 1e-5
        prob_digit_1 = prob_digit_1 + eps
        with torch.no_grad():
            Z1 = torch.sum(prob_digit_1, dim=-1, keepdim=True) #nx1
        prob_digit_1 = prob_digit_1 / Z1  # Normalization
        prob_digit_2 = prob_digit_2 + eps
        with torch.no_grad():
            Z2 = torch.sum(prob_digit_2, dim=-1, keepdim=True)
        prob_digit_2 = prob_digit_2 / Z2  # Normalization
        prob_digit_3 = prob_digit_3 + eps
        with torch.no_grad():
            Z3 = torch.sum(prob_digit_3, dim=-1, keepdim=True)
        prob_digit_3 = prob_digit_3 / Z3  # Normalization

        return torch.stack([prob_digit_1, prob_digit_2, prob_digit_3], dim=1).view(-1, 3, self.n_facts)# due tensori batchx2x5
    
    @staticmethod
    def get_loss(args):
            return ADDMNIST_DPL(ADDMNIST_Cumulative)
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)