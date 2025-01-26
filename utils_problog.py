from itertools import product
import torch


def build_worlds_queries_matrix(sequence_len=0, n_digits=0, task='addition'):
    """Build Worlds-Queries matrix""" #matrici 2 entrsnte mondi e domande
    if task == 'addition':
        possible_worlds = list(product(range(n_digits), repeat=sequence_len)) #lista a due a due dei possibili addendi, product è il prodotto cartesiano, reapeat è la lunghezza delle combinazioni, per righe
        n_worlds = len(possible_worlds) #numero di coppie possibili
        n_queries = len(range(0, 9))#numero di possibili risultati
        look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)} #avendo il numero delle possibili coppie, dizionario che restituisce la coppia caratteristica
        w_q = torch.zeros(n_worlds, n_queries)  # (25, 9)
        for w in range(n_worlds):
            digit1, digit2 = look_up[w] #ricavo le 2 digit 
            for q in range(n_queries):
                if digit1 + digit2 == q:
                    w_q[w, q] = 1
        return w_q
    
    elif task == 'score' or task=='position':
        possible_worlds = list(product(range(n_digits), repeat=sequence_len)) #lista a due a due dei possibili addendi, product è il prodotto cartesiano, reapeat è la lunghezza delle combinazioni
        n_worlds = len(possible_worlds) #numero di coppie possibili
        n_queries = len(range(0, 3))#numero di possibili risultati
        look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)} #avendo il numero delle possibili coppie, dizionario che restituisce la coppia caratteristica
        w_q = torch.zeros(n_worlds, n_queries)  # (100, 3)
        for w in range(n_worlds):
            digit1, digit2 = look_up[w] #ricavo le 2 digit
            result=int(digit1<=digit2)
            add=int(digit1==digit2)
            result=result+add
            for q in range(n_queries):
                if result == q:
                    w_q[w, q] = 1
        return w_q
    
        
    else:
        NotImplementedError('Wrong choice')

