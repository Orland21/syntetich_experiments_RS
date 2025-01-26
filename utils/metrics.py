import torch
import numpy as np
from utils.dpl_loss import ADDMNIST_DPL
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_mix(true, pred):
    ac = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='weighted')
    # pc = precision_score(true, pred)
    # rc = recall_score(true, pred)
    
    return ac, f1 #, pc, rc 

def evaluate_metrics(model, loader, args, _loss=None, last=False):
    L = len(loader)  # dimensione suddivisione
    tloss, cacc, yacc, tot_loss, cacc_scambio = 0, 0, 0, 0, 0
    for i, data in enumerate(loader):
        latents, labels, concepts, representation = data
        latents, labels, concepts, representation = latents.to(model.device), labels.to(model.device), concepts.to(model.device), representation.to(model.device)

        out_dict = model(representation)
        out_dict.update({'INPUTS': latents, 'LABELS': labels, 'CONCEPTS': concepts, 'REPRESENTATION': representation})

        if _loss is not None:
            loss, losses = _loss(out_dict, args)
            tot_loss += loss.item()

        if last and i == 0:
            y_true = labels.detach().cpu().numpy().astype(int)
            c_true = concepts.detach().cpu().numpy().astype(int)
            y_pred = out_dict['YS'].detach().cpu().numpy()
            c_pred = out_dict['CS'].detach().cpu().numpy()
        elif last and i > 0:
            y_true = np.concatenate([y_true, labels.detach().cpu().numpy().astype(int)], axis=0)
            c_true = np.concatenate([c_true, concepts.detach().cpu().numpy().astype(int)], axis=0)
            y_pred = np.concatenate([y_pred, out_dict['YS'].detach().cpu().numpy()], axis=0)
            c_pred = np.concatenate([c_pred, out_dict['CS'].detach().cpu().numpy()], axis=0)

        if not last:
            loss, ac, acc = ADDMNIST_eval_tloss_cacc_acc(out_dict, concepts)  # , cacc_scambio
            tloss += loss.item()
            cacc += ac
            yacc += acc
            # cacc_scambio += ac_scambio
        else:
            NotImplementedError()

    if last:
        ys = np.argmax(y_pred, axis=1)
        gs = np.split(c_true, c_true.shape[1], axis=1)

        cs = np.split(c_pred, c_pred.shape[1], axis=1)
        # cs_flip=np.copy(cs)
        # cs_flip[0],cs_flip[1]=cs[1],cs[0]

        assert len(gs) == len(cs), f'gs: {gs.shape}, cs: {cs.shape}'

        gs = np.concatenate(gs, axis=0).squeeze(1)  # prima tutti i digit 1, poi tutti i digit 2 e infine tutti i digit 3
        cs = np.concatenate(cs, axis=0).squeeze(1).argmax(axis=1)
        # cs_flip=np.concatenate(cs_flip, axis=0).squeeze(1).argmax(axis=1)

        assert gs.shape == cs.shape, f'gs: {gs.shape}, cs: {cs.shape}'

        return y_true, gs, ys, cs  # , cs_flip
    else:
        return tloss / L, cacc / L, yacc / L, tot_loss / L  # , cacc_scambio / L


def pair_concepts_acc(gs,cs):

    gs=torch.from_numpy(gs)
    cs=torch.from_numpy(cs)

    # Split tensors into two halves
    n = len(gs) // 2
    gs_1, gs_2 = gs[:n], gs[n:]
    cs_1, cs_2 = cs[:n], cs[n:]

    # Create pair tensors
    g = torch.cat((gs_1.unsqueeze(1), gs_2.unsqueeze(1)), dim=1)
    c = torch.cat((cs_1.unsqueeze(1), cs_2.unsqueeze(1)), dim=1)

    # Check for correct pairs and swapped pairs
    correct_pairs = (gs_1 == cs_1) & (gs_2 == cs_2)
    swapped_pairs = (gs_1 == cs_2) & (gs_2 == cs_1) & (cs_1 != cs_2)  # Avoid double-counting identical pairs

    # Check for element-wise correctness
    correct_single = (g == c)
    single_right = correct_single.sum(dim=1)  # Count matches per row
    swap_index = torch.nonzero(single_right == 0).squeeze(1)  # Rows without matches

    if swap_index.numel() > 0:
        g_s, c_s = g[swap_index, :].flip(1), c[swap_index, :]
        correct_single_swap = (g_s == c_s)
        single_swap = correct_single_swap.sum().item()
    else:
        single_swap = 0

    # Calculate total results
    correct_pair = correct_pairs.sum().item()  # Correct pairs
    swapped_pair = swapped_pairs.sum().item()  # Swapped pairs
    total_pairs = correct_pair + swapped_pair  # Total correct pairs (ordered or swapped)
    correct_single_total = single_right.sum().item() + single_swap  # Total correct elements

    return correct_pair, swapped_pair, correct_single_total, total_pairs

def ADDMNIST_eval_tloss_cacc_acc(out_dict, concepts):
    logits = out_dict['CS']
    L = len(logits)
    
    objs = torch.split(logits, 1, dim=1) # tupla di 3 elementi nx1xn_classes
    g_objs = torch.split(concepts, 1, dim=1) #tupla di 3 elementi nx1
    
    assert len(objs) == len(g_objs), f'{len(objs)}-{len(g_objs)}'
        
    loss, cacc= 0, 0 #, cacc_scambio= 0
    for j in range(len(objs)):
        # enconding + ground truth
        obj_enc = objs[j].squeeze(dim=1) #nxn_classes
        g_obj   = g_objs[j].to(torch.long).view(-1) # (n,)

        # evaluate loss on concepts
        loss += torch.nn.CrossEntropyLoss()(obj_enc, g_obj)
        
        # concept accuracy of object
        c_pred = torch.argmax(obj_enc, dim=1)
        #print(g_obj)
        #print(c_pred)
        
        assert c_pred.size() == g_obj.size(), f'size c_pred: {c_pred.size()}, size g_objs: {g_obj.size()}'
        
        correct = (c_pred == g_obj).sum().item()
        cacc += correct / len(objs[j])

        #confondo i concetti destra e sinistra? siamo a 3 concetti
        # obj_enc_scambio = objs[1-j].squeeze(dim=1)
        # c_pred_scambio = torch.argmax(obj_enc_scambio, dim=1)
        # correct_scambio= (c_pred_scambio == g_obj).sum().item()
        # cacc_scambio += correct_scambio / len(objs[1-j])
        
    y = out_dict['YS']
    y_true = out_dict['LABELS']

    y_pred = torch.argmax(y, dim=-1)

    assert y_pred.size() == y_true.size(), f'size c_pred: {c_pred.size()}, size g_objs: {g_obj.size()}' 

    acc = (y_pred == y_true).sum().item() / len(y_true)
    
    return loss / len(objs), cacc / len(objs) * 100, acc * 100#, cacc_scambio / len(objs) * 100

def evaluate_model_output(model, loader):
    L = len(loader)#dimensione suddivisione
    for i, data in enumerate(loader):
        latents, labels, concepts, representation =data
        latents, labels, concepts, representation= latents.to(model.device), labels.to(model.device), concepts.to(model.device),representation.to(model.device)
            
        out_dict = model(representation)
        out_dict.update({'INPUTS': latents, 'LABELS': labels, 'CONCEPTS': concepts, 'REPRESENTATION': representation})
        
        if i == 0:
            y_true = labels.detach().cpu().numpy().astype(int)
            c_true = concepts.detach().cpu().numpy().astype(int)
            y_pred = out_dict['YS'].detach().cpu().numpy()
            c_pred = out_dict['CS'].detach().cpu().numpy() #batchX3xn_classes
        elif i > 0:
            y_true = np.concatenate([y_true, labels.detach().cpu().numpy().astype(int)], axis=0)
            c_true = np.concatenate([c_true, concepts.detach().cpu().numpy().astype(int)], axis=0)
            y_pred = np.concatenate([y_pred, out_dict['YS'].detach().cpu().numpy()], axis=0)
            c_pred = np.concatenate([c_pred, out_dict['CS'].detach().cpu().numpy()], axis=0)

                
       
    ys = np.argmax(y_pred, axis=1)
    gs = np.split(c_true, c_true.shape[1], axis=1) #batchxx5 2 tensori
    cs = np.split(c_pred, c_pred.shape[1], axis=1)
                
    assert len(gs) == len(cs), f'gs: {gs.shape}, cs: {cs.shape}'
                
    g1 = gs[0].squeeze(1)
    g2= gs[1].squeeze(1)
    g3= gs[2].squeeze(1)
    c1 = cs[0].squeeze(1).argmax(axis=1)
    c2= cs[1].squeeze(1).argmax(axis=1)
    c3= cs[2].squeeze(1).argmax(axis=1)
        
                
    return y_true, g1, g2,g3, ys, c1, c2, c3
