'''train'''

from itertools import product
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import seaborn as sns
import wandb


from utils.dpl_loss import ADDMNIST_DPL
from utils.status import progress_bar
from utils.wandb_logger import *
from utils.metrics import evaluate_metrics, evaluate_mix, evaluate_model_output, pair_concepts_acc
from utils.data_generation import get_data_loaders

from warmup_scheduler import GradualWarmupScheduler

def monitor_gradients(model):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms

def train(model, _loss:ADDMNIST_DPL, args,seed,p,count):

    model.to(model.device)
    train_loader,val_loader,test_loader=get_data_loaders(args,p)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(model.opt,args.exp_decay)
    w_scheduler=GradualWarmupScheduler(model.opt,1.0,args.warmup_steps)
    torch.autograd.set_detect_anomaly(True)

    if args.wandb is not None:
        print('\n--wandb on-- \n')
        wandb.init(project=args.project, entity=args.wandb, name=args.dataset+'_42k_'+str(seed)+f'_{p}_', config=args)

    print('\n-- Start of Training --\n')

    model.opt.zero_grad()
    model.opt.step()


    for epoch in range (args.n_epochs):
        all_concepts = []
        all_labels = []
        tot_loss=0
        L=len(train_loader)
        for i, data in enumerate(train_loader):
            model.train()

            latents, labels, concepts, representation =data
            latents, labels, concepts, representation= latents.to(model.device), labels.to(model.device), concepts.to(model.device),representation.to(model.device)
            
            out_dict = model(representation)
            out_dict.update({'INPUTS': latents, 'LABELS': labels, 'CONCEPTS': concepts, 'REPRESENTATION': representation})

            all_concepts.append(concepts)
            all_labels.append(labels)

            model.opt.zero_grad()
            loss, losses=_loss(out_dict,args)
            tot_loss+=loss.item()
            loss.backward()
            model.opt.step()
            # grad_norms = monitor_gradients(model)
            # print(f"Epoch {epoch}: Max grad: {max(grad_norms.values()):.4f}, Min grad: {min(grad_norms.values()):.4f}")
            if args.wandb is not None:
                wandb_log_step(i,epoch,loss.item(),losses)

            if i%10==0: progress_bar(i,len(train_loader)-9,epoch, loss.item())
        
        # concepts = torch.cat(all_concepts, dim=0)
        # labels = torch.cat(all_labels, dim=0)
        # tot = int(torch.max(labels))
        # class_sample_count = np.array([len(torch.nonzero(labels == t)) for t in torch.unique(labels)])
        # print(class_sample_count)

        # pairs = list(product(range(5), repeat=3)) 
        # concepts=np.array([tuple(row.tolist()) for row in concepts])
        # pair_sample_count = np.array([np.sum(np.all(concepts == t, axis=1)) for t in pairs])
        # print(pair_sample_count)
        # print(sum(pair_sample_count))
        model.eval()
        with torch.no_grad():
            c_loss, cacc, yacc, vloss = evaluate_metrics(model, val_loader, args ,_loss) # media metriche cacc_scambio
            y_t, g1, g2, g3, ys, c1, c2, c3 =evaluate_model_output(model, val_loader)
        
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item()
        print(f"Total gradient norm: {total_grad_norm:.4f}")

        # update at end of the epoch 
        if epoch < args.warmup_steps:  
            w_scheduler.step()
        else:          
            scheduler.step()
    

        ### LOGGING ###

        print('  ACC C', cacc, '  ACC Y', yacc ) #, '  ACC C SCAMBIO', cacc_scambio

        if args.wandb is not None:
            wandb_log_epoch(epoch=epoch, acc=yacc, cacc=cacc,vloss=vloss,
                            closs=c_loss, totloss=tot_loss/L,
                            lr=float(scheduler.get_last_lr()[0])) #, cacc_scambio=cacc_scambio
    
    # Evaluate performances on val or test
    model.eval()
    if args.validate:
        y_true, c_true, y_pred, c_pred  = evaluate_metrics(model, val_loader, args, last=True) #, c_pred_flip
    else:
        y_true, c_true, y_pred, c_pred = evaluate_metrics(model, test_loader, args, last=True)  #, c_pred_flip
        y_t, g1, g2, g3, ys, c1, c2, c3 = evaluate_model_output(model, test_loader)
        #correct_pair, swapped_pair, correct_single_total, total_pairs=pair_concepts_acc(c_true,c_pred)


    # if args.wandb is not None:
        
    #     wandb.log({'test-right_concepts':correct_pair , 'test-swapped_concepts': swapped_pair, 'test-pair_concepts_up_to_permutation': total_pairs,'test-single_concepts_up_to_permutation':correct_single_total} )
    c_tuple_true = np.concatenate([g1.reshape(-1,1), g2.reshape(-1,1), g3.reshape(-1,1)], axis=1)
    c_tuple_pred = np.concatenate([c1.reshape(-1,1), c2.reshape(-1,1), c3.reshape(-1,1) ], axis=1)
    c_encod_true=encoding_tuple(c_tuple_true)
    c_encod_pred=encoding_tuple(c_tuple_pred)

    yac, yf1 = evaluate_mix(y_true, y_pred)
    cac, cf1 = evaluate_mix(c_true, c_pred)
    if yf1*100>= 90:
        count+=1
        with open("best_models_"+args.dataset+"_summary.txt", "a") as f:
            f.write(f"{count})Seed: {seed}, Batch Size: {args.batch_size}, CF1: {cf1}, CAC: {cac}, YF1: {yf1}, YAC: {yac}\n")

    #cac_swap, cf1_swap = evaluate_mix(c_true, c_pred_swap)
    print('concetti veri<-->predetti')
    print([c_true, c_pred])
    
    print(f'Concepts:\n    ACC: {cac}, F1: {cf1}')
    #print(f'Concepts swapped:\n    ACC: {cac_swap}, F1: {cf1_swap}')
    print(f'Labels:\n      ACC: {yac}, F1: {yf1}')
    
    if args.wandb is not None:
        K = max(max(y_pred), max(y_true))
        
        wandb.log({'test-y-acc': yac*100, 'test-y-f1': yf1*100})
        wandb.log({'test-c-acc': cac*100, 'test-c-f1': cf1*100})
     #   wandb.log({'test-c-acc-with_exchange':cac_swap*100,'test-c-f1-with_exchange': cf1_swap*100})
        
        wandb.log({
            'cf-labels': wandb.plot.confusion_matrix(None, y_true.astype(int), y_pred,  class_names=[str(i) for i in range(K+1)]),
            })
        # K = max(np.max(c_pred), np.max(c_true))
        # wandb.log({
        #     'cf-concepts': wandb.plot.confusion_matrix(None, c_true.astype(int), c_pred,  class_names=[str(i) for i in range(K+1)]),
        # })

        K = max(np.max(c1), np.max(g1))#confusion matrix digit 1
        wandb.log({
            'cf-concept_1': wandb.plot.confusion_matrix(None, g1.astype(int),c1.astype(int),  class_names=[str(i) for i in range(K+1)]),
        })
        K = max(np.max(c2), np.max(g2))#confusion matrix digit 2
        wandb.log({
            'cf-concepts_2': wandb.plot.confusion_matrix(None, g2.astype(int),c2.astype(int),  class_names=[str(i) for i in range(K+1)]),
        })
        K = max(np.max(c3), np.max(g3))#confusion matrix digit 3
        wandb.log({
            'cf-concepts_3': wandb.plot.confusion_matrix(None, g3.astype(int),c3.astype(int),  class_names=[str(i) for i in range(K+1)]),
        })
        #confusion matrix totale 5 da 25x25
        # K=max(np.max(c_encod_true),np.max(c_encod_pred))
        # wandb.log({
        #     'cf-concept_tuples': wandb.plot.confusion_matrix(None, c_encod_true.astype(int),c_encod_pred.astype(int),  class_names=[f"({c1},{c2},{c3})" for c1 in range(5) for c2 in range(5) for c3 in range(5)]),
        # })
        
        conf_mat = confusion_matrix(c_encod_true.astype(int),c_encod_pred.astype(int), labels=list(range(125)))  # Calcola la confusion matrix
        conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        # Crea la heatmap
        x_labels = ["(0,0,0)"]+[""]*4+["(0,1,0)"]+[""]*4+["(0,2,0)"]+[""]*4+["(0,3,0)"]+[""]*4+["(0,4,0)"]+[""]*4+["(1,0,0)"]+[""]*4+["(1,1,0)"]+[""]*4+["(1,2,0)"]+[""]*4+["(1,3,0)"]+[""]*4+["(1,4,0)"]+[""]*4+["(2,0,0)"]+[""]*4+["(2,1,0)"]+[""]*4+["(2,2,0)"]+[""]*4+["(2,3,0)"]+[""]*4+["(2,4,0)"]+[""]*4+["(3,0,0)"]+[""]*4+["(3,1,0)"]+[""]*4+["(3,2,0)"]+[""]*4+["(3,3,0)"]+[""]*4+["(3,4,0)"]+[""]*4+["(4,0,0)"]+[""]*4+["(4,1,0)"]+[""]*4+["(4,2,0)"]+[""]*4+["(4,3,0)"]+[""]*4+["(4,4,0)"]+[""]*3+["(4,4,4)"]
        y_labels = ["(0,0,0)"]+[""]*4+["(0,1,0)"]+[""]*4+["(0,2,0)"]+[""]*4+["(0,3,0)"]+[""]*4+["(0,4,0)"]+[""]*4+["(1,0,0)"]+[""]*4+["(1,1,0)"]+[""]*4+["(1,2,0)"]+[""]*4+["(1,3,0)"]+[""]*4+["(1,4,0)"]+[""]*4+["(2,0,0)"]+[""]*4+["(2,1,0)"]+[""]*4+["(2,2,0)"]+[""]*4+["(2,3,0)"]+[""]*4+["(2,4,0)"]+[""]*4+["(3,0,0)"]+[""]*4+["(3,1,0)"]+[""]*4+["(3,2,0)"]+[""]*4+["(3,3,0)"]+[""]*4+["(3,4,0)"]+[""]*4+["(4,0,0)"]+[""]*4+["(4,1,0)"]+[""]*4+["(4,2,0)"]+[""]*4+["(4,3,0)"]+[""]*4+["(4,4,0)"]+[""]*3+["(4,4,4)"]
        plt.figure(figsize=(15, 12))  # Larghezza e altezza
        plt.figure(1)
        sns.heatmap(conf_mat_normalized, cbar=True, xticklabels=x_labels, yticklabels=y_labels)

        # Titolo e etichette degli assi con fontsize personalizzato
        plt.title("Confusion Matrix Concepts (c_1,c_2,c_3)", fontsize=20)  # Ingrandisci titolo
        plt.xlabel("Predicted", fontsize=16)  # Ingrandisci etichetta X
        plt.ylabel("True", fontsize=16)  # Ingrandisci etichetta Y

        # Salva la heatmap come immagine
        plt.savefig("confusion_matrix.png")
        wandb.log({"confusion_matrix_heatmap": wandb.Image("confusion_matrix.png")})
        plt.close()
        wandb.finish()

    print('--- Training Finished ---')
