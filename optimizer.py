import optuna
import torch
import torch.optim as optim
from utils.status import progress_bar
from utils.wandb_logger import *
from utils.data_generation import get_data_loaders
from warmup_scheduler import GradualWarmupScheduler
from argparse import ArgumentParser
from utils.conf import *
from nets.DPL_model import DPL_synthetic
from utils.metrics import evaluate_metrics,evaluate_mix
import  wandb


def parse_arg():
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='type of task.')
    parser.add_argument('--path', type=str, required=True, help='data path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--wandb', type=str, default=None,  
                        help='Enable wandb logging -- set name of project')
    parser.add_argument('--project', type=str, default=None,  
                        help='-- set name of project')
    parser.add_argument('--debug', action='store_true', help='Concepts supervision')

    args = parser.parse_args()

    return args


# Funzione obiettivo per Optuna
def objective(trial, args):
    # **Scelta degli iperparametri**
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    exp_decay = trial.suggest_categorical('exp_decay', [0.95, 0.96, 0.97, 0.98])
    warmup_steps = trial.suggest_int('warmup_steps', 2, 8)
    epochs = trial.suggest_int('epochs', 10, 50)
    args.batch_size = batch_size

    # Modello
    model = DPL_synthetic(args)
    _loss = model.get_loss(args)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # DataLoader
    model.to(model.device)
    train_loader, val_loader, _ = get_data_loaders(args, p=0)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)
    w_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=warmup_steps)

    # WandB Config
    wandb.init(
        project=args.project,
        entity=args.wandb,
        name=f"trial_{trial.number}",
        config={'lr': lr, 'batch_size': batch_size, 'exp_decay': exp_decay, 'warmup': warmup_steps, 'epochs': epochs}
    )

    # **Training**
    for epoch in range(epochs):
        model.train()
        tot_loss = 0
        for i, data in enumerate(train_loader):
            latents, labels, concepts, representation = data
            latents, labels, concepts, representation = (
                latents.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
                representation.to(model.device),
            )
            out_dict = model(representation)
            out_dict.update(
                {'INPUTS': latents, 'LABELS': labels, 'CONCEPTS': concepts, 'REPRESENTATION': representation}
            )
            optimizer.zero_grad()
            loss, losses = _loss(out_dict, args)
            tot_loss += loss.item()
            loss.backward()
            optimizer.step()
        # Update learning rate
        if epoch < warmup_steps:
            w_scheduler.step()
        else:
            scheduler.step()
        # Log training loss
        wandb.log({'train_loss': tot_loss / len(train_loader), 'epoch': epoch})

    # **Validation**
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            latents, labels, concepts, representation = data
            latents, labels, concepts, representation = (
                latents.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
                representation.to(model.device),
            )
            out_dict = model(representation)
            out_dict.update(
                {'INPUTS': latents, 'LABELS': labels, 'CONCEPTS': concepts, 'REPRESENTATION': representation}
            )
            loss, _ = _loss(out_dict, args)
            val_loss += loss.item()

        # Metriche
        y_true, c_true, y_pred, c_pred = evaluate_metrics(model, val_loader, args, last=True)
        yac, yf1 = evaluate_mix(y_true, y_pred)
        cac, cf1 = evaluate_mix(c_true, c_pred)

        # Log delle metriche
        wandb.log({'y-acc': yac, 'c-acc': cac, 'y-f1': yf1, 'c-f1': cf1, 'val-loss': val_loss})
        wandb.finish()

        # Metriche extra per Optuna
        trial.set_user_attr('y_acc', yac)
        trial.set_user_attr('y_f1', yf1)
        trial.set_user_attr('c_acc', cac)
        trial.set_user_attr('c_f1', cf1)

    return val_loss



if __name__ == '__main__':
    args = parse_arg()
    set_random_seed(42)

    # **Ottimizzazione con Optuna**
    study = optuna.create_study(direction='minimize')  # Minimizza la validation loss
    study.optimize(lambda trial: objective(trial,args), n_trials=5)  # Testa 20 combinazioni

    # Migliori iperparametri
    with open("best_parameters_"+args.dataset+"_summary.txt", "a") as f:
        f.write(f"Best parameters: {study.best_params}\n")
        f.write(f"Best validation loss: {study.best_value}\n")
        f.write(f"Best y-acc: {study.best_trial.user_attrs['y_acc']}\n")
        f.write(f"Best y-f1: {study.best_trial.user_attrs['y_f1']}\n")
        f.write(f"Best c-acc: {study.best_trial.user_attrs['c_acc']}\n")
        f.write(f"Best c-f1: {study.best_trial.user_attrs['c_f1']}\n")



