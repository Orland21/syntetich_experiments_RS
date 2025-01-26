''' main'''
from argparse import ArgumentParser
from utils.conf import *
from utils.train import train
from utils.data_generation import data_download
from nets.DPL_model import DPL_synthetic


def parse_arg():
    parser=ArgumentParser()

    parser.add_argument('--project', type=str, required=True,
                        help='name of the project.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='type of task.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility.')
    parser.add_argument('--download_data', action='store_true',
                        help='to download the data')
    parser.add_argument('--path', type=str, required=True,
                        help ='data path')
   #optimization parameter
    parser.add_argument('--lr',default=0.005, type=float,
                        help='learning rate.')
    parser.add_argument('--warmup_steps', type=int, default=2,
                        help='warmup epovhs')
    parser.add_argument('--exp_decay', type=float, default=0.95,
                        help='decay factor')
    # learning hyperams
    parser.add_argument('--n_epochs',   type=int, default=50,
                         help='Number of epochs per task.')
    parser.add_argument('--batch_size', type=int, default=64,
                         help='Batch size.')
    # logging
    parser.add_argument('--wandb', type=str, default=None,  
                        help='Enable wandb logging -- set name of project')
    
    #parser.add_argument('--posthoc',  action='store_true', default=False, help='Used to evaluate only the loaded model')
    parser.add_argument('--debug',  action='store_true', help='Concepts supervision')
    parser.add_argument('--validate', action='store_true', default=False, help='Used to evaluate on the validation set for hyperparameters search')

    
    args = parser.parse_args()
  
    return args
    
def main_f (args,seed,p,count):

    set_random_seed(seed)
    
    model=DPL_synthetic(args)
    loss=model.get_loss(args)
    model.start_optim(args)
    
    print('    Chosen device:', model.device)
#    if args.posthoc: pass
#    else: train(model, loss, args,seed,p)
    train(model, loss, args,seed,p,count)

    print('\n ### Closing ###')

if __name__=='__main__':

    args=parse_arg()
    permute=range(6)
    if args.download_data: 
        data_download(args)
        print('data downloaded')
    else:
        count=0
        for p in permute:
            args.project='new_synthetic_'+args.dataset+f'_p={p}_'+'RS_best_models'
            for seed in range(0,50):
                print(args)
                print(f'seed:{seed}')
                seed=seed
                main_f(args,seed,p,count)

