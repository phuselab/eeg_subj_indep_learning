from pipelines.analysis_blocks import run_lbm, run_cbramod_diva, run_alignment
from utils import argparser
import random
import numpy as np
import torch
import os
from utils.helper import SEED



def set_seed():
    # 1. Python standard library
    random.seed(SEED)
    
    # 2. Numpy
    np.random.seed(SEED)
    
    # 3. PyTorch (CPU e tutte le GPU)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # 4. Variabili d'ambiente per hash e algoritmi deterministici
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    # 5. Configurazione per il backend CuDNN (fondamentale per GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        

set_seed()



def main():
    args = argparser.parse_training_args()
    if args.analysis_block == "lbm":
        run_lbm(args)
    if args.analysis_block == "disentanglement":
        run_cbramod_diva(args)
    if args.analysis_block == "alignment":
        run_alignment(args)
    


if __name__ == "__main__":
    main()