import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

from braindecode.models import EEGNet
from braindecode import EEGClassifier

from data.dataloaders.shared_loader import CustomLoaderShared
from data import create_disjoint_loaders
from utils.helper import fit_clf_fn
from models.backbones import create_backbone
from models.disentanglement.core import DisentangledEEGModel

def get_xy(loader):
    """
    Extract (X, y) numpy arrays from a loader.
    """
    all_x, all_y = [], []
    for batch in loader:
        all_x.append(batch[1])
        all_y.append(batch[3])
    X = torch.cat(all_x).numpy()
    y = torch.cat(all_y).numpy()
    return X, y

def evaluate_competitors_on_data(X_train, y_train, X_test, y_test, name_tag=""):
    """
    Trains and evaluates XGBoost and EEGNet on the provided data.
    """
    print(f"\n--- Evaluating {name_tag} ---")

    # 1. XGBoost Baseline
    print("Training XGBoost...")
    # X_train/X_test
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    xgb_clf = fit_clf_fn(X_train_flat, y_train)
    y_pred_xgb = xgb_clf.predict(X_test_flat)
    xgb_acc = balanced_accuracy_score(y_test, y_pred_xgb)
    print(f"XGBoost {name_tag} Acc: {xgb_acc:.4f}")

    # 2. EEGNet Baseline
    X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32)

    net = EEGClassifier(
        EEGNet,
        module__n_chans=X_train.shape[1],
        module__n_outputs=len(np.unique(y_train)),
        module__n_times=X_train.shape[2],
        module__F1=8,
        module__D=2,
        max_epochs=30,
        lr=0.001,
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[],
        train_split=None
    )
    
    print(f"Training EEGNet on {len(X_train_t)} samples...")
    net.fit(X_train_t, y_train_t)
    y_pred_eeg = net.predict(X_test_t)
    eeg_acc = balanced_accuracy_score(y_test, y_pred_eeg)
    print(f"EEGNet {name_tag} Acc: {eeg_acc:.4f}")

    return {"XGB_Acc": xgb_acc, "EEG_Acc": eeg_acc}

def main():

    model_path = Path('/home/juser/projects/shared/eeg_disentanglement/experiments/MI/MI_CBraMod_weighted_STFT01_1_newgammadeltaweighting/last_model.pt')
    run_config = torch.load(model_path.parent / 'run_config.pt', weights_only=False)
    args = argparse.Namespace(**run_config)

    MI_translated_path = '/home/juser/projects/shared/eeg_disentanglement/data/translated_data/MI/translated_test_gamma_MI_single_train_subject_no_swap.pt'
    MI_original_path = '/mnt/pve/Rita-Storage-2/disentangleData/processed_data/MI_eeg_cbramod.pt'

    # Data Loading
    print("Loading data dictionaries...")
    translated_data_dict = torch.load(MI_translated_path, weights_only=False)
    original_data_dict = torch.load(MI_original_path, weights_only=False)

    # Loader Initialization 
    # Original test set via disjoint loaders
    train_loader, val_loader, test_loader, loader_info = create_disjoint_loaders(
        original_data_dict, 64, args.disjoint_split_ratio, exclude_tasks=[args.exclude_tasks]
    )

    # Translated dataset via CustomLoaderShared
    trans_dataset = CustomLoaderShared(translated_data_dict, exclude_tasks=[args.exclude_tasks])
    print(f"\nTranslated Dataset Stats:")
    print(f"  Size: {len(trans_dataset)}")
    print(f"  Tasks: {trans_dataset.unique_tasks}")
    print(f"  Subjects: {len(trans_dataset.unique_subjects)}")

    # Get data for evaluation
    print("\nPreparing training data...")
    X_train, y_train = get_xy(train_loader)

    # Evaluation on Original Test Set
    print("Processing Original Test Set...")
    X_te_orig, y_te_orig = get_xy(test_loader)
    res_orig = evaluate_competitors_on_data(X_train, y_train, X_te_orig, y_te_orig, "Original")

    # Evaluation on Translated Test Set
    print("Processing Translated Test Set...")
    _, X_test_tran, _, y_test_tran, _ = trans_dataset.sample_batch(len(trans_dataset))
    
    # Convert samples to numpy to match get_xy output format
    X_test_tran_np = X_test_tran.numpy()
    y_test_tran_np = y_test_tran.numpy()
    
    res_tran = evaluate_competitors_on_data(X_train, y_train, X_test_tran_np, y_test_tran_np, "Translated")

    # --- 5. Summary Results ---
    print("\n" + "="*40)
    print("FINAL COMPARISON SUMMARY for Original vs Translated Test Sets:")
    print("="*40)
    results_df = pd.DataFrame([res_orig, res_tran], index=["Original Set", "Translated Set"])
    print(results_df.to_string())
    

if __name__ == "__main__":
    main()