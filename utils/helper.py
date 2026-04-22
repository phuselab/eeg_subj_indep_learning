import torch
from typing import List, Tuple
import logging
import numpy as np
from pathlib import Path
import sys
import xgboost as xgb
from collections import Counter
import torch.nn.functional as F
from torch import linalg
import re


SEED = 42

BOTTLE_F_DIM = 16


def clean_ch_names(ch_names):
    if ch_names is None:
        return None
    cleaned = []
    for name in ch_names:
        # 1. Strip extra chars and force uppercase
        # "EEG Fp1-Ref" -> "FP1"
        name = re.sub(r'(?i)eeg\s*|[-_]ref|[-_]le|\.', '', name).strip().upper()
        
        # 2. Fix the 'z' for midline channels (e.g., FPZ -> Fpz, CZ -> Cz)
        if name.endswith('Z') and len(name) > 1:
            name = name[:-1] + 'z'
            
        # 3. Fix the 'p' in Fp channels (FP1 -> Fp1)
        if name.startswith('FP'):
            name = 'Fp' + name[2:]
            
        cleaned.append(name)
    return cleaned

def segment_to_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, C, T = x.shape

    n_patches = T // patch_size
    if n_patches == 0:
        raise ValueError(f"patch_size {patch_size} larger than signal length {T}")
    x = x[:, :, : n_patches * patch_size]
    return x.reshape(B, C, n_patches, patch_size).contiguous()

def get_optimal_patch_size(signal_length: int, min_patches: int = 4, max_patches: int = 16) -> int:
    candidates = []
    for n_patches in range(min_patches, max_patches + 1):
        patch_size = signal_length // n_patches
        if signal_length % patch_size < 0.2 * patch_size:
            candidates.append((patch_size, n_patches))
    if not candidates:
        return signal_length // 8
    return min(candidates, key=lambda x: abs(x[1] - 8))[0]

def calculate_train_stats(data_tensor: torch.Tensor, train_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    train_data = data_tensor[train_indices]
    mu = train_data.mean().detach().clone()
    std = train_data.std().detach().clone()
    return mu, std

def setup_logging(save_dir: str):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    log_file = save_path / 'training_log.txt'
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        force=True,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def infer_num_classes(custom_loader) -> Tuple[int, int]:
    num_subjects = len(custom_loader.unique_subjects)
    num_tasks = len(custom_loader.unique_tasks)
    return num_subjects, num_tasks

def validate_label_ranges(train_loader, num_subjects: int, num_tasks: int):
    logging.info("VALIDATING LABEL RANGES")
    logging.info(f"Expected ranges: Subjects [0, {num_subjects-1}], Tasks [0, {num_tasks-1}]")
    
    all_subjects = []
    all_tasks = []
    
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        all_subjects.append(batch[2])
        all_tasks.append(batch[3])
    
    all_subjects = torch.cat(all_subjects)
    all_tasks = torch.cat(all_tasks)
    
    subject_min, subject_max = all_subjects.min().item(), all_subjects.max().item()
    task_min, task_max = all_tasks.min().item(), all_tasks.max().item()
    
    logging.info(f"Observed ranges: Subjects [{subject_min}, {subject_max}], Tasks [{task_min}, {task_max}]")
    
    errors = []
    if subject_min < 0:
        errors.append(f"Subject labels have negative values: {subject_min}")
    if subject_max >= num_subjects:
        errors.append(f"Subject labels exceed num_classes: {subject_max} >= {num_subjects}")
    if task_min < 0:
        errors.append(f"Task labels have negative values: {task_min}")
    if task_max >= num_tasks:
        errors.append(f"Task labels exceed num_classes: {task_max} >= {num_tasks}")
    
    if errors:
        for error in errors:
            logging.error(f"  - {error}")
        raise ValueError("Label ranges do not match configured num_classes.")
    
    logging.info("All labels within valid ranges")
    
    
    
def compute_class_weights(dataloader, device):
    # Count occurrences of each class in the 'task' label
    all_labels = []
    for batch in dataloader:
        all_labels.append(batch[3]) # batch[3] is 'task'
    
    y = torch.cat(all_labels).numpy()
    classes, counts = np.unique(y, return_counts=True)
    
    # Simple Inverse Frequency
    weights = len(y) / (len(classes) * counts)
    
    # Normalize weights so they average to 1.0 (preserves learning rate scale)
    weights = weights / weights.mean()
    
    logging.info(f"Class counts: {dict(zip(classes, counts))}")
    logging.info(f"Computed class weights: {dict(zip(classes, weights))}")
    
    return torch.tensor(weights, dtype=torch.float, device=device)
    
    

class FreezeUnfreeze:
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True



class XGBWrapper:
    
    def __init__(self, xgb_object):
        self.xgb_object = xgb_object
    
    def fit(self, X, y):
        X = X.reshape(X.shape[0], -1)
        self.translation_dict = {l: i for i, l in enumerate(np.unique(y))}
        self.retranslation_dict = {i: l for i, l in enumerate(np.unique(y))}
        y = np.vectorize(self.translation_dict.get)(y)
        class_counts = Counter(y)
        class_weights = {i: min(class_counts.values()) / class_counts[i] for i in class_counts.keys()}
        class_weights_arr = np.vectorize(class_weights.get)(y)
        self.xgb_object.fit(X, y, sample_weight=class_weights_arr)
    
    def predict(self, X):
        X = X.reshape(X.shape[0], -1)
        y_pred = self.xgb_object.predict(X)
        y_pred = np.vectorize(self.retranslation_dict.get)(y_pred)
        return y_pred
    
    def score(self, X, y):
        X = X.reshape(X.shape[0], -1)
        y = np.vectorize(self.translation_dict.get)(y)
        score = self.xgb_object.score(X, y)
        return score


def fit_clf_fn(X, y, num_classes=None):
    X = X.reshape(X.shape[0], -1)
    # If num_classes is provided, pass it to the XGBClassifier; otherwise, let it infer from the data
    if num_classes is not None:
        clf = XGBWrapper(xgb.XGBClassifier(n_estimators=300, max_bin=100, learning_rate=0.3, grow_policy='depthwise', objective='multi:softmax', tree_method='hist', n_jobs=-1, num_class=num_classes))
    else:
        clf = XGBWrapper(xgb.XGBClassifier(n_estimators=300, max_bin=100, learning_rate=0.3, grow_policy='depthwise', objective='multi:softmax', tree_method='hist', n_jobs=-1))
    clf.fit(X, y)
    return clf


# def apply_euclidean_alignment(x: torch.Tensor) -> torch.Tensor:
#     """
#     Euclidean Alignment (EA).
#     Formula: y = Sigma^{-1/2} x [cite: 216-218]
#     Assumes all samples in the input batch 'x' belong to the SAME subject.
#     """
#     original_shape = x.shape
    
#     # Flatten patches into time dimension if input is (B, C, NP, T)
#     if x.dim() == 4:
#         B, C, NP, T = x.shape
#         x_work = x.view(B, C, NP * T)
#     else:
#         x_work = x

#     # Recenter per electrode (mean across the temporal dimension)
#     x_work = x_work - x_work.mean(dim=2, keepdim=True)
    
#     # Rescale using total trial std (Average Global Field Power)
#     gfp = x_work.std(dim=(1, 2), keepdim=True)
#     x_work = x_work / (gfp + 1e-6)

#     # Compute Spatial Covariance: (C, C)
#     C = x_work.shape[1]
#     # Flatten across batch and time: (C, B * T')
#     x_work_flat = x_work.transpose(0, 1).reshape(C, -1)
#     cov = torch.cov(x_work_flat)

#     # Covariance matrix square root inverse: Sigma^{-1/2}
#     L, V = torch.linalg.eigh(cov)
#     L = torch.clamp(L, min=1e-6) # Threshold for numerical stability
#     inv_sqrt_cov = V @ torch.diag(1.0 / torch.sqrt(L)) @ V.T

#     # Apply alignment: y = Sigma^{-1/2} x_work
#     out = torch.einsum('ij,njt->nit', inv_sqrt_cov, x_work)

#     # Restore original shape (B, C, NP, T) or (B, C, T)
#     return out.view(original_shape)


def apply_euclidean_alignment(x: torch.Tensor) -> torch.Tensor:
    """
    Euclidean Alignment (EA).
    Formula: y = Sigma^{-1/2} x
    Assumes all samples in the input batch 'x' belong to the SAME subject.
    """
    original_shape = x.shape
    
    # Flatten patches into time dimension if input is (B, C, NP, T)
    if x.dim() == 4:
        B, C, NP, T = x.shape
        x_work = x.view(B, C, NP * T)
    else:
        x_work = x

    # Recenter per electrode (mean across the temporal dimension)
    x_work = x_work - x_work.mean(dim=2, keepdim=True)
    
    # Rescale using total trial std (Average Global Field Power)
    gfp = x_work.std(dim=(1, 2), keepdim=True)
    x_work = x_work / (gfp + 1e-6)

    # Compute Spatial Covariance: (C, C)
    C = x_work.shape[1]
    # Flatten across batch and time: (C, B * T')
    x_work_flat = x_work.transpose(0, 1).reshape(C, -1)
    cov = torch.cov(x_work_flat)

    # -- MODIFICA QUI: Decomposizione di Cholesky come richiesto dal paper --
    # Aggiungiamo un epsilon alla diagonale per garantire che la matrice sia 
    # definita positiva, un requisito per Cholesky (aiuta la stabilità numerica)
    cov = cov + torch.eye(C, device=cov.device, dtype=cov.dtype) * 1e-6
    
    # L è la matrice triangolare inferiore tale che L @ L.T = cov
    L = torch.linalg.cholesky(cov)
    
    # Calcoliamo l'inversa della matrice L (cioè Sigma^{-1/2})
    inv_sqrt_cov = torch.linalg.inv(L)

    # Apply alignment: y = Sigma^{-1/2} x_work
    out = torch.einsum('ij,njt->nit', inv_sqrt_cov, x_work)

    # Restore original shape (B, C, NP, T) or (B, C, T)
    return out.view(original_shape)