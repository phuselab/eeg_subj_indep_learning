import torch
from sklearn.model_selection import train_test_split
import logging
import numpy as np
from collections import Counter

# Local imports will need adjustment
from data.dataloaders.shared_loader import CustomLoaderShared
from utils.helper import SEED



def create_disjoint_loaders(data_dict: dict, batch_size: int,
                           disjoint_ratio: float, num_samples_per_epoch: int = 1000, 
                           exclude_tasks=[], seed=SEED, by_subject=False, by_subject_inference=False,
                           divisor=100, exclude_subjects=[]):
    """
    Create disjoint Train, Validation, and Test loaders.
    The Test set contains entirely different subjects from Train and Validation.
    """
    LoaderClass = CustomLoaderShared
    
    # --- PREPROCESSING ---
    if 'features' in data_dict and 'data' not in data_dict:
        data_dict['data'] = data_dict.pop('features')
    if 'ch_names' in data_dict:
        data_dict.pop('ch_names')
    if 'fold_info' in data_dict:
        data_dict.pop('fold_info')
    if 'tasks' not in data_dict and ('y' in data_dict or 'Y' in data_dict):
        data_dict['tasks'] = data_dict.get('y', data_dict.get('Y'))
    if 'labels' in data_dict:
        data_dict.pop('labels')

    # exclude subjects if specified
    if len(exclude_subjects) > 0:
        print(f"Excluding subjects: {exclude_subjects}")
        valid_indices_subject = np.where(~torch.isin(torch.tensor(data_dict["subjects"]), torch.tensor(exclude_subjects)))[0]
        data_dict = {k: v[valid_indices_subject] for k, v in data_dict.items()}

    # Remap tasks to 0-indexed if necessary
    if data_dict['tasks'].min() != 0:
        unique_tasks = data_dict['tasks'].clone().unique().tolist()
        task_mapping = {old: new for new, old in enumerate(unique_tasks)}
        data_dict['tasks'] = torch.tensor([task_mapping[t.item()] for t in data_dict['tasks']])
        
    # Remap subjects to 0-indexed if necessary
    if data_dict['subjects'].min() != 0:
        unique_subjects = data_dict['subjects'].clone().unique().tolist()
        subject_mapping = {old: new for new, old in enumerate(unique_subjects)}
        data_dict['subjects'] = torch.tensor([subject_mapping[s.item()] for s in data_dict['subjects']])

    # --- 1. SPLIT SUI SOGGETTI (TEST DISGIUNTO) ---
    unique_subjects = data_dict["subjects"].clone().unique().tolist()
    
    # Usa disjoint_ratio per decidere quanti SOGGETTI vanno nel test (es. 0.2 o 0.3)
    train_val_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=disjoint_ratio, random_state=SEED
    )
    logging.info(f"Total unique subjects in train/val: {len(train_val_subjects)} | in test: {len(test_subjects)}")
    
    test_mask = torch.isin(data_dict["subjects"], torch.tensor(test_subjects))
    train_val_mask = torch.isin(data_dict["subjects"], torch.tensor(train_val_subjects))
    
    test_data_dict = {k: v[test_mask] for k, v in data_dict.items()}
    train_val_data_dict = {k: v[train_val_mask] for k, v in data_dict.items()}

    # --- 2. SPLIT TRAIN/VALIDATION (BILANCIATO PER SOGGETTO E TASK) ---
    # Definisci qui il ratio di validation rispetto al blocco train_val (es. 0.2 = 20%)
    val_ratio = 0.2 
    
    merged_info = [f"{s.item()}_{t.item()}" for s, t in 
                   zip(train_val_data_dict["subjects"], train_val_data_dict["tasks"])]
    merged_info = np.array(merged_info)
    counts = Counter(merged_info)

    # Identifica le combinazioni con un solo esempio (che fanno crashare lo stratify)
    singletons = {k for k, v in counts.items() if v < 2}

    if singletons:
        logging.warning(f"Trovati {len(singletons)} gruppi Soggetto_Task con un solo elemento. Verranno forzati nel Training set.")
        keep_mask = np.array([m not in singletons for m in merged_info])
        
        # Indici sicuri da splittare
        indices_to_split = np.arange(len(merged_info))[keep_mask]
        merged_info_filtered = merged_info[keep_mask]
        
        train_idx, val_idx = train_test_split(
            indices_to_split, 
            test_size=val_ratio, 
            random_state=SEED, 
            stratify=merged_info_filtered
        )
        # Recupera i singletons e aggiungili al train
        singleton_indices = np.where(~keep_mask)[0]
        train_idx = np.concatenate([train_idx, singleton_indices])
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(merged_info)), 
            test_size=val_ratio, 
            random_state=SEED, 
            stratify=merged_info
        )

    # Crea i dizionari finali per Train e Validation
    train_data_dict = {k: v[train_idx] for k, v in train_val_data_dict.items()}
    val_data_dict = {k: v[val_idx] for k, v in train_val_data_dict.items()}

    #logging.info(f"Samples finali -> Train: {len(train_data_dict['data'])}, Val: {len(val_data_dict['data'])}, Test: {len(test_data_dict['data'])}")

    # --- 3. CREAZIONE DATALOADERS ---
    custom_train = LoaderClass(train_data_dict, divisor=divisor, exclude_tasks=exclude_tasks)
    custom_val = LoaderClass(val_data_dict, divisor=divisor, exclude_tasks=exclude_tasks)
    custom_test = LoaderClass(test_data_dict, divisor=divisor, exclude_tasks=exclude_tasks)

    train_loader = custom_train.get_dataloader(
        num_total_samples=None,
        batch_size=batch_size,
        property=None, 
        random_sample=True,
        by_subject=by_subject
    )
    val_loader = custom_val.get_dataloader(
        num_total_samples=None,
        batch_size=batch_size,
        property=None,
        random_sample=False,
        by_subject=by_subject_inference
    )
    test_loader = custom_test.get_dataloader(
        num_total_samples=None,
        batch_size=batch_size,
        property=None,
        random_sample=False,
        by_subject=by_subject_inference
    )

    loader_info = {
        "train": {"num_subjects": len(custom_train.unique_subjects), "num_tasks": len(custom_train.unique_tasks)},
        "val": {"num_subjects": len(custom_val.unique_subjects), "num_tasks": len(custom_val.unique_tasks)},
        "test": {"num_subjects": len(custom_test.unique_subjects), "num_tasks": len(custom_test.unique_tasks)}
    }
    
    return train_loader, val_loader, test_loader, loader_info



