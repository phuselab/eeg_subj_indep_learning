from unittest import loader

import torch 
import numpy as np
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')
from data.dataloaders.shared_loader import CustomLoaderShared
from tqdm.auto import tqdm
from data import create_disjoint_loaders
from utils.helper import fit_clf_fn
import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from torch import nn
from models.backbones import create_backbone
from models.disentanglement.core import DisentangledEEGModel

def normalize(data, mean, std):
    return (data - mean) / (std + 1e-6)

def translation():
    model_path = '/home/juser/projects/shared/eeg_disentanglement/experiments/MI/MI_CBraMod_weighted_STFT01_1_newgammaweighting/last_model.pt'
    backbone_weights = '/mnt/pve/Rita-Storage-2/disentangleData/weights/cbramod_pretrained_weights.pth'
    data = '/mnt/pve/Rita-Storage-2/disentangleData/processed_data/MI_eeg_cbramod.pt'
    # output_dir = '/home/juser/projects/shared/eeg_disentanglement/data/translated_data/MI/translated_test_gamma_MI_self_recon_train_dataset.pt'
    EXPERIMENT_NAME = 'MI_paper_choice'
    output_dir = f'/mnt/pve/Rita-Storage-2/disentangleData/processed_data/translated_data/{EXPERIMENT_NAME}'
    # check if output_dir exists, if not create it
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    # data = '/home/user/projects/eeg_disentanglement/data/processed_data/MI_eeg_cbramod.pt'
    data_dict = torch.load(data, weights_only=False)
    loader = CustomLoaderShared(data_dict, exclude_tasks=[4]) 
    
    device = f'cuda:{torch.cuda.device_count() - 1}' if torch.cuda.is_available() else 'cpu'
    
    #global_mean = loader.data_mean.to(device)
    #global_std = loader.data_std.to(device)


    # ── 1. Load the checkpoint ────────────────────────────────────────────────────
    checkpoint = torch.load(model_path, weights_only=False)
    run_config  = torch.load(Path(model_path).parent / 'run_config.pt', weights_only=False)

    config = checkpoint['config']
    data_shape = run_config['data_shape']
    args = argparse.Namespace(**run_config)
    args.batch_size = 4  # Override batch size for translation phase
    args.backbone_weights = backbone_weights  # Ensure backbone weights are not loaded again
    
    backbone = create_backbone(args, data_shape, config=config, use_identity_for_reconstruction=False)
    proj_out = backbone.model.proj_out if hasattr(backbone.model, 'proj_out') else None

    backbone.model.proj_out = nn.Identity()  # replace with identity to get direct reconstruction

    model = DisentangledEEGModel(
        backbone,
        config=config,
        phase_name='DVAE',
        reconstruction_decoder=proj_out,
        classifier_type='diva_classifier'
    )
    model.baseline_classifier = None  # Ensure no classifier is used during translation
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"STFTLoss: Detected {num_devices} CUDA device(s). Using device: {torch.cuda.get_device_name(0)}")
    # select last GPU if multiple are available
    model = model.to(device)


    # Now I want to get each sample from the test set, pass it through the model, get a sample from the training set, pass it through the model, and then swap the latent representations
    # to get a new sample with the subject belonging to the training set but the task belonging to the test set.

    train_loader, val_loader, test_loader, loader_info = create_disjoint_loaders(
            data_dict, args.batch_size, args.disjoint_split_ratio, exclude_tasks=[args.exclude_tasks]
        )

    # Self reconstruction train
    x_hat_list = []
    y_task_list = []
    synthetic_subject_list = []
    true_subject_list = []
    runs_list = []
    for batch in tqdm(train_loader, desc="Training"):
        with torch.no_grad():
            output = model(batch[1].to(device), do_classification=False)
            recon = output['eeg_reconstruction']
            x_hat_list.append(recon.cpu())
            y_task_list.append(batch[3].cpu())  # Assuming batch[3] contains the labels of the tasks
            
            true_subject_list.append(batch[2].cpu())  # Assuming batch[2] contains the true subject IDs
            synthetic_subject_list.append(batch[2].cpu())  # For self-reconstruction, synthetic subject is the same as true subject
            runs_list.append(batch[4].cpu())  # Assuming batch[4] contains the run information
    output_training_data = {
        'data': torch.cat(x_hat_list, dim=0),
        'tasks': torch.cat(y_task_list, dim=0),
        'subjects': torch.cat(true_subject_list, dim=0),
        'synthetic_subjects': torch.cat(synthetic_subject_list, dim=0),
        'runs': torch.cat(runs_list, dim=0)
    }

    torch.save(output_training_data, output_dir + '_self_recon_train.pt')
    print(f"Self-reconstructed samples saved to {output_dir}")
    
    # Self reconstruction test 
    x_hat_list = []
    y_task_list = []
    synthetic_subject_list = []
    true_subject_list = []
    runs_list = []
    for batch in tqdm(test_loader, desc="Testing"):
        with torch.no_grad():
            output = model(batch[1].to(device), do_classification=False)
            recon = output['eeg_reconstruction']
            x_hat_list.append(recon.cpu())
            y_task_list.append(batch[3].cpu())  # Assuming batch[3] contains the labels of the tasks
            
            true_subject_list.append(batch[2].cpu())  # Assuming batch[2] contains the true subject IDs
            synthetic_subject_list.append(batch[2].cpu())  # For self-reconstruction, synthetic subject is the same as true subject
            runs_list.append(batch[4].cpu())  # Assuming batch[4] contains the run information
    output_self_recon_test_data = {
        'data': torch.cat(x_hat_list, dim=0),
        'tasks': torch.cat(y_task_list, dim=0),
        'subjects': torch.cat(true_subject_list, dim=0),
        'synthetic_subjects': torch.cat(synthetic_subject_list, dim=0),
        'runs': torch.cat(runs_list, dim=0)
    }

    torch.save(output_self_recon_test_data, output_dir + '_self_recon_test.pt')
    print(f"Self-reconstructed test samples saved to {output_dir}")
    
    # Cross reconstruction of test samples
    x_hat_list = []
    y_task_list = []
    synthetic_subject_list = []
    true_subject_list = []
    runs_list = []
    for batch in tqdm(test_loader, desc="Testing"):
        test_samples = batch[1].to(device)
        # print range of data values to check normalization, mean and std
        # print(f"Test samples range: {test_samples.min().item()} to {test_samples.max().item()}")
        # print(f"Test samples mean: {test_samples.mean().item()}, std: {test_samples.std().item()}")
        current_batch_size = test_samples.size(0)
        # test_sample = test_loader[i]
        # Get the corresponding training sample (you can implement a strategy to select it, e.g., random or based on some criteria)
        random_batch = False  # Set to True to sample a random batch from the training set
        if random_batch:
            train_batch = train_loader.dataset.sample_batch(current_batch_size)
        else:
            # always retrieve same subject for all test samples, e.g., subject 0
            fixed_subject_original_id = train_loader.dataset.loader.unique_subjects[0]
            train_batch = train_loader.dataset.loader.get_batch_by_subject(fixed_subject_original_id, current_batch_size)
        #print(type(train_loader))           # DataLoader?
        #print(type(train_loader.dataset))   # DelegatedLoader?
        #print(dir(train_loader.dataset))    # what attributes does DelegatedLoader expose?
            
        train_samples_raw = train_batch[1].to(device)
        #print(f"data_mean used: {global_mean.item():.6f}")
        #print(f"data_std used: {global_std.item():.6f}")
        #print(f"loader.data_mean: {loader.data_mean}")
        #print(f"loader.data_std: {loader.data_std}")
        #train_samples = normalize(train_samples_raw, global_mean, global_std)
        # ADDED THIS LINE FOR TRAIN LOADER TEST 

        train_samples = train_samples_raw  # Assuming the model was trained on non-normalized data, we keep it as is for translation
    
        # Pass both samples through the model to get their latent representations
        with torch.no_grad():
            test_output = model(test_samples.to(device), do_classification=False)
            train_output = model(train_samples.to(device), do_classification=False)

        # Swap the latent representations (you can choose which parts to swap based on your model's architecture)
        train_res = train_output['encoder_body_residuals']
        train_bottleneck = train_output['var_features_dict']

        test_res = test_output['encoder_body_residuals']
        test_bottleneck = test_output['var_features_dict']

        swap_strategy = True  # Set to True to swap both task and subject, False to keep original representations
        if swap_strategy == True:
            test_task_train_subj_res = {
                'task': test_res['task'],  # Keep task from test sample
                'subject': train_res['subject'],  # Keep subject from train sample
                'noise': train_res['noise'],  # Keep noise from train sample
            }

            test_task_train_subj_bottleneck = {
                'task': test_bottleneck['task'],  # Keep task from test sample
                'subject': train_bottleneck['subject'],  # Keep subject from train sample
                'noise': train_bottleneck['noise'],  # Keep noise from test sample
            }
        else:
            # just take original to see whether self reconstruction works
            test_task_train_subj_res = test_res
            test_task_train_subj_bottleneck = test_bottleneck

        # Now you can pass the swapped representations through the generator to get the new sample
        generated = model.decode_dvae(test_task_train_subj_bottleneck, test_task_train_subj_res)

        x_hat_list.append(generated.cpu())
        y_task_list.append(batch[3].cpu())  # Assuming batch[3] contains the labels of the tasks
        true_subject_list.append(batch[2].cpu())  # Assuming batch[2] contains the true subject IDs
        synthetic_subject_list.append(train_batch[2].cpu())  # Assuming batch[2] contains the subject IDs
        # append zeros to runs list since we don't have run information for the generated samples
        runs_list.append(torch.zeros(current_batch_size, dtype=torch.long))  # Assuming run information
        
    # Save the generated samples and their corresponding labels
    output_data = {
        'data': torch.cat(x_hat_list, dim=0),
        'tasks': torch.cat(y_task_list, dim=0),
        'subjects': torch.cat(true_subject_list, dim=0),
        'synthetic_subjects': torch.cat(synthetic_subject_list, dim=0),
        'runs': torch.cat(runs_list, dim=0)
    }
    torch.save(output_data, output_dir + 'test_swapped.pt')
    print(f"Translated samples saved to {output_dir}")