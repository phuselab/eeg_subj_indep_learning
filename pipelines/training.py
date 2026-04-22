import torch 
import numpy as np
import wandb
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from collections import defaultdict
from typing import Dict, Optional, List
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models.disentanglement.core import DisentangledEEGModel
from models.losses import DisentanglementLoss
from models.classifiers.classifiers import ExternalClassifierHead, SimpleFeaturesClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
import openTSNE
from utils.helper import fit_clf_fn
from sklearn.model_selection import train_test_split



class DVAETrainer:
    """
    Main DVAE trainer - handles Modes A, B, C (Stage 2), D
    """
    def __init__(self, model: DisentangledEEGModel, loss_fn: DisentanglementLoss,
                 optimizer: torch.optim.Optimizer, device: str = 'cuda',
                 save_dir: str = 'experiments', phase_name: str = "BB_FT", logPerClass: bool = False,
                 save_every_epochs: int = 10, tsne_every_epochs: int = 200, xgb_classifier: bool = True, tsne: bool = True):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optimizer_D = None
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_history = []
        self.val_history = []
        self.test_history = []
        
        self.plot_dir = self.save_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True)

        self.phase_name = phase_name
        self.logPerClass = logPerClass

        self.bestModel = None
        self.save_every_epochs = save_every_epochs
        self.tsne_every_epochs = tsne_every_epochs
        self.xgb_classifier = xgb_classifier
        self.tsne = tsne

        print(f'\n\nUsing xgb_cls: {self.xgb_classifier} and tsne: {self.tsne} in DVAETrainer for phase {self.phase_name}\n\n')

        if self.loss_fn.config.adversarial:
            if hasattr(self.model, 'discriminator') and self.model.discriminator is not None:
                d_params = self.model.discriminator.parameters()
                base_lr = self.optimizer.param_groups[0]['lr']
                self.optimizer_D = torch.optim.AdamW(
                    d_params, 
                    lr=base_lr, 
                    weight_decay=self.optimizer.param_groups[0].get('weight_decay', 0.01)
                )
                logging.info(f"Adversarial Mode ON: Initialized optimizer_D with LR {base_lr}.")
            else:
                 logging.error("Adversarial loss is enabled, but model.discriminator is missing or None!")
                 self.loss_fn.config.adversarial = False
    
  

    def tmp_split_optimizer(self, train_backbone: bool = False):
        current_lr = self.optimizer.param_groups[0]['lr']

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'feature_extractor' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        if train_backbone:
            self.optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': current_lr},
                {'params': other_params, 'lr': current_lr * 5}
            ], weight_decay=5e-2)
        else:
            self.optimizer = torch.optim.AdamW([
                {'params': other_params, 'lr': current_lr}
            ], weight_decay=5e-2)
    

    def validate(self, dataloader: DataLoader, epoch: int = 0, plot_sub_dir: Optional[Path] = None) -> Dict[str, float]:
        self.model.eval()
        epoch_losses = defaultdict(list)
        
        all_predictions = {name: [] for name in self.model.encoder_names 
                        if self.model.config.encoders[name].num_classes is not None}
        all_var_predictions = {name: [] for name in self.model.encoder_names 
                        if self.model.config.encoders[name].num_classes is not None}
        all_labels = {name: [] for name in all_predictions.keys()}
        
        # xgboost evaluation flag - only if we have a binary classification task in the encoder (e.g., task with 2 classes)
        if self.model.config.encoders['task'].num_classes == 2:
            pass_classes_to_xgboost = True
            print("Binary classification detected in 'task' encoder. Will pass class labels to XGBoost for evaluation.")
        else: 
            pass_classes_to_xgboost = False
            print(f"{self.model.config.encoders['task'].num_classes} classes detected in 'task' encoder. Will NOT pass class labels to XGBoost.")

        all_z_subject = []
        all_z_task = []
        all_subject_labels = []
        all_task_labels = []
        all_z_noise = []

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                inputs = batch[1].to(self.device)
                labels = {
                    'subject': batch[2].to(self.device),
                    'task': batch[3].to(self.device)
                }

                outputs = self.model(inputs)
                outputs['inputs'] = inputs
                losses = self.loss_fn.compute_loss(outputs, labels, self.model)

                for k, v in losses.items():
                    if torch.is_tensor(v):
                        epoch_losses[k].append(v.item())
                
                if outputs['var_features_dict'] is not None:
                    if 'subject' in outputs['var_features_dict']:
                        z_subject = outputs['var_features_dict']['subject']['z']
                        all_z_subject.append(z_subject.cpu())
                        all_subject_labels.append(labels['subject'].cpu())

                    if 'task' in outputs['var_features_dict']:
                        all_z_task.append(outputs['var_features_dict']['task']['z'].cpu())
                        all_task_labels.append(labels['task'].cpu())
                    if 'noise' in outputs['var_features_dict']:
                        z_noise = outputs['var_features_dict']['noise']['z']
                        all_z_noise.append(z_noise.cpu())
                    
                for name in all_predictions.keys():
                    if name in outputs['logits_dict'] and outputs['logits_dict'][name] is not None:
                        preds = outputs['logits_dict'][name].argmax(dim=-1)
                        all_predictions[name].append(preds.cpu())
                        all_labels[name].append(labels[name].cpu())
                
                for name in all_var_predictions.keys():
                    if outputs['var_logits_dict'] and name in outputs['var_logits_dict'] and outputs['var_logits_dict'][name] is not None:
                        var_preds = outputs['var_logits_dict'][name].argmax(dim=-1)
                        all_var_predictions[name].append(var_preds.cpu())
                        # all_labels[name].append(labels[name].cpu())

        metrics = {k: np.mean(v) for k, v in epoch_losses.items()}
        
            
        for name in all_predictions.keys():
            if all_predictions[name]:
                preds = torch.cat(all_predictions[name])
                labs = torch.cat(all_labels[name])
                accuracy = (preds == labs).float().mean().item()
                metrics[f'accuracy_{name}'] = accuracy

        for key in all_predictions.keys():
            if all_predictions[key]:
                preds = torch.cat(all_predictions[key])
                labs = torch.cat(all_labels[key])
                if len(labs.unique()) > 1:  # only compute these metrics if we have more than one class
                    precision, recall, f1, _ = precision_recall_fscore_support(labs.numpy(), preds.numpy(), average='weighted', zero_division=0)
                    metrics[f'precision_{key}'] = precision
                    metrics[f'recall_{key}'] = recall
                    metrics[f'f1_{key}'] = f1
                # add also balanced accuracy for multi-class classification
                if len(labs.unique()) > 1:
                    balanced_acc = balanced_accuracy_score(labs.numpy(), preds.numpy())
                    metrics[f'balanced_accuracy_{key}'] = balanced_acc
            if all_var_predictions[key] and len(all_var_predictions[key]) > 0:
                var_preds = torch.cat(all_var_predictions[key])
                labs = torch.cat(all_labels[key])
                if len(labs.unique()) > 1:
                    precision, recall, f1, _ = precision_recall_fscore_support(labs.numpy(), var_preds.numpy(), average='weighted', zero_division=0)
                    metrics[f'var_precision_{key}'] = precision
                    metrics[f'var_recall_{key}'] = recall
                    metrics[f'var_f1_{key}'] = f1
                if len(labs.unique()) > 1:
                    balanced_acc = balanced_accuracy_score(labs.numpy(), var_preds.numpy())
                    metrics[f'var_balanced_accuracy_{key}'] = balanced_acc
                
                

        for name in all_var_predictions.keys():
            if all_var_predictions[name]:
                var_preds = torch.cat(all_var_predictions[name])
                labs = torch.cat(all_labels[name])
                var_accuracy = (var_preds == labs).float().mean().item()
                metrics[f'var_accuracy_{name}'] = var_accuracy
                    
        # if 'accuracy_subject' in metrics:
        #     metrics.pop('accuracy_subject')
        # check for nan values in all_z_subject and all_z_task before visualization

        if self.phase_name == 'DVAE' and self.xgb_classifier:
            if epoch % self.tsne_every_epochs == 0:
            # We want to use an xgboost classifier to evaluate the quality of the z spaces also for disentanglement
                if len(all_z_subject) > 0 and len(all_subject_labels) > 0:
                    z_subject = torch.cat(all_z_subject).cpu().numpy()
                    subject_labels = torch.cat(all_subject_labels).cpu().numpy()
                    z_task = torch.cat(all_z_task).cpu().numpy()
                    task_labels = torch.cat(all_task_labels).cpu().numpy()

                    z_subject = z_subject.mean(axis=2)
                    z_task = z_task.mean(axis=2)
                    
                    # I don't want to use all the data for the xgboost evaluation, otherwise it takes too long - I'll use a subset of 1000 samples randomly selected from the validation set
                    # I want to make sure that the subset is balanced across classes, so I'll use stratified sampling based on the subject labels for the subject space evaluation and on the task labels for the task space evaluation

                    # subject space analysis
                    z_subject_train, z_subject_test, y_subject_train, y_subject_test = train_test_split(z_subject, subject_labels, test_size=0.2, random_state=42)
                    clf_subject_on_subject = fit_clf_fn(z_subject_train, y_subject_train)
                    subj_on_subj_score = clf_subject_on_subject.score(z_subject_test, y_subject_test)

                    
                    z_subject_train, z_subject_test, y_task_train, y_task_test = train_test_split(z_subject, task_labels, test_size=0.2, random_state=42)
                    if pass_classes_to_xgboost:
                        clf_task_on_subject = fit_clf_fn(z_subject_train, y_task_train, num_classes=self.model.config.encoders['task'].num_classes)
                    else:
                        clf_task_on_subject = fit_clf_fn(z_subject_train, y_task_train)
                    task_on_subj_score = clf_task_on_subject.score(z_subject_test, y_task_test)
                

                    # task space analysis
                    z_task_train, z_task_test, y_task_train, y_task_test = train_test_split(z_task, task_labels, test_size=0.2, random_state=42)
                    if pass_classes_to_xgboost:
                        clf_task_on_task = fit_clf_fn(z_task_train, y_task_train, num_classes=self.model.config.encoders['task'].num_classes)
                    else:
                        clf_task_on_task = fit_clf_fn(z_task_train, y_task_train)
                    task_on_task_score = clf_task_on_task.score(z_task_test, y_task_test)

                    z_task_train, z_task_test, y_subject_train, y_subject_test = train_test_split(z_task, subject_labels, test_size=0.2, random_state=42)
                    if pass_classes_to_xgboost:
                        clf_subj_on_task = fit_clf_fn(z_task_train, y_subject_train, num_classes=self.model.config.encoders['subject'].num_classes)
                    else:
                        clf_subj_on_task = fit_clf_fn(z_task_train, y_subject_train)
                    subj_on_task_score = clf_subj_on_task.score(z_task_test, y_subject_test)

                    # add to metrics
                    metrics['subject_on_subject_score'] = subj_on_subj_score
                    metrics['task_on_subject_score'] = task_on_subj_score
                    metrics['task_on_task_score'] = task_on_task_score
                    metrics['subject_on_task_score'] = subj_on_task_score


        if self.phase_name == 'DVAE' and epoch % self.tsne_every_epochs == 0 and not (any(torch.isnan(z).any() for z in all_z_subject) or any(torch.isnan(z).any() for z in all_z_task)):
            if self.tsne:
                if len(all_z_subject) > 0:
                    self._visualize_latent_spaces(
                        epoch, 
                        all_z_subject, all_subject_labels,
                        all_z_task, all_task_labels,
                        all_z_noise, 
                        plot_dir= plot_sub_dir if plot_sub_dir is not None else self.plot_dir
                    )
            
   
        
            # create confusion matrix and save it
            cm = confusion_matrix(labs.numpy(), var_preds.numpy())
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labs.unique().numpy())
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix for {name} - Epoch {epoch}')
            plt.savefig(f'{self.plot_dir}/var_confusion_matrix_{name}_epoch_{epoch}.png')
            plt.close()
        return metrics
    
    
    def train_epoch(self, dataloader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        self.model.train()
        self.loss_fn.set_epoch(epoch) 
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        debug_first_batch = (epoch == 0)
        # initialize rD infinity
        rD = np.inf
        rG = np.inf
        nu = 5
        split_losses = {'G': [], 'D': []}
        G_count = 0
        D_count = 0

        all_z_subject_ref = []
        all_z_task_ref = []
        all_subject_labels_ref = []
        all_task_labels_ref = []
        all_z_noise_ref = []

        all_z_subject_cmp = []
        all_z_task_cmp = []
        all_z_noise_cmp = []

        all_z_subject_cross = []
        all_z_task_cross = []
        all_subject_labels_cross = []
        all_task_labels_cross = []
        all_z_noise_cross = []

        all_predictions = {name: [] for name in self.model.encoder_names 
                        if self.model.config.encoders[name].num_classes is not None}
        all_var_predictions = {name: [] for name in self.model.encoder_names
                        if self.model.config.encoders[name].num_classes is not None}
        all_labels = {name: [] for name in all_predictions.keys()}
        # la domanda rimane sul dataloader
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if batch[1].shape[0] == 0:
                continue  # skip empty batches
            num_batches += 1
            inputs = batch[1].to(self.device)
            labels = {
                'subject': batch[2].to(self.device),
                'task': batch[3].to(self.device)
            }
            
            outputs = self.model(inputs)
            if i == 0:
                logging.info(f"Batch {num_batches}: Input std: {outputs['backbone_features'].std().item():.4f}, mean: {outputs['backbone_features'].mean().item():.4f}")
                if outputs['eeg_reconstruction'] is not None:
                    logging.info(f"Batch {num_batches}: Reconstruction std: {outputs['eeg_reconstruction'].std().item():.4f}, mean: {outputs['eeg_reconstruction'].mean().item():.4f}")

            # calculate accuracy 
            for key in outputs['logits_dict'].keys():
                if outputs['logits_dict'][key] is not None:
                    preds = outputs['logits_dict'][key].argmax(dim=-1)
                    all_predictions[key].append(preds.cpu())
                    all_labels[key].append(labels[key].cpu())
            
            if outputs['var_logits_dict'] is not None:
                for key in outputs['var_logits_dict'].keys():
                    if outputs['var_logits_dict'][key] is not None:
                        var_preds = outputs['var_logits_dict'][key].argmax(dim=-1)
                        all_var_predictions[key].append(var_preds.cpu())
                        # all_labels[key].append(labels[key].cpu())
            
            # add initial inputs for cross-subject losses to outputs
            outputs['inputs'] = inputs

            if self.phase_name == 'DVAE' and epoch % self.tsne_every_epochs == 0:
                b, c, n_patches, patch_size = outputs['backbone_features'].shape
                # apply the model to the reconstructed signals to get their latent representations for visualization
                reconstruction = outputs['eeg_reconstruction']
                # reconstruction is the reconstruction of the features vector extracted by CBraMod, not the EEG signal
                recon_outputs = self.model(reconstruction, extract_features=True)
                if 'subject' in recon_outputs['var_features_dict'] and 'subject' in outputs['var_features_dict']:
                    all_z_subject_cmp.append(recon_outputs['var_features_dict']['subject']['z'].detach().cpu())
                    all_z_subject_ref.append(outputs['var_features_dict']['subject']['z'].detach().cpu())
                    all_subject_labels_ref.append(labels['subject'].detach().cpu())
                    
                all_z_task_cmp.append(recon_outputs['var_features_dict']['task']['z'].detach().cpu())
                all_z_task_ref.append(outputs['var_features_dict']['task']['z'].detach().cpu())
                all_task_labels_ref.append(labels['task'].detach().cpu())
                
                if 'noise' in outputs['var_features_dict']:
                    all_z_noise_cmp.append(recon_outputs['var_features_dict']['noise']['z'].detach().cpu()  )
                    all_z_noise_ref.append(outputs['var_features_dict']['noise']['z'].detach().cpu())

                
            if self.loss_fn.config.cross_subject_intra_class or self.loss_fn.config.cross_subject_cross_class or self.loss_fn.config.knowledge_distillation:
                if self.loss_fn.config.cross_subject_intra_class:
                    # sample till the batch size/2 is reached
                    n_tasks = len(dataloader.dataset.loader.unique_tasks)
                    batch_size = max(inputs.shape[0], 2)
                    sampling_repeats = max(1, (batch_size//2) // n_tasks)
                    samplesA_batch = torch.zeros((batch_size//2, inputs.shape[1], inputs.shape[2]))
                    samplesB_batch = torch.zeros((batch_size//2, inputs.shape[1], inputs.shape[2]))
                    for i in range(sampling_repeats - 1):
                        start_idx = i * n_tasks
                        end_idx = start_idx + n_tasks
                        samplesA_batch[start_idx:end_idx] = dataloader.dataset.sample_by_property('task')[1] # n_tasks, 64, 800
                        samplesB_batch[start_idx:end_idx] = dataloader.dataset.sample_by_property('task')[1] # n_tasks, 64, 800
                    
                    samplesA = samplesA_batch.to(self.device)[:batch_size//2]
                    samplesB = samplesB_batch.to(self.device)[:batch_size//2]


                    outputA = self.model(samplesA.to(self.device))
                    outputB = self.model(samplesB.to(self.device))

                    encoded_A = outputA['var_features_dict'] 
                    residual_A = outputA['encoder_body_residuals']

                    encoded_B = outputB['var_features_dict']
                    residual_B = outputB['encoder_body_residuals']

                    encoded_A_swapped = {}
                    residual_A_swapped = {}
                    encoded_B_swapped = {}
                    residual_B_swapped = {}
                    for name in self.model.encoder_names:
                        if name == 'task':
                            encoded_A_swapped[name] = encoded_B[name]['z']
                            residual_A_swapped[name] = residual_B[name]

                            encoded_B_swapped[name] = encoded_A[name]['z']
                            residual_B_swapped[name] = residual_A[name]
                        else:
                            encoded_A_swapped[name] = encoded_A[name]['z']
                            residual_A_swapped[name] = residual_A[name]

                            encoded_B_swapped[name] = encoded_B[name]['z']
                            residual_B_swapped[name] = residual_B[name]

                    reconstruction_A = self.model.generator(encoded_A_swapped, encoder_body_residuals=residual_A_swapped)
                    reconstruction_B = self.model.generator(encoded_B_swapped, encoder_body_residuals=residual_B_swapped)

                    outputs['cross_intra_reconstruction_A'] = reconstruction_A
                    outputs['cross_intra_reconstruction_B'] = reconstruction_B
                    outputs['cross_intra_target_A'] = samplesA
                    outputs['cross_intra_target_B'] = samplesB
                    
                if self.loss_fn.config.cross_subject_cross_class:
                    n_subjects = len(dataloader.dataset.loader.unique_subjects)
                    batch_size = max(inputs.shape[0], 3)
                    sampling_repeats = max(1, (batch_size//3) // n_subjects)
                    samplesA_batch = torch.zeros((batch_size, inputs.shape[1], inputs.shape[2]))
                    samplesB_batch = torch.zeros((batch_size, inputs.shape[1], inputs.shape[2]))
                    samplesC_batch = torch.zeros((batch_size, inputs.shape[1], inputs.shape[2]))
                    for i in range(sampling_repeats - 1):
                        start_idx = i * n_subjects
                        end_idx = min(start_idx + n_subjects, batch_size)
                        shifts = np.random.choice(range(n_subjects), size=3, replace=False)
                        samplesA_batch[start_idx:end_idx] = dataloader.dataset.sample_by_property('subject', shift=shifts[0])[1] # n_subjects, 64, 800
                        samplesB_batch[start_idx:end_idx] = dataloader.dataset.sample_by_property('subject', shift=shifts[1])[1] # n_subjects, 64, 800
                        samplesC_batch[start_idx:end_idx] = dataloader.dataset.sample_by_property('subject', shift=shifts[2])[1] # n_subjects, 64, 800


                    samplesA = samplesA_batch.to(self.device)[:batch_size//3]
                    samplesB = samplesB_batch.to(self.device)[:batch_size//3]
                    samplesC = samplesC_batch.to(self.device)[:batch_size//3]

                    outputA = self.model(samplesA.to(self.device), do_reconstruction=False, do_classification=False)
                    outputB = self.model(samplesB.to(self.device), do_reconstruction=False, do_classification=False)
                    outputC = self.model(samplesC.to(self.device), do_reconstruction=False, do_classification=False)

                    encoded_A, residual_A = outputA['var_features_dict'], outputA['encoder_body_residuals']
                    encoded_B, residual_B = outputB['var_features_dict'], outputB['encoder_body_residuals']
                    encoded_C, residual_C = outputC['var_features_dict'], outputC['encoder_body_residuals']

                    encoded_ABC = encoded_A.copy()
                    residual_ABC = residual_A.copy()
                    
                    encoded_BCA = encoded_B.copy()
                    residual_BCA = residual_B.copy()
                    
                    encoded_CAB = encoded_C.copy()
                    residual_CAB = residual_C.copy()
                    

                    # Do all the necessary permutations for cross cross losses and loss (14) CC_cycle_C
                    encoded_ABC['subject'] = encoded_A['subject']['z']
                    encoded_ABC['task'] = encoded_B['task']['z']
                    encoded_ABC['noise'] = encoded_C['noise']['z']  # optional, depending on model config

                    residual_ABC['subject'] = residual_A['subject']
                    residual_ABC['task'] = residual_B['task']
                    residual_ABC['noise'] = residual_C['noise']
                    
                    encoded_BCA['subject'] = encoded_B['subject']['z']
                    encoded_BCA['task'] = encoded_C['task']['z']
                    encoded_BCA['noise'] = encoded_A['noise']['z']  # optional, depending on model config
                    
                    residual_BCA['subject'] = residual_B['subject']
                    residual_BCA['task'] = residual_C['task']
                    residual_BCA['noise'] = residual_A['noise']

                    encoded_CAB['subject'] = encoded_C['subject']['z']
                    encoded_CAB['task'] = encoded_A['task']['z']
                    encoded_CAB['noise'] = encoded_B['noise']['z']  # optional, depending on model config



                    # ABC 
                    # apply generator to mixed latent codes
                    recon_ABC = self.model.generator(encoded_ABC, encoder_body_residuals=residual_ABC)
                    # apply model, the input are already CBraMod features, so we set extract_features=False to avoid re-extracting features and just get the latent representations and logits
                    outputs_ABC = self.model(recon_ABC, extract_features=True, do_reconstruction=False)
                    
                    z_subjects_ABC = outputs_ABC['var_features_dict']['subject']['z']
                    z_tasks_ABC = outputs_ABC['var_features_dict']['task']['z']
                    z_noise_ABC = outputs_ABC['var_features_dict']['noise']['z']
                    
                    # BCA
                    recon_BCA = self.model.generator(encoded_BCA, encoder_body_residuals=residual_BCA)
                    outputs_BCA = self.model(recon_BCA, extract_features=True, do_reconstruction=False, do_classification=False)
                    
                    z_subjects_BCA = outputs_BCA['var_features_dict']['subject']['z']
                    z_tasks_BCA = outputs_BCA['var_features_dict']['task']['z']
                    z_noise_BCA = outputs_BCA['var_features_dict']['noise']['z']
                    

                    # CAB
                    recon_CAB = self.model.generator(encoded_CAB, encoder_body_residuals=residual_CAB)
                    outputs_CAB = self.model(recon_CAB, extract_features=True, do_reconstruction=False, do_classification=False)
                    
                    z_subjects_CAB = outputs_CAB['var_features_dict']['subject']['z']
                    z_tasks_CAB = outputs_CAB['var_features_dict']['task']['z']
                    z_noise_CAB = outputs_CAB['var_features_dict']['noise']['z']

                    
                    # ABC
                    outputs['cross_cross_z_subjects_ABC'] = z_subjects_ABC
                    outputs['cross_cross_z_tasks_ABC'] = z_tasks_ABC
                    outputs['cross_cross_z_noise_ABC'] = z_noise_ABC
                    outputs['cross_cross_z_subjects_target_ABC'] = encoded_A['subject']['z']
                    outputs['cross_cross_z_tasks_target_ABC'] = encoded_B['task']['z']
                    outputs['cross_cross_z_noise_target_ABC'] = encoded_C['noise']['z']

                    # we only need this once (for ABC), not for the other variants 
                    outputs['cross_cross_var_logits'] = outputs_ABC['var_logits_dict']
                    outputs['cross_cross_logits'] = outputs_ABC['logits_dict']
                    
   

                    # LA GRANDE PERMUTAZIONE 
                    AAA = encoded_ABC.copy()
                    AAA['subject'] = z_subjects_ABC
                    AAA['task'] = z_tasks_BCA
                    AAA['noise'] = z_noise_CAB
                    AAA_residual = residual_ABC.copy()
                    AAA_residual['subject'] = residual_ABC['subject']
                    AAA_residual['task'] = residual_BCA['task']
                    AAA_residual['noise'] = residual_CAB['noise']

                    A_hat = self.model.generator(AAA, encoder_body_residuals=AAA_residual)

                    BBB = encoded_BCA.copy()
                    BBB['subject'] = z_subjects_BCA
                    BBB['task'] = z_tasks_CAB
                    BBB['noise'] = z_noise_ABC
                    BBB_residual = residual_BCA.copy()
                    BBB_residual['subject'] = residual_BCA['subject']
                    BBB_residual['task'] = residual_CAB['task']
                    BBB_residual['noise'] = residual_ABC['noise']

                    B_hat = self.model.generator(BBB, encoder_body_residuals=BBB_residual)

                    CCC = encoded_CAB.copy()
                    CCC['subject'] = z_subjects_CAB
                    CCC['task'] = z_tasks_ABC
                    CCC['noise'] = z_noise_BCA
                    CCC_residual = residual_CAB.copy()
                    CCC_residual['subject'] = residual_CAB['subject']
                    CCC_residual['task'] = residual_ABC['task']
                    CCC_residual['noise'] = residual_BCA['noise']

                    C_hat = self.model.generator(CCC, encoder_body_residuals=CCC_residual)

                    outputs['cross_cross_cycle_rec_A'] = A_hat
                    outputs['cross_cross_cycle_rec_B'] = B_hat
                    outputs['cross_cross_cycle_rec_C'] = C_hat
                    outputs['cross_cross_cycle_target_A'] = samplesA
                    outputs['cross_cross_cycle_target_B'] = samplesB
                    outputs['cross_cross_cycle_target_C'] = samplesC


                    if self.loss_fn.config.adversarial:
                        outputs['cross_cross_adv_fake'] = recon_ABC
                        outputs['cross_cross_adv_real'] = samplesA

                    # add to list for tSNE visualization
                    all_z_subject_cross.append(z_subjects_ABC.detach().cpu())
                    all_subject_labels_cross.append(labels['subject'].detach().cpu())
                    all_z_task_cross.append(z_tasks_ABC.detach().cpu())
                    all_task_labels_cross.append(labels['task'].detach().cpu())
                    if 'noise' in encoded_ABC:
                        all_z_noise_cross.append(encoded_ABC['noise'].detach().cpu())
                    
                
            
            if self.loss_fn.config.adversarial:
                losses_G = self.loss_fn.compute_loss(outputs, labels, self.model, adversarial_step='G')
                split_losses['G'].append(losses_G['total'].item())
                losses_D = self.loss_fn.compute_loss(outputs, labels, self.model, adversarial_step='D')
                split_losses['D'].append(losses_D['total'].item())

                whole_losses = {**losses_G, **losses_D}

                
                if len(split_losses['D']) >= 2:
                    rD = np.abs((split_losses['D'][-1] - split_losses['D'][-2]) / (split_losses['D'][-2] + 1e-8))
                    rG = np.abs((split_losses['G'][-1] - split_losses['G'][-2]) / (split_losses['G'][-2] + 1e-8))

                self.optimizer.zero_grad(set_to_none=True)

                if rD > nu * rG:
                    D_count += 1
                else:
                    G_count += 1
                logging.info(f"Adversarial Step - rD: {rD:.4f}, rG: {rG:.4f}, nu: {nu}, D_count: {D_count}, G_count: {G_count}")

                loss = losses_D['total'] if rD > nu * rG else losses_G['total']
                epoch_losses['total'] += split_losses['G'][-1]
                epoch_losses['total'] += split_losses['D'][-1]       
            else:
                losses = self.loss_fn.compute_loss(outputs, labels, self.model, adversarial_step='G')
                loss = losses['total']
                epoch_losses['total'] += loss.item()
                whole_losses = losses
        
            loss.backward()
        
            # if debug_first_batch and num_batches == 1:
            #     self.check_gradient_flow(after_backward=True)
        
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
        
            

            for k, v in whole_losses.items():
                if k != 'total' and torch.is_tensor(v):
                    epoch_losses[k] += v.item()
            
                # if self.loss_fn.config.adversarial:
                #     epoch_losses['adversarial_discriminator'] += loss_D.item()
                #     epoch_losses['gradient_penalty'] += gradient_penalty.item()
        
                torch.cuda.empty_cache()
        
        logging.info(f"Epoch {epoch} Completed.")
            
        if self.phase_name == 'DVAE' and epoch % self.tsne_every_epochs == 0 and self.tsne:
            if len(all_z_subject_ref) > 0:
                self._visualize_train_latent_spaces(
                        epoch, all_z_subject_ref, all_z_subject_cmp, all_subject_labels_ref, all_z_task_ref, 
                        all_z_task_cmp, all_task_labels_ref, all_z_noise_ref, all_z_noise_cmp, 'train'
                    )
                #self._visualize_signal_vs_reconstruction_spectogram_of_example_sample(
                #    epoch, samplesA = samplesA, reconA = A_hat , plot_dir = self.plot_dir,
                #    fft_size = 1024, shift_size = 120, win_length = 600, window="hann_window"
                #)

        # compute accuracy for the epoch
        for key in all_predictions.keys():
            if all_predictions[key]:
                preds = torch.cat(all_predictions[key])
                labs = torch.cat(all_labels[key])
                accuracy = (preds == labs).float().mean().item()
                epoch_losses[f'accuracy_{key}'] = accuracy * num_batches
            if all_var_predictions[key] and len(all_var_predictions[key]) > 0:
                var_preds = torch.cat(all_var_predictions[key])
                labs = torch.cat(all_labels[key])
                var_accuracy = (var_preds == labs).float().mean().item()
                epoch_losses[f'var_accuracy_{key}'] = var_accuracy * num_batches
            # compute all the standard metrics for classification tasks
            for key in all_predictions.keys():
                if all_predictions[key]:
                    preds = torch.cat(all_predictions[key])
                    labs = torch.cat(all_labels[key])
                    acc = (preds == labs).float().mean().item()
                    epoch_losses[f'accuracy_{key}'] = acc * num_batches
                    if len(labs.unique()) > 1:  # only compute these metrics if we have more than one class
                        precision, recall, f1, _ = precision_recall_fscore_support(labs.numpy(), preds.numpy(), average='weighted', zero_division=0)
                        epoch_losses[f'precision_{key}'] = precision * num_batches
                        epoch_losses[f'recall_{key}'] = recall * num_batches
                        epoch_losses[f'f1_{key}'] = f1 * num_batches
                    # add also balanced accuracy for multi-class classification
                    if len(labs.unique()) > 1:
                        balanced_acc = balanced_accuracy_score(labs.numpy(), preds.numpy())
                        epoch_losses[f'balanced_accuracy_{key}'] = balanced_acc * num_batches
                if all_var_predictions[key] and len(all_var_predictions[key]) > 0:
                    var_preds = torch.cat(all_var_predictions[key])
                    labs = torch.cat(all_labels[key])
                    var_acc = (var_preds == labs).float().mean().item()
                    epoch_losses[f'var_accuracy_{key}'] = var_acc * num_batches
                    if len(labs.unique()) > 1:
                        precision, recall, f1, _ = precision_recall_fscore_support(labs.numpy(), var_preds.numpy(), average='weighted', zero_division=0)
                        epoch_losses[f'var_precision_{key}'] = precision * num_batches
                        epoch_losses[f'var_recall_{key}'] = recall * num_batches
                        epoch_losses[f'var_f1_{key}'] = f1 * num_batches
                    if len(labs.unique()) > 1:
                        balanced_acc = balanced_accuracy_score(labs.numpy(), var_preds.numpy())
                        epoch_losses[f'var_balanced_accuracy_{key}'] = balanced_acc * num_batches

        return {k: v / num_batches for k, v in epoch_losses.items()}
    
 
    
    def train(self, train_loader, val_loader, test_loader, num_epochs: int, 
              freeze_at_epoch: Optional[int] = None, wandb_run=None):
        best_val_loss = float('inf')
        
        logging.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_history.append(train_metrics)
            
            
            val_metrics = self.validate(val_loader, epoch, plot_sub_dir='val')
            test_metrics = self.validate(test_loader, epoch, plot_sub_dir='test')

            self.val_history.append(val_metrics)
            self.test_history.append(test_metrics)

            best_metrics = None
            best_epoch = 0
            
            if wandb_run is not None:
                log_data = {
                    f'epoch': epoch,
                    **{f'{self.phase_name}/train/{k}': v for k, v in train_metrics.items()},
                    **{f'{self.phase_name}/val/{k}': v for k, v in val_metrics.items()},
                    **{f'{self.phase_name}/test/{k}': v for k, v in test_metrics.items()},
                }
                wandb.log(log_data)
            
            if 'total' in train_metrics:
                logging.info(f"Train Loss: {train_metrics['total']:.4f}")
            if 'total' in val_metrics:
                logging.info(f"Val Loss: {val_metrics['total']:.4f}")
            if 'total' in test_metrics:
                logging.info(f"Test Loss: {test_metrics['total']:.4f}")

            for key in train_metrics:
                if 'accuracy' in key:
                    logging.info(f"Train {key}: {train_metrics[key]:.4f}")
            for key in val_metrics:
                if 'accuracy' in key:
                    logging.info(f"Val {key}: {val_metrics[key]:.4f}")
            for key in test_metrics:
                if 'accuracy' in key:
                    logging.info(f"Test {key}: {test_metrics[key]:.4f}")

            # if val_metrics['total'] < best_val_loss:
            #     best_val_loss = val_metrics['total']
            #     self.best_model = self.model.state_dict()
            #     best_epoch = epoch
            #     best_metrics = val_metrics
                
            

            # if (epoch + 1) % self.save_every_epochs == 0 or epoch == num_epochs - 1:
            #     if best_metrics is not None:
            #         self.save_checkpoint('best_model.pt', best_epoch, best_metrics)

            #     self.save_checkpoint('last_model.pt', epoch, val_metrics)
            #     # self.save_model_weights('last_model_weights.pt')
            #     print(f"Modello epoch {epoch} salvato")
                



    def _visualize_latent_spaces(
    self,
    epoch,
    z_subj_list,
    y_subj_list,
    z_task_list,
    y_task_list,
    z_noise_list=None,
    plot_dir: Optional[Path] = None,
):
        if not z_subj_list or not z_task_list:
            return

        # ---------------------------------------------------------
        # Convert to numpy
        # ---------------------------------------------------------
        z_subj = torch.cat(z_subj_list).cpu().numpy()
        z_subj = z_subj.mean(axis=2)  # average over patch
        shape = z_subj.shape
        z_subj = z_subj.reshape(shape[0], -1)  # flatten if needed
        y_subj = torch.cat(y_subj_list).cpu().numpy()

        z_task = torch.cat(z_task_list).cpu().numpy()
        z_task = z_task.mean(axis=2)  # average over patch and time dimensions if needed
        z_task = z_task.reshape(shape[0], -1)  # flatten if needed

        y_task = torch.cat(y_task_list).cpu().numpy()

        if z_noise_list is not None and len(z_noise_list) > 0:
            z_noise = torch.cat(z_noise_list).cpu().numpy()
            z_noise = z_noise.mean(axis=2)  # average over patch and time dimensions if needed
            z_noise = z_noise.reshape(shape[0], -1)  # flatten if needed
            y_noise_subj = y_subj[:len(z_noise)]
            y_noise_task = y_task[:len(z_noise)]

        # ---------------------------------------------------------
        # Setup
        # ---------------------------------------------------------
        num_subjects = len(np.unique(y_subj))
        unique_tasks = np.unique(y_task)

        # Subject colormap: MANY COLORS, NO COLLISIONS
        cmap_subj = plt.cm.get_cmap('hsv', num_subjects)
        norm_subj = plt.cm.colors.BoundaryNorm(
            boundaries=np.arange(num_subjects + 1) - 0.5,
            ncolors=num_subjects
        )

        # Task colormap: few, strong colors
        cmap_task = plt.cm.get_cmap('tab10', len(unique_tasks))

        nrows = 3 if z_noise_list is not None else 2
        fig, axes = plt.subplots(nrows, 2, figsize=(16, 6 * nrows))

        tsne = TSNE(n_components=2, random_state=42)

        # ---------------------------------------------------------
        # SUBJECT LATENT (colored by SUBJECT)
        # ---------------------------------------------------------
        z_subj_2d = tsne.fit_transform(z_subj)
        sc_ss = axes[0, 0].scatter(
            z_subj_2d[:, 0], z_subj_2d[:, 1],
            c=y_subj, cmap=cmap_subj, norm=norm_subj,
            s=10, alpha=0.6
        )
        axes[0, 0].set_title(f"Subject Latent (by Subject) - Ep {epoch}")
        cbar_ss = plt.colorbar(sc_ss, ax=axes[0, 0])
        cbar_ss.set_label("Subject ID")

        # ---------------------------------------------------------
        # SUBJECT LATENT (colored by TASK)  ← residui task
        # ---------------------------------------------------------
        for i, task_id in enumerate(unique_tasks):
            idx = y_task == task_id
            axes[0, 1].scatter(
                z_subj_2d[idx, 0], z_subj_2d[idx, 1],
                color=cmap_task(i),
                label=f"Task {int(task_id)}",
                s=12, alpha=0.6
            )
        axes[0, 1].set_title("Subject Latent (by Task)")
        axes[0, 1].legend(title="Tasks")
        axes[0, 1].grid(alpha=0.3)

        # ---------------------------------------------------------
        # TASK LATENT (colored by TASK)
        # ---------------------------------------------------------
        z_task_2d = tsne.fit_transform(z_task)
        for i, task_id in enumerate(unique_tasks):
            idx = y_task == task_id
            axes[1, 0].scatter(
                z_task_2d[idx, 0], z_task_2d[idx, 1],
                color=cmap_task(i),
                label=f"Task {int(task_id)}",
                s=12, alpha=0.6
            )
        axes[1, 0].set_title("Task Latent (by Task)")
        axes[1, 0].legend(title="Tasks")
        axes[1, 0].grid(alpha=0.3)

        # ---------------------------------------------------------
        # TASK LATENT (colored by SUBJECT) ← residui subject
        # ---------------------------------------------------------
        sc_ts = axes[1, 1].scatter(
            z_task_2d[:, 0], z_task_2d[:, 1],
            c=y_subj, cmap=cmap_subj, norm=norm_subj,
            s=10, alpha=0.6
        )
        axes[1, 1].set_title("Task Latent (by Subject)")
        cbar_ts = plt.colorbar(sc_ts, ax=axes[1, 1])
        cbar_ts.set_label("Subject ID")

        # ---------------------------------------------------------
        # NOISE
        # ---------------------------------------------------------
        if z_noise_list is not None and len(z_noise_list) > 0:
            z_noise_2d = tsne.fit_transform(z_noise)

            # Noise by Subject
            sc_ns = axes[2, 0].scatter(
                z_noise_2d[:, 0], z_noise_2d[:, 1],
                c=y_noise_subj, cmap=cmap_subj, norm=norm_subj,
                s=10, alpha=0.6
            )
            axes[2, 0].set_title("Noise Latent (by Subject)")
            cbar_ns = plt.colorbar(sc_ns, ax=axes[2, 0])
            cbar_ns.set_label("Subject ID")

            # Noise by Task
            for i, task_id in enumerate(unique_tasks):
                idx = y_noise_task == task_id
                axes[2, 1].scatter(
                    z_noise_2d[idx, 0], z_noise_2d[idx, 1],
                    color=cmap_task(i),
                    label=f"Task {int(task_id)}",
                    s=12, alpha=0.6
                )
            axes[2, 1].set_title("Noise Latent (by Task)")
            axes[2, 1].legend(title="Tasks")
            axes[2, 1].grid(alpha=0.3)

        plt.tight_layout()

        # ---------------------------------------------------------
        # Save
        # ---------------------------------------------------------
        if plot_dir is None:
            plot_dir = self.save_dir / "plots"
        else:
            plot_dir = self.save_dir / "plots" / plot_dir
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_dir / f"latent_epoch_{epoch}.png")
        plt.close()

    def _visualize_train_latent_spaces(
        self,
        epoch,
        z_subj_ref_list,
        z_subj_cmp_list,
        y_subj_list,
        z_task_ref_list,
        z_task_cmp_list,
        y_task_list,
        z_noise_ref_list=None,
        z_noise_cmp_list=None,
        plot_dir: Optional[Path] = None,
    ):
        if not z_subj_ref_list or not z_task_ref_list:
            return

        # ---------------------------------------------------------
        # Convert to numpy
        # ---------------------------------------------------------
        z_subj_ref = torch.cat(z_subj_ref_list).cpu().numpy()
        print("BEFORE SHAPE =  ", z_subj_ref.shape)
        z_subj_ref = z_subj_ref.mean(axis=2)  # average over patch and time dimensions if needed

        shape = z_subj_ref.shape
        print("SHAPE =  ", shape)
     

        # RESHAPE 
        z_subj_ref = z_subj_ref.reshape(shape[0], -1)
        z_subj_cmp = torch.cat(z_subj_cmp_list).cpu().numpy()
        z_subj_cmp = z_subj_cmp.mean(axis=2)  # average over patch and time dimensions if needed
        z_subj_cmp = z_subj_cmp.reshape(shape[0], -1)
        y_subj = torch.cat(y_subj_list).cpu().numpy()

        z_task_ref = torch.cat(z_task_ref_list).cpu().numpy()
        z_task_ref = z_task_ref.mean(axis=2)  # average over patch and time dimensions if needed
        z_task_ref = z_task_ref.reshape(shape[0], -1)
        z_task_cmp = torch.cat(z_task_cmp_list).cpu().numpy()
        z_task_cmp = z_task_cmp.mean(axis=2)  # average over patch and time dimensions if needed
        z_task_cmp = z_task_cmp.reshape(shape[0], -1)
        y_task = torch.cat(y_task_list).cpu().numpy()

        if z_noise_ref_list is not None and z_noise_cmp_list is not None and len(z_noise_ref_list) > 0 and len(z_noise_cmp_list) > 0:
            z_noise_ref = torch.cat(z_noise_ref_list).cpu().numpy()
            z_noise_ref = z_noise_ref.mean(axis=2)  # average over patch and time dimensions if needed
            z_noise_ref = z_noise_ref.reshape(shape[0], -1)
            z_noise_cmp = torch.cat(z_noise_cmp_list).cpu().numpy()
            z_noise_cmp = z_noise_cmp.mean(axis=2)  # average over patch and time dimensions if needed
            z_noise_cmp = z_noise_cmp.reshape(shape[0], -1)
            y_noise_subj = y_subj[: len(z_noise_ref)]
            y_noise_task = y_task[: len(z_noise_ref)]

        max_samples = 1000
        if z_subj_ref.shape[0] > max_samples:
            idx = np.random.choice(z_subj_ref.shape[0], max_samples, replace=False)
            z_subj_ref = z_subj_ref[idx]
            z_subj_cmp = z_subj_cmp[idx]
            y_subj = y_subj[idx]
            z_task_ref = z_task_ref[idx]
            z_task_cmp = z_task_cmp[idx]
            y_task = y_task[idx]
            if z_noise_ref_list is not None and z_noise_cmp_list is not None and len(z_noise_ref_list) > 0 and len(z_noise_cmp_list) > 0:
                z_noise_ref = z_noise_ref[idx]
                z_noise_cmp = z_noise_cmp[idx]
                y_noise_subj = y_noise_subj[idx]
                y_noise_task = y_noise_task[idx]

        # ---------------------------------------------------------
        # Setup colormaps
        # ---------------------------------------------------------
        num_subjects = len(np.unique(y_subj))
        unique_tasks = np.unique(y_task)

        cmap_subj = plt.cm.get_cmap("hsv", num_subjects)
        norm_subj = plt.cm.colors.BoundaryNorm(
            boundaries=np.arange(num_subjects + 1) - 0.5,
            ncolors=num_subjects
        )

        cmap_task = plt.cm.get_cmap("tab10", len(unique_tasks))

        # ---------------------------------------------------------
        # Figure
        # ---------------------------------------------------------
        nrows = 3 if z_noise_ref_list is not None else 2
        fig, axes = plt.subplots(nrows, 2, figsize=(18, 6 * nrows))

        # ---------------------------------------------------------
        # TSNE SUBJECT (shared space)
        # ---------------------------------------------------------
        tsne_subj = openTSNE.TSNE(
            n_components=2,
            perplexity=30,
            initialization="pca",
            random_state=42,
            n_jobs=4,
            verbose=False,
        )

        z_subj_ref_2d = tsne_subj.fit(z_subj_ref)
        z_subj_cmp_2d = z_subj_ref_2d.transform(z_subj_cmp)

        # Subject by Subject
        sc_ss = axes[0, 0].scatter(
            z_subj_ref_2d[:, 0], z_subj_ref_2d[:, 1],
            c=y_subj, cmap=cmap_subj, norm=norm_subj,
            s=12, alpha=0.6, label="Ref"
        )
        axes[0, 0].scatter(
            z_subj_cmp_2d[:, 0], z_subj_cmp_2d[:, 1],
            c=y_subj, cmap=cmap_subj, norm=norm_subj,
            s=12, alpha=0.6, marker="x", label="Cmp"
        )
        axes[0, 0].set_title(f"Subject Latent (by Subject) – Ep {epoch}")
        axes[0, 0].legend()
        cbar_ss = plt.colorbar(sc_ss, ax=axes[0, 0])
        cbar_ss.set_label("Subject ID")

        # Subject by Task
        for i, task_id in enumerate(unique_tasks):
            idx = y_task == task_id
            axes[0, 1].scatter(
                z_subj_ref_2d[idx, 0], z_subj_ref_2d[idx, 1],
                color=cmap_task(i), alpha=0.6, s=14,
                label=f"Task {int(task_id)} (Ref)"
            )
            axes[0, 1].scatter(
                z_subj_cmp_2d[idx, 0], z_subj_cmp_2d[idx, 1],
                color=cmap_task(i), alpha=0.6, s=14, marker="x"
            )
        axes[0, 1].set_title("Subject Latent (by Task)")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # ---------------------------------------------------------
        # TSNE TASK (shared space)
        # ---------------------------------------------------------
        tsne_task = openTSNE.TSNE(
            n_components=2,
            perplexity=30,
            initialization="pca",
            random_state=42,
            n_jobs=4,
            verbose=False,
        )

        z_task_ref_2d = tsne_task.fit(z_task_ref)
        z_task_cmp_2d = z_task_ref_2d.transform(z_task_cmp)

        # Task by Task
        for i, task_id in enumerate(unique_tasks):
            idx = y_task == task_id
            axes[1, 0].scatter(
                z_task_ref_2d[idx, 0], z_task_ref_2d[idx, 1],
                color=cmap_task(i), alpha=0.6, s=14,
                label=f"Task {int(task_id)} (Ref)"
            )
            axes[1, 0].scatter(
                z_task_cmp_2d[idx, 0], z_task_cmp_2d[idx, 1],
                color=cmap_task(i), alpha=0.6, s=14, marker="x"
            )
        axes[1, 0].set_title("Task Latent (by Task)")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Task by Subject
        sc_ts = axes[1, 1].scatter(
            z_task_ref_2d[:, 0], z_task_ref_2d[:, 1],
            c=y_subj, cmap=cmap_subj, norm=norm_subj,
            s=12, alpha=0.6, label="Ref"
        )
        axes[1, 1].scatter(
            z_task_cmp_2d[:, 0], z_task_cmp_2d[:, 1],
            c=y_subj, cmap=cmap_subj, norm=norm_subj,
            s=12, alpha=0.6, marker="x", label="Cmp"
        )
        axes[1, 1].set_title("Task Latent (by Subject)")
        axes[1, 1].legend()
        cbar_ts = plt.colorbar(sc_ts, ax=axes[1, 1])
        cbar_ts.set_label("Subject ID")

        # ---------------------------------------------------------
        # NOISE (shared space)
        # ---------------------------------------------------------
        if z_noise_ref_list is not None and z_noise_cmp_list is not None and len(z_noise_ref_list) > 0 and len(z_noise_cmp_list) > 0:
            tsne_noise = openTSNE.TSNE(
                n_components=2,
                perplexity=30,
                initialization="pca",
                random_state=42,
                n_jobs=4,
                verbose=False,
            )

            z_noise_ref_2d = tsne_noise.fit(z_noise_ref)
            z_noise_cmp_2d = z_noise_ref_2d.transform(z_noise_cmp)

            # Noise by Subject
            sc_ns = axes[2, 0].scatter(
                z_noise_ref_2d[:, 0], z_noise_ref_2d[:, 1],
                c=y_noise_subj, cmap=cmap_subj, norm=norm_subj,
                s=12, alpha=0.6, label="Ref"
            )
            axes[2, 0].scatter(
                z_noise_cmp_2d[:, 0], z_noise_cmp_2d[:, 1],
                c=y_noise_subj, cmap=cmap_subj, norm=norm_subj,
                s=12, alpha=0.6, marker="x", label="Cmp"
            )
            axes[2, 0].set_title("Noise Latent (by Subject)")
            axes[2, 0].legend()
            cbar_ns = plt.colorbar(sc_ns, ax=axes[2, 0])
            cbar_ns.set_label("Subject ID")

            # Noise by Task
            for i, task_id in enumerate(unique_tasks):
                idx = y_noise_task == task_id
                axes[2, 1].scatter(
                    z_noise_ref_2d[idx, 0], z_noise_ref_2d[idx, 1],
                    color=cmap_task(i), alpha=0.6, s=14,
                    label=f"Task {int(task_id)} (Ref)"
                )
                axes[2, 1].scatter(
                    z_noise_cmp_2d[idx, 0], z_noise_cmp_2d[idx, 1],
                    color=cmap_task(i), alpha=0.6, s=14, marker="x"
                )
            axes[2, 1].set_title("Noise Latent (by Task)")
            axes[2, 1].legend()
            axes[2, 1].grid(alpha=0.3)

        plt.tight_layout()

        # ---------------------------------------------------------
        # Save
        # ---------------------------------------------------------
        if plot_dir is None:
            plot_dir = self.save_dir / "plots"
        else:
            plot_dir = self.save_dir / "plots" / plot_dir

        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_dir / f"latent_epoch_{epoch}.png")
        plt.close()
        
    def _visualize_signal_vs_reconstruction_spectogram_of_example_sample(
        self, epoch: int, samplesA: torch.Tensor, reconA: torch.Tensor, plot_dir: Optional[Path] = None,
        fft_size: int = 1024, shift_size: int = 120, win_length: int = 600, window="hann_window"
    ):
        """
        Visualizes STFT magnitudes for all channels of the first sample in the batch.
        
        Dimensions:
        - samplesA / reconA: (B, C, T)
        - x_stft: (Freq, Frames) where Freq = (fft_size // 2) + 1
        """
        # 1. Selection: Extract the first sample [0, C, T]
        # We move to CPU and detach for plotting
        orig = samplesA[0].detach().cpu() 
        recon = reconA[0].detach().cpu()
        
        num_channels = orig.shape[0]

        fig, axes = plt.subplots(num_channels, 2, figsize=(12, 3 * num_channels), squeeze=False)
        fig.suptitle(f"Epoch {epoch}: STFT Comparison (Original vs Reconstructed)", fontsize=16)

        for c in range(num_channels):
            for i, (signal, title) in enumerate([(orig[c], "Original"), (recon[c], "Reconstruction")]):
                # 2. Compute STFT
                # Input: (T,) -> Output: (Freq, Frames)
                stft = torch.stft(
                    signal, 
                    n_fft=fft_size, 
                    hop_length=shift_size, 
                    win_length=win_length, 
                    window=window, 
                    return_complex=True
                )
                
                # 3. Magnitude to Log Scale (dB) for visualization
                # Formula: 20 * log10(abs(S) + epsilon)
                magnitude = torch.abs(stft)
                log_spectrogram = 20 * torch.log10(magnitude + 1e-7).numpy()

                ax = axes[c, i]
                im = ax.imshow(log_spectrogram, aspect='auto', origin='lower', cmap='magma')
                ax.set_title(f"Ch {c} - {title}")
                if i == 0:
                    ax.set_ylabel("Frequency Bin")
                if c == num_channels - 1:
                    ax.set_xlabel("Frame Index")
                
                plt.colorbar(im, ax=ax, format="%+2.0f dB")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if plot_dir:
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_dir / f"stft_epoch_{epoch}.png")
            plt.close()
        else:
            plt.show()

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.model.config
        }
        torch.save(checkpoint, self.save_dir / filename)
        
    def save_model_weights(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        return checkpoint['epoch'], checkpoint['metrics']
    
    def check_gradient_flow(self, after_backward: bool = False):
        has_gradients = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                param_norm = param.data.norm().item()
                
                if grad_norm > 1e-8:
                    has_gradients = True
        
        if not has_gradients and after_backward:
            logging.warning("NO GRADIENTS DETECTED!")

    def analyze_encoder_outputs(self, outputs: Dict, labels: Dict):
        encoded = outputs['encoded']
        
        for name, enc_output in encoded.items():
            mu = enc_output['mu']
            logvar = enc_output['logvar']
            z = enc_output['z']
            
            if abs(mu.mean().item()) < 0.01 and abs(mu.std().item() - 1.0) < 0.1:
                logging.warning(f"{name} encoder collapsed to N(0,1) prior!")
            
            if 'logits' in enc_output and name in labels:
                logits = enc_output['logits']
                preds = logits.argmax(dim=-1)
                acc = (preds == labels[name]).float().mean().item()
                
                unique_preds = torch.unique(preds).numel()
                total_classes = logits.shape[1]
                
                if unique_preds == 1:
                    logging.warning(f"Predicting only 1 class! Model collapsed.")
                elif acc < 0.3 and total_classes == 4:
                    logging.warning(f"Below random chance for this Motor imagery 4-class task!")
