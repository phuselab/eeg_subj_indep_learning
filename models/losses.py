import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import numpy as np

# Local imports will need adjustment
from configs.config import LossConfig
from models.disentanglement.core import DisentangledEEGModel
from models.classifiers.classifiers import SimpleFeaturesClassifier
from torch import autograd



import torch
import torch.nn as nn
import torch.nn.functional as F

class STFTLoss(nn.Module):
    """
    Calcola la Spectral Convergence Loss e la Log Magnitude Loss 
    per una singola risoluzione STFT.
    """
    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        # Registriamo la finestra come buffer per farla spostare automaticamente su CPU/GPU
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, x_hat):
        # Sposta la finestra senza riassegnare self.window (che è un buffer)
        # window = self.window.to(x.device)

        # Reshape se 3D: (B, C, T) -> (B*C, T)
        if x.dim() == 3:
            b, c, t = x.shape
            x = x.reshape(-1, t)
            x_hat = x_hat.reshape(-1, t)

        # STFT -> Risultato 3D: (B*C, Freq, Frames)
        x_stft = torch.stft(x, n_fft=self.fft_size, hop_length=self.shift_size, 
                            win_length=self.win_length, window=self.window, 
                            return_complex=True)
        
        #freq_bins_weights = torch.linspace(0.1, 1.0, steps=freq_bins, device=x.device).view(1, freq_bins, 1)
        x_hat_stft = torch.stft(x_hat, n_fft=self.fft_size, hop_length=self.shift_size, 
                                win_length=self.win_length, window=self.window, 
                                return_complex=True)
        
        apply_weights = False # TODO for now, now band weighting
        if apply_weights: 
            freq_bins, frames = x_stft.shape[1], x_stft.shape[2]
            
            #  Frequency-band-aware weights for EEG (200 Hz, bins up to fft_size//2+1) 
            # Map bin index -> Hz: hz = bin * (sample_rate / fft_size)
            # Instead of raw linear weights, define band-based weights:
            #   delta (0–4 Hz), theta (4–8), alpha (8–13), beta (13–30), gamma (30+)
            hz_per_bin = 200.0 / self.fft_size
            bin_freqs = torch.arange(freq_bins, device=x.device) * hz_per_bin

            weights = torch.ones(freq_bins, device=x.device)
            weights = torch.where(bin_freqs < 4,   weights * 1.3,  weights)  # delta
            weights = torch.where((bin_freqs >= 4)  & (bin_freqs < 8),  weights * 1.0, weights)  # theta
            weights = torch.where((bin_freqs >= 8)  & (bin_freqs < 13), weights * 1.5, weights)  # alpha
            weights = torch.where((bin_freqs >= 13) & (bin_freqs < 30), weights * 2.0, weights)  # beta
            weights = torch.where(bin_freqs >= 30,  weights * 3.0, weights)  # gamma — boosted but bounded
            weights = torch.where(bin_freqs >= 60, weights * 0.1, weights)
            weights = weights.view(1, freq_bins, 1)  # broadcast over (B*C, Freq, Frames)
            
            x_mag = torch.abs(x_stft) + 1e-7
            #x_mag = x_mag * freq_bins_weights  # Applichiamo i pesi ai bin di frequenza
            x_hat_mag = torch.abs(x_hat_stft) + 1e-7
        

            # 2. Spectral Convergence Loss
            # Usiamo la norma globale (equivalente a Frobenius su più dimensioni)
            # Apply weights to the ERROR, not to the signals themselves
            
            sc_loss = torch.linalg.norm((x_mag - x_hat_mag) * weights) / (torch.linalg.norm(x_mag) + 1e-9)
        else: 
            x_mag = torch.abs(x_stft) + 1e-7
            x_hat_mag = torch.abs(x_hat_stft) + 1e-7

            # 2. Spectral Convergence Loss
            # Usiamo la norma globale (equivalente a Frobenius su più dimensioni)
            sc_loss = torch.linalg.norm(x_mag - x_hat_mag) / (torch.linalg.norm(x_mag) + 1e-9)

        # 3. Log Magnitude Loss
        log_mag_loss = F.l1_loss(torch.log(x_mag), torch.log(x_hat_mag), reduction='mean')

        return sc_loss, log_mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Calcola la STFT Loss mediata su M risoluzioni differenti.
    """
    def __init__(self, segment_length=800):
        super(MultiResolutionSTFTLoss, self).__init__()
        # Inizializzazione corretta per EEG a 200Hz con chunk da 800 sample
        win_lengths = [segment_length // (2 ** i) for i in range(1, 4)]  # [400, 200, 100]
        # come fft_size voglio la potenza di 2 più vicina a win_length, quindi 512, 256, 128
        fft_sizes = [2 ** int(np.ceil(np.log2(wl))) for wl in win_lengths]  # [512, 256, 128]
        hop_sizes = [wl // 4 for wl in win_lengths]  # hop size tipico è win_length // 4, quindi [200, 100, 50]
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), \
            "Le liste dei parametri STFT devono avere la stessa lunghezza (M)."
        
        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, ss, wl))

    def forward(self, x, x_hat):
        """
        x: tensore originale [Batch, Time] o [Batch * Channels, Time]
        x_hat: tensore ricostruito, stessa shape di x
        """
        sc_loss_total = 0.0
        mag_loss_total = 0.0
        M = len(self.stft_losses)

        for f in self.stft_losses:
            sc_l, mag_l = f(x, x_hat)
            sc_loss_total += sc_l
            mag_loss_total += mag_l

        # Equazione (2): Media delle loss sulle M risoluzioni
        sc_loss = sc_loss_total / M
        mag_loss = mag_loss_total / M

        # Somma L_sc e L_mag come richiesto dall'equazione
        return sc_loss + mag_loss




class DisentanglementLoss(nn.Module):
    """Comprehensive loss function with all disentanglement objectives."""
    
    def __init__(self, config: LossConfig, discriminator=None, class_weights: Dict[str, torch.Tensor] = None, segment_length=800):
        super().__init__()
        self.config = config
        self.state = {'prev_losses': {}, 'epoch': 0}
        self.discriminator = discriminator
        self.class_weights = class_weights
        self.stft_loss = MultiResolutionSTFTLoss(segment_length=segment_length)  # Example segment length for EEG at 200Hz
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            print(f"STFTLoss: Detected {num_devices} CUDA device(s). Using device: {torch.cuda.get_device_name(0)}")
        # select last GPU if multiple are available
        gpu = f'cuda:{torch.cuda.device_count() - 1}' if torch.cuda.is_available() else 'cpu'
        self.stft_loss.to(gpu)  # Ensure the loss module is on the correct device

        
    def set_epoch(self, epoch: int):
        self.state['epoch'] = epoch
    
    def get_dynamic_classification_weight(self) -> float:
        epoch = self.state.get('epoch', 0)
        base_weight = self.config.classification_weight
        
        if epoch < 10:
            return base_weight * (0.2 + 0.08 * epoch)
        return base_weight
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_elements = 1 + logvar - mu.pow(2) - logvar.exp()
        B = mu.size(0)
        kl_flat = kl_elements.view(B, -1)
        return -0.5 * torch.sum(kl_flat, dim=1).mean()
    
    def self_reconstruction_loss(self, reconstruction: torch.Tensor, 
                                target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(reconstruction, target)
    
    
    def self_eeg_reconstruction_loss(self, reconstruction: torch.Tensor, 
                                   target: torch.Tensor) -> torch.Tensor:
        return self.stft_loss(reconstruction, target)
    
    def self_eeg_reconstruction_loss_mse(self, reconstruction: torch.Tensor, 
                                      target: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(reconstruction, target)

    
    
   
    def classification_loss(self, logits: torch.Tensor,
                            labels: torch.Tensor, name: str) -> torch.Tensor:
        # Retrieve weights if they exist for this specific head
        weight = self.class_weights.get(name) if self.class_weights else None
        return F.cross_entropy(logits, labels, weight=weight)
    


    
    def self_cycle_loss(self, original: torch.Tensor, cycled: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(original, cycled)
    
    def cross_subject_intra_class_loss(self, reconstruction: torch.Tensor,
                                      target_features: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(reconstruction, target_features)
    
    def cross_subject_cross_class_loss(self, reconstruction: torch.Tensor,
                                      target_features: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(reconstruction, target_features)
    
    def knowledge_distillation_loss(self, student_logits: torch.Tensor,
                                   teacher_logits: torch.Tensor) -> torch.Tensor:
        return F.kl_div(student_logits, teacher_logits, reduction='batchmean', log_target=True)
    
    def compute_gradient_penalty_eeg(self, real_eeg, fake_eeg, device):
        """
        Calcola la Gradient Penalty per input EEG 3D.
        Input shape attesa: (Batch, NumChannels, Temporal)
        """
        batch_size = real_eeg.size(0)
        
        # 1. Gestione di Alpha per il broadcasting 3D
        # alpha deve avere dimensioni (B, 1, 1) per spalmare lo stesso 
        # scalare su tutti i canali e il tempo di un singolo campione.
        alpha = torch.rand(batch_size, 1, 1, device=device)
        
        # 2. Interpolazione
        # Shape: (B, C, T) = (B, 1, 1) * (B, C, T) + ...
        interpolates = (alpha * real_eeg + ((1 - alpha) * fake_eeg)).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates)
        
        # Tensore dummy per il calcolo dei gradienti
        dummy = torch.ones_like(d_interpolates, requires_grad=False, device=device)
        
        # 3. Calcolo Gradienti
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=dummy,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # 4. Flattening
        # Da (B, C, T) diventa (B, C * T)
        # Questo calcola la norma L2 del gradiente per ogni intero campione EEG
        gradients = gradients.reshape(batch_size, -1)
        
        # Calcolo penalità: (||nabla||_2 - 1)^2
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty


    def discriminator_loss_eeg(self, real_eeg, fake_eeg, lambda_gp=10.0, drift_weight=0.001, device='cpu'):
        """
        Loss Discriminatore WGAN-GP per tensori EEG 3D
        Input shape attesa: (Batch, NumChannels, Temporal)
        """
        # Assicuriamoci che abbiano la stessa shape prima di fare qualsiasi cosa
        if real_eeg.shape != fake_eeg.shape:
            real_eeg = real_eeg.view_as(fake_eeg)

        # Output Discriminatore: si assume shape (Batch, 1)
        d_real = self.discriminator(real_eeg)
        d_fake = self.discriminator(fake_eeg.detach()) # stacca il generatore dal grafo!
        
        # Wasserstein Loss: D(fake) - D(real)
        wgan_loss = d_fake.mean() - d_real.mean()
        
        # Gradient Penalty (versione 3D)
        gp_loss = self.compute_gradient_penalty_eeg(real_eeg, fake_eeg, device)
        
        # Drift Term: penalizza valori di output del discriminatore troppo grandi in modulo
        drift_term = ((d_real + d_fake) ** 2).mean()
        
        total_loss = wgan_loss + (lambda_gp * gp_loss) + (drift_weight * drift_term)
        
        return total_loss, wgan_loss

    def generator_loss_eeg(self, real_eeg, fake_eeg, drift_weight=0.001):
        """
        Loss Generatore per tensori EEG a patch (Eq. 7)
        Input: (Batch, Channels, NumPatches, PatchSize)
        """
        d_fake = self.discriminator(fake_eeg)
        
        # D(real) serve per il drift term dell'Eq (7)
        with torch.no_grad():
            d_real = self.discriminator(real_eeg)
        
        # Adversarial: -D(fake)
        adv_loss = -d_fake.mean()
        
        # Drift Term
        drift_term = ((d_real + d_fake) ** 2).mean()
        
        total_loss = adv_loss + (drift_weight * drift_term)
        
        return total_loss
    
 
    
    
    
    def compute_loss(self, outputs: Dict[str, Any], labels: Dict[str, torch.Tensor],
                    model: DisentangledEEGModel, adversarial_step=None) -> Dict[str, torch.Tensor]:
        losses = {}
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
        # select last GPU if multiple are available
        gpu = f'cuda:{torch.cuda.device_count() - 1}' if torch.cuda.is_available() else 'cpu'
        total_loss = torch.tensor(0.0, device=gpu)
        
        encoded = outputs['var_features_dict']
        # print(f"Reconstruction shape in loss: {reconstruction.shape}")
        # features = outputs['encoder_body_features_dict']
        logits_dict = outputs['logits_dict']
        backbone_features = outputs.get('backbone_features', None)
        inputs = outputs.get('inputs', None) # shape (B, C, T)
        var_logits_dict = outputs.get('var_logits_dict', None)
        eeg_reconstruction = outputs.get('eeg_reconstruction', None)
        backbone_eeg_reconstruction = outputs.get('backbone_eeg_reconstruction', None)
        encoder_body_residuals = outputs.get('encoder_body_residuals', None)

        if adversarial_step == 'G' or adversarial_step is None:
            if self.config.self_reconstruction:
    
                recon_loss = self.self_eeg_reconstruction_loss(eeg_reconstruction, inputs)
                losses['self_reconstruction'] = recon_loss
                total_loss += self.config.self_reconstruction_weight * recon_loss
                
                recon_loss_mse = self.self_eeg_reconstruction_loss_mse(eeg_reconstruction, inputs)
                losses['self_reconstruction_mse'] = recon_loss_mse
                total_loss += self.config.self_reconstruction_weight_mse * recon_loss_mse
            
            if self.config.kl_divergence:
                kl_total = 0
                for name, enc_output in encoded.items():
                    kl_loss = self.kl_divergence(enc_output['mu'], enc_output['logvar'])
                    losses[f'kl_{name}'] = kl_loss
                    
                    weight = (self.config.noise_kl_weight if name == 'noise' 
                            else self.config.kl_weight)
                    kl_total +=  kl_loss
                
                losses['kl_total'] = kl_total
                total_loss += kl_total * weight
                        
            if self.config.classification:
                for name, logits in logits_dict.items():
                    if name in labels:
                        #cls_loss = self.classification_loss(logits, labels[name])
                        cls_loss = self.classification_loss(logits, labels[name], name)
                        losses[f'classification_{name}'] = cls_loss
                        total_loss += cls_loss

            if self.config.var_classification:
                # also do classification on latent codes after variational head
                for name, enc_output in encoded.items():
                    if name in labels:
                        cls_loss = self.classification_loss(var_logits_dict[name], labels[name], name)
                        losses[f'classification_on_z_{name}'] = cls_loss
                        total_loss += cls_loss
            

            # Equation 9 and Figure 3: under the name of Latent Consistency, compares the noise with a sampled noise N(0,1) 
            if self.config.self_cycle:
                var_feats_w_sampled_noise = encoded.copy()
                # Override noise with sampled noise from normal N(0,1)
                sampled_noise = torch.randn_like(encoded['noise']['z'])
                var_feats_w_sampled_noise['noise']['z'] = sampled_noise
                var_feats_w_sampled_noise = {k: v['z'] for k, v in var_feats_w_sampled_noise.items()}
                cycle_reconstruction_w_rand_noise = model.generator(var_feats_w_sampled_noise, encoder_body_residuals=encoder_body_residuals)
                features = model.feature_extractor(cycle_reconstruction_w_rand_noise)
                noise_body, _ = model.encoder_bodies['noise'](features)
                noise_encoded = model.variational_heads['noise'](noise_body)
                cycle_loss = F.l1_loss(sampled_noise, noise_encoded['z'])
                losses['self_cycle'] = cycle_loss
                total_loss += self.config.self_cycle_weight * cycle_loss
                
            # Equation 11
            if self.config.cross_subject_intra_class and 'cross_intra_reconstruction_A' in outputs:
                intra_recon_A = outputs['cross_intra_reconstruction_A']
                intra_target_A = outputs['cross_intra_target_A']
                
                intra_recon_B = outputs['cross_intra_reconstruction_B']
                intra_target_B = outputs['cross_intra_target_B']

                intra_loss_A = self.cross_subject_intra_class_loss(intra_recon_A, intra_target_A)
                intra_loss_B = self.cross_subject_intra_class_loss(intra_recon_B, intra_target_B)
                intra_loss = (intra_loss_A + intra_loss_B) / 2
                losses['cross_subject_intra_class'] = intra_loss
                total_loss += self.config.cross_subject_intra_class_weight * intra_loss

            # Equation 13
            if self.config.cross_subject_cross_class and 'cross_cross_z_subjects_ABC' in outputs:
                cross_loss_subjects = self.cross_subject_cross_class_loss(outputs['cross_cross_z_subjects_target_ABC'], outputs['cross_cross_z_subjects_ABC'])
                losses['cross_subject_cross_class_subjects'] = cross_loss_subjects
                cross_loss_tasks = self.cross_subject_cross_class_loss(outputs['cross_cross_z_tasks_target_ABC'], outputs['cross_cross_z_tasks_ABC'])
                losses['cross_subject_cross_class_tasks'] = cross_loss_tasks
                cross_loss = (cross_loss_subjects + cross_loss_tasks) / 2
                losses['cross_subject_cross_class'] = cross_loss
                total_loss += self.config.cross_subject_cross_class_weight * cross_loss

                # Equation 12
                if self.config.adversarial and 'cross_cross_adv_fake' in outputs:
                    adv_fake = outputs['cross_cross_adv_fake']
                    adv_real = outputs['cross_cross_adv_real'].detach()
                    adv_loss = self.generator_loss_eeg(
                        adv_real, adv_fake,
                        drift_weight=0.001
                    )
                    losses['adversarial_generator_cross_cross'] = adv_loss
                    total_loss += self.config.adversarial_weight * adv_loss
                
                # Equation 14
                if self.config.cross_cross_cycle:
                    cc_A_loss = self.self_eeg_reconstruction_loss(outputs['cross_cross_cycle_rec_A'], outputs['cross_cross_cycle_target_A'])
                    cc_B_loss = self.self_eeg_reconstruction_loss(outputs['cross_cross_cycle_rec_B'], outputs['cross_cross_cycle_target_B'])
                    cc_C_loss = self.self_eeg_reconstruction_loss(outputs['cross_cross_cycle_rec_C'], outputs['cross_cross_cycle_target_C'])
                    cross_cross_cycle_loss = (cc_A_loss + cc_B_loss + cc_C_loss) / 3
                    losses['cross_cross_cycle'] = cross_cross_cycle_loss
                    total_loss += self.config.cross_cross_cycle_weight * cross_cross_cycle_loss
                    
    
            # Equation 15
            if 'cross_cross_logits' in outputs and 'cross_cross_var_logits' in outputs and self.config.knowledge_distillation:
                for name in logits_dict.keys():
                    if name in labels:
                        cc_logits = outputs['cross_cross_logits'][name].detach()
                        cc_var_logits = outputs['cross_cross_var_logits'][name]

                        # transform to log probabilities
                        cc_logits = F.log_softmax(cc_logits, dim=-1)
                        cc_var_logits = F.log_softmax(cc_var_logits, dim=-1)
                        
                        cc_knowledge_distillation_kl = self.knowledge_distillation_loss(
                            cc_var_logits, cc_logits
                        )
                        losses[f'knowledge_distillation_{name}'] = cc_knowledge_distillation_kl
                        total_loss += self.config.kd_weight * cc_knowledge_distillation_kl
                        
            
            if self.config.adversarial:
                adv_gen_loss = self.generator_loss_eeg(
                    inputs, eeg_reconstruction.reshape(inputs.shape),
                    drift_weight=0.001
                )
                losses['adversarial_generator'] = adv_gen_loss
                total_loss += self.config.adversarial_weight * adv_gen_loss

                
        elif adversarial_step == 'D':
            if self.config.adversarial:
                # compute adversarial discriminator loss
                # Here detach on both because it's not the input but the CBraMod output (which is not frozen)
                adv_disc_loss, wgan_loss = self.discriminator_loss_eeg(
                    inputs, eeg_reconstruction.detach(),
                    lambda_gp=10.0,
                    drift_weight=0.001,
                    device=inputs.device
                )
                losses['adversarial_discriminator'] = adv_disc_loss
                losses['adversarial_wgan'] = wgan_loss
                total_loss += self.config.adversarial_weight * adv_disc_loss

                if self.config.cross_subject_cross_class and 'cross_cross_adv_fake' in outputs:
                    adv_fake = outputs['cross_cross_adv_fake'].detach()
                    adv_real = outputs['cross_cross_adv_real'].detach() # Here detach because it's not the input but the CBraMod output (which is not frozen)
                    adv_disc_cc_loss, wgan_cc_loss = self.discriminator_loss_eeg(
                        adv_real, adv_fake,
                        lambda_gp=10.0,
                        drift_weight=0.001,
                        device=inputs.device
                    )
                    losses['adversarial_wgan_cross_cross'] = wgan_cc_loss
                    losses['adversarial_discriminator_cross_cross'] = adv_disc_cc_loss
                    total_loss += self.config.adversarial_weight * adv_disc_cc_loss

        losses['total'] = total_loss
        return losses

        

    def compute_loss_classification_only(self, outputs: Dict[str, Any], labels: Dict[str, torch.Tensor],
                    model: SimpleFeaturesClassifier) -> Dict[str, torch.Tensor]:
        losses = {}
        total_loss = torch.tensor(0.0, device='cuda:0')
        
        logits = outputs

        if self.config.classification:
            cls_loss = F.cross_entropy(logits, labels['task'])
            losses['classification'] = cls_loss
            total_loss += self.config.classification_weight * cls_loss
        losses['total'] = total_loss
        return losses





