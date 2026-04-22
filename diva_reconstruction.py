"""
eval_disentanglement_spectral.py
─────────────────────────────────────────────────────────────────────────────
Spectral evaluation of the DisentangledEEGModel reconstruction quality and
disentanglement effectiveness.

Three evaluation blocks:
  1. Reconstruction quality  — PSD overlay, spectrogram diff, per-band energy
  2. Latent swap             — swap z_subject / z_task and measure spectral
                               change in each EEG frequency band
  3. Linear probe (bonus)    — cross-prediction accuracy from each latent space
─────────────────────────────────────────────────────────────────────────────
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# ── make sure project root is on the path ────────────────────────────────────
#if os.getcwd().endswith("notebooks"):
#    os.chdir("..")

from models.backbones import create_backbone
from models.disentanglement.core import DisentangledEEGModel
from data.dataloaders.shared_loader import CustomLoaderShared

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — edit these four lines
# ═══════════════════════════════════════════════════════════════════════════
RECON_MODEL_PATH = "/home/juser/projects/shared/eeg_disentanglement/experiments/MI/PoolFilterSize1/best_model.pt"
RECON_MODEL_PATH = "/home/juser/projects/shared/eeg_disentanglement/experiments/MI/MI_CBraMod_weighted_STFT01_1_newgammaweighting/last_model.pt"
RECON_MODEL_PATH = "/home/juser/projects/shared/eeg_disentanglement/experiments/MI/MI_CBraMod_weighted_STFT01_1/last_model.pt"

BACKBONE_WEIGHTS_PATH = "/mnt/pve/Rita-Storage-2/disentangleData/weights/cbramod_pretrained_weights.pth"
DATA_PATH = "/mnt/pve/Rita-Storage-2/disentangleData/processed_data/MI_eeg_cbramod.pt"
FS = 200                      # EEG sampling rate (Hz)
N_PLOT_SAMPLES = 3            # samples used for visual panels
N_PROBE_SAMPLES = 200          # samples collected for linear probe eval
# put it in recon_model folder to keep things organized
SAVE_DIR = Path(RECON_MODEL_PATH).parent / "eval_figures" # where to save figures
SAVE_DIR.mkdir(exist_ok=True)

# EEG bands
BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Load model
# ═══════════════════════════════════════════════════════════════════════════
def load_model(model_path: str, backbone_weights: str = None):
    checkpoint = torch.load(model_path, weights_only=False)
    run_config = torch.load(Path(model_path).parent / "run_config.pt", weights_only=False)
    

    config = checkpoint["config"]
    data_shape = run_config["data_shape"]
    args = argparse.Namespace(**run_config)
    
    if backbone_weights:
        print(f"Initializing backbone with weights from: {args.backbone_weights} -> {backbone_weights}")
        args.backbone_weights = backbone_weights  # Ensure backbone weights are not loaded again
        print(f"Overriding backbone weights path in config: {backbone_weights}")
    

    backbone = create_backbone(args, data_shape, config=config, use_identity_for_reconstruction=False)
    print(f"Backbone created with feature dim: {backbone}")
    proj_out = backbone.model.proj_out if hasattr(backbone.model, "proj_out") else None
    #print(f"Backbone proj_out layer: {proj_out}")
    backbone.model.proj_out = nn.Identity()

    model = DisentangledEEGModel(
        backbone, 
        config=config,
        phase_name="DVAE",
        reconstruction_decoder=proj_out,
        classifier_type="diva_classifier"
    )
    
    print("Loading model state dict …")

    # remove the baseline_classifier key if it exists to avoid loading issues
 

    model.load_state_dict(checkpoint["model_state_dict"], strict=False) # strict false ist okay hier, da wir baseline_classifier wirklich nicht brauche
    model.eval()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # count devices and take cuda:1 for everything if there are multiple gpus (to avoid conflicts with training which uses cuda:0)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")
    return model.to(device), device, config


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Spectral helpers
# ═══════════════════════════════════════════════════════════════════════════
def compute_psd(signal_np: np.ndarray, fs: int = 200):
    """signal_np: (T,) -> freqs, psd"""
    freqs, psd = welch(signal_np, fs=fs, nperseg=min(256, len(signal_np)))
    return freqs, psd


def band_energy(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(np.trapezoid(psd[mask], freqs[mask]))


def compute_stft_mag(signal_t: torch.Tensor, fs: int = 200):
    """
    signal_t: (T,) torch tensor on any device.
    Returns (freq_bins, time_frames) numpy magnitude in dB.
    """
    win_len = min(256, signal_t.shape[-1])
    n_fft = 2 ** int(np.ceil(np.log2(win_len)))
    hop = win_len // 4
    window = torch.hann_window(win_len, device=signal_t.device)
    stft = torch.stft(signal_t, n_fft=n_fft, hop_length=hop,
                         win_length=win_len, window=window, return_complex=True)
    mag_db  = 20 * torch.log10(torch.abs(stft) + 1e-6)
    # Keep only up to 45 Hz (gamma)
    freq_res = fs / n_fft                   # Hz per bin
    max_bin = int(np.ceil(45 / freq_res))
    return mag_db[:max_bin].cpu().numpy(), n_fft, hop, freq_res


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Data collection helpers
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def collect_batches(model, loader, device, n_samples: int):
    """
    Returns:
        inputs (N, C, T)
        recons (N, C, T)
        latents {name: (N, latent_dim)}
        task_labels (N,)
        subj_labels (N,)
    """
    inputs_list, recons_list = [], []
    latents_dict = {}
    task_labels_list, subj_labels_list = [], []

    collected = 0
    for batch in loader:
        # batch layout: [samples, self.data[samples_torch], self.y_subjects_tensor[samples_torch], self.y_tasks_tensor[samples_torch], self.y_runs_tensor[samples_torch]] 
        subj_label = batch[2].to(device)
        eeg = batch[1].to(device)
        task_label = batch[3].to(device)

        out = model(eeg)

        inputs_list.append(eeg.cpu())
        recons_list.append(out["eeg_reconstruction"].cpu())
        task_labels_list.append(task_label.cpu())
        subj_labels_list.append(subj_label.cpu())

        if out["var_features_dict"] is not None:
            for name, vf in out["var_features_dict"].items():
                z_flat = vf["z"].reshape(eeg.shape[0], -1).cpu()
                latents_dict.setdefault(name, []).append(z_flat) # store as list of batches, will concatenate at the end

        collected += eeg.shape[0] # keep track of how many samples we've collected so far
        if collected >= n_samples: 
            break

    inputs = torch.cat(inputs_list)[:n_samples] # concatenate all batches and trim to n_samples
    recons = torch.cat(recons_list)[:n_samples] 
    task_y = torch.cat(task_labels_list)[:n_samples].numpy()
    subj_y = torch.cat(subj_labels_list)[:n_samples].numpy()
    latents = {k: torch.cat(v)[:n_samples].numpy() for k, v in latents_dict.items()} # concatenate latent batches and trim to n_samples

    return inputs, recons, latents, task_y, subj_y


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PLOT A — Reconstruction quality
#     Per sample: (a) PSD overlay  (b) Spectrogram side-by-side + diff
#                 (c) Per-band relative error bar chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_reconstruction_quality(inputs, recons, fs=200, n_samples=3,
                                n_channels=3, save_dir=SAVE_DIR):
    """
    inputs, recons: (N, C, T) torch tensors (on CPU)
    """
    band_names = list(BANDS.keys())
    band_ranges = list(BANDS.values())

    for s_idx in range(min(n_samples, inputs.shape[0])):
        fig = plt.figure(figsize=(18, n_channels * 4 + 4), constrained_layout=True)
        fig.suptitle(f"Reconstruction quality — sample {s_idx}", fontsize=14, weight=500)

        outer = gridspec.GridSpec(n_channels + 1, 1, figure=fig,
                                  hspace=0.45)

        for ch in range(min(n_channels, inputs.shape[1])):
            orig_np = inputs[s_idx, ch].numpy()
            recon_np = recons[s_idx, ch].numpy()

            inner = gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=outer[ch], wspace=0.35)

            # ── (a) PSD overlay ─────────────────────────────────────────
            ax_psd = fig.add_subplot(inner[0])
            f_o, p_o = compute_psd(orig_np, fs)
            f_r, p_r = compute_psd(recon_np, fs)
            ax_psd.semilogy(f_o, p_o, label="Original",     color="#185FA5", lw=1.5)
            ax_psd.semilogy(f_r, p_r, label="Reconstructed",color="#D85A30", lw=1.5,
                            linestyle="--")
            for bname, (blo, bhi) in BANDS.items():
                ax_psd.axvspan(blo, bhi, alpha=0.07, color="gray")
                ax_psd.text((blo + bhi) / 2, ax_psd.get_ylim()[0],
                            bname[:3], ha="center", fontsize=7, color="gray",
                            va="bottom")
            ax_psd.set_xlim(0, 45)
            ax_psd.set_xlabel("Frequency (Hz)", fontsize=9)
            ax_psd.set_ylabel("PSD (V²/Hz)", fontsize=9)
            ax_psd.set_title(f"Ch {ch} — PSD", fontsize=10)
            ax_psd.legend(fontsize=8)

            # ── (b) Spectrogram comparison ───────────────────────────────
            mag_o, n_fft, hop, fres = compute_stft_mag(inputs[s_idx, ch], fs)
            mag_r, *_               = compute_stft_mag(recons[s_idx, ch], fs)

            vmin = min(mag_o.min(), mag_r.min())
            vmax = max(mag_o.max(), mag_r.max())

            t_ax = np.arange(mag_o.shape[1]) * hop / fs
            f_ax = np.arange(mag_o.shape[0]) * fres

            ax_orig  = fig.add_subplot(inner[1])
            ax_recon = fig.add_subplot(inner[2])

            extent = [t_ax[0], t_ax[-1], f_ax[0], f_ax[-1]]
            kw = dict(aspect="auto", origin="lower", cmap="viridis",
                      vmin=vmin, vmax=vmax, extent=extent)
            im1 = ax_orig.imshow(mag_o,  **kw)
            im2 = ax_recon.imshow(mag_r, **kw)
            ax_orig.set_title(f"Ch {ch} — Spectrogram (orig)", fontsize=10)
            ax_recon.set_title(f"Ch {ch} — Spectrogram (recon)", fontsize=10)
            for ax_ in [ax_orig, ax_recon]:
                ax_.set_xlabel("Time (s)", fontsize=9)
                ax_.set_ylabel("Freq (Hz)", fontsize=9)
            fig.colorbar(im2, ax=ax_recon, label="dB", shrink=0.8)

        # ── (c) Per-band relative error ─────────────────────────────────
        ax_bar = fig.add_subplot(outer[n_channels])
        n_ch_use = min(n_channels, inputs.shape[1])
        bar_width = 0.15
        x = np.arange(len(band_names))

        for ch_i in range(n_ch_use):
            orig_np = inputs[s_idx, ch_i].numpy()
            recon_np = recons[s_idx, ch_i].numpy()
            f_o, p_o = compute_psd(orig_np, fs)
            f_r, p_r = compute_psd(recon_np, fs)
            rel_err = []
            for blo, bhi in band_ranges:
                e_o = band_energy(f_o, p_o, blo, bhi)
                e_r = band_energy(f_r, p_r, blo, bhi)
                rel_err.append(abs(e_o - e_r) / (e_o + 1e-12))
            ax_bar.bar(x + ch_i * bar_width, rel_err, bar_width,
                       label=f"Ch {ch_i}")

        ax_bar.set_xticks(x + bar_width * (n_ch_use - 1) / 2)
        ax_bar.set_xticklabels(band_names, fontsize=10)
        ax_bar.set_ylabel("Relative energy error", fontsize=10)
        ax_bar.set_title("Per-band reconstruction error (lower = better)", fontsize=11)
        ax_bar.legend(fontsize=9)
        ax_bar.set_ylim(0, 1)
        ax_bar.axhline(0.1, color="gray", linestyle="--", linewidth=0.8,
                       label="10 % threshold")

        fig.savefig(save_dir / f"recon_quality_sample{s_idx}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: recon_quality_sample{s_idx}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5.  PLOT B — Latent swap spectral analysis
#     For each pair of samples that differ in one factor:
#       • swap z_subject -> which bands change?
#       • swap z_task -> which bands change?
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def plot_latent_swap(model, inputs, task_labels, subj_labels, device,
                     fs=200, n_pairs=4, save_dir=SAVE_DIR):
    """
    Picks pairs where:
      same_task_diff_subj  → swap z_subject
      same_subj_diff_task  → swap z_task
    For each swap: reconstructs the swapped signal and plots which EEG bands change.
    """
    encoder_names = model.encoder_names  # ['noise', 'subject', 'task']

    def encode_one(eeg):
        """eeg: (1, C, T) on device -> var_features_dict, encoder_body_residuals"""
        out = model(eeg)
        return out["var_features_dict"], out["encoder_body_residuals"]

    def decode_swapped(vf_dict, residuals, swap_from_vf, swap_key):
        """Replace z for swap_key and decode."""
        new_vf = {k: {"z": v["z"].clone()} for k, v in vf_dict.items()}
        new_vf[swap_key]["z"] = swap_from_vf[swap_key]["z"].clone()
        latent_codes = {k: new_vf[k]["z"] for k in encoder_names}
        return model.generator(latent_codes, residuals)

    def spectral_change(orig_t, swapped_t, fs=200):
        """Returns per-band absolute relative change in PSD energy."""
        orig_np = orig_t.squeeze().cpu().numpy()     # (C, T)
        swapped_np = swapped_t.squeeze().cpu().numpy()
        changes = {b: [] for b in BANDS}
        for ch in range(orig_np.shape[0]):
            f_o, p_o = compute_psd(orig_np[ch], fs)
            f_s, p_s = compute_psd(swapped_np[ch], fs)
            for bname, (blo, bhi) in BANDS.items():
                e_o = band_energy(f_o, p_o, blo, bhi)
                e_s = band_energy(f_s, p_s, blo, bhi)
                changes[bname].append(abs(e_o - e_s) / (e_o + 1e-12))
        return {b: float(np.mean(v)) for b, v in changes.items()}

    # ── find pairs ────────────────────────────────────────────────────────
    # same-task, different subject
    pairs_subj_swap, pairs_task_swap = [], []
    N = len(task_labels)
    for i in range(N):
        for j in range(i + 1, N):
            if task_labels[i] == task_labels[j] and subj_labels[i] != subj_labels[j]:
                pairs_subj_swap.append((i, j))
            if subj_labels[i] == subj_labels[j] and task_labels[i] != task_labels[j]:
                pairs_task_swap.append((i, j))
            if len(pairs_subj_swap) >= n_pairs and len(pairs_task_swap) >= n_pairs:
                break
        if len(pairs_subj_swap) >= n_pairs and len(pairs_task_swap) >= n_pairs:
            break

    band_names = list(BANDS.keys())

    def run_swap_analysis(pairs, swap_key, label):
        all_changes = {b: [] for b in BANDS}

        for (i, j) in pairs[:n_pairs]:
            eeg_i = inputs[i:i+1].to(device)
            eeg_j = inputs[j:j+1].to(device)

            vf_i, res_i = encode_one(eeg_i)
            vf_j, _ = encode_one(eeg_j)

            recon_swapped = decode_swapped(vf_i, res_i, vf_j, swap_key)
            ch = recon_swapped.shape[1]
            orig_full_t = eeg_i[:, :ch, :]
            changes = spectral_change(orig_full_t, recon_swapped, fs)
            for b, v in changes.items():
                all_changes[b].append(v)

        mean_changes = {b: np.mean(v) for b, v in all_changes.items()}
        std_changes = {b: np.std(v) for b, v in all_changes.items()}
        return mean_changes, std_changes

    swap_configs = []
    if "subject" in encoder_names and pairs_subj_swap:
        swap_configs.append(("subject", "z_subject swap\n(same task, diff subject)",
                             "#185FA5", pairs_subj_swap))
    if "task" in encoder_names and pairs_task_swap:
        swap_configs.append(("task", "z_task swap\n(same subject, diff task)",
                             "#D85A30", pairs_task_swap))

    if not swap_configs:
        print("Not enough pairs found for latent swap analysis.")
        return

    fig, axes = plt.subplots(1, len(swap_configs),
                             figsize=(7 * len(swap_configs), 5),
                             sharey=False)
    if len(swap_configs) == 1:
        axes = [axes]

    fig.suptitle("Spectral change after latent swap\n"
                 "(which EEG bands are affected?)", fontsize=13, weight=500)

    for ax, (swap_key, title, color, pairs) in zip(axes, swap_configs):
        mean_c, std_c = run_swap_analysis(pairs, swap_key, title)
        means = [mean_c[b] for b in band_names]
        stds = [std_c[b]  for b in band_names]
        x = np.arange(len(band_names))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                       color=color, alpha=0.75, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(band_names, fontsize=11)
        ax.set_ylabel("Mean relative PSD change (across channels & pairs)", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(means) * 1.4 + 0.05)

        # annotate values
        for bar_, m in zip(bars, means):
            ax.text(bar_.get_x() + bar_.get_width() / 2,
                    bar_.get_height() + 0.01,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_dir / "latent_swap_spectral.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: latent_swap_spectral.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6.  PLOT C — Spectrogram before/after swap (visual, 1 example pair)
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def plot_swap_spectrogram(model, inputs, task_labels, subj_labels, device,
                          fs=200, ch_to_show=0, save_dir=SAVE_DIR):
    """
    Finds one pair (same task, diff subject), swaps z_subject, and plots:
    [orig A]  [orig B]  [A with B's subject]  [diff]
    """
    encoder_names = model.encoder_names
    N = len(task_labels)
    pair = None
    for i in range(N):
        for j in range(i + 1, N):
            if task_labels[i] == task_labels[j] and subj_labels[i] != subj_labels[j]:
                pair = (i, j)
                break
        if pair:
            break

    if pair is None:
        print("No same-task, different-subject pair found for spectrogram swap plot.")
        return

    i, j = pair
    eeg_A = inputs[i:i+1].to(device)
    eeg_B = inputs[j:j+1].to(device)

    out_A = model(eeg_A)
    out_B = model(eeg_B)
    vf_A, res_A = out_A["var_features_dict"], out_A["encoder_body_residuals"]
    vf_B = out_B["var_features_dict"]

    # Swap z_subject
    if "subject" in encoder_names:
        new_vf = {k: {"z": v["z"].clone()} for k, v in vf_A.items()}
        new_vf["subject"]["z"] = vf_B["subject"]["z"].clone()
        latent_codes = {k: new_vf[k]["z"] for k in encoder_names}
        recon_swapped = model.generator(latent_codes, res_A)
    else:
        print("No 'subject' encoder found.")
        return

    # self-reconstruction of A for comparison
    latent_self = {k: vf_A[k]["z"] for k in encoder_names}
    recon_self = model.generator(latent_self, res_A)

    def spec(t, ch):
        sig = t.squeeze()[ch] if t.dim() > 2 else t[ch]
        mag, n_fft, hop, fres = compute_stft_mag(sig, fs)
        t_ax = np.arange(mag.shape[1]) * hop / fs
        f_ax = np.arange(mag.shape[0]) * fres
        return mag, t_ax, f_ax

    signals = {
        "Original A": eeg_A[:, :recon_swapped.shape[1], :],
        "Original B\n(donor subject)": eeg_B[:, :recon_swapped.shape[1], :],
        "A + B's subject\n(swap)": recon_swapped,
        "Self-recon A": recon_self,
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f"Spectrogram — z_subject swap  (channel {ch_to_show})",
                 fontsize=13, weight=500)

    mags_all = []
    for title, sig_t in signals.items():
        mag, t_ax, f_ax = spec(sig_t, ch_to_show)
        mags_all.append(mag)

    vmin = min(m.min() for m in mags_all)
    vmax = max(m.max() for m in mags_all)

    for ax, (title, sig_t), mag in zip(axes, signals.items(), mags_all):
        t_ax = np.arange(mag.shape[1]) * (256 // 4) / fs
        f_ax = np.arange(mag.shape[0]) * (fs / 512)
        extent = [t_ax[0], t_ax[-1], f_ax[0], f_ax[-1]]
        im = ax.imshow(mag, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Freq (Hz)", fontsize=9)
        fig.colorbar(im, ax=ax, label="dB", shrink=0.8)

    plt.tight_layout()
    fig.savefig(save_dir / "swap_spectrogram.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: swap_spectrogram.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7.  PLOT D — Linear probe cross-prediction matrix
#     For each latent space, predict both task and subject labels.
# ═══════════════════════════════════════════════════════════════════════════
def plot_linear_probe(latents, task_labels, subj_labels, save_dir=SAVE_DIR):
    """
    latents: {encoder_name: (N, latent_dim) np array}
    Produces a heatmap of cross-val accuracy:
        rows = latent spaces, cols = predicted label type
    """
    label_sets  = {"task": task_labels, "subject": subj_labels}
    encoder_nms = sorted(latents.keys())
    results = np.zeros((len(encoder_nms), len(label_sets)))

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")),
    ])

    print("\nLinear probe results (5-fold cross-val accuracy):")
    print(f"{'Latent':>12}  {'task':>8}  {'subject':>8}")
    for i, enc_name in enumerate(encoder_nms):
        Z = latents[enc_name]
        for j, (label_name, y) in enumerate(label_sets.items()):
            if len(np.unique(y)) < 2:
                results[i, j] = float("nan")
                continue
            scores = cross_val_score(clf, Z, y, cv=5, scoring="accuracy")
            results[i, j] = scores.mean()
        print(f"{enc_name:>12}  "
              f"{results[i, 0]:.3f}      {results[i, 1]:.3f}")

    fig, ax = plt.subplots(figsize=(5, len(encoder_nms) + 1))
    im = ax.imshow(results, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(label_sets)))
    ax.set_xticklabels(list(label_sets.keys()), fontsize=11)
    ax.set_yticks(range(len(encoder_nms)))
    ax.set_yticklabels(encoder_nms, fontsize=11)
    ax.set_title("Linear probe accuracy\n(diagonal = good disentanglement)", fontsize=11)
    fig.colorbar(im, ax=ax, label="Accuracy")

    for i in range(len(encoder_nms)):
        for j in range(len(label_sets)):
            val = results[i, j]
            txt = f"{val:.2f}" if not np.isnan(val) else "n/a"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=12, weight=500,
                    color="white" if val < 0.5 else "black")

    plt.tight_layout()
    fig.savefig(save_dir / "linear_probe.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: linear_probe.png")


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__" or True:  # works both as script and notebook cell

    # ── load ──────────────────────────────────────────────────────────────
    print("Loading model …")
    model, device, config = load_model(RECON_MODEL_PATH, BACKBONE_WEIGHTS_PATH)

    print("Loading data …")
    data_dict = torch.load(DATA_PATH, weights_only=False)
    loader_obj = CustomLoaderShared(data_dict, exclude_tasks=[4])
    loader = loader_obj.get_dataloader(
        num_total_samples=None,
        batch_size=8,
        property=None,
        random_sample=True,
    )

    # ── collect ───────────────────────────────────────────────────────────
    print(f"Collecting {N_PROBE_SAMPLES} samples …")
    inputs, recons, latents, task_labels, subj_labels = collect_batches(
        model, loader, device, N_PROBE_SAMPLES
    )
    print(f" inputs : {inputs.shape}")
    print(f" recons : {recons.shape}")
    print(f" latents : { {k: v.shape for k, v in latents.items()} }")

    # ── compute overall spectral MSE ──────────────────────────────────────
    recon_errs = []
    for s in range(inputs.shape[0]):
        for ch in range(inputs.shape[1]):
            f_o, p_o = compute_psd(inputs[s, ch].numpy(), FS)
            f_r, p_r = compute_psd(recons[s, ch].numpy(), FS)
            # interpolate onto same grid
            p_r_i = np.interp(f_o, f_r, p_r)
            recon_errs.append(np.mean((np.log(p_o + 1e-12) - np.log(p_r_i + 1e-12))**2))
    print(f"\nMean log-PSD MSE (all channels, all samples): {np.mean(recon_errs):.4f}")

    # ── Plot A: Reconstruction quality ────────────────────────────────────
    print("\n[Plot A] Reconstruction quality …")
    plot_reconstruction_quality(inputs, recons, fs=FS,
                                n_samples=N_PLOT_SAMPLES, n_channels=3)

    # ── Plot B: Latent swap spectral change ───────────────────────────────
    print("\n[Plot B] Latent swap spectral analysis …")
    plot_latent_swap(model, inputs, task_labels, subj_labels,
                     device, fs=FS, n_pairs=4)

    # ── Plot C: Spectrogram before/after swap ─────────────────────────────
    print("\n[Plot C] Swap spectrogram …")
    plot_swap_spectrogram(model, inputs, task_labels, subj_labels,
                          device, fs=FS, ch_to_show=0)

    # ── Plot D: Linear probe ──────────────────────────────────────────────
    print("\n[Plot D] Linear probe …")
    plot_linear_probe(latents, task_labels, subj_labels)