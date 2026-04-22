### 

import os
import scipy.io
import numpy as np
import mne
import torch
import cvxeda

def get_cvx_component(signal, fs, component_type='tonic'):
    """Decomposes EDA using convex optimization (cvxEDA)."""
    # Standardize signal
    # use min max normalization
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val > min_val:
        signal = (signal - min_val) / (max_val - min_val)
        
    if np.any(np.isnan(signal)):
        signal = np.nan_to_num(signal, nan=np.nanmean(signal))
                
    # Run cvxEDA algorithm: r: phasic, t: tonic, obj: objective function minimized [cite: 1145]
    r, p, t, l, d, e, obj = cvxeda.cvxEDA(np.asarray(signal, dtype=np.float64), 1.0 / fs)
    
    if component_type == 'tonic':
        return t
    elif component_type == 'phasic':
        return r
    else:
        return signal
    
def align_start_times(eeg_times, gsr_times):
    # Find which starts later and align to that
    if eeg_times[0] > gsr_times[0]:
        start_time = eeg_times[0]
    else:
        start_time = gsr_times[0]
    eeg_start_idx = np.searchsorted(eeg_times, start_time)
    gsr_start_idx = np.searchsorted(gsr_times, start_time)
    print(f"Aligning to start time: {start_time:.2f}s")
    print(f"EEG starts at {eeg_times[eeg_start_idx]:.2f}s (index {eeg_start_idx}), GSR starts at {gsr_times[gsr_start_idx]:.2f}s (index {gsr_start_idx})")
    return int(eeg_start_idx), int(gsr_start_idx)



def build_vr_arousal_dataset(base_dir, component='tonic', target_sfreq=200, WINDOW_SIZE=30, gsr_fs=20):
    """
    Builds a dataset from extracted folders S0xxx.
    """
    all_x, all_y, all_subjs = [], [], []
    ch_names = ['FC5', 'FC1', 'CP5', 'CP1', 'CP2', 'CP6', 'FC2', 'FC6']
    
    # Get subject directories (S0xxx)
    subject_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('S')])
    print(f"Found {len(subject_dirs)} subject directories.")

    for subj_str in subject_dirs:
        try:
            subj_id = int(subj_str[1:])
        except ValueError: # NO zip files here! ->_ zippp are skipped
            continue
            
        subj_path = os.path.join(base_dir, subj_str)
        file_list = os.listdir(subj_path)
        
        # Map indices (1, 2) to files
        eeg_mats = {f.split('_')[-1].replace('.mat', ''): f for f in file_list if 'EEG' in f.upper() and f.endswith('.mat')}
        phys_mats = {f.split('_')[-1].replace('.mat', ''): f for f in file_list if 'PHYSIO' in f.upper() and f.endswith('.mat')}
        
        common_sessions = set(eeg_mats.keys()) & set(phys_mats.keys()) # {"_1", "_2"}
        
        if not common_sessions:
            #print(f"Skipping {subj_str}: No common EEG/Physio sessions.")
            continue
            
        for session in sorted(common_sessions):
            # --- 1. LOAD DATA FROM DISK ---
            phys_file = os.path.join(subj_path, phys_mats[session])
            eeg_file = os.path.join(subj_path, eeg_mats[session])
            print(f"Processing {subj_str} Session {session}: EEG file: {eeg_file}, Physio file: {phys_file}")
            
            # --- 2. PROCESS EDA ---
            physio_mat = scipy.io.loadmat(phys_file)
            physio_key = [k for k in physio_mat.keys() if not k.startswith('__')][0]
            physio_data = physio_mat[physio_key]
            if physio_data.shape[0] != 5:
                physio_data = physio_data.T
                
            raw_gsr = physio_data[2, :] # Row 2 is GSR 
            gsr_times = physio_data[0, :] # Row 0 is timestamps
            if np.any(np.isnan(raw_gsr)):
                raw_gsr = np.nan_to_num(raw_gsr, nan=np.nanmean(raw_gsr))
                
            # Decomposition identifies tonic (slow) and phasic (fast) responses 
            eda_processed = get_cvx_component(raw_gsr, fs=gsr_fs, component_type=component)
            
            
            # --- 3. PROCESS EEG ---
            eeg_mat = scipy.io.loadmat(eeg_file)
            raw_eeg_mat = eeg_mat['rawEEG']
            eeg_data = raw_eeg_mat[1:9, :] # Channels 1-8
            eeg_times = raw_eeg_mat[0, :]
            # Align EDA and EEG start times (if timestamps are available)
            eeg_idx, gsr_idx = align_start_times(eeg_times, gsr_times)
            
            # Align both signals to the same start time and trim to the same length
            eeg_data = eeg_data[:, eeg_idx:]
            eda_processed = eda_processed[gsr_idx:]
            
            
            # EDA is sampled at 20Hz, so we create labels for each 30s window (600 samples) based on the mean EDA in that window.
            samples_per_window_eda = int(WINDOW_SIZE * gsr_fs)
            n_epochs_eda = len(eda_processed) // samples_per_window_eda
            print(f"Total EDA samples: {len(eda_processed)}, which gives {n_epochs_eda} full epochs of {WINDOW_SIZE}s each.")
            
            epoch_means = np.mean(eda_processed[:n_epochs_eda * samples_per_window_eda].reshape(-1, samples_per_window_eda), axis=1)
            print(f"Mean EDA per epoch (first 5): {epoch_means[:5]}, SHAPE: {epoch_means.shape}")
            # print how many 0, 1, 2 labels we have
            
            # Labeling captures low, medium, and high arousal levels 
            #q33, q66 = np.percentile(eda_processed, [33.3, 66.6])
            q33, q66 = 0.33, 0.66
            eda_labels = [0 if m <= q33 else 1 if m <= q66 else 2 for m in epoch_means]
            print(f"EDA labels distribution: {np.bincount(eda_labels)}")

            # EEG preprocessing: Bandpass filter and resample to target_sfreq (e.g., 200Hz)
            info = mne.create_info(ch_names=ch_names, sfreq=500, ch_types='eeg')
            raw = mne.io.RawArray(eeg_data, info, verbose=False)
            
            raw.filter(l_freq=0.3, h_freq=45, verbose=False)
            raw.resample(target_sfreq, verbose=False)
            
            # 30s, 500 Hz -> 6000 samples; at 200Hz -> 6000 * (200/500) = 2400 samples per epoch
            epochs = mne.make_fixed_length_epochs(raw, duration=WINDOW_SIZE, preload=True, verbose=False)
            ep_data = epochs.get_data() # (n_epochs, n_channels, n_times)
            
            # AMount labels to match the number of epochs we have in EEG 
            print(f"EEG epochs shape: {ep_data.shape}, EDA labels length: {len(eda_labels)}")

            # Synchronize epoch count
            min_len = min(len(ep_data), len(eda_labels))
            
            all_x.append(ep_data[:min_len])
            all_y.extend(eda_labels[:min_len])
            all_subjs.extend([subj_id] * min_len)
            print(f"Processed {subj_str} S{session}: {min_len} epochs.")

    X = torch.tensor(np.concatenate(all_x, axis=0), dtype=torch.float32)
    y = torch.tensor(np.array(all_y), dtype=torch.long)
    subjects = torch.tensor(np.array(all_subjs), dtype=torch.long)
    runs = torch.zeros_like(y)
    return {'X': X, 'y': y, 'subjects': subjects, 'runs': runs}

data_dict = build_vr_arousal_dataset("/mnt/turingDatasets/MMDAPBE/") 