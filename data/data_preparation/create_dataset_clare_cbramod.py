import os
import torch
import mne
import numpy as np
import pandas as pd
from pathlib import Path

# Parameters
ORIGINAL_FS = 256  
TARGET_FS = 200
SEGMENT_SEC = 10
SAMPLES_PER_SEGMENT = SEGMENT_SEC * TARGET_FS  # 2000 samples
CH_NAMES = ['TP9', 'AF7', 'AF8', 'TP10'] # Standard Muse configuration for CLARE
PICK_CH_NAMES = ['AF7', 'AF8'] # Focus on these channels due to noise in others

def get_cognitive_load_label(load):
    """
    Thresholding at 5 treats the lower-middle of the 
    scale as 'Low'. 
    """
    return 0 if load <= 5 else 1

def process_clare(root_path):
    root = Path(root_path)
    eeg_dir = root / "EEG"
    label_dir = root / "Labels"
    
    all_signals, all_labels, all_subjects, all_runs = [], [], [], []

    for p_folder in sorted(eeg_dir.iterdir()):
        if not p_folder.is_dir(): continue
        print(f"Processing participant: {p_folder.name}")
        # exclude participant 1818
        if p_folder.name == "1818":
            print("Skipping participant 1818 due to known data issues.")
            continue
        
        subj_id = int(p_folder.name)
        label_file = label_dir / f"{p_folder.name}.csv"
        
        if not label_file.exists():
            print(f"Label file missing: {label_file}")
            continue

        # Load labels once per subject: shape usually (N_labels, 4_sessions)
        try:
            df_labels = pd.read_csv(label_file)
        except Exception as e:
            print(f"Error reading labels {label_file}: {e}")
            continue
        
        for session_idx in range(4):
            eeg_file = p_folder / f"eeg_data_exp_{session_idx}.csv"
            if not eeg_file.exists(): continue

            # 1. Robust Loading: Use pandas to handle empty strings/NaNs
            # We skip the first column (Timestamp)
            try:
                df_eeg = pd.read_csv(eeg_file)
                # Drop timestamp, handle NaNs via interpolation to preserve signal continuity
                raw_data = df_eeg.iloc[:, 1:].interpolate(method='linear').fillna(0).values.T
            except Exception as e:
                print(f"Failed to load {eeg_file}: {e}")
                continue
            
            # Extract labels for this session (column index session_idx)
            # Skeptic check: Ensure label indices match columns level_0 to level_3
            col_name = f"level_{session_idx}"
            if col_name not in df_labels.columns:
                # Fallback to integer indexing if column names differ
                session_labels = df_labels.iloc[:, session_idx].values
            else:
                session_labels = df_labels[col_name].values

            binary_labels = np.array([get_cognitive_load_label(l) for l in session_labels])
            
            # 2. Resampling & Preprocessing
            # Fix: ch_names must be a list of strings
            info = mne.create_info(ch_names=CH_NAMES, sfreq=ORIGINAL_FS, ch_types='eeg')
            raw = mne.io.RawArray(raw_data, info, verbose=False)
            
            # pick only 'AF7', 'AF8' channels as others are dirty
            pick_ch_names = ['AF7', 'AF8']
            raw = raw.pick_channels(pick_ch_names)

            
            # Bandpass filtering is highly recommended for NNs to remove DC offset/high-freq noise
            raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
            raw.resample(TARGET_FS, verbose=False)
            processed_data = raw.get_data()

            # 3. Windowing (Synchronization)
            # Dimension Check: raw_data (C, T) -> S_i (C, 2000)
            for i, label in enumerate(binary_labels):
                start = i * SAMPLES_PER_SEGMENT
                end = start + SAMPLES_PER_SEGMENT
                
                if end <= processed_data.shape[1]:
                    segment = processed_data[:, start:end]
                    # Z-score normalization per segment (crucial for neural net convergence)
                    # segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-6)
                    print("Appending segment with shape:", segment.shape, "and label:", label, f"(Subject {subj_id}, Session {session_idx}, Segment {i})")
                    
                    all_signals.append(segment)
                    all_labels.append(label)
                    all_subjects.append(subj_id)
                    all_runs.append(session_idx)
                    
    unique_subjects = np.unique(all_subjects)
    subject_to_idx = {orig_id: i for i, orig_id in enumerate(unique_subjects)}
    mapped_subjects = [subject_to_idx[s] for s in all_subjects]

    # 4. Dimension Finalization
    # Output X will be (N, C, T)
    X = torch.tensor(np.stack(all_signals), dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.long)
    sub = torch.tensor(mapped_subjects, dtype=torch.long)
    run = torch.tensor(all_runs, dtype=torch.long)

    return X, y, sub, run

if __name__ == "__main__":
    CLARE_PATH = "/mnt/pve/Turing-Storage2/CLARE"
    X, y, sub, run = process_clare(CLARE_PATH)
    
    dataset = {'X': X, 'y': y, 'ch_names': PICK_CH_NAMES, 'subjects': sub, 'runs': run}
    torch.save(dataset, "/mnt/pve/Rita-Storage-2/disentangleData/processed_data/clare_processed_thresh_5_cbramod_18Parts_2channels.pt")
    print(f"Saved {X.shape[0]} segments with shape {X.shape[1:]}")