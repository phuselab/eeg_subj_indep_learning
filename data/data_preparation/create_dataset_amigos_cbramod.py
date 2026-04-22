from scipy.signal import butter, iirnotch, resample_poly, lfilter, sosfilt
import mne
import torch
import numpy as np
import scipy.io as sio
import os


SAMPLING_RATE = 128  # AMIGOS EEG native SR
CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
TARGET_SAMPLING_RATE = 200
SEGMENT_SECONDS = 10
SAMPLES_PER_SEGMENT = SEGMENT_SECONDS * TARGET_SAMPLING_RATE # 4000


def get_quadrant_label(v, a):
    """
    Categorizes Valence and Arousal (1-5 scale) into 4 quadrants.
    """
    if v == 5 or a == 5:
        return None  # o np.nan

    if v > 5 and a > 5:
        return 0  # HVHA
    elif v < 5 and a > 5:
        return 1  # LVHA
    elif v < 5 and a < 5:
        return 2  # LVLA
    else:
        return 3  # HVLA
    

def process_single_amigos_file(file_path):
    # data is a NpzFile 'Data_Preprocessed_P14.npz' with keys: VideoIDs, joined_data, labels_ext_annotation, labels_selfassessment
    data = np.load(file_path, allow_pickle=True)
    subject_id = int(os.path.basename(file_path).split("Preprocessed_P")[1].split(".")[0])
    
    # Extract EEG data and labels
    eeg_data = data['joined_data'][:15] # List of 16 arrays, each of shape (14, T) where T is the number of time points for that video
    labels = data['labels_selfassessment'][:15]  # List of 16 or more (crop in case), each of shape (21), use only the first 2 values for arousal and valence
    video_ids = data['VideoIDs'][:15]  # List of 16 video IDs, can be used to track which video corresponds to which segment

    info = mne.create_info(ch_names=CHANNELS, sfreq=SAMPLING_RATE, ch_types='eeg')

    all_signals = []
    all_labels = []
    all_subjects = []
    all_runs = []  # AMIGOS does not have runs, but we can create a dummy run label if needed (e.g., all zeros)
    # Process each video segment
    for i in range(len(eeg_data)):
        video_eeg = eeg_data[i]  # Shape (14, T)       
        if np.isnan(video_eeg).any():
            print(f"NaN già presenti nel video {i}: {np.isnan(video_eeg).sum()}") 
            continue  # Skip this video if it contains NaN values
        
        if video_eeg.shape[1] == 17:
            # transpose to get shape (17, N) and then select the first 14 channels
            video_eeg = video_eeg.T
        elif video_eeg.shape[0] == 17:
            # already in shape (17, N), just select the first 14 channels
            pass
        else:
            print(f"Unexpected shape for video {i}: {video_eeg.shape}, skipping this video.")
            continue
        video_eeg = video_eeg[:14, :]  # Ensure we only have the 14 EEG channels

        raw = mne.io.RawArray(video_eeg, info)
        raw.set_eeg_reference('average')
        # raw.filter(l_freq=0.3, h_freq=None)
        # raw.notch_filter(freqs=60) 
        raw.resample(TARGET_SAMPLING_RATE)
        
        video_eeg_resampled = raw.get_data()

        # check if there are nan values in the resampled data
        if np.isnan(video_eeg_resampled).any():
            print(f" the count of nan values in the resampled data for video {i} is: {np.isnan(video_eeg_resampled).sum()}")
            continue  # Skip this video if it contains NaN values after resampling

        print(labels[i].shape)  # Debug: Check the shape of the labels for each video
        v = labels[i][0][0]  # Valence
        a = labels[i][0][1]  # Arousal
        label = get_quadrant_label(v, a)
        
        if label is not None:
            # Segment the resampled EEG data into 30-second segments
            total_samples = video_eeg_resampled.shape[1]
            for start in range(0, total_samples, SAMPLES_PER_SEGMENT):
                end = start + SAMPLES_PER_SEGMENT
                if end <= total_samples:
                    segment = video_eeg_resampled[:, start:end]  # Shape (14, SAMPLES_PER_SEGMENT)
                    all_signals.append(segment)
                    all_labels.append(label)
                    all_subjects.append(subject_id)
                    all_runs.append(0)  # Dummy run label since AMIGOS does not have runs
    return all_signals, all_labels, all_subjects, all_runs




def process_amigos(file_path):
    """
    Iterate in the folder to read .npz files, robustly load them and extract EEG segments with quadrant labels.
    """
    print(f"Loading {file_path}...")
    all_signals = []
    all_labels = []
    all_subjects = []
    all_runs = []  # AMIGOS does not have runs, but we can create a dummy run label if needed (e.g., all zeros)
    for file in os.listdir(file_path):
        if file.endswith(".npz"):
            # extract the subject ID from the filename, e.g., "Data_preprocessed_P01.npz" -> 1
            path_to_file = os.path.join(file_path, file)
            signals, labels, subjects, runs = process_single_amigos_file(path_to_file)
            all_signals.extend(signals)
            all_labels.extend(labels)
            all_subjects.extend(subjects)
            all_runs.extend(runs)

    # Convert lists to tensors
    X = torch.tensor(all_signals, dtype=torch.float32)  # Shape [N, 14, SAMPLES_PER_SEGMENT]
    y = torch.tensor(all_labels, dtype=torch.long)  # Shape [N]
    subjects = torch.tensor(all_subjects, dtype=torch.long)  # Shape [N]
    runs = torch.tensor(all_runs, dtype=torch.long)  # Shape [N]

    return X, y, subjects, runs  # No runs in AMIGOS dataset


# ============================================================================
# EXECUTION & SAVING 
# ============================================================================

if __name__ == "__main__":
    path_to_files = "/mnt/turingDatasets/AMIGOS/pre_processed_py" 
    
    X, y, subjects, runs = process_amigos(path_to_files)
    
    # Final data validation
    print("\n--- Final Dataset Summary ---")
    print(f"Total Samples: {X.shape[0]}")
    print(f"Tensor Shape [N, Channels, Time]: {X.shape}")
    
    # Class Distribution Analysis
    unique, counts = np.unique(y.numpy(), return_counts=True)
    class_map = {0: "HVHA", 1: "LVHA", 2: "LVLA", 3: "HVLA"}
    for u, c in zip(unique, counts):
        print(f"{class_map[u]}: {c} samples")

    dataset_dict = {
        'X': X,
        'y': y,
        'subjects': subjects, 
        'runs': runs,
    }
    
    torch.save(dataset_dict, "/home/user/projects/eeg_disentanglement/data/processed_data/amigos_eeg_quadrants_10s_cbramod.pt")
    print("\nSuccess: Processed data saved.")