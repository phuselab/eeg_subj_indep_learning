from scipy.signal import butter, iirnotch, resample_poly, lfilter, sosfilt
import mne
import torch
import numpy as np
import scipy.io as sio


SAMPLING_RATE = 128  # DREAMER EEG native SR
CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
THRESHOLD = 3
TARGET_SAMPLING_RATE = 200
SEGMENT_SECONDS = 10
SAMPLES_PER_SEGMENT = SEGMENT_SECONDS * TARGET_SAMPLING_RATE # 4000


def get_quadrant_label2(v, a):
    """
    Categorizes Valence and Arousal (1-5 scale) into 4 quadrants.
    """
    if v > 3 and a > 3: return 0    # HVHA (Happy)
    elif v <= 3 and a > 3: return 1  # LVHA (Angry)
    elif v <= 3 and a <= 3: return 2 # LVLA (Sad)
    else: return 3                   # HVLA (Relaxed)
    
def get_quadrant_label(v, a):
    """
    Categorizes Valence and Arousal (1-5 scale) into 4 quadrants. Ignnore 3 and remove from dataset if either v or a is exactly 3.
    """
    if v == 3 or a == 3:
        return None  # Return None to indicate this sample should be ignored
    if v > 3 and a > 3: return 0    # HVHA (Happy)
    elif v < 3 and a > 3: return 1  # LVHA (Angry)
    elif v < 3 and a < 3: return 2 # LVLA (Sad)
    else: return 3                   # HVLA (Relaxed)
    


def process_dreamer(file_path, window_seconds=60, sampling_rate=128):
    """
    Robustly loads DREAMER.mat and extracts EEG segments with quadrant labels.
    """
    print(f"Loading {file_path}...")
    try:
        data_mat = sio.loadmat(file_path)
    except Exception as e:
        raise FileNotFoundError(f"Could not load .mat file: {e}")

    # Enter the nested MATLAB structure
    # DREAMER -> Data (1, 23)
    dreamer = data_mat['DREAMER'][0, 0]
    data_struct = dreamer['Data'][0] # Array of 23 subjects
    
    all_signals = []
    all_labels = []
    all_subjects = []
    
    # MNE Setup: Create info structure once
    info = mne.create_info(ch_names=CHANNELS, sfreq=SAMPLING_RATE, ch_types='eeg')
    
    #target_window_size = window_seconds * TARGET_SAMPLING_RATE
    #window_size = window_seconds * sampling_rate

    for s_idx in range(23):
        subject_data = data_struct[s_idx]
        # Access EEG trials and self-assessment scores
        # scores are (18, 1) or (1, 18)
        eeg_struct = subject_data['EEG'][0, 0]
        v_scores = subject_data['ScoreValence'][0, 0]
        a_scores = subject_data['ScoreArousal'][0, 0]
        stimuli_list = eeg_struct['stimuli'][0, 0]
        
        # Access the stimuli and baseline object arrays
        #baseline_list = eeg_struct['baseline'][0, 0]

        print(f"Processing Subject {s_idx + 1}/23...")

        for t_idx in range(18):
            # Extract raw EEG (Samples, 14)
            eeg_raw = stimuli_list[t_idx, 0]
            
            if eeg_raw.size == 0:
                continue
            
            raw = mne.io.RawArray(eeg_raw.T, info)
            raw.set_eeg_reference('average')
            raw.filter(l_freq=0.3, h_freq=None)
            raw.notch_filter(freqs=60) 
            raw.resample(TARGET_SAMPLING_RATE)
            
            processed_data = raw.get_data()
            
            # Determine labels
            v_val, a_val = float(v_scores[t_idx]), float(a_scores[t_idx])
            label = get_quadrant_label(v_val, a_val) # Some samples may be None if v or a is exactly 3
            
            if label is None:
                continue # Skip samples with valence or arousal exactly 3
            
            # 6. Segment the whole video into 20s blocks
            total_samples = processed_data.shape[1]
            num_segments = total_samples // SAMPLES_PER_SEGMENT
            
            for i in range(num_segments):
                start = i * SAMPLES_PER_SEGMENT
                end = start + SAMPLES_PER_SEGMENT
                
                segment = processed_data[:, start:end]
                
                all_signals.append(segment)
                all_labels.append(label)
                all_subjects.append(s_idx + 1)
            

    # Convert to Tensors
    X = torch.tensor(np.array(all_signals), dtype=torch.float32)
    y = torch.tensor(np.array(all_labels), dtype=torch.long)
    subjects = torch.tensor(np.array(all_subjects), dtype=torch.int)
    runs = torch.zeros_like(y) # Dummy runs for pipeline compatibility
    
    return X, y, subjects, runs

# ============================================================================
# EXECUTION & SAVING 
# ============================================================================

if __name__ == "__main__":
    path_to_mat = "/mnt/turingDatasets/DREAMER/DREAMER.mat" 
    path_to_mat = "/mnt/pve/Turing-Storage2/DREAMER/DREAMER.mat" # Alternative path if the first one doesn't work
    X, y, subjects, runs = process_dreamer(path_to_mat)
    
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
        'ch_names': CHANNELS,
        'subjects': subjects, 
        'runs': runs,
    }
    
    #torch.save(dataset_dict, "/home/user/projects/eeg_disentanglement/data/processed_data/dreamer_eeg_quadrants_wo_cross_10s_cbramod.pt")
    torch.save(dataset_dict, "/mnt/pve/Rita-Storage-2/disentangleData/processed_data/dreamer_eeg_quadrants_wo_cross_10s_cbramod2.pt")
    print("\nSuccess: Processed data saved.")