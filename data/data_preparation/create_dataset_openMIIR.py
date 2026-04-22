import os
import torch
import mne
import numpy as np
from pathlib import Path

# Parameters
TARGET_FS = 200
WINDOW_SEC = 4
SAMPLES_PER_WINDOW = int(WINDOW_SEC * TARGET_FS)  # 800
STRIDE_SAMPLES = int(2 * TARGET_FS)               # 400  (50% overlap)

# Event IDs are 3-digit: hundreds=stimulus, tens+units=condition
# e.g. 111 -> stimulus=11, condition=1
# e.g. 241 -> stimulus=24, condition=1
STIM_MAP = {1:0, 2:1, 3:2, 4:3, 11:4, 12:5, 13:6, 14:7, 21:8, 22:9, 23:10, 24:11}
DURATIONS = {
    1: 13.3, 2: 7.7, 3: 9.7, 4: 11.6,
    11: 13.5, 12: 7.7, 13: 9.0, 14: 12.2,
    21: 8.3, 22: 16.0, 23: 9.2, 24: 6.9
}

def decode_trigger(trigger):
    """
    Decode 3-digit trigger into (stim_id, condition).
    Format: stim_id = trigger // 10, condition = trigger % 10
    Examples: 111 -> (11, 1), 241 -> (24, 1), 34 -> (3, 4)
    """
    stim_id = trigger // 10
    condition = trigger % 10
    return stim_id, condition

def process_openmiir_fixed_64(root_path):
    root = Path(root_path)
    raw_files = sorted(list(root.rglob("*-raw.fif")))

    all_signals, all_stims, all_conds, all_subjects, all_runs = [], [], [], [], []

    for subj_idx, file_path in enumerate(raw_files):
        print(f"\n--- Processing {file_path.name} ---")

        try:
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

            # Pick EEG channels by type — this is robust across subjects
            eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
            eeg_ch_names = [raw.ch_names[i] for i in eeg_picks]
            print(f"Found {len(eeg_picks)} EEG channels: {eeg_ch_names[:5]}...")

            if len(eeg_picks) != 64:
                print(f"WARNING: Expected 64 EEG channels, got {len(eeg_picks)}. Forcing first 64.")
                # Fallback: force first 64 channels to EEG type and use them
                eeg_picks = list(range(64))
                mapping = {raw.ch_names[i]: 'eeg' for i in eeg_picks}
                raw.set_channel_types(mapping)
                eeg_ch_names = [raw.ch_names[i] for i in eeg_picks]

            # Find events BEFORE resampling (on original 512 Hz data)
            events = mne.find_events(
                raw, stim_channel='STI 014',
                shortest_event=1, verbose=False
            )
            print(f"Found {len(events)} events. Unique triggers: {sorted(set(events[:, 2]))}")

            # Filter and resample only EEG channels
            old_fs = raw.info['sfreq']
            raw.filter(l_freq=1.0, h_freq=40.0, picks=eeg_picks, verbose=False)
            raw.resample(TARGET_FS, verbose=False)

            # Rescale event sample indices to new sampling rate
            events[:, 0] = np.round(events[:, 0] * (TARGET_FS / old_fs)).astype(int)

            trial_counts = {}
            segments_in_file = 0

            for i in range(len(events)):
                onset_sample, _, trigger = events[i]
                stim_id, condition = decode_trigger(trigger)

                if stim_id not in STIM_MAP:
                    continue  # Skip non-stimulus events (1000, 1111, 2001, etc.)

                total_dur_samples = int(DURATIONS[stim_id] * TARGET_FS)
                trial_key = (stim_id, condition)

                curr_start = onset_sample
                while (curr_start + SAMPLES_PER_WINDOW) <= (onset_sample + total_dur_samples):
                    end_idx = curr_start + SAMPLES_PER_WINDOW

                    if end_idx <= raw.n_times:
                        segment = raw.get_data(picks=eeg_picks, start=curr_start, stop=end_idx)

                        if segment.shape == (64, SAMPLES_PER_WINDOW):
                            # Z-score normalization per segment
                            segment = (segment - segment.mean()) / (segment.std() + 1e-6)

                            all_signals.append(segment)
                            all_stims.append(STIM_MAP[stim_id])
                            all_conds.append(condition)
                            all_subjects.append(subj_idx)
                            all_runs.append(trial_counts.get(trial_key, 0))
                            segments_in_file += 1

                    curr_start += STRIDE_SAMPLES

                # Increment trial count after processing all windows for this trial
                trial_counts[trial_key] = trial_counts.get(trial_key, 0) + 1

            print(f"Done. Extracted {segments_in_file} segments.")

        except Exception as e:
            import traceback
            print(f"Failed {file_path.name}: {e}")
            traceback.print_exc()

    if not all_signals:
        print("No signals extracted.")
        return None

    X   = torch.tensor(np.stack(all_signals), dtype=torch.float32)
    y_s = torch.tensor(all_stims,    dtype=torch.long)
    y_c = torch.tensor(all_conds,    dtype=torch.long)
    sub = torch.tensor(all_subjects, dtype=torch.long)
    run = torch.tensor(all_runs,     dtype=torch.long)

    return X, y_s, y_c, sub, run, eeg_ch_names

if __name__ == "__main__":
    DATA_PATH = "/mnt/pve/Turing-Storage2/OpenMIIR-RawEEG_v1"
    SAVE_PATH = "/mnt/pve/Rita-Storage-2/disentangleData/processed_data/openmiir_force64_3.pt"

    res = process_openmiir_fixed_64(DATA_PATH)
    if res:
        X, y_s, y_c, sub, run, chs = res
        dataset = {
            'X': X, 'y': y_s, 'y_condition': y_c,
            'subjects': sub, 'runs': run, 'ch_names': chs
        }
        torch.save(dataset, SAVE_PATH)
        print(f"\nSUCCESS! Shape: {X.shape}")
        print(f"Stimuli:    {y_s.unique()}")
        print(f"Conditions: {y_c.unique()}")
        print(f"Subjects:   {sub.unique()}")