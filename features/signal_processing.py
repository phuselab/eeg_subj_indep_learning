import torch
import torch.nn as nn
import numpy as np
import scipy
import scipy.signal as signal
from scipy.stats import entropy
from scipy.fft import fft, fftfreq
import logging
from typing import Dict, Optional

class SignalProcessingBackbone(nn.Module):
    """
    Extracts traditional time-domain and frequency-domain features 
    and returns a fixed-length feature vector.
    """
    def __init__(self, n_chans: int, n_times: int, sfreq: float = 128.0):
        super().__init__()
        
        self.sfreq = sfreq
        self.n_chans = n_chans
        self.feature_names = None  # To be populated after first forward pass
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # The feature dimension is determined by the number of features extracted per channel.
        # Based on the advanced extraction logic, this is approximately 10-15 features/channel.
        # We will dynamically set feature_dim after the first pass.
        self.feature_dim = 0 
        
    def _extract_channel_features(self, ch_data: np.ndarray) -> Dict[str, float]:
        """Core logic to extract advanced temporal and spectral features for one channel."""
        features = {}
        
        # 1. TEMPORAL FEATURES
        features['mean'] = np.mean(ch_data)
        features['std'] = np.std(ch_data)
        features['skew'] = scipy.stats.skew(ch_data)
        features['kurtosis'] = scipy.stats.kurtosis(ch_data) 
        features['rms'] = np.sqrt(np.mean(ch_data**2))
        
        # 2. FREQUENCY FEATURES (PSD via Welch's method)
        freq_bands = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12),
            'beta': (12, 30), 'gamma': (30, 45)
        }
        
        # nperseg should be <= length of data. 
        nperseg = min(256, len(ch_data)) # TODO make this data dependent?
        freqs, psd = signal.welch(ch_data, fs=self.sfreq, nperseg=nperseg)
        total_power = np.trapezoid(psd, freqs)

        for band_name, (low_freq, high_freq) in freq_bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.trapezoid(psd[band_mask], freqs[band_mask])
            features[f'{band_name}_power'] = band_power
            
            # Relative Power
            features[f'{band_name}_relative'] = band_power / total_power if total_power > 1e-6 else 0
            
        # 3. ENTROPY & PEAK
        features['entropy'] = entropy(np.histogram(ch_data, bins=10)[0])
        peak_freq_idx = np.argmax(psd)
        features['peak_freq'] = freqs[peak_freq_idx]

        return features

    def forward(self, x: torch.Tensor, save_path: Optional[str] = None) -> torch.Tensor:
        """
        Converts (B, C, T) raw EEG segments into (B, F) feature vectors.
        """
        logging.info("WE WILL USE Z_NORM IN THE END!!!...")
        B, C, T = x.shape
        x_np = x.cpu().numpy()
        all_features_np = []
        print(f"Extracting Signal Processing features from input of shape: {x.shape}")
        
        if C != self.n_chans:
             logging.warning(f"Channel count mismatch: {C} in data vs {self.n_chans} in config.")

        for seg_idx in range(B):
            print(f"  Processing segment {seg_idx+1}/{B}", end='\r')
            segment_features = {}
            # Iterate through channels and extract features
            for ch_idx in range(C):
                try:
                    ch_features = self._extract_channel_features(x_np[seg_idx, ch_idx, :])
                    for k, v in ch_features.items():
                        segment_features[f'ch{ch_idx}_{k}'] = v
                except Exception as e:
                    logging.error(f"Error extracting features for segment {seg_idx}, channel {ch_idx}: {e}")
                    # Insert default zeros if calculation fails
                    if not segment_features: segment_features = {f'ch{ch_idx}_zero': 0.0}
            
            all_features_np.append(segment_features)
            print(" " * 50, end='\r')  # Clear line after processing

        # 4. Final Conversion to Tensor and Dimension Check
        if not all_features_np:
            return torch.zeros(B, self.feature_dim)
            
        # Convert list of dicts to NumPy array (ensuring consistent feature ordering)
        if self.feature_names is None:
            self.feature_names = sorted(all_features_np[0].keys())
            self.feature_dim = len(self.feature_names)
        
        feature_vector = np.array([
            [d.get(name, 0.0) for name in self.feature_names] for d in all_features_np
        ], dtype=np.float32)
        
        # Z-normalize the features 
        z_norm = True 
        if z_norm:
            feature_vector = (feature_vector - np.mean(feature_vector, axis=0)) / (
                np.std(feature_vector, axis=0) + 1e-6)
        
        # add z_norm to save_path
        # split before .pt and insert _znorm
        save_path = f"{save_path[:-3]}_znorm.pt" if save_path else None
        
        if save_path:
            try:
                # Per salvare i feature e i nomi per il debug/ricarico
                feature_dict = {
                    'features': feature_vector,
                    'feature_names': self.feature_names
                }
                torch.save(feature_dict, save_path)
                logging.info(f"Signal Processing features saved to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save Signal Processing features to {save_path}: {e}")
        
        return torch.from_numpy(feature_vector).to(x.device)

