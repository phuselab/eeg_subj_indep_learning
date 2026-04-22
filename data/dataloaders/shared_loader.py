import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from data.dataloaders.delegated_loader import DelegatedLoader
import torch

# CustomLoaderShared è ora un'istanza di torch.utils.data.Dataset
class CustomLoaderShared(Dataset):
    '''
    Custom data loader for shared EEG data with filtering and indexing capabilities.
    Risolve il Data Leakage calcolando/applicando mean/std solo sul training set.
    ---
    '''

    def __init__(self, data_dict, exclude_tasks=[], divisor: float = 100.0, exclude_subjects=[]):
        print("Initializing CustomLoaderShared...")
        self.exclude_tasks = exclude_tasks
        self.exclude_subjects = exclude_subjects

        # --- 1. Gestione Iniziale dei Dati ---
        if "data" not in data_dict:
            if "features" in data_dict:
                data_dict["data"] = data_dict.pop("features")
            elif "X" in data_dict:
                data_dict["data"] = data_dict.pop("X")
            else:
                raise KeyError("Data dictionary must contain 'data', 'features', or 'X' key")
        
        self.data_tensor = data_dict["data"]
        
        if "y" in data_dict:
            self.tasks = data_dict["y"].numpy()
        elif "tasks" in data_dict:
            self.tasks = data_dict["tasks"].numpy()
        elif "labels" in data_dict:
            self.tasks = data_dict["labels"].numpy()
        else:
            raise KeyError("Data dictionary must contain 'y' or 'labels' key")
        
        # Filtro task invalido (0 che corrisponde a index 4)
        valid_indices = np.where(~np.isin(self.tasks, self.exclude_tasks))[0]
        valid_indices_subject = np.where(~torch.isin(torch.tensor(data_dict["subjects"]), torch.tensor(self.exclude_subjects)))[0]
        valid_indices = np.intersect1d(valid_indices, valid_indices_subject)
        
        
        self.data_tensor = self.data_tensor[valid_indices]
        self.tasks = self.tasks[valid_indices]
        self.subjects = data_dict["subjects"].numpy()[valid_indices]
        self.runs = data_dict["runs"].clamp(min=1).numpy()[valid_indices]
         
        # Applica il filtro e sposta su CPU/Float
        # self.data = self.data_tensor.float().contiguous().detach().clone() 
        self.data = self.data_tensor.float().contiguous().detach().clone() / divisor
        
        self.subjects = np.ascontiguousarray(self.subjects)
        self.tasks = np.ascontiguousarray(self.tasks)
        self.runs = np.ascontiguousarray(self.runs)
        self.size = len(self.data)
        
        # print(f"Split '{split_subjects}': {len(self.unique_subjects_filter)} subjects selected. Total samples: {self.size}")

        # --- 4. Normalizzazione (Anti-Leakage) ---
        self.data_mean = self.data.mean().detach().clone().contiguous()
        self.data_std = self.data.std().detach().clone().contiguous()
        self.data_std[self.data_std == 0] = 1.0 
        print(f"Calculated new mean/std for training split.")

        # --- 5. ENCODING (La correzione è qui) ---
        
        # A. Subjects: Mappa Global ID (es. 70) -> Local Label (es. 0) per la Loss
        # Manteniamo self.subjects come ORIGINALI per riferimento, usiamo self.y_subjects per training
        unique_subjects_sorted = np.sort(np.unique(self.subjects))

        self.unique_subjects = unique_subjects_sorted.tolist()
        self.unique_tasks = np.sort(np.unique(self.tasks)).tolist()

        self.subject_map = {original_id: new_id for new_id, original_id in enumerate(unique_subjects_sorted)}
        self.y_subjects = np.array([self.subject_map[s] for s in self.subjects], dtype=np.int64)
        

        # B. Tasks: Mappa Tasks (es. 1, 2) -> Local Label (es. 0, 1)
        # Nota: Ho corretto il bug nel dizionario (prima era invertito id:new_id)
        unique_tasks_sorted = np.sort(np.unique(self.tasks))
        self.task_map = {original_id: new_id for new_id, original_id in enumerate(unique_tasks_sorted)}
        self.y_tasks = np.array([self.task_map[t] for t in self.tasks], dtype=np.int64)

        # C. Runs
        unique_runs_sorted = np.sort(np.unique(self.runs))
        self.run_map = {original_id: new_id for new_id, original_id in enumerate(unique_runs_sorted)}
        self.y_runs = np.array([self.run_map[r] for r in self.runs], dtype=np.int64)

        # --- 6. Index Mapping ---
        # Usiamo gli ID originali per le chiavi dei dizionari (più intuitivo per property sampling)
        self.subject_indices = defaultdict(list)
        self.task_indices = defaultdict(list)
        self.run_indices = defaultdict(list)
        self.full_indices = defaultdict(lambda: defaultdict(list))

        for i, (s, t, r) in enumerate(zip(self.subjects, self.tasks, self.runs)):
            self.subject_indices[s].append(i)
            self.task_indices[t].append(i)
            self.run_indices[r].append(i)
            self.full_indices[s][t].append(i)
        
        # Convert to Tensor for fast indexing
        # self.subjects rimangono gli ID originali (per metadata)
        self.subjects_tensor = torch.as_tensor(self.subjects, dtype=torch.long)
        # self.y_subjects sono le LABEL per il training (0..N-1)
        self.y_subjects_tensor = torch.as_tensor(self.y_subjects, dtype=torch.long)
        self.y_tasks_tensor = torch.as_tensor(self.y_tasks, dtype=torch.long)
        self.y_runs_tensor = torch.as_tensor(self.y_runs, dtype=torch.long)

        self.reset_sample_counts()
    
    # --- Metodi Essenziali di PyTorch Dataset ---
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        
        # Applica normalizzazione
        normalized_data = (sample_data - self.data_mean) / (self.data_std + 1e-6)

        return normalized_data, self.y_subjects_tensor[idx], self.y_tasks_tensor[idx], self.y_runs_tensor[idx], idx

    # --- Metodi Aggiuntivi per la DelegatedLoader (Funzionalità Invariata) ---

    def reset_sample_counts(self):
        self.total_samples = 0
    
    def get_dataloader(self, num_total_samples=None, batch_size=None, property=None, random_sample=True, by_subject=False):
        # L'assunzione è che DelegatedLoader esista e funzioni come un wrapper del Dataset
        if num_total_samples is None:
            num_total_samples = self.size
        if by_subject:
            return DataLoader(DelegatedLoader(self, batch_size=batch_size, length=num_total_samples, by_subject=True), batch_size=None, pin_memory=True, num_workers=0)
        

        delegated_loader = DelegatedLoader(self, property=property, batch_size=batch_size if random_sample else None, length=num_total_samples)
        if not random_sample and batch_size is not None:
            return DataLoader(delegated_loader, batch_size=batch_size, pin_memory=True, num_workers=0)
        return DataLoader(delegated_loader, batch_size=None, pin_memory=True, num_workers=0)
    
    def sample_by_condition(self, subjects, tasks):
        samples = []
        for s, t in zip(subjects, tasks):
            i = np.random.choice(self.full_indices[s][t])
            samples.append(i)
        samples = np.array(samples)
        self.total_samples += len(samples)
        # NOTA: Qui si restituisce il dato NON normalizzato (coerente con l'originale, ma da verificare se voluto)
        return self.data[samples]    
    
    def sample_by_property(self, property, shift=0):
        
        property = property.lower()
        if property.startswith("s"):
            property_indices = self.subject_indices
        elif property.startswith("t"):
            property_indices = self.task_indices
        elif property.startswith("r"):
            property_indices = self.run_indices
        else:
            raise ValueError("Invalid property")
        
        # shift the property indices by 'shift' positions
        property_keys = list(property_indices.keys())
        shifted_keys = property_keys[shift:] + property_keys[:shift]
        property_indices = {k: property_indices[k] for k in shifted_keys}
        
        samples = []
        for indices in property_indices.values():
            i = np.random.choice(indices)
            samples.append(i)
        samples = np.array(samples)
        self.total_samples += len(samples)
        # NOTA: Anche qui si restituisce il dato NON normalizzato
        return samples, self.data[samples], self.y_subjects_tensor[samples], self.y_tasks_tensor[samples], self.y_runs_tensor[samples]
    
    def sample_batch(self, batch_size):
        samples = np.random.randint(0, self.size, size=batch_size)
        samples_torch = torch.as_tensor(samples, dtype=torch.long)
        self.total_samples += batch_size
        # NOTA: Anche qui si restituisce il dato NON normalizzato
        return samples, self.data[samples_torch], self.y_subjects_tensor[samples_torch], self.y_tasks_tensor[samples_torch], self.y_runs_tensor[samples_torch]
    
    def get_batch_by_subject(self, subject_id, batch_size):
        if subject_id not in self.subject_indices:
            raise ValueError(f"Subject ID {subject_id} not found in dataset")
        indices = self.subject_indices[subject_id]
        if len(indices) < batch_size:
            raise ValueError(f"Not enough samples for subject {subject_id} to create a batch of size {batch_size}")
        selected_indices = np.random.choice(indices, size=batch_size, replace=False)
        selected_indices_torch = torch.as_tensor(selected_indices, dtype=torch.long)
        self.total_samples += batch_size
        # NOTA: Anche qui si restituisce il dato NON normalizzato
        return selected_indices, self.data[selected_indices_torch], self.y_subjects_tensor[selected_indices_torch], self.y_tasks_tensor[selected_indices_torch], self.y_runs_tensor[selected_indices_torch]
    
    def iterator(self):
        for i in range(self.size):
            self.total_samples += 1
            # NOTA: Anche qui si restituisce il dato NON normalizzato
            yield i, self.data[i], self.y_subjects_tensor[i], self.y_tasks_tensor[i], self.y_runs_tensor[i]

    def batch_iterator(self, batch_size, length):
        # 1. Crea una lista di indici mescolati (es. [5, 0, 9, 2, ...])
        # Questo si fa UNA volta all'inizio dell'epoca
        indices = np.random.permutation(self.size)
        
        # 2. Scorre gli indici a blocchi (batch)
        for start_idx in range(0, self.size, batch_size):
            # Prende i prossimi N indici
            if length is not None and start_idx >= length:
                break
            batch_indices = indices[start_idx : start_idx + batch_size]
            
            # Converte in tensor per indicizzare i dati
            samples_torch = torch.as_tensor(batch_indices, dtype=torch.long)
            
            # Restituisce il batch
            yield (batch_indices, 
                self.data[samples_torch], 
                self.y_subjects_tensor[samples_torch], 
                self.y_tasks_tensor[samples_torch], 
                self.y_runs_tensor[samples_torch])
    
    def property_iterator(self, property, length):
        num_samples = 0
        num_per = 0
        while True:
            if length is not None and num_samples + num_per >= length:
                break
            yield self.sample_by_property(property)
            if length is not None:
                if num_per == 0:
                    property = property.lower()
                    if property.startswith("s"):
                        num_per = len(self.subject_indices)
                    elif property.startswith("t"):
                        num_per = len(self.task_indices)
                    elif property.startswith("r"):
                        num_per = len(self.run_indices)
                    else:
                        raise ValueError("Invalid property")
                num_samples += num_per

                
    def subject_batch_iterator(self, batch_size, length=None):
        """
        Iterator custom for Latent Alignment.
        Guarantees that every batch contains only samples from the same subject.
        """
        batches = []
        # 1. Group indices in batch for each subject
        for subj_id, indices in self.subject_indices.items():
            # Copy and shuffle indices for this subject
            indices_copy = list(indices)
            np.random.shuffle(indices_copy)
            
            # Divide into batches
            for i in range(0, len(indices_copy), batch_size):
                batch = indices_copy[i : i + batch_size]
                
                if len(batch) >= batch_size // 2:
                    batches.append(batch)
        
        # 2. Shuffle the batches to mix subjects
        np.random.shuffle(batches)
        
        # 3. Yield batch
        num_samples_yielded = 0
        for batch_indices in batches:
            if length is not None and num_samples_yielded >= length:
                break
                
            samples_torch = torch.as_tensor(batch_indices, dtype=torch.long)
            
            # NOTE: Return non normalized data
            # (or normalized /100 as defined in init). 
            yield (batch_indices, 
                   self.data[samples_torch], 
                   self.y_subjects_tensor[samples_torch], 
                   self.y_tasks_tensor[samples_torch], 
                   self.y_runs_tensor[samples_torch])
                   
            num_samples_yielded += len(batch_indices)