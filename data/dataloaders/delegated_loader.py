import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, IterableDataset


class DelegatedLoader(IterableDataset):
    '''
    Wrapper to delegate loading to custom loader with optional batching/property filtering.'''
    
    def __init__(self, loader, property=None, batch_size=None, length=None, by_subject=False):
        self.loader = loader
        self._property = property
        self._batch_size = batch_size
        self._length = length
        self._by_subject = by_subject
    
    def __len__(self):
        if self._batch_size is not None or self._property is not None:
            if self._length is not None:
                if self._batch_size is not None:
                    return self._length // self._batch_size
                return self._length
            return None
        return len(self.loader)
    
    def __iter__(self):
        if self._by_subject:
            return self.loader.subject_batch_iterator(batch_size=self._batch_size, length=self._length)
        if self._batch_size is not None or self._property is not None:
            if self._batch_size is not None:
                return self.loader.batch_iterator(self._batch_size, self._length)
            elif self._property is not None:
                return self.loader.property_iterator(self._property, self._length)
        else:
            return self.loader.iterator()

    def sample_by_property(self, property, shift=0):
        return self.loader.sample_by_property(property, shift=shift)

    def sample_batch(self, batch_size):
        return self.loader.sample_batch(batch_size)
