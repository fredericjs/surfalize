from pathlib import Path
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
from .surface import Surface

try:
    # Optional import
    from tqdm import tqdm
except ImportError:
    logger.info('tqdm not found, no progessbars will be shown.')
    # If tqdm is not defined, replace with dummy function
    tqdm = lambda x, *args, **kwargs: x
    

#TODO batch image export
class Batch:
    
    def __init__(self, filepaths):
        """
        Initializies the Batch object from filepaths to topography files.

        Parameters
        ----------
        filepaths: list of filepaths
        """
        self._filepaths = [Path(file) for file in filepaths]
        self._load_surfaces()
        
    def _load_surfaces(self):
        self._surfaces = dict()
        for file in tqdm(self._filepaths, desc='Loading files'):
            self._surfaces[file] = Surface.load(file)
            
    def fill_nonmeasured(self, mode='nearest'):
        for surface in tqdm(self._surfaces.values(), desc='Filling non-measured points'):
            surface.fill_nonmeasured(mode=mode, inplace=True)
        return self
            
    def level(self):
        for surface in tqdm(self._surfaces.values(), desc='Leveling'):
            surface.level(inplace=True)
        return self
            
    def filter(self, cutoff, *, mode, cutoff2=None, inplace=False):
        for surface in tqdm(self._surfaces.values(), desc='Filtering'):
            surface.filter(cutoff, mode=mode, cutoff2=None, inplace=True)
        return self

    def roughness_parameters(self, parameters=None):
        if parameters is None:
            parameters = list(Surface.AVAILABLE_PARAMETERS)
        df = pd.DataFrame({'filepath': [file.name for file in self._filepaths]})
        df = df.set_index('filepath')
        df[list(parameters)] = np.nan
        for file, surface in tqdm(self._surfaces.items(), desc='Calculating parameters'):
            results = surface.roughness_parameters(parameters)
            for k, v in results.items():
                df.loc[file.name][k] = v
        return df.reset_index()