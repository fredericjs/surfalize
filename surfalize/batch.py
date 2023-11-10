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
    
    def __init__(self, filepaths, additional_data=None):
        """
        Initializies the Batch object from filepaths to topography files. The batch object
        is used to calculate quantitative surface parameters for a batch of supplied files
        at once and return them as a pd.DataFrame. If the caller wants to supply additional
        parameters for each file, such as fabrication data, they can specify the path to an
        excel file containing that data using the 'additional_data' keyword argument.
        The excel file should contain a column 'filename' of the format 'name.extension'.
        Otherwise, an arbitrary number of additional columns can be supplied.

        Parameters
        ----------
        filepaths: list[pathlib.Path | str]
            List of filepaths of topography files
        additional_data: str, pathlib.Path
            Path to an excel file containing additional parameters, such as
            input parameters. Excel file must contain a column 'filename' with
            the filename including the file extension. Otherwise, an arbitrary
            number of additional columns can be supplied.
        
        """
        self._filepaths = [Path(file) for file in filepaths]
        self._additional_data = additional_data
        self._load_surfaces()
        
    def _load_surfaces(self):
        """
        Loads each files into a surfalize.Surface object.
        """
        self._surfaces = dict()
        for file in tqdm(self._filepaths, desc='Loading files'):
            self._surfaces[file] = Surface.load(file)
            
    def fill_nonmeasured(self, mode='nearest'):
        """
        Calls the 'fill_nonmeasured' method of each surface in the batch with the inplace argument
        specified as True.
        """
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
    
    def rotate(self, angle):
        for surface in tqdm(self._surfaces.values(), desc='Rotating'):
            surface.rotate(angle, inplace=True)
        return self
    
    def align(self):
        for surface in tqdm(self._surfaces.values(), desc='Aligning'):
            surface.align(inplace=True)
        return self

    def roughness_parameters(self, parameters=None):
        if parameters is None:
            parameters = list(Surface.AVAILABLE_PARAMETERS)
        df = pd.DataFrame({'filename': [file.name for file in self._filepaths]})
        df = df.set_index('filename')
        df[list(parameters)] = np.nan
        for file, surface in tqdm(self._surfaces.items(), desc='Calculating parameters'):
            results = surface.roughness_parameters(parameters)
            for k, v in results.items():
                df.loc[file.name][k] = v
        df = df.reset_index()
        if self._additional_data is None:
            return df
        dfin = pd.read_excel(self._additional_data)
        return pd.merge(dfin, df, on='filename')