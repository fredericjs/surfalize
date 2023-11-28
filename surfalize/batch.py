from multiprocessing import Pool
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

class _Operation:

    def __init__(self, identifier, args=None, kwargs=None):
        self.identifier = identifier
        self.args = tuple() if args is None else args
        self.kwargs = dict() if kwargs is None else kwargs

    def execute_on(self, surface):
        method = getattr(surface, self.identifier)
        method(*self.args, **self.kwargs)

class _Parameter:

    def __init__(self, identifier, args=None, kwargs=None):
        self.identifier = identifier
        self.args = tuple() if args is None else args
        self.kwargs = dict() if kwargs is None else kwargs

    def calculate_from(self, surface):
        method = getattr(surface, self.identifier)
        return method(*self.args, **self.kwargs)


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
        self._operations = []
        self._parameters = []
        
    def execute(self, multiprocessing=False):
        def task(filepath, operations, parameters):
            surface = Surface.load(filepath)
            for operation in operations:
                operation.execute_on(surface)
            results = dict()
            for parameter in parameters:
                result = parameter.calculate_from(surface)
                results[parameter.identifier] = result
            return results

        results = []
        if multiprocessing:
            with Pool() as pool:
                for result in pool.map(task, self._filepaths):
                    results.append(result)
            return results

        for filepath in self._filepaths:
            results.append(task(filepath))
        return results

    def zero(self):
        operation = _Operation('zero', kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self

    def center(self):
        operation = _Operation('center', kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self
            
    def fill_nonmeasured(self, mode='nearest'):
        """
        Calls the 'fill_nonmeasured' method of each surface in the batch with the inplace argument
        specified as True.
        """
        operation = _Operation('fill_nonmeasured', kwargs=dict(mode=mode, inplace=True))
        self._operations.append(operation)
        return self
            
    def level(self):
        operation = _Operation('level', kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self
            
    def filter(self, cutoff, *, mode, cutoff2=None, inplace=False):
        operation = _Operation('filter', args=(cutoff, ), kwargs=dict(mode=mode, cutoff2=cutoff2, inplace=True))
        self._operations.append(operation)
        return self
    
    def rotate(self, angle):
        operation = _Operation('rotate', args=(angle,), kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self
    
    def align(self):
        operation = _Operation('align', kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self

    def zoom(self, factor):
        operation = _Operation('zoom', args=(factor,), kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self

    def __getattr__(self, attr):
        # This is probably a questionable implementation
        # The call to getattr checks if the attribute exists in this class. If not, it checks whether the attribute
        # is part of the available roughness parameters of the surfalize.Surface class. If it is not, it raises
        # the original AttributeError again. If it is a parameter of surfalize.Surface class though, it constructs
        # a dummy method that is returned to the caller which, instead of calling the actual method from the Surface
        # class, registers the parameter with the corresponding arguments in this class for later execution.
        try:
            return self.__dict__[attr]
        except KeyError:
            if attr not in Surface.AVAILABLE_PARAMETERS:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
        def parameter_dummy_method(*args, **kwargs):
            parameter = _Parameter(attr, args=args, kwargs=kwargs)
            self._parameters.append(parameter)
            return self

        return parameter_dummy_method

    # To be reimplemented

    # def roughness_parameters(self, parameters=None):
    #     if parameters is None:
    #         parameters = list(Surface.AVAILABLE_PARAMETERS)
    #     df = pd.DataFrame({'filename': [file.name for file in self._filepaths]})
    #     df = df.set_index('filename')
    #     df[list(parameters)] = np.nan
    #     for file, surface in tqdm(self._surfaces.items(), desc='Calculating parameters'):
    #         results = surface.roughness_parameters(parameters)
    #         for k, v in results.items():
    #             df.loc[file.name][k] = v
    #     df = df.reset_index()
    #     if self._additional_data is None:
    #         return df
    #     dfin = pd.read_excel(self._additional_data)
    #     return pd.merge(dfin, df, on='filename')