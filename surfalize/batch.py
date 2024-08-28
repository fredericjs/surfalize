from collections import defaultdict
from multiprocessing.pool import ThreadPool
from functools import partial
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .surface import Surface
from .utils import is_list_like
from .file import supported_formats
from .exceptions import BatchError, CalculationError

class ParsingError(Exception):
    """
    Is raised when an error occurs during filename parsing.
    """
    pass


class _Token:
    """
    Class representing a token in a filename string.
    """

    def __init__(self, token_str):
        self.token_str = token_str
        self.extract()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.token_str == other.token_str
        return False

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.token_str}")'

    def extract(self):
        split_token = self.token_str.split('|')
        if len(split_token) < 2:
            raise ParsingError('Template expression must be of the form  "<name|type>" or <name|type|prefix|suffix>"')
        split_token += [''] * (4 - len(split_token))
        self.name, self.dtype, self.prefix, self.suffix = split_token


class FilenameParser:
    """
    Parser class that parses filenames according to a template string.

    The template can specify parameters by specifying their name, datatype, prefix (optional) and suffix (optional).
    The name is used to label the resulting column in the dataframe. The patterns have the general syntax:

    <name|datatype|prefix|suffix>

    Both prefix and suffix can be omitted. If only a suffix is defined, the prefix must be indicated as an empty
    string. A pattern to match a filename could look like this:

    filename: 'P90_N10_F1.21_FREP10kHz.vk4'
    pattern: '<power|float|P>_<pulses|int|N>_<fluence|float|F>_<frequency|float|FREP|kHz>'
    """
    TYPES = {
        'float': r'\d+(?:(?:\.|,)\d+)?',
        'int': r'\d+',
        'str': r'.+'
    }

    def __init__(self, template_str):
        self.template_str = template_str

    def parse_template(self):
        """
        Parses the template string into separate tokens and constructs a regex to match the filename from these tokens.

        Returns
        -------
        tokens, separators :  list[_Token], list[str]
            List of tokens and list of string separators
        """
        tokens = []
        separators = []
        token_started = False
        separator = ''
        token = ''
        for char in self.template_str:
            if char == '<':
                if token_started:
                    raise ParsingError(f'Unclosed expression found: "<{token}"')
                token = ''
                token_started = True
                separators.append(separator)
                separator = ''
                continue
            elif char == '>':
                token_started = False
                tokens.append(_Token(token))
                continue
            if token_started:
                token += char
            else:
                separator += char

        if token_started:
            raise ParsingError(f'Unclosed expression found: "<{token}"')
        separators.append(separator)
        return tokens, separators

    def construct_regex(self, tokens, separators):
        """
        Construct a regex from the tokens and separators to match a filename.

        Parameters
        ----------
        tokens : list[_Token]
            List of tokens obtained from parsing the template string.
        separators : list[str]
            List of string obtained from parsing the template string.

        Returns
        -------
        regex : str
        """
        patterns = []
        for token in tokens:
            s = token.prefix
            s += f'(?P<{token.name}>'
            try:
                s += self.TYPES[token.dtype]
            except KeyError:
                raise ParsingError(f'The datatype {token.dtype} is invalid. '
                                   f'Possible datatypes are "int", "float", "str"') from None
            s += ')'
            s += token.suffix
            patterns.append(s)
        regex = ''
        for pattern, separator in zip(patterns, separators):
            regex += separator + pattern
        regex += separators[-1]
        return regex

    def extract_from(self, df, column):
        """
        Extracts the parameters from a column of a dataframe into a new dataframe, where each column represents one
        parameter.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame object that contains a column with filenames
        column : str
            Name of the column which contains the filenames
        Returns
        -------
        pd.DataFrame
        """
        tokens, separators = self.parse_template()
        regex = self.construct_regex(tokens, separators)
        extracted = df[column].str.extract(regex)
        for token in tokens:
            if token.dtype == 'float':
                extracted[token.name] = extracted[token.name].str.replace(',', '.')
            extracted[token.name] = extracted[token.name].astype(token.dtype)
        return extracted

    def apply_on(self, df, column, insert_after_column=True):
        """
        Extracts the parameters from a column of a dataframe and adds them to the dataframe. Each parameter in the
        filename will be represented by a new column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame object that contains a column with filenames
        column : str
            Name of the column which contains the filenames
        insert_after_column : bool, default True
            If True, inserts the new columns directly after the filename column, if False, appends them at the end of
            the dataframe

        Returns
        -------
        pd.Dataframe
            Original dataframe with added columns
        """

        extracted = self.extract_from(df, column)
        if insert_after_column:
            cols = extracted.columns
            df = df.copy()
            idx = df.columns.get_loc(column) + 1
            for col in cols[::-1]:
                if col not in df.columns:
                    df.insert(idx, col, '')
        return df.assign(**extracted)


class BatchResult:

    def __init__(self, df):
        self.__df = df.copy()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.__df, attr)

    def __getitem__(self, item):
        return self.__df.__getitem__(item)

    def __setitem__(self, item, value):
        self.__df.__setitem__(item, value)

    def get_dataframe(self):
        return self.__df

    def extract_from_filename(self, pattern):
        parser = FilenameParser(pattern)
        self.__df = parser.apply_on(self.__df, 'file')

class Operation:
    """
    Class that holds the identifier and arguments to register a call to a surface method that operates on its data.
    This class is used to implement lazy processing of topography files.

    Parameters
    ----------
    identifier : str
        Name of the method. Must be identical to a method of the Surface class.
    args : tuple
        Tuple of positional arguments that should be passed to the Surface method.
    kwargs : dict
        Dictionary of keyword arguments that should be passed to the Surface method.
    """
    def __init__(self, identifier, args=None, kwargs=None):
        self.identifier = identifier
        self.args = tuple() if args is None else args
        self.kwargs = dict() if kwargs is None else kwargs

    def execute_on(self, surface):
        """
        Executes the registered method from the surface file the positional and keyword arguments.

        Parameters
        ----------
        surface : surfalize.Surface
            surface object on which to execute the registered method.

        Returns
        -------
        None.
        """
        method = getattr(surface, self.identifier)
        method(*self.args, **self.kwargs)

class Parameter:
    """
    Class that holds the identifier and arguments to register a call to a surface method that returns a parameter.
    This class is used to implement lazy processing of topography files.

    Parameters
    ----------
    identifier : str
        Name of the method. Must be identical to a method of the Surface class.
    args : tuple
        Tuple of positional arguments that should be passed to the Surface method.
    kwargs : dict
        Dictionary of keyword arguments that should be passed to the Surface method.

    Examples
    --------
    Customize the arguments for `Surface.homogeneity()` to include Sa, Sdr, Sk and Sal.

    >>> homogeneity = Parameter('homogeneity', kwargs=dict(parameters=['Sa', 'Sdr', 'Sk', 'Sal']))
    >>> homogeneity.calculate_from(surface)

    The intended use is to supply it to `Batch.roughness_parameters()` instead of a string.

    >>> batch.roughness_parameters(['Sa', 'Sq', 'Sz', homogeneity])
    >>> batch.execute()
    """
    def __init__(self, identifier, args=None, kwargs=None, custom_name=None):
        self.identifier = identifier
        self.name = custom_name if custom_name is not None else identifier
        self.args = tuple() if args is None else args
        self.kwargs = dict() if kwargs is None else kwargs

    def calculate_from(self, surface, ignore_errors=True):
        """
        Executes the registered method from the surface file the positional and keyword arguments. Returns a dictionary
        containing the identifier as a key and the value returned from the method as value. If a method returns multiple
        values, this method must be supplied with a decorator that registers labels for the different return values in
        the order they are returned. Each value in the return dictionary will then have a key that consists of the
        identifier as well as the corresponding return value label joined by an underscore:

        Parameters
        ----------
        surface : surfalize.Surface
            surface object on which to execute the registered method.

        Returns
        -------
        None
        """
        method = getattr(surface, self.identifier)
        try:
            result = method(*self.args, **self.kwargs)
        except CalculationError as error:
            if not ignore_errors:
                raise error
            if hasattr(method, 'return_labels'):
                result = [np.nan] * len(method.return_labels)
            else:
                result = np.nan
        if is_list_like(result):
            try:
                labels = method.return_labels
            except AttributeError:
                raise BatchError(f"No return labels registered for Surface.{self.identifier}.") from None
            if len(result) != len(labels):
                raise BatchError("Number of registered return labels do not match number of returned values.") from None
            return {f'{self.name}_{label}': value for value, label in zip(result, labels)}
        return {self.name: result}

def _task(filepath, operations, parameters, ignore_errors):
    """
    Task that loads a surface from file, executes a list of operations and calculates a list of parameters.
    This function is used to split the processing load of a Batch between CPU cores.

    Parameters
    ----------
    filepath : str | pathlib.Path
        Filepath pointing to the measurement file.
    operations : list[Operation]
        List of operations to execute on the surface.
    parameters : list[Parameter]
        List of parameters to calculate from the surface.

    Returns
    -------
    results : dict[str: value]
        Dictionary containing the values for each invokes parameter, with the parameter's method identifier as
        key.
    """
    surface = Surface.load(filepath)
    for operation in operations:
        operation.execute_on(surface)
    results = dict(file=filepath.name)
    for parameter in parameters:
        result = parameter.calculate_from(surface, ignore_errors=ignore_errors)
        results.update(result)
    return results

#TODO batch image export
class Batch:
    """
   The batch class is used to perform operations and calculate quantitative surface parameters for a batch of
   topography files. The implementation allows to register operations and parameters for lazy calculation by invoking
   methods defined by this class. Every operation method that is defined by Surface can be invoked on the batch class,
   which then registers the method and the passed arguments for later execution. Similarly, every roughness parameter
   can be called on the Batch class. The __getattr__ method is responsible for checking if an invoked method constitutes
   a roughness parameter and if so, automatically wraps the method in a Parameter class, which is registered for later
   calculation. This means that roughness parameters can be invoked on the Batch object despite not being explicitly
   defined in the code.

   All methods can be chained, since they implement the builder design pattern, where every method returns the object
   itself. For exmaple, the operations levelling, filtering and aligning as well as the calculation of roughness
   parameters Sa, Sq and Sz can be registered for later calculation in the following manner:

   >>> batch = Batch(filespaths)
   >>> batch.level().filter(filter_type='lowpass', cutoff=10).align().Sa().Sq().Sz()

   Or on separate lines:
   >>> batch.level().filter(filter_type='lowpass', cutoff=10).align()
   >>> batch.Sa()
   >>> batch.Sq()
   >>> batch.Sz()

   Upon invoking the execute method, all registered operations and parameters are performed.
   >>> batch.execute()

   If the caller wants to supply additional parameters for each file, such as fabrication data, they can specify the
   path to an Excel file containing that data using the 'additional_data' keyword argument. The excel file should
   contain a column 'filename' of the format 'name.extension'. Otherwise, an arbitrary number of additional columns can
   be supplied.

   Parameters
   ----------
   filepaths : list[pathlib.Path | str]
       List of filepaths of topography files
   additional_data : str, pathlib.Path
       Path to an Excel file containing additional parameters, such as
       input parameters. Excel file must contain a column 'file' with
       the filename including the file extension. Otherwise, an arbitrary
       number of additional columns can be supplied.

    Examples
    --------
    >>> from pathlib import Path
    >>> files = Path().cwd().glob('*.vk4')
    >>> batch = Batch(filespaths, addition_data='additional_data.xlsx')
    >>> batch.level().filter('lowpass', 10).Sa().Sq().Sdr()
   """
    
    def __init__(self, filepaths, additional_data=None):
        self._filepaths = [Path(file) for file in filepaths]
        if additional_data is None:
            self._additional_data = None
        else:
            self._additional_data = pd.read_excel(additional_data)
            if 'file' not in self._additional_data.columns:
                raise ValueError("File specified by 'additional_data' does not contain column named 'file'.")
        self._operations = []
        self._parameters = []
        self._filename_pattern = None

    @classmethod
    def from_dir(cls, dir_path, file_extensions=None, additional_data=None):
        """
        Alternative constructor for Batch class that takes a directory path as well as a string or list of strings
        of file extensions as positional arguments.

        Parameters
        ----------
        dir_path : str | pathlib.Path
            Path to the directory containing the files
        file_extensions : str | list-like, optional
            File extension or list of file extensions to be searched for, eg. '.vk4', '.plu'. The file extension must
            be prefixed by a dot. If no file extensions are specified, all files are added to the batch that have a file
            extension that corresponds to a supported file format.
        additional_data : str, pathlib.Path, optional
            Path to an Excel file containing additional parameters, such as
            input parameters. Excel file must contain a column 'file' with
            the filename including the file extension. Otherwise, an arbitrary
            number of additional columns can be supplied.

        Examples
        --------
        >>> directory = 'C:\\topography_files'
        >>> batch = Batch.from_dir(directory)

        Returns
        -------
        Batch
        """
        dir_path = Path(dir_path)
        filepaths = []
        if file_extensions is None:
            file_extensions = supported_formats
        elif isinstance(file_extensions, str):
            file_extensions = [file_extensions]
        for extension in file_extensions:
            filepaths.extend(list(dir_path.glob(f'*{extension}')))
        return cls(filepaths, additional_data=additional_data)


    def _disptach_tasks(self, multiprocessing=True, ignore_errors=True):
        """
        Dispatches the individual tasks between CPU cores if multiprocessing is True, otherwise executes them
        sequentially.

        Notes
        -----
        This implementation has switched from a true multiprocessing pool to a thread pool. The reason for the
        change is that multiprocessing relies on pickling, which causes multiple issues when using Jupyter Notebooks.
        Also, if the batch.execute method is not called in a main guard, an infinite spawning of child processes may
        occur. These issues are avoided when using threads while maintaining most of the speedup, since most numpy-based
        computations release the GIL anyway.

        Parameters
        ----------
        multiprocessing : bool, default True
            If True, dispatches the task among CPU cores, otherwise sequentially computes the tasks.

        Returns
        -------
        results : dict[str: value]
            Dictionary containing the values for each invokes parameter, with the parameter's method identifier as
            key.
        """
        results = []
        if multiprocessing:
            with ThreadPool() as pool:
                task = partial(_task, operations=self._operations, parameters=self._parameters, ignore_errors=ignore_errors)
                with tqdm(total=len(self._filepaths), desc='Processing files') as progress_bar:
                    for result in pool.imap_unordered(task, self._filepaths):
                        results.append(result)
                        progress_bar.update()
                pool.close()
                pool.join()
            return results

        for filepath in tqdm(self._filepaths, desc='Processing'):
            results.append(_task(filepath, self._operations, self._parameters))
        return results

    def _construct_dataframe(self, results, filename_pattern=None):
        """
        Constructs a pandas DataFrame from the result dictionary of the _dispatch_tasks method. This method is also
        responsible for merging the additional data if specified.

        Parameters
        ----------
        results : dict[str: any]

        Returns
        -------
        pd.DataFrame
        """
        df = pd.DataFrame(results)
        if self._additional_data is not None:
            df = pd.merge(self._additional_data, df, on='file')
        if self._filename_pattern is not None:
            parser = FilenameParser(self._filename_pattern)
            df = parser.apply_on(df, 'file')
        return df

    def execute(self, multiprocessing=True, ignore_errors=True, saveto=None):
        """
        Executes the Batch processing and returns the obtained data as a pandas DataFrame. The dataframe can be saved
        as an Excel file.

        Examples
        --------
        >>> pattern = ''
        >>> batch.execute(saveto='C:/users/example/documents/data.xlsx')

        Parameters
        ----------
        multiprocessing : bool, default True
            If True, dispatches the task among CPU cores, otherwise sequentially computes the tasks.
        ignore_errors : bool, default True
            Errors that are raised during the calculation of parameters are ignored if True. Missing parameter values
            are filled with nan values. If False, the batch processing is interrupted when an error is raised.
        saveto : str | pathlib.Path, default None
            Path to an Excel file where the data is saved to. If the Excel file does already exist, it will be
            overwritten.

        Returns
        -------
        pd.DataFrame
        """
        if not self._parameters and not self._operations:
            raise BatchError('No operations of parameters defined.')
        # Check for duplicate parameters without custom names and raise an error.
        parameter_dict = defaultdict(int)
        for parameter in self._parameters:
            if parameter_dict[parameter.name] > 0:
                raise BatchError(f'The parameter "{parameter.identifier}" is computed twice. If this was not a mistake,'
                                 f' consider giving it an alternate name using the keyword argument "custom_name".')
            parameter_dict[parameter.name] += 1
        results = self._disptach_tasks(multiprocessing=multiprocessing, ignore_errors=ignore_errors)
        df = self._construct_dataframe(results)
        if saveto is not None:
            df.to_excel(saveto)
        return BatchResult(df)

    def extract_from_filename(self, pattern):
        """
        Extracts parameters that are encoded in filenames into their own columns. For instance a filename might encode
        different fabrication parameters of the measured surface:

        filename: 'Sample1_P50_N12_F1.23_FREP10kHz.vk4'

        The pattern can encode parameters by specifying their name, datatype, prefix (optional) and suffix (optional).
        The name is used to label the resulting column in the dataframe. The patterns have the general syntax:

        <name|datatype|prefix|suffix>

        Both prefix and suffix can be omitted. If only a suffix is defined, the prefix must be indicated as an empty
        string. A pattern to match the above filename could look like this:

        pattern: '<power|float|P>_<pulses|int|N>_<fluence|float|F>_<frequency|float|FREP|kHz>'

        This pattern is parsed and constructs a regex that searches the filename for the defined parameters.
        The parameters are  extracted and converted to their respective datatype. The values are added as new columns
        to the dataframe.

        Parameters
        ----------
        filename_pattern : str | None
            Pattern with which to extract parameters from filename.

        Returns
        -------
        self
        """
        self._filename_pattern = pattern
        return self

    def zero(self):
        """
        Registers Surface.zero for later execution. Inplace is True by default.

        Returns
        -------
        self
        """
        operation = Operation('zero', kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self

    def center(self):
        """
        Registers Surface.center for later execution. Inplace is True by default.

        Returns
        -------
        self
        """
        operation = Operation('center', kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self

    def threshold(self, threshold=0.5):
        """
        Registers Surface.thresold for later execution. Inplace is True by default.

        Parameters
        ----------
        threshold : float, default 0.5
            Threshold argument from Surface.threshold

        Returns
        -------
        self
        """
        operation = Operation('threshold', kwargs=dict(threshold=threshold, inplace=True))
        self._operations.append(operation)
        return self

    def remove_outliers(self, n=3, method='mean'):
        """
        Registers Surface.remove_outliers for later execution. Inplace is True by default.

        Parameters
        ----------
        n : float, default 3
            n argument from Surface.remove_outliers
        method : {'mean', 'median'}, default 'mean'
            method argument from Surface.remove_outliers

        Returns
        -------
        self
        """
        operation = Operation('remove_outliers', kwargs = dict(n=n, method=method, inplace=True))
        self._operations.append(operation)
        return self
            
    def fill_nonmeasured(self, method='nearest'):
        """
        Registers Surface.fill_nonmesured for later execution. Inplace is True by default.

        Parameters
        ----------
        method : {'linear', 'nearest', 'cubic'}, default 'nearest'
            method argument from Surface.fill_nonmeasured

        Returns
        -------
        self
        """
        operation = Operation('fill_nonmeasured', kwargs=dict(method=method, inplace=True))
        self._operations.append(operation)
        return self

    def crop(self, box):
        """
        Registers Surface.crop for later execution. Inplace is True by default.

        Parameters
        ----------
        box : tuple[float, float, float, float]
            The crop rectangle, as a (x0, x1, y0, y1) tuple.

        Returns
        -------
        self
        """
        operation = Operation('crop', args=(box,), kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self
            
    def level(self):
        """
        Registers Surface.level for later execution. Inplace is True by default.

        Returns
        -------
        self
        """
        operation = Operation('level', kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self
            
    def filter(self, filter_type, cutoff, cutoff2=None):
        """
        Registers Surface.filter for later execution. Inplace is True by default. The filter_type both cannot be used
        for batch analysis.

        Parameters
        ----------
        filter_type : str
            Mode of filtering. Possible values: 'highpass', 'lowpass', 'bandpass'.
        cutoff : float
            Cutoff frequency in 1/µm at which the high and low spatial frequencies are separated.
            Actual cutoff will be rounded to the nearest pixel unit (1/px) equivalent.
        cutoff2 : float | None, default None
            Used only in mode='bandpass'. Specifies the lower cutoff frequency of the bandpass filter. Must be greater
            than cutoff.

        Returns
        -------
        self
        """
        operation = Operation('filter', args=(filter_type, cutoff),
                              kwargs=dict(cutoff2=cutoff2, inplace=True))
        self._operations.append(operation)
        return self
    
    def rotate(self, angle):
        """
        Registers Surface.rotate for later execution. Inplace is True by default.

        Parameters
        ----------
        angle : float
            Angle in degrees.

        Returns
        -------
        self
        """
        operation = Operation('rotate', args=(angle,), kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self
    
    def align(self, axis='y'):
        """
        Registers Surface.align for later execution. Inplace is True by default.

        Parameters
        ----------
        axis : {'x', 'y'}, default 'y'
            The axis with which to align the texture with.

        Returns
        -------
        self
        """
        operation = Operation('align', kwargs=dict(inplace=True, axis=axis))
        self._operations.append(operation)
        return self

    def zoom(self, factor):
        """
        Registers Surface.zoom for later execution. Inplace is True by default.

        Parameters
        ----------
        factor : float
            Factor by which the surface is magnified

        Returns
        -------
        self
        """
        operation = Operation('zoom', args=(factor,), kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self

    def stepheight_level(self):
        """
        Registers Surface.stepheight_level for later execution. Inplace is True by default.

        Returns
        -------
        self
        """
        operation = Operation('stepheight_level', kwargs=dict(inplace=True))
        self._operations.append(operation)
        return self

    def __getattr__(self, attr):
        # This is probably a questionable implementation
        # The call to getattr checks if the attribute exists in this class and returns it if True. If not, it checks
        # whether the attribute is part of the available roughness parameters of the surfalize.Surface class. If it is
        # not, it raises the original AttributeError again. If it is a parameter of surfalize.Surface class though, it
        # constructs a dummy method that is returned to the caller which, instead of calling the actual method from the
        # Surface class, registers the parameter with the corresponding arguments in this class for later execution.
        try:
            return self.__dict__[attr]
        except KeyError:
            if attr not in Surface.AVAILABLE_PARAMETERS:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
        def parameter_dummy_method(*args, custom_name=None, **kwargs):
            parameter = Parameter(attr, args=args, kwargs=kwargs, custom_name=custom_name)
            self._parameters.append(parameter)
            return self

        return parameter_dummy_method

    def roughness_parameters(self, parameters=None):
        """
        Registers multiple roughness parameters for later execution. Corresponds to Surface.roughness_parameters.
        If parameters is None, all available roughness and periodic parameters are registered. Otherwise, a list of
        parameters can be passed as argument, which contains the parameter method identifier, which must be equal to
        the method name of the parameter in the Surface class.
        If a parameter is given as a string, it is registered with its default keyword argument values. In the case that
        the user wants to specify a parameter with keyword arguments, there are two options. Either register that
        parameter explicitly by calling Batch.parameter(args, kwargs) or by passing a Parameter class to this method
        instead of a string.

        Examples
        --------
        Here, only the specified parameters will be calculated.

        >>> batch = Batch(filepaths)
        >>> batch.roughness_parameters(['Sa', 'Sq', 'Sz', 'Sdr', 'Vmc'])

        In this case, all available parameters will be calculated.

        >>> batch = Batch(filepaths)
        >>> batch.roughness_parameters()

        Here, we define a custom Parameter class that allows for the specification of keyword arguments. Note that we
        are passing the Parameter to the method instead of the string version.

        >>> from surfalize.batch import Parameter
        >>> Vmc = Parameter('Vmc', kwargs=dict(p=5, q=95))
        >>> batch.roughness_parameters(['Sa', 'Sq', 'Sz', 'Sdr', Vmc])

        Parameters
        ----------
        parameters : list[str | surfalize.batch.Parameter]
            List of parameters to be registered, either as a string identifier or as a Parameter class.

        Returns
        -------
        self
        """
        if parameters is None:
            parameters = list(Surface.ISO_PARAMETERS)
        for parameter in parameters:
            if isinstance(parameter, str):
                parameter = Parameter(parameter)
            self._parameters.append(parameter)
        return self