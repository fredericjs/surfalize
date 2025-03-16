import inspect
import io
from multiprocessing.pool import ThreadPool
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from .surface import Surface
from .utils import is_list_like, remove_parameter_from_docstring
from .file import supported_formats_read
from .exceptions import BatchError, CalculationError

class ParsingError(Exception):
    """
    Is raised when an error occurs during filename parsing.
    """
    pass


@dataclass
class FileInput:
    """
    Class that wraps a file-like object, adding a name and an optional format specifier for use in batch processing.
    """
    name: str
    data: io.IOBase
    format: Optional[str] = None


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

    """
    Class that wraps the DataFrame returned by `Batch.execute`. Provides a method to get the underlying DateFrame object
    and a method to apply filename extraction on the DataFrame.

    Parameters
    ----------
    df : pd.DateFrame
        Pandas DataFrame object.
    """

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
        """Returns the underlying DataFrame object"""
        return self.__df

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
        pattern : str | None
            Pattern with which to extract parameters from filename.

        Returns
        -------
        None
        """
        parser = FilenameParser(pattern)
        self.__df = parser.apply_on(self.__df, 'file')


class _Operation:
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


class _Parameter:
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

    >>> homogeneity = _Parameter('homogeneity', kwargs=dict(parameters=['Sa', 'Sdr', 'Sk', 'Sal']))
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


class _CustomParameter:

    def __init__(self, func):
        self.func = func
        self.name = id(func)

    def calculate_from(self, surface, ignore_errors=True):
        try:
            result = self.func(surface)
        except CalculationError as error:
            if not ignore_errors:
                raise error
        return result


class _CustomOperation:

    def __init__(self, func):
        self.func = func

    def execute_on(self, surface):
        self.func(surface)


def _task(file, steps, ignore_errors, preserve_chaining_order):
    """
    Task that loads a surface from file, executes a list of operations and calculates a list of parameters.
    This function is used to split the processing load of a Batch between CPU cores.

    Parameters
    ----------
    file : str | pathlib.Path
        Filepath pointing to the measurement file or FileInput object.
    operations : list[_Operation, _Parameter, _CustomParameter]
        List of steps to execute on the surface.
    preserve_chaining_order : bool
        Whether to preserve the order the different operations and parameter calculations are called on the batch
        obeject. If True, operations and parameters can be applied in arbitrary order
        (e.g batch.operation().parameter().operation()). If False, all operations will be performed before the
        parameter calculations, irrespective of the order they were called on the batch. The order within the
        operations and parameters themselves will be preserved nonetheless.

    Returns
    -------
    results : dict[str: value]
        Dictionary containing the values for each invokes parameter, with the parameter's method identifier as
        key.
    """
    if isinstance(file, FileInput):
        surface = Surface.load(file.data, format=file.format)
    else:
        surface = Surface.load(file)
    results = dict(file=file.name)
    if preserve_chaining_order:
        for step in steps:
            if isinstance(step, (_Operation, _CustomOperation)):
                step.execute_on(surface)
            elif isinstance(step, (_Parameter, _CustomParameter)):
                result = step.calculate_from(surface, ignore_errors=ignore_errors)
                results.update(result)
    else:
        operations = [step for step in steps if isinstance(step, (_Operation, _CustomOperation))]
        parameters = [step for step in steps if isinstance(step, (_Parameter, _CustomParameter))]
        for operation in operations:
            operation.execute_on(surface)
        for parameter in parameters:
            result = parameter.calculate_from(surface, ignore_errors=ignore_errors)
            results.update(result)
    return results

#TODO batch image export
class Batch:
    """
    The batch class is used to perform operations and calculate quantitative surface parameters for a batch of
    topography files. The implementation allows to register operations and parameters for lazy calculation by invoking
    methods defined by the `Surface` class. Every operation method that is defined by Surface can be invoked on the
    batch class, which then registers the method and the passed arguments for later execution. Similarly, every
    roughness parameter can be called on the Batch class. The methods of the `Surface` class do not appear in the this
    class's documentation. However, their docstring can be accessed through `help(Batch.method)` or `Batch.method?` in
    Jupyter.

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
    files : list[pathlib.Path | str | FileInput]
       List of filepaths or FileInput objects. For file-like objects, a FileInput object must be constructed that holds
       a name and the file-like object.
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
    
    def __init__(self, files, additional_data=None):
        self._files = []
        for file in files:
            if isinstance(file, FileInput):
                self._files.append(file)
            else:
                self._files.append(Path(file))
        if additional_data is None:
            self._additional_data = None
        else:
            self._additional_data = pd.read_excel(additional_data)
            if 'file' not in self._additional_data.columns:
                raise ValueError("File specified by 'additional_data' does not contain column named 'file'.")
        self._steps = []
        self._filename_pattern = None

        for name, method in Surface.__dict__.items():
            if hasattr(method, '_batch_type'):
                self._create_batch_method(name, method)

    def __len__(self):
        return len(self._files)

    def _create_batch_method(self, name, method):
        """Create a batch method from a Surface method."""

        def batch_method(*args, **kwargs):
            for param_name, param_value in method._fixed.items():
                kwargs[param_name] = param_value
            if method._batch_type == 'operation':
                step = _Operation(name, args=args, kwargs=kwargs)
            elif method._batch_type == 'parameter':
                custom_name = kwargs.pop('custom_name', None)
                step = _Parameter(name, args=args, kwargs=kwargs, custom_name=custom_name)

            self._add_step(step)
            return self

        # Create batch docstring
        batch_doc = (f'Batch version of Surface.{name}.\nThis method registers the {name} {method._batch_type} for batch '
                     f'processing. The actual computation occurs when execute() is called.\n\nParameters that cannot be '
                     f'used on Batch objects are: {", ".join(method._fixed.keys())}.')

        if hasattr(method, '_batch_doc'):
            batch_doc += f"\n{method._batch_doc}\n"

        remove_parameters = {'self', *method._fixed.keys()}
        if method.__doc__:
            # Remove inplace from original docstring
            filtered_doc = method.__doc__
            for param in remove_parameters:
                filtered_doc = remove_parameter_from_docstring(param, filtered_doc)
            batch_doc += f"\n{filtered_doc}"

        batch_method.__doc__ = inspect.cleandoc(batch_doc)

        sig = inspect.signature(method)
        params = [p for name, p in list(sig.parameters.items())
                  if name not in remove_parameters]  # Skip 'self'
        batch_method.__signature__ = sig.replace(parameters=params)

        # Add method to instance
        setattr(self, name, batch_method)

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
            file_extensions = supported_formats_read
        elif isinstance(file_extensions, str):
            file_extensions = [file_extensions]
        for extension in file_extensions:
            filepaths.extend(list(dir_path.glob(f'*{extension}')))
        return cls(filepaths, additional_data=additional_data)

    def add_files(self, files):
        """
        Add files to Batch after initialization.

        Parameters
        ----------
        files: str | pathlib.Path | FileInput | list-like[str | pathlib.Path | FileInput]
            Files to add to the Batch.

        Returns
        -------
        self
        """
        if not is_list_like(files):
            files = [files]
        for file in files:
            if isinstance(file, FileInput):
                self._files.append(file)
            else:
                self._files.append(Path(file))
        return self

    def add_dir(self, dir_path, file_extensions=None):
        """
        Add all files in a directory to Batch after initialization. If

        Parameters
        ----------
        dir_path : str | pathlib.Path
            Path to the directory containing the files
        file_extensions : str | list-like, optional
            File extension or list of file extensions to be searched for, eg. '.vk4', '.plu'. The file extension must
            be prefixed by a dot. If no file extensions are specified, all files are added to the batch that have a file
            extension that corresponds to a supported file format.

        Returns
        -------
        self
        """
        dir_path = Path(dir_path)
        filepaths = []
        if file_extensions is None:
            file_extensions = supported_formats_read
        elif isinstance(file_extensions, str):
            file_extensions = [file_extensions]
        for extension in file_extensions:
            self._files.extend(list(dir_path.glob(f'*{extension}')))
        return self

    def _disptach_tasks(self, multiprocessing=True, ignore_errors=True, on_file_complete=None,
                        preserve_chaining_order=True):
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
        ignore_errors : bool, default True
            Errors that are raised during the calculation of parameters are ignored if True. Missing parameter values
            are filled with nan values. If False, the batch processing is interrupted when an error is raised.
        on_file_complete: Callable
            Hook for a Callable that is executed for every surface that has finished processing. The Callable must take
            a results parameter that is passed a dictionary of the results of the surface calculation. The dictionary
            will at least hold a key 'file' with the respective filename.
        preserve_chaining_order : bool
            Whether to preserve the order the different operations and parameter calculations are called on the batch
            obeject. If True, operations and parameters can be applied in arbitrary order
            (e.g batch.operation().parameter().operation()). If False, all operations will be performed before the
            parameter calculations, irrespective of the order they were called on the batch. The order within the
            operations and parameters themselves will be preserved nonetheless.

        Returns
        -------
        results : dict[str: value]
            Dictionary containing the values for each invokes parameter, with the parameter's method identifier as
            key.
        """
        results = []
        if multiprocessing:
            with ThreadPool() as pool:
                task = partial(_task, steps=self._steps, ignore_errors=ignore_errors,
                               preserve_chaining_order=preserve_chaining_order)
                with tqdm(total=len(self._files), desc='Processing files') as progress_bar:
                    for result in pool.imap_unordered(task, self._files):
                        results.append(result)
                        if on_file_complete is not None:
                            on_file_complete(result)
                        progress_bar.update()
                pool.close()
                pool.join()
            return results

        for filepath in tqdm(self._files, desc='Processing'):
            results.append(_task(filepath, steps=self._steps, ignore_errors=ignore_errors,
                               preserve_chaining_order=preserve_chaining_order))
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

    def _add_step(self, step):
        """
        Adds a step (operation, parameter) to the list of registered steps. For parameters, it checks whether a
        parameter with the same name is already present in the list and raises an error if this is the case.

        Parameters
        ----------
        step: _Operation | _Parameters
            Step to add to the list of steps.

        Returns
        -------
        None
        """
        if isinstance(step, (_Parameter, _CustomParameter)):
            if step.name in {s.name for s in self._steps if isinstance(s, (_Parameter, _CustomParameter))}:
                raise BatchError(f'The parameter "{step.identifier}" is already registered. Consider giving it an '
                                 f'alternate name using the keyword argument "custom_name".')
        self._steps.append(step)


    def execute(self, multiprocessing=True, ignore_errors=True, saveto=None, on_file_complete=None,
                preserve_chaining_order=True):
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
        on_file_complete: Callable
            Hook for a Callable that is executed for every surface that has finished processing. The Callable must take
            a results parameter that is passed a dictionary of the results of the surface calculation. The dictionary
            will at least hold a key 'file' with the respective filename.
        preserve_chaining_order : bool
            Whether to preserve the order the different operations and parameter calculations are called on the batch
            obeject. If True, operations and parameters can be applied in arbitrary order
            (e.g batch.operation().parameter().operation()). If False, all operations will be performed before the
            parameter calculations, irrespective of the order they were called on the batch. The order within the
            operations and parameters themselves will be preserved nonetheless.

        Returns
        -------
        pd.DataFrame
        """
        if not self._steps:
            raise BatchError('No operations of parameters defined.')
        results = self._disptach_tasks(multiprocessing=multiprocessing,
                                       ignore_errors=ignore_errors,
                                       on_file_complete=on_file_complete,
                                       preserve_chaining_order=preserve_chaining_order)
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

    def custom_parameter(self, func):
        """
        Add a custom parameter calculation in the form of a simple function to the batch calculation. The function
        must take the surface object as its only parameter and return a dictionary, where the keys are the parameter
        names and parameter values. If the parameter consists of only one value, the dictionary should have only one
        entry. The keys of the returned dictionary will be used as the column names in the resulting DataFrame.

        Examples
        --------
        An examplary function might look like this:

        >>> def median(surface):
        ...    median = np.median(surface.data)
        ...    return {'height_median': median}

        Or with multiple parameters:

        >>> def mean_std(surface):
        ...    mean = np.mean(surface.data)
        ...    std = np.std(surface.data)
        ...    return {'mean_value': mean, 'std_value': std}

        Parameters
        ----------
        func: callable
            Function to be executed. Must take a surface object as the only argument and return a dictionary.

        Returns
        -------
        self
        """
        self._add_step(_CustomParameter(func))
        return self

    def custom_operation(self, func):
        """
        Add a custom parameter operation in the form of a simple function to the batch calculation. The function
        must take the surface object as its only parameter and modify the surface object in place, returning None.

        Examples
        --------
        An examplary function might look like this:

        >>> def remove_specific_outliers(surface):
        ...    outlier_value = 1001
        ...    surface.data[surface.data == outlier_value] = np.nan

        Parameters
        ----------
        func: callable
            Function to be executed. Must take a surface object as the only argument and return None.

        Returns
        -------
        self
        """
        self._add_step(_CustomOperation(func))
        return self

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

        >>> from surfalize.batch import _Parameter
        >>> Vmc = _Parameter('Vmc', kwargs=dict(p=5, q=95))
        >>> batch.roughness_parameters(['Sa', 'Sq', 'Sz', 'Sdr', Vmc])

        Parameters
        ----------
        parameters : list[str | surfalize.batch._Parameter]
            List of parameters to be registered, either as a string identifier or as a Parameter class.

        Returns
        -------
        self
        """
        if parameters is None:
            parameters = list(Surface.ISO_PARAMETERS)
        for parameter in parameters:
            if isinstance(parameter, str):
                parameter = _Parameter(parameter)
            self._add_step(parameter)
        return self