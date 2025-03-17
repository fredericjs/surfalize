import io
from pathlib import Path
from dataclasses import dataclass

try:
    import requests
    from requests.exceptions import ConnectionError
except ImportError:
    raise ImportError("Please install requests library to load example files")

EXAMPLE_FILES_URL = "https://api.github.com/repos/fredericjs/surfalize/contents/tests/test_files"


@dataclass
class ExampleFile:
    """
    Class that holds a reference to an example file from the github repo.
    The surface object for the file can be loaded using the load method.
    """
    name: str
    url: str

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    def load(self):
        """
        Downloads and instantiates a Surface object from the github repo example files.

        Returns
        -------
        Surface
        """
        from .surface import Surface
        try:
            response = requests.get(self.url)
        except ConnectionError:
            raise ConnectionError('Could not connect to github!') from None
        if Path(self.name).suffix:
            format_ = Path(self.name).suffix
        else:
            format_ = None
        return Surface.load(io.BytesIO(response.content), format=format_)


def list_examples(format=None):
    """
    Returns a list of available example files stored in the github repo.

    Parameters
    ----------
    format : str | None
        Restricts the returned example files to files of the specified format.
        The format must be given as a file suffix, e.g. '.vk4'.

    Returns
    -------
    list[ExampleFile]
    """
    try:
        response = requests.get(EXAMPLE_FILES_URL)
    except ConnectionError:
        raise ConnectionError('Could not connect to github!') from None
    files = [ExampleFile(f['name'], f['download_url']) for f in response.json()]
    if format is not None:
        files = [f for f in files if Path(f.name).suffix == format]
    return files