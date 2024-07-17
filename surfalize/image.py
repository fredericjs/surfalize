import PIL
import numpy as np

class Image:
    """
    Thin wrapper around numpy ndarray object representing an image.
    Provides methods to show and save the image to disk.
    """
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return (f'Image({self.data.shape[1]} x {self.data.shape[0]}, Bitdepth: {self.data.dtype.itemsize * 8}, '
                f'Mode: {"RGB" if self.data.ndim == 3 else "Grayscale"})')

    def save(self, path):
        """
        Save the image to disk using pillow.

        Parameters
        ----------
        path : str | Path
            Path where the image should be saved. The extension determines the file format.

        Returns
        -------
        None
        """
        PIL.Image.fromarray(self.data).save(path)

    def show(self):
        """
        Constructs a pillow image from the underlying array and shows in Jupyter.

        Returns
        -------
        PIL.Image
        """
        return PIL.Image.fromarray(self.data)

    @staticmethod
    def is_grayscale(array):
        """
        Determines whether a three dimensional array is grayscale by checking if all RGB channels have the same values.

        Parameters
        ----------
        array : ndarray
            ndarray of ndim 3.

        Returns
        -------
        True if grayscale, else False
        """
        return np.all((array[:, :, 0] == array[:, :, 1]) & (array[:, :, 0] == array[:, :, 2]))