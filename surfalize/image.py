import PIL
import numpy as np

class Image:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return (f'Image({self.data.shape[1]} x {self.data.shape[0]}, Bitdepth: {self.data.dtype.itemsize * 8}, '
                f'Mode: {"RGB" if self.data.ndim == 3 else "Grayscale"})')

    def save(self, path):
        PIL.Image.fromarray(self.data).save(path)

    def show(self):
        return PIL.Image.fromarray(self.data)

    @staticmethod
    def is_grayscale(array):
        return np.all((array[:, :, 0] == array[:, :, 1]) & (array[:, :, 0] == array[:, :, 2]))