import numpy as np

# Define general sinusoid function for fitting
sinusoid = lambda x, a, p, xo, yo: a*np.sin((x-xo)/p*2*np.pi) + yo

def register_returnlabels(labels):
    def wrapper(function):
        function.return_labels = labels
        return function
    return wrapper