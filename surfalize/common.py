import numpy as np

# Define general sinusoid function for fitting
sinusoid = lambda x, a, p, xo, yo: a*np.sin((x-xo)/p*2*np.pi) + yo