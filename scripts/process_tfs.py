'''
Script to process the through-focus series into a stack of 2D
 Fourier transforms.
'''

import numpy as np
import pyCTF
from pyCTF.fourier import Fourier

# Path to image stack.
path = 'assets\\example_TFS.tif'

# Define radii for background subtraction.
r1 = 4
r2 = 4

print('Processing TFS.')

print('Importing through-focus series.')
stack =  Fourier.import_stack( path )

print('Performing FFT.')
FFT = Fourier.fft_stack( stack )
FFT = Fourier.log_mod( FFT )

print('Removing background.')
FFT = Fourier.remove_bckg_stack( FFT, r1, r2 )

print('Radial profile...')
prof = Fourier.profile_fft_stack( FFT )

fig, ax = plt.subplots()
ax.imshow(prof)

np.save('TFS', prof)