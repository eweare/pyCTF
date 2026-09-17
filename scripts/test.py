'''
Standard tests for PyCTF.
'''

# Timing variables for testing
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# t1 = time.perf_counter()
# t2 = time.perf_counter()
# diff = t2 - 1

# Test importing package.
import pyCTF


#from numba import jit, config
#if config.DISABLE_JIT == True:
#	print('Numba JIT disabled.')
#else:
#	print('Numba JIT enabled.')

# Test 1D simulations
def _test_1D_sim():
	print('Testing 1D simulations.')

	from pyCTF.simulation import CTFSimulation1D
	import matplotlib.pyplot as plt

	CTF = CTFSimulation1D( 5.0, int(2000), 200, -500,
		delta_current=0.0001,
		current=5.0,
		beta=0.1,#mrad
		delta_E=3.0,
		delta_voltage=0.0001,
		aperture=2.5,
		mode=0)

	fig, ax = CTF.plot_ctf()
	#ax.set_prop_cycle(color=['red', 'green', 'blue'])
	ax.set_ylim([0,None])

	CTF.show_all()

	CTF.print_aberrations()
	return

# Test 2D simulations
def _test_2D_sim():
	print('Testing 2D simulations.')

	from pyCTF.simulation import CTFSimulation2D
	image_size = int(256) # Side length of image in pixels.
	acc_voltage = 200 # Accelerating voltage in kV.
	max_frequency = 6 # Maximum spatial frequency in nm-1.
	defocus = -300 # Defocus in nm.

	CTF2D = CTFSimulation2D( max_frequency,
	                         image_size,
	                         acc_voltage,
	                         defocus,
	                         aperture=3.0,
	                         C12a = 400,
	                         C12b = 0,
	                         phi = np.deg2rad(45),
	                         mode=0 )
	CTF2D.defocus=1000e-9
	CTF2D.update()
	CTF2D.plot_ctf()
	CTF2D.show_all()
	CTF2D.print_aberrations()
	print(CTF2D.scale)
	return


# Test defocus measurement.
def _test_defocus( filepath ):
	print('Testing defocus measurements.')

	from pyCTF.image import ElectronImage
	from pyCTF.utils import show_image
	from pyCTF.utils import normalise_data_range

	
	CTF = pyCTF.image.import_ctf( np.array( Image.open( filepath )), 
												200, 
												0.0066127 )

	CTF.remove_background( 8, 4 )

	from pyCTF.image import measure_defocus
	from pyCTF.image import print_Cs_results
	measure_defocus( CTF,
	                 polynomial=20,# Polynomial for Savitsky-Golay smoothing.
	                 window=3,# Window size for Savitsky-Golay smoothing.
	                 f_limits=[0.1,3.0],# Range of freqeuncy to fit.
	                     underfocus=True,# False for overfocus.
	                     xlim=[0.75,3.0],# Exclude all minima outside this range.
	                     start=2 )# First index for fitting.
	print_Cs_results( CTF, verbose=True )

	plt.show()
	return


# Testing astig functions.
def _test_astig( filepath ):
	print('Testing astigmatism measurements.')

	from pyCTF.image import ElectronImage
	from pyCTF.utils import show_image

	# Import the CTF and remove the background.
	CTF = pyCTF.image.import_ctf( np.array( Image.open( filepath )), 
												200, 
												0.0066127 )
	CTF.remove_background(10, 10)

	from pyCTF.astig import astig_angle
	from pyCTF.astig import Astig
	# Measure the angle of twofold astigmatism.
	astig_angle( CTF )
	Astig.plot_angles( CTF )
	print( CTF.amax )
	return


def _test_other( filepath ):
	print('Testing overlaying images.')
	from pyCTF import image
	from pyCTF.utils import composite_image, show_image, normalise_data_range
	from pyCTF.simulation import CTFSimulation2D

	# First, import the experimental CTF.
	CTF = pyCTF.image.import_ctf( np.array( Image.open( filepath )), 
												200, 
												0.0066127 )
	CTF.remove_background(10, 10)
	# Then, simulate a 2D CTF.
	image_size = CTF.width
	acc_voltage = 200
	max_frequency = CTF.max_freq_inscribed*2
	defocus = -484
	simCTF = CTFSimulation2D(max_frequency, image_size, acc_voltage, defocus)
	simCTF.Cs=1.3e-3
	simCTF.update()

	# Then create the overlay.
	composite = composite_image( CTF.image,# Base image.
	                 simCTF.square_CTF,# Image to overlay.
	                 int(CTF.width/2) )# Size of output.
	show_image( composite, scale=CTF.scale )
	return

def _test_fourier( filepath ):
	print('Testing Fourier functions.')
	from pyCTF.fourier import Fourier
	from pyCTF.utils import show_image

	image =  np.array( Image.open( filepath ))
	FFT = Fourier.imfft( image )
	FFT = Fourier.crop( FFT, 600 )
	FFT = Fourier.log_mod( FFT )

	show_image( FFT, scale=0.14782, length=5 )
	return


def _test_profiles( filepath ):
	print('Testing line profiles.')

	from pyCTF.image import ElectronImage
	from pyCTF.utils import show_image
	from pyCTF.utils import normalise_data_range

	
	CTF = pyCTF.image.import_ctf( np.array( Image.open( filepath )), 
												200, 
												0.0066127 )
	CTF.image = normalise_data_range(CTF.image)
	CTF.image = pyCTF.utils._mask_dc_frequency( CTF.image, 0.4, 3)
	CTF.get_profiles(f_limits=[0, 3.0])

	#CTF.plot_profiles()
	fig, ax = plt.subplots(1)
	ax.plot( CTF.frequency, CTF.radial_profile )
	ax.set_box_aspect(1)
	ax.set_ylabel('Intensity / a.u.', fontsize = 16)
	ax.set_xlabel('Frequency / $nm^{-1}$', fontsize = 16)
	ax.set_xlabel('Frequency / $nm^{-1}$', fontsize = 16)
	ax.set_xticks([])
	ax.legend()
	plt.show()
	return


def _test_astig_measure(filepath):
	from pyCTF.image import ElectronImage
	from pyCTF.astig import Astig
	from pyCTF.utils import normalise_data_range

	CTF = pyCTF.image.import_ctf( np.array( Image.open( filepath )), 
												200, 
												0.0066127*2)
	CTF.remove_background( 15, 10 )

	#from pyCTF.astig import astig_angle
	#astig_angle( CTF )
	#Astig.plot_angles( CTF )
	#print( CTF.amax )

	#from pyCTF.utils import show_image
	#show_image(CTF.image)

	from pyCTF.simulation import CTFSimulation2D
	image_size = CTF.image.shape[0] # Side length of image in pixels.
	#CTF.image[int(CTF.image.shape[0]/2), int(CTF.image.shape[0]/2)] = 0

	CTF2D = CTFSimulation2D( CTF.image.shape[0]*0.0066127*2,
	                         image_size,#size
	                         200,#kv
	                         -650)# defocus
	CTF2D.phi = np.deg2rad(40.3)
	CTF2D.update()

	slices = 15

	vals, a, polar_list = Astig.magnitude_measure( normalise_data_range(CTF.image),
													slices,
													100,
													CTF2D )
	Astig.plot_results( CTF, vals, slices, a, CTF2D )
	return


# Script starts here.
print('Starting tests.')

path = 'C:\\Users\\pczbw2\\Desktop\\git\\pyCTF\\assets\\'

#_test_astig_measure( path+'example_astigmatism2.tif' )

#_TFS( path+'\\wip\\example_TFS.tif' )



#_test_1D_sim()
#_test_2D_sim()
#_test_show_image( path+'example_image.tif' )
#_test_defocus( path + 'example_CTF.tif' )
#_test_profiles( path + 'example_CTF.tif' )
#_test_background_subtraction( path+'example_CTF.tif' )
#_test_astig( 'assets\\example_astigmatism.tif' )
#_test_chromatic()
#_test_other( path+'example_CTF.tif' )
#_test_fourier('assets\\example_image.tif')
#_test_TFS( 'assets\\example_TFS.tif' )
#_test_custom( 'C:\\Users\\pczbw2\\Desktop\\git\\pyCTF\\assets\\test_img.tif' )
'''
from pyCTF.figures import zernike_polynomials

zernike = zernike_polynomials()

size = 1024

arr = np.ones([size, size])

from pyCTF.utils import find_iradius_itheta

iradius, itheta = find_iradius_itheta( arr, 1 )

from pyCTF.image import ElectronImage
from pyCTF.utils import show_image
from pyCTF.utils import normalise_data_range

	
CTF = pyCTF.image.import_ctf( np.array( Image.open( path + 'example_CTF.tif'  )), 
												200, 
												0.0066127 )
CTF.C12 = 50e-9
CTF.defocus = 1000e-9
CTF.Cs = 1.0e-3

#polynomials = zernike._phase_plate( CTF, CTF.image.shape[0]/2, CTF.iradius, CTF.itheta )
polynomials = zernike._phase_plate( arr, arr.shape[0]/2, iradius, itheta )

#fig, ax = zernike._plot_phase_plate( polynomials, 0.01 )
fig, ax = zernike._plot_all(polynomials)
'''
print('Tests finished.')

plt.show()