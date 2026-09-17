'''
Script to measure the defocus at each step in a through focus series.
'''

# Timing variables for testing
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import pyCTF
import skimage
from skimage.filters import gaussian
from pyCTF.zeros import Zeros
from scipy.optimize import curve_fit


def hyperbola( x, a, b, c, d ):
	y = (a/(b*x)) + x*c + d
	return y


# Load data and set variables.
yscale = 6.25 #in nm
xscale = 0.06632 # in nm-1
path = 'C:\\Users\\pczbw2\\Desktop\\git\\pyCTF\\assets\\'

arr = np.load( path+'\\wip\\TFS.npy' )
arr = np.rot90(arr)

arr = arr[:, 1:50]


# Apply Gaussian blur to TFS.
arr = gaussian(arr, sigma=1)


# Create variables to hold results of processing.
freq = np.linspace(0, arr.shape[1])
output = np.zeros([arr.shape[0], 8])


# Find the minima at each step of the TFS.
for n in range(arr.shape[0]):
	minima, _ = Zeros.calc_zeros( arr[n, :] )
	minima = Zeros.filter_zeros(minima,
								arr[n, :],
								freq,
								[10,40],
								[None,0.4] )
	for m in range(minima.shape[0]):
		output[n, m] = minima[m]


# Replace all zeros with NaN. 
output[output == 0] = np.nan

def linear(x, m, c):
	y = m*x + c
	return y

# Determine Cs and delta-f.

y_min = [0,0,0,0]
for n in (0, 1, 2, 3):
	y_min[n] = (n+1)/output[0][n]**2


popt4, pcov4 = curve_fit( linear,
						(output[0,0:4])**2,
						y_min,
						maxfev=5000,
						nan_policy='omit' )

	#results, Cs, defocus = Zeros.fit_numpy( output[0,0:4]**2, y_min, lamb=pyCTF.utils.kv_to_lamb(200) )
	#print(str(Cs), str(defocus))
#except:
#	print('Error: could not determine defocus.')


# Then fit to hyperbola.
# Filter based on frequency
step = np.linspace((-yscale*arr.shape[0]),
					yscale*arr.shape[0],
					arr.shape[0])


# Filter the region near zero focus.
overfocus = (step > 50)
underfocus = (step < -50)


# Params stored as overfocus, underfocus.
fitparams = [[],[],[],[]]

# Fit the hyperbola.
for n in (1, 2, 3, 4):
	popt1, pcov1 = curve_fit( hyperbola,
							(output[:,n-1][overfocus])**2,
							n/output[:,n-1][overfocus]**2,
							maxfev=5000,
							nan_policy='omit' )
	popt2, pcov2 = curve_fit( hyperbola,
							(output[:,n-1][underfocus])**2,
							-n/output[:,n-1][underfocus]**2,
							maxfev=5000,
							nan_policy='omit' )
	fitparams[n-1] = popt1, pcov1, popt2, pcov2



# Plot results.
fig, ax = plt.subplots(1, 2)

# Axis 1.
ax[0].imshow(arr)

# Plot the picked minima on the TFS.
ax[0].plot( output[:,0], range(0, 200),  'x', color='k', label='n=|1|')
ax[0].plot( output[:,1], range(0, 200), 'x', color='y', label='n=|2|')
ax[0].plot( output[:,2], range(0, 200), 'x', color='m', label='n=|3|')
ax[0].plot( output[:,3], range(0, 200), 'x', color='c', label='n=|4|')
ax[0].set_title('Detected minima')

ax[0].legend()

# Set the scale of the x- abd y-axes correctly.
xscale = 0.06632
ticks = np.array([0,\
				arr.shape[1]*0.25,\
				arr.shape[1]*0.5,\
				arr.shape[1]*0.75,\
				arr.shape[1]])

labels = np.array([ str(0),\
        str(np.round(arr.shape[1]*0.25*xscale,2)),\
        str(np.round(arr.shape[1]*0.5*xscale,2)),\
        str(np.round(arr.shape[1]*0.75*xscale,2)),\
        str(np.round(arr.shape[1]*xscale,2))])

ax[0].set_xticks( ticks, labels=labels)


ticks = np.array([0,\
				arr.shape[0]*0.25,\
				arr.shape[0]*0.5,\
				arr.shape[0]*0.75,\
				arr.shape[0]])

labels = np.array([str(np.round(-arr.shape[0]*0.5*yscale,2)),\
        str(np.round(-arr.shape[0]*0.25*yscale,2)),\
        str(0),
        str(np.round(arr.shape[0]*0.25*yscale,2)),\
        str(np.round(arr.shape[0]*0.5*yscale,2))])

ax[0].set_yticks( ticks, labels=labels)

ax[0].set_xlabel('Frequency / nm$^{-1}$')
ax[0].set_ylabel('Defocus / nm')


# Plot hyperbola.
colours = ('k','y','m','c')

for n in (0, 1, 2, 3):
	ax[1].plot(hyperbola(range(0, 2000),
				*fitparams[n][0]),
				c=colours[n],
				ls='-',
				label='overfocus' )
	ax[1].plot(hyperbola(range(0, 2000),
				*fitparams[n][2]),
				c=colours[n],
				ls='-',
				label='underfocus' )

'''
for n, m in zip((1, 2, 3, 4), colours):
	ax[1].plot( output[:,n-1][overfocus]**2, n/(output[:,n-1][overfocus])**2, 
		'x', color=m,
		label='n='+str(n))
	ax[1].plot( output[:,n-1][underfocus]**2, -n/(output[:,n-1][underfocus])**2, 
		'x', color=m,
		label='n='+str(-n))
'''

ax[1].set_ylim([-0.015, 0.015])
ax[1].set_xlim([0, 2000])

#ax[1].legend()

ax[1].set_xlabel('${nu}^2$')
ax[1].set_ylabel('$n/{nu}^2$')
ax[1].grid()
ax[1].set_title('Fitted CTF minima')

ax[1].plot(linear(range(0, 2000),
				*popt4),
				c='r',
				ls='--',
				label='underfocus' )
ax[1].plot( (output[0,0:4])**2, y_min,'x',
				c='k' )

plt.show()