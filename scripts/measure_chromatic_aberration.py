'''
Script to measure Cc
'''

import pyCTF
from pyCTF.chromatic import chromaticAberration

voltage = 200
voltage_data = np.array([199.70, 199.75, 199.80, 199.85, 199.90, 199.95, 200])
defocus_data =   np.array([ -714.09, -433.04, -197.66, 0, 240.20, 458.89, 688.94 ])*1e-9

Cc = chromaticAberration( voltage,# Primary accelerating voltage in kV.
                          voltage_data,# Series of accelerating voltages in kV.
                          defocus_data)# Series of defocuses, in m.
fit_method = 'lmfit'
Cc.fit( method=fit_method )
fig, ax = Cc.plot_figure( method=fit_method )
Cc.print_results( method=fit_method )