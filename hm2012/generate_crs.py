import numpy as np
from scipy import integrate, interpolate

import phfit2 # verner et. al. 1996 fit

output_filename = "hm12_avg_crs.dat"
data_file       = "hm2012uvb.dat"

angstrom_to_cm = 1.0E-8
c = 2.998E10        # cm / s
h = 4.135667662E-15 # ev * s

Mb_to_cgs = 1.0E-18 # cm**2

v_hi         = 13.59840 / h
vscale       = v_hi
v_hi         = v_hi / vscale
v_hei        = 24.58740 / h / vscale
v_heii       = 54.51776 / h / vscale
v_heii_lim   = 62.0 / h / vscale

nshell       = 7

def phfit_wrapper(z, ne, ishell, photon_energy):
    return phfit2.phfit2(z, ne, ishell, photon_energy)

def total_crs(z, ne, photon_energy):
    crs = 0.0

    for ishell in np.arange(1, nshell + 1):
        crs += phfit_wrapper(z, ne, ishell, photon_energy)

    return crs * Mb_to_cgs

def HI_crs(freq):
    E_photon = freq * h
    z        = 1
    ne       = 1

    return total_crs(z, ne, E_photon)

def HeI_crs(freq):
    E_photon = freq * h
    z        = 2
    ne       = 2

    return total_crs(z, ne, E_photon)

def HeII_crs(freq):
    E_photon = freq * h
    z        = 2
    ne       = 1

    return total_crs(z, ne, E_photon)


#
# load redshift bins
#
z = np.genfromtxt(data_file, skip_header = 3, max_rows=1)
nz = np.size(z)
#
# load the rest of the data
#
data       = np.genfromtxt(data_file, skip_header = 4)
wavelength = data[:,0]
frequency  = c / (angstrom_to_cm * wavelength[::-1])

#
# for each redshift, compute crs for HI, HeI, HeII
#
hi_avg_crs   = np.zeros(nz)
hi_avg_crs_2 = np.zeros(nz)
hei_avg_crs  = np.zeros(nz)
heii_avg_crs = np.zeros(nz)


outfile = open(output_filename, 'w')
outfile.write("#z hi_avg_crs hi_avg_crs_2 hei_avg_crs heii_avg_crs\n")

for i in np.arange(nz):

    f = interpolate.interp1d(frequency / vscale, data[:, i+1][::-1], kind='linear')

#
# HI : integrate from HI to HeII
#

    numerator_integrand   = lambda x : (f(x) / x) * HI_crs(x*vscale)
    denominator_integrand = lambda x : (f(x) / x)

    num   = integrate.quad(numerator_integrand, v_hi, v_heii)[0]
    denom = integrate.quad(denominator_integrand, v_hi, v_heii)[0]

    if num == 0.0 or denom == 0.0:
        hi_avg_crs[i] = 0.0
    else:
        hi_avg_crs[i] = num / denom
    print 'HI', num, denom

#
# HI : integrate from HI to HeI
#

    num   = integrate.quad(numerator_integrand, v_hi, v_hei)[0]
    denom = integrate.quad(denominator_integrand, v_hi, v_hei)[0]

    if num == 0.0 or denom == 0.0:
        hi_avg_crs_2[i] = 0.0
    else:
        hi_avg_crs_2[i] = num / denom
    print 'HI 2', num, denom
#
# HeI : integrate from HeI to HeII
#

    numerator_integrand   = lambda x : (f(x) / x) * HeI_crs(x*vscale)
   
    num   = integrate.quad(numerator_integrand, v_hei, v_heii)[0]
    denom = integrate.quad(denominator_integrand, v_hei, v_heii)[0]

    if num == 0.0 or denom == 0.0:
        hei_avg_crs[i] = 0.0
    else:
        hei_avg_crs[i] = num / denom
    print 'HeI', num, denom
#
# HeII : integrate from HeII to last frequency bin
#

    numerator_integrand   = lambda x : (f(x) / x) * HeII_crs(x*vscale)
    
    num   = integrate.quad(numerator_integrand, v_heii, v_heii_lim)[0]
    denom = integrate.quad(denominator_integrand, v_heii, v_heii_lim)[0]

    if num == 0.0 or denom == 0.0:
        heii_avg_crs[i] = 0.0
    else:
        heii_avg_crs[i] = num / denom

    outfile.write('%.5E %8.8E %8.8E %8.8E %8.8E\n'%(z[i], hi_avg_crs[i], hi_avg_crs_2[i], hei_avg_crs[i], heii_avg_crs[i]))

outfile.close()
