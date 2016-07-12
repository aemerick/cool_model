#
# Adapted from pyGrackle example script of the same name
#
#

from matplotlib import pyplot
import os
import yt
import numpy as np

from pygrackle import \
    FluidContainer, \
    chemistry_data, \
    evolve_constant_density

from pygrackle.utilities.physical_constants import \
    mass_hydrogen_cgs, \
    sec_per_Myr, \
    cm_per_mpc

tiny_number = 1.0E-20

safety_factor    = 0.4
current_redshift = 0.0

#metallicities = [0.0001, 0.001]
metallicities = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.005, 0.0075, 0.01, 0.015]

#metallicities = [0.01, 0.015]
for z in metallicities:

    cloudy_file      = "CloudyData_UVB=HM2012.h5"
    metal_fraction   = z
    outname          = "uvb_eq_Z=%.5f_z=%.1f.txt"%(metal_fraction, current_redshift)

    i = 0
    npoints = 100

    densities = np.logspace(-4, 3.0, npoints)
    eq_temperature = np.zeros(npoints)
    eq_density     = np.zeros(npoints)
    eq_n_density   = np.zeros(npoints)
    eq_pressure    = np.zeros(npoints)

    file = open(outname, "w")
    file.write("#n rho T P\n")



    for n in densities:
        # initial values (loop over this)
        density = n * 1.0

        initial_temperature = 4000.0
        final_time          = 1000.0 # Myr

        # chemistry
        my_chemistry = chemistry_data()
        my_chemistry.use_grackle            = 1
        my_chemistry.with_radiative_cooling = 1
        my_chemistry.primordial_chemistry   = 1
        my_chemistry.metal_cooling          = 1
        my_chemistry.self_shielding_method  = 3
        my_chemistry.UVbackground           = 1

        grackle_dir = '/home/emerick/code/grackle'
        my_chemistry.grackle_data_file = grackle_dir + '/input/' + cloudy_file

        # Set units
        my_chemistry.comoving_coordinates = 0 # proper units
        my_chemistry.a_units = 1.0
        my_chemistry.a_value = 1. / (1. + current_redshift) / \
            my_chemistry.a_units
        my_chemistry.density_units =  mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g
        my_chemistry.length_units  =  cm_per_mpc         # 1 Mpc in cm
        my_chemistry.time_units    =  sec_per_Myr          # 1 Myr in s
        my_chemistry.velocity_units = my_chemistry.a_units * \
            (my_chemistry.length_units / my_chemistry.a_value) / \
            my_chemistry.time_units

        # self-shielding cross sections in CGS
        my_chemistry.hi_pi_avg_cross_section  = 2.49E-18
        my_chemistry.hi_ph_avg_cross_section  = 2.49E-18
        my_chemistry.hei_ph_avg_cross_section = 4.1294e-18
        my_chemistry.hei_pi_avg_cross_section = 4.1294e-18
        my_chemistry.heii_ph_avg_cross_section = 0.0
        my_chemistry.heii_pi_avg_cross_section = 0.0


        rval = my_chemistry.initialize()

        fc = FluidContainer(my_chemistry, 1)
        fc["density"][:] = density
        if my_chemistry.primordial_chemistry > 0:
            fc["HI"][:] = 0.76 * fc["density"]
            fc["HII"][:] = tiny_number * fc["density"]
            fc["HeI"][:] = (1.0 - 0.76) * fc["density"]
            fc["HeII"][:] = tiny_number * fc["density"]
            fc["HeIII"][:] = tiny_number * fc["density"]
        if my_chemistry.primordial_chemistry > 1:
            fc["H2I"][:] = tiny_number * fc["density"]
            fc["H2II"][:] = tiny_number * fc["density"]
            fc["HM"][:] = tiny_number * fc["density"]
            fc["de"][:] = tiny_number * fc["density"]
        if my_chemistry.primordial_chemistry > 2:
            fc["DI"][:] = 2.0 * 3.4e-5 * fc["density"]
            fc["DII"][:] = tiny_number * fc["density"]
            fc["HDI"][:] = tiny_number * fc["density"]
        if my_chemistry.metal_cooling == 1:
            fc["metal"][:] = metal_fraction * fc["density"] 


        fc["x-velocity"][:] = 0.0
        fc["y-velocity"][:] = 0.0
        fc["z-velocity"][:] = 0.0
  
        fc["energy"][:] = initial_temperature / \
            fc.chemistry_data.temperature_units
        fc.calculate_temperature()
        fc["energy"][:] *= initial_temperature / fc["temperature"]


        # let gas cool at constant density
        data = evolve_constant_density(
            fc, final_time=final_time,
            safety_factor=safety_factor, verbose=False, ignore_time_evolution = True)

        
        eq_temperature[i] = data["temperature"][-1]
        eq_density[i]     = data["density"][-1]
        eq_n_density[i]   = data["density"][-1] / (fc.calculate_mean_molecular_weight()[0] * mass_hydrogen_cgs)
        eq_pressure[i]    = data["pressure"][-1]

        file.write("%8.8E %8.8E %8.8E %8.8E\n"%(eq_n_density[i], eq_density[i], eq_temperature[i], eq_pressure[i]))
        i = i + 1
        
    file.close()
