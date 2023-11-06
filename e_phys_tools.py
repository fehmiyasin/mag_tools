#!/usr/bin/env python
'''
A module that includes functions for calculating certain physical variables
of an electron prepared at a given input energy. Includes relatavistic
wavelength, spatial and angular separation due to grating diffraction, velocity,
phase imparted onto wave due to specimen with a given input mean inner
potential, V_i, and it's  conjugate.  
'''
__author__ = "Fehmi Yasin"
__version__ = "1.0.0"
__date__ = "18/03/06"
__maintainer__ = "Fehmi Yasin"
__email__ = "fyasin@uoregon.edu"

import numpy as np

E_0 = 511. #keV
c = 2.9979*10**(8) #m/s

def e_velocity(U):
    '''U is the accelerating voltage of the 
    electron gun, or the potential used to accelerate the electron.'''
    E_0 = 511. #keV
    c = 2.99792458*10**(8) #m/s
    
    v = c*np.sqrt(1. - (1./(1.+ U/E_0)**2)) #m/s
    return v

def e_wavelength(U):
    '''
    v_e is the electron's relativistic velocity down the column.
    '''
    m_0 = 9.1091*10**(-31) # kg (electron's rest mass)
    h = 6.6256*10**(-34)/(1.602*10**(-16)) # keV s
    E_0 = 511. # keV
    c = 2.9979*10**(8) #m/s
    
#     p_e = m_0*v_e # kg m/s
    p_e = (1./c)*np.sqrt(2.*U*E_0 + U**2) # kg m/s
    
    lamda_e = h/p_e 
    
    
    return lamda_e

def spatial_and_angular_sep(d, lamda_e, z, M):
    '''
    This function calculates the angular and real-space separation between
    beams diffracted from a grating-like diffraction hologram, much the same as
    the ones fabricated by the McMorran Lab. d is the pitch of the grating (the
    peak to peak separation in the grating pattern, which is typically
    sinusoidal), while lamda_e is the wavelength of the electron's travelling
    down the column, which can be calculated from the accelerating voltage using 
    the previously defined function e_wavelength(). z is the physiccal distance
    between the diffraction hologram and the specimen plane and M is the
    magnification of the probe-forming optics.
    '''
    
    del_theta = lamda_e/d
    
    del_x = z*del_theta/M
    
    return del_x, del_theta

def t_from_phi(phi, E=300, V_i=0.0078):
    '''
    E is the acceleration voltage times e, or the energy of
    the accelerated electron, phi is the phase of the material
    you want to convert to thickness, and V_i is the mean
    inner potential in kV
    '''
    import numpy as np
    
    E_0 = 511 # keV
    E = E # keV
    h = 6.6256*10**(-34)/(1.602*10**(-16)) # keV s
    c = 2.99792458*10**(8) #m/s
    V_i = V_i # kV
    p_e = (1./c)*np.sqrt(2.*E*E_0 + E**2) # kg m/s
    
    e_wavelength = h/p_e # m
    
    print('e_wavelength = ' + str(e_wavelength))
    t = (2*E_0 + E)/(E_0 + E)*((E*e_wavelength)/(V_i*2*np.pi))*phi
    
    return t

def phi_from_t(t, E=300, V_i=0.0078):
    '''
    E is the acceleration voltage times e, or the energy of
    the accelerated electron, phi is the phase of the material
    you want to convert to thickness, and V_i is the mean
    inner potential in kV
    '''
    import numpy as np
    
    E_0 = 511 # keV
    E = E # keV
    h = 6.6256*10**(-34)/(1.602*10**(-16)) # keV s
    c = 2.99792458*10**(8) #m/s
    V_i = V_i # kV
    p_e = (1./c)*np.sqrt(2.*E*E_0 + E**2) # kg m/s
    
    e_wavelength = h/p_e # m
    
    print('e_wavelength = ' + str(e_wavelength))
    phi = t*(E_0 + E)/(2*E_0 + E)*((V_i*2*np.pi)/(E*e_wavelength))
    
    return phi
