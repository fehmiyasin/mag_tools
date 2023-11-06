"""
This python module stores some functions that are useful in electron microscopy
simulations. Many of these were developed through research necessity and/or
an e-optics course that Alice, Cameron and I developed and co-taught each other
from 171020 to 180601."""
__author__ = "Fehmi Yasin"
__creation_date__ = "18/02/21"
__update__ = "21/01/15"
__version__ = "1.1"
__maintainer__ = "Fehmi Yasin"
__email__ = "fyasin@uoregon.edu"

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kn

def fft(array):
    fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(array)))
    return(fft)

def ifft(array):
    ifft = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(array)))
    return(ifft)

def multislice_CTEM(trans_sp,pix_x=1,pix_y=1,del_z=1,
                    lamda=0.0025080098785156674,PAD=True,TEST=False):
    """
    This function inputs a specimen transmission function array with shape
    (N_slices,N_Y,N_X), where N_slices*del_z is the thickness of the specimen in
    nm. A plane wave interacts with a slice, then the exit wave is
    propagated to the next real space slice via forward and inverse FFTs. This
    procedure is looped over the thickness of the specimen, and the resulting
    output wave function is returned. The specimen transmission function is
    symmetrically bandwidth limitted to 2/3 of the field of view to prevent
    aliasing.
    lamda is the electron wavelength in nm.
    """
    
    # define the height and width of the slices pythonically
    if PAD==True:
        # PAD the array (quadruple the size) in order to avoid
        # edge-induced FFT artifacts within the output array
        N_PIX_KY_ORIG = np.shape(trans_sp)[1]
        N_PIX_KX_ORIG = np.shape(trans_sp)[2]
        X_PAD = int(4*N_PIX_KX_ORIG) 
        SPEC = np.pad(np.copy(trans_sp),pad_width=((0,0),(X_PAD//2,X_PAD//2),
                                                   (X_PAD//2,X_PAD//2)),
                      mode = 'reflect', reflect_type = 'even')
##                      mode='constant', constant_values = np.mean(trans_sp))
        if TEST==True:
            print("PAD = ", str(X_PAD//2), "\n SPEC shape = ", np.shape(SPEC))
            plt.figure(figsize = (10,10))
            plt.imshow(np.angle(np.sum(SPEC,axis = 0)))
            plt.show()
    else:
        SPEC = np.copy(trans_sp)
    for i, slice_2d in enumerate(SPEC):
        h, w = slice_2d.shape[:2]
        radius = int(2 / 3 * np.min([h / 2, w / 2]))
        SPEC[i] = symmetric_bw(slice_2d, radius = radius)
    
    # Initialize the incident wave function psi_0(x,y) = 1
    BEAM = np.ones((np.shape(SPEC)[-2:]),dtype=np.complex64)
    LAMDA=lamda
    N_SLICES = np.shape(SPEC)[0]
    N_PIX_KY = np.shape(SPEC)[1]
    N_PIX_KX = np.shape(SPEC)[2]
    PIX_KY = 1./(N_PIX_KY*pix_y)
    PIX_KX = 1./(N_PIX_KX*pix_x)
    KY_MAX = PIX_KY*N_PIX_KY/2
    KX_MAX = PIX_KX*N_PIX_KX/2
    KY_1D =  np.linspace(-KY_MAX,KY_MAX,N_PIX_KY)
    KX_1D =  np.linspace(-KX_MAX,KX_MAX,N_PIX_KX)
    KX, KY = np.meshgrid(KX_1D,KY_1D)
    DEFOCUS = np.linspace(-N_SLICES*del_z/2,+N_SLICES*del_z/2,N_SLICES)
    #DEFOCUS = np.linspace(-N_SLICES*del_z,0,N_SLICES)

    for i,DF in enumerate(DEFOCUS):
        PROP = np.exp(np.array(-1.0j*np.pi*LAMDA*(KX**2+KY**2)*DF
                               ,dtype=np.complex64))
        BEAM = ifft(PROP*fft(np.copy(BEAM)*SPEC[i]))
    if PAD==True:
        MID_X = N_PIX_KX//2
        MID_Y = N_PIX_KY//2
        BOX = slice(MID_Y-N_PIX_KY_ORIG//2,
                    MID_Y+N_PIX_KY_ORIG//2),slice(MID_X-N_PIX_KX_ORIG//2,
                                                  MID_X+N_PIX_KX_ORIG//2)
        return(BEAM[BOX])
    else:
        return(BEAM)

def multislice_STEM(probe,trans_sp,pix_kx=1,pix_ky=1,del_z=1,
                    lamda=0.025080098785156674):
    """
    This function inputs a electron probe wavefunction array and specimen
    transmission function array with shape (N_slices,N_Y,N_X), where
    N_slices*del_z is the thickness of the specimen in Angstroms. The probe
    interacts with a slice, then the exit wave is propagated to the next real
    space slice via forward and inverse FFTs. This procedure is looped over the
    thickness of the specimen, and the resulting output wave function is
    returned.
    lamda is the electron wavelength in Angstroms.
    """
    PROBE=probe
    SPEC = trans_sp
    LAMDA=lamda
    N_SLICES = np.shape(trans_sp)[0]
    N_PIX_KY = np.shape(probe)[0]
    N_PIX_KX = np.shape(probe)[1]
    KY_MAX = pix_ky*N_PIX_KY/2
    KX_MAX = pix_kx*N_PIX_KX/2
    KY_1D =  np.linspace(-KY_MAX,KY_MAX,N_PIX_KY)
    KX_1D =  np.linspace(-KX_MAX,KX_MAX,N_PIX_KX)
    KX, KY = np.meshgrid(KX_1D,KY_1D)
    DEFOCUS = np.linspace(-N_SLICES*del_z/2,+N_SLICES*del_z/2,N_SLICES)

    for i,DF in enumerate(DEFOCUS):
        PROP = np.exp(np.array(-1.0j*np.pi*LAMDA*(KX**2+KY**2)*DF
                               ,dtype=np.float32))
        PROBE = ifft(PROP*fft(PROBE*SPEC[i]))
    return(PROBE)

def scan_probe(ap, pix_k=1, x_p=0, y_p=0,N_e=10**8):
    """
    This function inputs a probe-forming aperture function and returns a real
    space probe wave function raster scanned to position (x_p,y_p) by
    introducing a phase in q-space (aperture plane). Aberrations must be
    included in the input aperture function. (x_p, y_p) are the dispacements of
    the probe from the (0,0) position located at the center of the image.
    """

    N_PIX = np.shape(ap)[0]
    K_MAX = pix_k*N_PIX/2
    K_1D =  np.linspace(-K_MAX,K_MAX,N_PIX)
    KX, KY = np.meshgrid(K_1D,K_1D)
    PIX_R = 1/(2*K_MAX)
    R_MAX = PIX_R*N_PIX/2
    if np.abs(x_p)>2*R_MAX or np.abs(y_p)>2*R_MAX:
        print("Warning: Your probe is out of bounds.")
        return(0)
    PSI_K = np.exp(-1.0j*2*np.pi*(KX*(x_p)+KY*(y_p)))*ap
    PSI_R = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(PSI_K)))
    NORM_CONST = np.sqrt(np.sum(np.abs(np.conj(PSI_R)*PSI_R)))
    PSI_R = np.copy(PSI_R)/NORM_CONST*N_e
    return(PSI_R)

def sim_STEM(ap, trans_sp, Nx_steps=1, Ny_steps=1, pix_kx=1, pix_ky=1,
             lamda=0.025080098785156674, del_z=1, real_space_im=True):
    """
    This function simulates the interaction between an electron probe formed via
    an aperture with function 'ap'and a specimen with transmission function
    trans_sp. The real space probe is scanned over the entire field of view with
    step size '2*R_MAX/n_steps' in real space.
    """
    PSI_AP = ap
    LAMDA = lamda
    DEL_Z = del_z
    N_PIX_Y = np.shape(ap)[0]
    N_PIX_X = np.shape(ap)[1]
    PIX_KX = pix_kx
    PIX_KY = pix_ky
    K_MAX = PIX_KX*N_PIX_X/2
    PIX_R = (1/PIX_KX)/N_PIX_X
    R_MAX = PIX_R*N_PIX_X/2
    if real_space_im==True:
        I_R = np.zeros((Ny_steps+1,Nx_steps+1,N_PIX_Y,N_PIX_X),dtype=np.float32)
    I_DET = np.zeros((Ny_steps+1,Nx_steps+1,N_PIX_Y,N_PIX_X),dtype=np.float32)
    
    SCAN_STEPX = PIX_R*N_PIX_X/Nx_steps
    SCAN_STEPY = PIX_R*N_PIX_Y/Ny_steps
    PROBE_SIZE = SCAN_STEPX/4
    for ny,row in enumerate(I_DET):
        for nx,_ in enumerate(row):
            #Calculate the real space probe from the probe forming aperture.
            #print"nx,ny=",nx,ny) 
            if nx==0:
                X_P = PROBE_SIZE
                #printSCAN_STEPX,PROBE_SIZE,R_MAX)
                #print("X_P=",X_P)
            if Nx_steps>1:
                if nx==0:
                    X_P = PROBE_SIZE
                elif nx==(len(row)-1):
                    X_P=nx*SCAN_STEPX-PROBE_SIZE
                else:
                    X_P=nx*SCAN_STEPX
            if ny==0:
                Y_P=PROBE_SIZE
            if Ny_steps>1:
                if ny==0:
                    Y_P=PROBE_SIZE
                elif ny==(len(I_DET)-1):
                    Y_P=ny*SCAN_STEPY-PROBE_SIZE
                else:
                    Y_P=ny*SCAN_STEPY
            X_P-=R_MAX
            Y_P-=R_MAX
            #print"X_P,Y_P = ",X_P,Y_P)
            PSI_R = scan_probe(ap=PSI_AP, pix_k=PIX_KX,
                               x_p=X_P,
                               y_p=Y_P)
            NORM_CONST = np.sqrt(np.sum(np.abs(np.conj(PSI_R)*PSI_R)))
            PSI_R = np.copy(PSI_R)/NORM_CONST
            if real_space_im==True:
                #Calculate the real space probe intensity, for visualization.
                I_R[ny,nx,:,:] = np.abs(np.conj(PSI_R)*PSI_R)
            #Calculate the exit wave function in real space
            if len(np.shape(trans_sp))==3:
                #print("Starting multislice")
                PSI_EX_R = multislice_STEM(probe=PSI_R,trans_sp=trans_sp,
                                           pix_kx=PIX_KX,pix_ky=PIX_KY,
                                           del_z=DEL_Z,
                                           lamda=LAMDA)
            elif len(np.shape(trans_sp))==2:
                PSI_EX_R = PSI_R*trans_sp
            #Propogate the wave function to the detector
            PSI_EX_K = fft(PSI_EX_R)
            #Calculate the intensity detected by the detector
            I_DET[ny,nx,:,:] = np.abs(np.conj(PSI_EX_K)*PSI_EX_K)

    if real_space_im==True:
        #Return the real space probe intensity, for visualization.
        return(I_DET, I_R)
    else:
        return(I_DET)

def sim_TEM(transfer_function, aperture, defocus, noise=True, multislice=True):
    """
    This function simulates transmission of a plane wave of electrons through a
    'transfer function' array with size (NZ, NY, NX) and passes the diffraction
    pattern, aka point spread function, through an objective aperture defined by
    the 2D array 'aperture.' The final image is returned as a real-valued array
    of intensities. 
    """


def simulate_EM_image(v_array, sigma=1.0, lambda_e = 1.96*10**(-12), C_s = 1.1,
                      del_f = 0., fov_x=50, fov_y=50):
    sigma = sigma
    lambda_e = lambda_e # m (300keV electrons)
    kx = np.linspace(-fov_x//2,fov_x//2, np.shape(v_array)[1])
    ky = np.linspace(-fov_y//2,fov_y//2, np.shape(v_array)[0])
    Kx,Ky = np.meshgrid(kx,ky)
    C_s = 1.1
    del_f = -0
    
    K = Kx**2 + Ky**2
    
    # v = v_1 + v_2 + v_10 + v_20
    v=v_1
    t = np.exp2(1.0j*sigma*v)
    
    # Objective lens transfer function
    chi = np.pi*lambda_e*K**2*(0.5*C_s*lambda_e**2*K**2 - del_f)
    A = K<100
    
    h_twid = np.exp2(-1.0j*chi)*A
    
    t_twid = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(t)))
    
    psi = h_twid*t_twid
    
    psi_twid = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(psi)))
    
    # Im = psi_twid*np.conj(psi_twid)
    Im = np.abs(psi_twid)**2
    return Im

def simulate_phase_plate(radius=25000, W = 40000, aperture_scale=1.,
                     CCD_pixels_side=2048,
                     phase=False, theta=np.pi/6.,
                     grating_efficiency_coeff=1./33.):
    # This function creates a grating with specifications input as keywords. 
    #It returns the grating array

    import numpy as np

    # Phase plate specs
    radius = radius
    print("radius = " + str(radius) + " nm.")
    aperture = radius*aperture_scale
    print("aperture = " + str(aperture*2./1000) + " um.")
    
    # Image specs
    L=2*W
    
    x = np.linspace(-W,W,CCD_pixels_side, endpoint=True)
    dx = x[1]-x[0]
    dy = dx
    X,Y = np.meshgrid(x,x)

    arg = X*np.cos(theta) - Y*np.sin(theta)
    arg2 = Y*np.cos(theta) + X*np.sin(theta)

    if phase==False:
        grating = (1+np.cos((2*np.pi/pitch)*
                            (arg +m*np.arctan2(arg2,arg)
                             )))/2.*(np.sqrt((X)**2+(Y)**2) < aperture)
    else:
        grating = np.exp(1.j*(1+np.cos(k*(arg + m*np.arctan2(arg2,arg))))/2.*
                         (grating_efficiency_coeff)) * (np.sqrt((X)**2+(Y)**2) < aperture)

    # Return the grating

    return grating

def simulate_grating(pitch=150, radius=25000, W = 40000, aperture_scale=1.,
                     CCD_pixels_side=2048, xo=0, yo=0,
                     phase=False, theta=np.pi/6.,
                     grating_efficiency_coeff=1./33., m=0):
    # This function creates a grating with specifications input as keywords. 
    #It returns the grating array

    import numpy as np

    # Grating specs
    pitch = pitch
    k = 2*np.pi/pitch
    print("pitch = " + str(pitch) + " nm.")
    radius = radius
    print("radius = " + str(radius) + " nm.")
    aperture = radius*aperture_scale
    print("aperture = " + str(aperture*2./1000) + " um.")
    
    # Image specs
    L=2*W
    
    x = np.linspace(-W,W,CCD_pixels_side, endpoint=True)
    dx = x[1]-x[0]
    dy = dx
    X,Y = np.meshgrid(x,x)

    arg = (X-xo)*np.cos(theta) - (Y-yo)*np.sin(theta)
    arg2 = (Y-yo)*np.cos(theta) + (X-xo)*np.sin(theta)

    if phase==False:
        grating = (1+np.cos(k*(arg +m*np.arctan2(arg2,arg)
                             )))/2.*(np.sqrt((X-xo)**2+(Y-yo)**2) < aperture)
    else:
        grating = np.exp(1.j*(1+np.cos(k*(arg + m*np.arctan2(arg2,arg))))/2.*
                         grating_efficiency_coeff) * (np.sqrt((X-xo)**2+(Y-yo)**2) < aperture)

    # Return the grating

    return grating

def sim_smooth_step_WPO_1D(phase_height, amp_h=0., N_pix=100, x_min=0, x_max=20,
                        step_width=10, step_pos=0):
    '''
    Returns a 1D array that simulated a smooth step function phase specimen. 
    Starts at xmin and ends at x_max with N_pix pixels. The step is step_width
    wide and phase height radians large.
    '''
    
    import matplotlib.pyplot as plt
    x = np.linspace(x_min, x_max, N_pix, dtype=float) - step_pos
    
    step = (6*(x/step_width)**5-15*(x/step_width)**4+
            10*(x/step_width)**3)*(x>0)*(x<=step_width)+(x>step_width)
    
    specimen = (1-amp_h*step)*np.exp(1.0j*phase_height*step)
    ftsize=20
    fig, ax1 = plt.subplots()
    ax1.plot(x, np.unwrap(np.angle(specimen)), label='phase profile', c='b')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel("Phase [rad]", color='b', fontsize=ftsize)
    ax1.tick_params('y', colors='b', labelsize=ftsize)

    ax2 = ax1.twinx()
    ax2.plot(x, np.abs(specimen), label='amplitude profile', c='r')
    ax2.set_ylabel('Amplitude', color='r', fontsize=ftsize)
    ax2.tick_params('y', colors='r', labelsize=ftsize)
    ax1.set_title("specimen " + str(phase_height) + "rad; A=%0.2g" %amp_h,
              fontsize=ftsize)
    fig.tight_layout()
    plt.show()

    
    return specimen

def sim_smooth_step_WPO_2D(phase_height, amp_h=0., N_pix=100, x_min=0, x_max=20,
                           step_width=10, step_pos=0, Y_step=False, function=0):
    '''Returns a 2D array that simulated a smooth step function phase specimen. 
    Starts at xmin and ends at x_max with N_pix pixels. The step is step_width
    wide and phase height radians large.'''
    
    import matplotlib.pyplot as plt
    from smooth_hsv_trh import smooth_hsv as hsv
    
    x = np.linspace(x_min, x_max, N_pix, dtype=float) - step_pos
    X,Y = np.meshgrid(x,x)
    
    step = (6*(X/step_width)**5-15*(X/step_width)**4+
            10*(X/step_width)**3)*(X>0)*(X<=step_width)+(X>step_width)
    if Y_step==True:
        step = ((6*(Y/step_width)**5-15*(Y/step_width)**4+
                10*(Y/step_width)**3)*(Y>0)*(Y<=step_width)+(Y>step_width))*function
        
    specimen = (1-amp_h*step)*np.exp(1.0j*phase_height*step)
    
    ftsize=20
    image = hsv(specimen)


    fig, ax1 = plt.subplots()
    p=ax1.imshow(image.data)
    ax1.set_title("specimen " + str(phase_height) + "rad; A=%0.2g" %amp_h,
              fontsize=ftsize)
    cbar = fig.colorbar(image.fake_data)
    
    fig.tight_layout()
    plt.show()

    
    return specimen
    
def symmetric_bw(image, radius=False):
    '''
    This function masks the input image, setting all pixels outside a circle of
    radius defined either by the user or by half the length of the smaller
    dimension to zero.
    '''
    
    # define the height and width of the image
    h, w = image.shape[:2]
    
    # define the center of the image
    midy = int(h / 2)
    midx = int(w / 2)
    center = (midx, midy)
    
    # define the radius of the circle mask
    if radius == False:
        radius = int(np.min([midy, midx]))

    # define the circle mask
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return(mask * image)

def twoD_Gaussian(x_y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    '''
    This function returns a 2D array that is a 2D gaussian function. The input
    params are:
    meshgrid x_y = (X,Y),
    amplitude => max value of gaussian (=1 default)
    center of gaussian (x0, y0)
    sigma_x => standard dev in x direction
    sigma_y => standard dev in y direction
    theta => rotation of gaussian (I think)
    offset => background offset from zero, where the gaussian falls off to
    '''

    x = x_y[0]
    y = x_y[1]
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) +
                                      c*((y-yo)**2)))
    return g.ravel().reshape(len(y), len(x))

def v_z(a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3, x_0, y_0, fov_x = 50,
        fov_y=50, N_pix_x=500, N_pix_y=500):
    '''
    This function returns a 2D array that is the atomic potential for one 
    atom with atomic number z.
    '''
    import numpy as np
    from scipy.special import kn
    
    x = np.linspace(-fov_x//2,fov_x//2, N_pix_x)
    y = np.linspace(-fov_y//2,fov_y//2, N_pix_y)
    X,Y = np.meshgrid(x,y)
    
    r = (X-x_0)**2 + (Y-y_0)**2
    a_0 = 0.529 # angstroms
    q_e = 14.4 # V*angstroms
    v1 = 4*np.pi**2*a_0*q_e*(a1*kn(0,2*np.pi*r*np.sqrt(b1)) + 
                            a2*kn(0,2*np.pi*r*np.sqrt(b2)) + 
                            a3*kn(0,2*np.pi*r*np.sqrt(b3)))
    v2 = 2*np.pi**2*a_0*q_e*(c1/d1*np.exp2(-np.pi**2*r**2/d1) +
                            c2/d2*np.exp2(-np.pi**2*r**2/d2) +
                            c3/d3*np.exp2(-np.pi**2*r**2/d3))
    
    v = v1 + v2
    
    return v
