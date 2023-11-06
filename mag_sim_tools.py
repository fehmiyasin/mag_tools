"""
This python module stores some functions that are useful for in-plane
magnetization, magnetic skyrmion, Lorentz TEM and differential phase contrast simulations. skyrmion_tor() and skyrmion were adapted and modified by Jordan Chess's home-built code he developed sometime in 2013-16.
"""
__author__ = "Fehmi Yasin"
__date__ = "23/03/22"
__version__ = "1.2"
__maintainer__ = "Fehmi Yasin"
__email__ = "fehmi.yasin@riken.jp"

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import smooth_hsv_jjc as smooth_hsv
import scipy as sp
from scipy import ndimage as nd
import vector_tools as vt
from skimage.restoration import unwrap_phase

class SpatialFrequencies(object):
    """
    Defines two reciprocal space variables of the real space object array.
    Note that 'x' is defined to be the pythonic horizontal variable, and
    therefore defined to be the '1' axis.
    """
    def __init__(self, array, pixel_size):

        self.x, self.y = np.meshgrid(np.fft.fftfreq(np.shape(array)[1],
                                                    pixel_size),
                                     np.fft.fftfreq(np.shape(array)[0],
                                                    pixel_size))
        self.squared = self.x ** 2 + self.y ** 2
        x = self.x.astype(float)
        y = self.y.astype(float)
        x[0, 0] = (x[0, 1] + x[1, 0] + x[1, 1])/3. * 0.0001
        y[0, 0] = (y[0, 1] + y[1, 0] + y[1, 1])/3. * 0.0001
        self.inverse2 = 1./(x**2 + y**2)

def calc_divergence(b_arr):
    '''
    Calculates the divergence of a magnetic vector field.
    b_arr has dimensions (3, nz, ny, nx) and b_arr[0:2] are b_x, b_y, and b_z,
    respectively.
    '''

    # Get the dimensions of the array
    _, nz, ny, nx = b_arr.shape
    
    # Calculate the divergence using finite differences
    div = np.zeros((nz, ny, nx), dtype=np.float64)
    
    # Calculate the divergence for each component
    for i in range(3):
        if i == 0:
            div += np.gradient(b_arr[i], axis=2)
        elif i == 1:
            div += np.gradient(b_arr[i], axis=1)
        else:
            div += np.gradient(b_arr[i], axis=0)
    
    return div

def calc_N_topo(bx_2d,by_2d,bz_2d,width=3,TEST=False):
    '''
    Takes a square array that's been preprocessed so that
    the spin texture is located near the center pixel of a 
    rectangular array and calculates the topological charge
    of the spin texture. Note that the input vectors are normalized
    element-wise. 
    '''
    #get the dimensions of the image
    ny,nx = bx_2d.shape
    if ny<nx:
        nx = bx_2d.shape[0]
    else:
        ny = bx_2d.shape[1] 
    print(nx,ny)
    #get the x and y center points of our image
    x_mid = int(nx/2)
    y_mid = int(ny/2)
    box_center = slice(y_mid-ny//2,y_mid+ny//2),slice(x_mid-nx//2,x_mid+nx//2)
    bx_2d = np.copy(bx_2d[box_center])
    by_2d = np.copy(by_2d[box_center])
    bz_2d = np.copy(bz_2d[box_center])
    theta = np.arccos(bz_2d)
    
    radius = np.min([x_mid,y_mid])
    y,x = np.ogrid[0:nx, 0:ny]
    
    # create a circle mask which is centered in the middle of
    # the image with input radius
    circle_mask0 = ((x-x_mid)**2 + (y-y_mid)**2 >= (width)**2)
    theta0_masked = np.copy(theta)
    theta0_masked[circle_mask0]=0
    theta_0 = np.average(bz_2d[theta0_masked!=0])
    if TEST==True:
        print('theta_0 = ',theta_0)
    if theta_0<0:
        topo_n = -calc_winding_number(bx_2d,by_2d,bz_2d)
    else:
        topo_n = calc_winding_number(bx_2d,by_2d,bz_2d)
    return(topo_n)

def calc_winding_number(bx_2d,by_2d,bz_2d):
    '''
    Takes a square array that's been preprocessed so that
    the spin texture is located near the center pixel of a 
    rectangular array and calculates the winding number
    of the spin texture.
    '''
    mag_i = np.array([bx_2d,by_2d,bz_2d])
    mag_j = np.roll(np.roll(mag_i,shift=-1,axis=1),shift=-1,axis=2)
    mag_k = np.roll(mag_i,shift=-1,axis=1)
    mag_k2 = np.roll(mag_i,shift=-1,axis=2)
    mag_i_1D = np.array([np.ravel(mag_i[0]),np.ravel(mag_i[1]),
                         np.ravel(mag_i[2])])
    mag_j_1D = np.array([np.ravel(mag_j[0]),np.ravel(mag_j[1]),
                         np.ravel(mag_j[2])])
    mag_k_1D = np.array([np.ravel(mag_k[0]),np.ravel(mag_k[1]),
                         np.ravel(mag_k[2])])
    mag_k2_1D = np.array([np.ravel(mag_k2[0]),np.ravel(mag_k2[1]),
                          np.ravel(mag_k2[2])])

    Omega = []
    for i in np.arange(np.size(mag_i[0])):
        numerator = np.inner(mag_i_1D[:,i],
                             np.cross(mag_j_1D[:,i],mag_k_1D[:,i]))
        denom = (1 + np.inner(mag_i_1D[:,i],mag_j_1D[:,i]) +
                 np.inner(mag_j_1D[:,i],mag_k_1D[:,i]) +
                 np.inner(mag_k_1D[:,i],mag_i_1D[:,i]))
        numerator2 = np.inner(mag_i_1D[:,i],
                              np.cross(mag_k2_1D[:,i],mag_j_1D[:,i]))
        denom2 = (1 + np.inner(mag_i_1D[:,i],mag_k2_1D[:,i]) +
                  np.inner(mag_k2_1D[:,i],mag_j_1D[:,i]) +
                  np.inner(mag_j_1D[:,i],mag_i_1D[:,i]))
        Omega.append(2*(np.arctan(numerator/denom)+
                        np.arctan(numerator2/denom2)))
    W = -(1/(4*np.pi))*np.sum(np.array(Omega))
    if density == True:
        return(-(1 / (4 * np.pi)) * np.array(Omega))
    else:
        return(W)

def calc_winding_number_density(bx_2d, by_2d, bz_2d):
    '''
    Takes a square array that's been preprocessed so that
    the spin texture is located near the center pixel of a
    rectangular array and calculates the winding number density
    of the spin texture.
    '''
    mag_i = np.array([bx_2d, by_2d, bz_2d])
    mag_j = np.roll(np.roll(mag_i, shift=-1, axis=1), shift=-1, axis=2)
    mag_k = np.roll(mag_i, shift=-1, axis=1)
    mag_k2 = np.roll(mag_i, shift=-1, axis=2)
    mag_i_1D = np.array([np.ravel(mag_i[0]), np.ravel(mag_i[1]),
                         np.ravel(mag_i[2])])
    mag_j_1D = np.array([np.ravel(mag_j[0]), np.ravel(mag_j[1]),
                         np.ravel(mag_j[2])])
    mag_k_1D = np.array([np.ravel(mag_k[0]), np.ravel(mag_k[1]),
                         np.ravel(mag_k[2])])
    mag_k2_1D = np.array([np.ravel(mag_k2[0]), np.ravel(mag_k2[1]),
                          np.ravel(mag_k2[2])])

    winding_number_density = np.zeros(bx_2d.shape[:2])
    for i in range(np.size(mag_i[0])):
        numerator = np.inner(mag_i_1D[:, i],
                             np.cross(mag_j_1D[:, i], mag_k_1D[:, i]))
        denom = (1 + np.inner(mag_i_1D[:, i], mag_j_1D[:, i]) +
                 np.inner(mag_j_1D[:, i], mag_k_1D[:, i]) +
                 np.inner(mag_k_1D[:, i], mag_i_1D[:, i]))
        numerator2 = np.inner(mag_i_1D[:, i],
                              np.cross(mag_k2_1D[:, i], mag_j_1D[:, i]))
        denom2 = (1 + np.inner(mag_i_1D[:, i], mag_k2_1D[:, i]) +
                  np.inner(mag_k2_1D[:, i], mag_j_1D[:, i]) +
                  np.inner(mag_j_1D[:, i], mag_i_1D[:, i]))
        winding_number_density += 2 * (np.arctan(numerator / denom) +
                                       np.arctan(numerator2 / denom2))

    winding_number_density *= -(1 / (4 * np.pi))

    return winding_number_density

def defl(Mx,My,Mz,E_0=200,t=100*(10**-9)):
    """
    Calculate the deflection of electrons due to an in-plane magnetization
    """
    import e_phys_tools as ept
    e=1.602*(10**-19) # C
    lam = ept.e_wavelength(E_0)*(10**9) # nm
    h = 6.626*(10**-16) # nm^2 kg/s
    CL = 15*(10**9) #nm
    mu_0=4*np.pi*(10**-16) #H/nm = N/A^2
    betay=e*lam*2*t*Mx*mu_0/h
    betax=-e*lam*2*t*My*mu_0/h
    delx=np.tan(betax)*CL #nm
    dely=np.tan(betay)*CL #nm
    return delx, dely

def field_from_phase(phase, pxsize, t):
    """Calculate the in plane magnetic field in T from the phase, 
    pxsize == pixel size in nm, t = sample thickness in nm"""
    e = 1.60217662e-19 #C
    hbar = 1.054571800e-34 #J s
    t = float(t)
    ddy, ddx = np.gradient(phase, pxsize)
    Bx = -ddy * (hbar/(e*t)) * 1.0e18 #T
    By = ddx * (hbar/(e*t)) * 1.0e18 #T
    return Bx, By

def load_DAT(file_name):
    """
    Opens a .dat file and loads the magnetization in a numpy array
    using numpy's genfromtxt function.
    
    np.genfromtxt loads a (N,5) array where array[:,0] and array[:,1]
    are the Y and X coordinates (indices) and array[:,2-4] are MY, MX
    and MZ components of the magnetization. The output array has the
    shape (3,YMAX,XMAX), where the 0, 1 and 2 indices represent MY, MX
    and MZ.
    """
    import numpy as np
    import os
    
    FILE_NAME = str(os.path.splitext(file_name)[0])
    EXT = str(os.path.splitext(file_name)[1])
    if EXT != '.dat' and EXT != "":
        print("File must be .dat.")
        return 0
    DATA = np.genfromtxt(str(FILE_NAME+'.dat'))
    YMAX = int(np.max(DATA[:,0])+1)
    XMAX = int(np.max(DATA[:,1])+1)
    M = np.zeros((3,YMAX,XMAX))
    for i,_ in enumerate(DATA[:,0]):
        M[0,int(DATA[i,0]),int(DATA[i,1])]=DATA[i,3]
        M[1,int(DATA[i,0]),int(DATA[i,1])]=DATA[i,2]
        M[2,int(DATA[i,0]),int(DATA[i,1])]=DATA[i,4]
    return M

def load_PNP(file_name, t_pix=1, n_pix=256, meta=False):
    """
    Opens a .pnp file and loads the magnetization into a numpy array
    using numpy's load function.
    
    np.load() loads a (2,) structure where array[0] is the data with shape
    (1,n_pix,n_pix,t_pix,3) where n_pix is the number of in-plane pixels and
    T_PIX is the number of thickness or z pixels. The final 0, 1 and 2 indices
    represent MY, MX and MZ, and array[1] is a dict structure
    with the metadata.
    """
    import numpy as np
    import os
    
    FILE_NAME = str(os.path.splitext(file_name)[0])
    EXT = str(os.path.splitext(file_name)[1])
    if EXT != '.pnp' and EXT != "":
        print("File must be .pnp.")
        return 0
    FILE = np.load(FILE_NAME+".pnp")
    DATA = np.array(FILE[0][0],dtype=np.float64)
    META_DATA = FILE[1]
    YMAX = int(np.max(DATA[:,0])+1)
    XMAX = int(np.max(DATA[:,1])+1)
    M=np.zeros((3,t_pix,n_pix,n_pix))
    M[0,:,:,:]=np.rollaxis(np.copy(DATA[:,:,:,1]),-1,0)
    M[1,:,:,:]=np.rollaxis(np.copy(DATA[:,:,:,0]),-1,0)
    M[2,:,:,:]=np.rollaxis(np.copy(DATA[:,:,:,2]),-1,0)
    if meta==False:
        return M
    if meta==True:
        return M, META_DATA

def load_ovf(file_name):
    """
    This function an .ovf file into a numpy array.
    """

    f = open(file_name,'r')
    lines = f.readlines()
    
    for i,line in enumerate(lines):
        if line[0]=="#":
            temp = line[2:].split(' ')
            if temp[0]=='xnodes:':
                NPIX_X = int(temp[1])
            if temp[0]=='ynodes:':
                NPIX_Y = int(temp[1])
            if temp[0]=='znodes:':
                NPIX_Z = int(temp[1])
            if temp[0]=='xstepsize:':
                XPIX = float(temp[1])
            if temp[0]=='ystepsize:':
                YPIX = float(temp[1])
            if temp[0]=='zstepsize:':
                ZPIX = float(temp[1])
    ARR=np.zeros((3,NPIX_Z*NPIX_Y*NPIX_X))
    print('np.shape(ARR) = ',np.shape(ARR))
    n=0
    for i,line in enumerate(lines):
        if line[0]!="#":
            ARR[:,n]=np.array(line.split(' ')[:3]).astype(float)
            n+=1
    ARR2 = np.reshape(ARR,(3,NPIX_Z,NPIX_Y,NPIX_X))[:,:,::-1,:]
    ARR2[1]*=-1
    print('np.shape(ARR2) = ', np.shape(ARR2))
    return(ARR2,XPIX,YPIX,ZPIX)

def npy_to_ovf(m_array,file_name="",xpix_len=1,ypix_len=1,zpix_len=1):
    """
    This function takes a numpy array with shape (3,NPIX_Z,NPIX_Y,NPIX_X), a
    file save name, and the pixel length in nm of each dimension and returns an
    .ovf file for use in OOMMF or mumax.
    """
    ARR=np.copy(m_array)#[:,:,::-1,:]
    #ARR[1]*=-1
    
    TITLE=file_name
    XPIX = xpix_len
    YPIX = ypix_len
    ZPIX = zpix_len
    NPIX_X = np.shape(m_array)[3]
    NPIX_Y = np.shape(m_array)[2]
    NPIX_Z = np.shape(m_array)[1]
    # Write the array to disk
    with open(str(str(TITLE)+'.ovf'), 'w', newline='\n') as outfile:
        # writing the OOMMF v2.0 header
        outfile.write('# OOMMF OVF 2.0\n'+
                      '# Segment count: 1\n'+
                      '# Begin: Segment\n'+
                      '# Begin: Header\n'+
                      '# Title: '+str(TITLE.split('/')[-1])+'\n'+
#                       '# Desc: "Description" tag, which may be used or ignored by postprocessing\n'+
#                       '# Desc: programs. You can put anything you want here, and can have as many\n'+
#                       '# Desc: "Desc" lines as you want.  The ## comment marker is disabled in\n'+
#                       '# Desc: description lines.\n'+
                      '# meshunit: nm\n'+
                      '# meshtype: rectangular\n'+
                      '# xbase: 0.\n'+
                      '# ybase: 0.\n'+
                      '# zbase: 0.\n'+
                      '# xnodes: '+str(NPIX_X)+'\n'+
                      '# ynodes: '+str(NPIX_Y)+'\n'+
                      '# znodes: '+str(NPIX_Z)+'\n'+
                      '# xstepsize: '+str(XPIX)+'\n'+
                      '# ystepsize: '+str(YPIX)+'\n'+
                      '# zstepsize: '+str(ZPIX)+'\n'+
                      '# xmin: 0.\n'+
                      '# ymin: 0.\n'+
                      '# zmin: '+str(-NPIX_Z*ZPIX/2)+'\n'+
                      '# xmax: '+str(NPIX_X*XPIX)+'\n'+
                      '# ymax: '+str(NPIX_Y*YPIX)+'\n'+
                      '# zmax: '+str(NPIX_Z*ZPIX/2)+'\n'+
                      '# valuedim: 3\n'+
                      '# valuelabels: m_x m_y m_z\n'+
                      '# valueunits: 1 1 1\n' +
                      '# End: Header\n'+
                      '# Begin: data text\n')
        
        # Iterating through the array and writing a magnetization mx my mz on
        # each line, iterated in x,y,z order.
        for k,T_TEMP in enumerate(ARR[0]):
            for j,Y_TEMP in enumerate(T_TEMP):
                for i,_ in enumerate(Y_TEMP):
                    outfile.write(str(str(ARR[0,k,j,i])+' '+
                                      str(ARR[1,k,j,i])+' '+
                                      str(ARR[2,k,j,i])+'\n'))
        # Writing out a break to indicate different slices...
        outfile.write('# End: data text\n'+
                      '# End: Segment')

def phase_from_magnetization(Mx, My, pixel_size, t, Ms=0.000384, PAD=True,
                             TEST=False):
    """pixel_size and t(hickness) should be in nm, Ms should be in A/nm """
    #Mansuripur algorithm, see Chess thesis pg 14-15
    import sim_tools as st
    const = 1.909168 * float(Ms) * float(t)  # 1.9... is np.pi*mu_0/(flux quantum)
    if PAD==True:
        # PAD the array (quadruple the size) in order to avoid
        # edge-induced FFT artifacts within the output array. This 
        N_PIX_KY_ORIG = np.shape(Mx)[0]
        N_PIX_KX_ORIG = np.shape(Mx)[1]
        X_PAD = int(4*N_PIX_KX_ORIG) 

        Mx = np.pad(np.copy(Mx),pad_width=((X_PAD//2,X_PAD//2),
                                                 (X_PAD//2,X_PAD//2)),
                       mode='edge')
        My = np.pad(np.copy(My),pad_width=((X_PAD//2,X_PAD//2),
                                                 (X_PAD//2,X_PAD//2)),
                       mode='edge')
        if TEST==True:
            print("PAD = ", str(X_PAD//2), "\n Mx shape = ", np.shape(Mx))
            print("Mx value ndtype: ",np.dtype(Mx[0,0]))
        N_PIX_KX = np.shape(Mx)[1]
        N_PIX_KY = np.shape(Mx)[0]
        MID_X = N_PIX_KX//2
        MID_Y = N_PIX_KY//2
        BOX = slice(MID_Y-N_PIX_KY_ORIG//2,
                    MID_Y+N_PIX_KY_ORIG//2),slice(MID_X-N_PIX_KX_ORIG//2,
                                                  MID_X+N_PIX_KX_ORIG//2)
    Mqx = (np.fft.fft2(Mx))
    Mqy = (np.fft.fft2(My))
    q = SpatialFrequencies(Mqy, float(pixel_size))
    Phi_q = (-1.0j * const * (Mqx * q.y - Mqy * q.x)) / (q.squared+(7./3-4./3-1))
    ##Phi_q[0, 0] = 0
    if PAD==True:
        phase = np.fft.ifft2(np.copy(Phi_q))[BOX]
    else:
        phase = np.fft.ifft2(np.copy(Phi_q))
    return(phase)

def plot_mag(mx,my,mz,FILE_SAVE_NAME,PIX=1,TITLE_NAME="",scale=20, width=1/200,
             M_STEP=10,ARR=True):
    import matplotlib as mpl
    from matplotlib.colors import colorConverter
    N_PIX = np.shape(mx)[0]
#     BOX = slice(0,N_PIX),slice(0,N_PIX) #full FOV
    MIDX = N_PIX//2
    MIDY = N_PIX//2
    IM_WIDTH = N_PIX//2
    BOX = slice(MIDY-IM_WIDTH,MIDY+IM_WIDTH),slice(MIDX-IM_WIDTH,MIDX+IM_WIDTH)

    MX_NORM=(np.copy(mx)/np.max(np.abs([mx,my,mz])))[BOX]
    MY_NORM=(np.copy(my)/np.max(np.abs([mx,my,mz])))[BOX]
    MZ_NORM=(np.copy(mz)/np.max(np.abs([mx,my,mz])))[BOX]

#     M_STEP=Mint(np.shape(MX_NORM)[0]//(2**6.))

    TEMPX=np.copy(MX_NORM[:-1:2*M_STEP,:-1:2*M_STEP])
    TEMPY=np.copy(MY_NORM[:-1:2*M_STEP,:-1:2*M_STEP])

    B_C = np.copy(MY_NORM) + 1.0j*np.copy(MX_NORM)
    B_C_AMP = np.abs(B_C)
    B_C_PHI = np.angle(B_C)
    
    FTSIZE=35
    TICKSIZE=0.75*FTSIZE
    FIG_SIZE=(13,13)

    NUM_PTS = 5
    INCREMENT = int(round(PIX*np.shape(B_C)[0]//NUM_PTS,-1)) #nm

    X1 = np.linspace(0, PIX*int(np.shape(B_C)[1]), np.shape(TEMPX)[1])
    Y1 = np.linspace(0, PIX*int(np.shape(B_C)[0]), np.shape(TEMPX)[0])

    X, Y = np.meshgrid(X1,Y1)

    X_LABS = np.round(np.arange(0,np.copy(X1)[-1],INCREMENT).astype(int),-1)

    fig, ax = plt.subplots(figsize=FIG_SIZE,frameon='False')
    # plt.figure(figsize=FIG_SIZE,frameon='False')

    if ARR==True:
        ax.quiver(X, Y,
                  nd.gaussian_filter(TEMPX,sigma=1),
                  nd.gaussian_filter(TEMPY,sigma=1),
                  scale=scale, width=width,
                   pivot='mid', color='w', alpha=0.7)

    EXTENT = np.min(X1), np.max(X1), np.min(Y1), np.max(Y1)
    ax.imshow((smooth_hsv.smooth_hsv(B_C)), interpolation='none',
               origin='lower', extent=EXTENT)

    # generate the colors for your colormap
    color1 = colorConverter.to_rgba('white')
    color2 = colorConverter.to_rgba('black')

    # make the colormaps
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color2,color1],256)

    cmap2._init() # create the _lut array, with rgba values

    # create your alpha array and fill the colormap with them.
    # here it is progressive, but you can create whathever you want
    alphas = np.linspace(0, 0.8, cmap2.N+3)
    cmap2._lut[:,-1] = alphas
    ax.imshow(MZ_NORM,cmap=cmap2,
             origin='lower', extent=EXTENT)
    # scalebar = ScaleBar(PIX, "nm", location="lower left", color='white',
    # #                     length_fraction=0.5,
    #                     frameon=False, pad=1,
    #                     font_properties={'size':95})
    # plt.gca().add_artist(scalebar)

    plt.xlabel('X [nm]', fontsize=0.85*FTSIZE)
    plt.xticks(X_LABS,labels=X_LABS.astype(str),fontsize=0.85*TICKSIZE)
    plt.ylabel('Y [nm]', fontsize=0.85*FTSIZE)
    plt.yticks(X_LABS,labels=X_LABS.astype(str),fontsize=0.85*TICKSIZE)
    plt.axis('equal')
    plt.title(TITLE_NAME, fontsize=FTSIZE)
#     plt.tight_layout()
    plt.savefig(str(FILE_SAVE_NAME+'.png'), fmt='png')
#     plt.show()
    plt.close('all')

def remove_outliers(data_1d,thresh=1):
    """
    Removes data points that are threshold*sigma away from the mean and replaces
    it with the mean value.
    """
    mean = np.mean(data_1d)
    std = np.std(data_1d)
    print('mean,std=',mean,std)
    threshold = thresh
    data_1d[np.logical_or(data_1d>(mean+threshold*std),
                          data_1d<(mean-threshold*std))]=mean
    return(data_1d)

def rotate_3d_vec(mag,angx,angy,angz,pix=1,TEST=False):
    """
    This function rotates a 3D vector array with shape (3,N_PIX,N_PIX) in
    3 directions, with the rotation angles defined within the arrays angx,
    angy and angz, which should all be the same length.
    """
    from tqdm import tqdm
    M = mag
    PIX=pix
    N_ANG = len(angx)
    #print(N_ANG)
    ANG = zip(angx,angy,angz)
    N_PIX_ORIG = np.shape(M)[2] #PIX
    N_PIX = np.shape(M)[2] #PIX
    X_ORIG = np.linspace(0,PIX*N_PIX,N_PIX_ORIG)
    X = np.linspace(0,PIX*N_PIX,N_PIX)
    BOX = slice(0,N_PIX_ORIG),slice(0,N_PIX_ORIG)
    M_ROTF = np.zeros((N_ANG, np.shape(M)[0],np.shape(M)[2],np.shape(M)[3]),dtype=np.float64)
    for n, ANGLE in enumerate(tqdm(ANG)):
        if TEST==True:
            print(n,ANGLE)
        M_TEMP = np.copy(M)
        M_ROT0 = np.empty((np.shape(M)),dtype=np.float64)
        
        #vector rotation
        for j,ANG_IT in enumerate(ANGLE):
            if TEST==True:
                print(ANG_IT)
            ANG_IT_rad = ANG_IT/180.*np.pi
            if ANG_IT != 0:
                for i,_ in enumerate(M[0]):
                    FX=sp.interpolate.interp2d(X_ORIG,X_ORIG,M[0][i][BOX])
                    FY=sp.interpolate.interp2d(X_ORIG,X_ORIG,M[1][i][BOX])
                    FZ=sp.interpolate.interp2d(X_ORIG,X_ORIG,M[2][i][BOX])
                    MX=FX(X,X)
                    MY=FY(X,X)
                    MZ=FZ(X,X)

                    MX_NORM=(np.copy(MX)/np.max(np.abs([MX,MY,MZ])))
                    MY_NORM=(np.copy(MY)/np.max(np.abs([MX,MY,MZ])))
                    MZ_NORM=(np.copy(MZ)/np.max(np.abs([MX,MY,MZ])))
                    if j==0:
                        M_ROT0[0,i],M_ROT0[1,i],M_ROT0[2,i]=vt.x_rot_2D(MX_NORM,MY_NORM,
                                                                        MZ_NORM,
                                                                        theta=ANG_IT_rad)
                    elif j==1:
                        M_ROT0[0,i],M_ROT0[1,i],M_ROT0[2,i]=vt.y_rot_2D(MX_NORM,MY_NORM,
                                                                        MZ_NORM,
                                                                        theta=ANG_IT_rad)
                    elif j==2:
                        M_ROT0[0,i],M_ROT0[1,i],M_ROT0[2,i]=vt.z_rot_2D(MX_NORM,MY_NORM,
                                                                        MZ_NORM,
                                                                        theta=ANG_IT_rad)
                    else:
                        print("Improper axis rotation")

                for l,M_ROT in enumerate(M_ROT0):
                    M_TEMP[l] = M_ROT
        #image rotation
        for i,AX in enumerate(M_TEMP):
            TEMP = nd.rotate(nd.rotate(np.copy(M_TEMP[i]),angle=ANGLE[1], axes=(0,2)),angle=ANGLE[0], axes=(0,1))
            if TEST==True:
                print(np.shape(TEMP))
            TEMP = nd.rotate(np.sum(TEMP,axis=0),angle=ANGLE[2])
            if TEST==True:
                print(np.shape(TEMP))
            LENY = np.shape(TEMP)[0]
            LENX = np.shape(TEMP)[1]
            TEMP_MIDY = LENY//2
            TEMP_MIDX = LENX//2
            MIDY = np.shape(M_ROTF)[2]//2
            MIDX = np.shape(M_ROTF)[3]//2
            if LENY >= np.shape(M_ROTF)[2]:
                LENY = np.shape(M_ROTF)[2]
            if LENX >= np.shape(M_ROTF)[3]:
                LENX = np.shape(M_ROTF)[3]
            if TEST==True:
                print("MIDY, MIDX, LENY, LENX", MIDY, MIDX, LENY, LENX)
            EXTY = LENY//2
            EXTX = LENX//2
            BOX_F = slice(MIDY-EXTY,MIDY+EXTY),slice(MIDX-EXTX,MIDX+EXTX)
            BOX_TEMP = slice(TEMP_MIDY-EXTY,
                             TEMP_MIDY+EXTY),slice(TEMP_MIDX-EXTX,
                                                   TEMP_MIDX+EXTX)
            M_ROTF[n,i][BOX_F] = TEMP[BOX_TEMP]
        for i,_ in enumerate(M_ROTF[0]):
            ###M_ROTF[n,i] = np.average(M_TEMP[i],axis=0)
            M_ROTF[n,i] = np.copy(M_ROTF[n,i])/np.max(np.abs(M_ROTF[n]))

###       ### ANGZ=360-90 #This final rotation is to match mumax output orientation.
###       ### ANGZ_rad=ANGZ/180.*np.pi
###       ### M_ROTF[n,0],M_ROTF[n,1],M_ROTF[n,2]=vt.z_rot_2D(np.copy(M_ROTF[n,0]),
###       ###                                                 np.copy(M_ROTF[n,1]),
###       ###                                                 np.copy(M_ROTF[n,2]),theta=-ANGZ_rad)
###       ### for i,_ in enumerate(M_ROTF[n]):
###       ###     M_ROTF[n,i]=nd.rotate(np.copy(M_ROTF[n,i]),angle=ANGZ)
    return M_ROTF

def rotate_3d_vec2(mag,angx,angy,angz,pix=1,PAD=True,spins_only=False,
                   TEST=False):
    """
    This function rotates a 3D vector array with shape (3,NY_PIX,NX_PIX) in
    3 directions, with the rotation angles defined within the arrays angx,
    angy and angz, which should all be the same length. The returned array
    is not normalized.
    """
    from tqdm import tqdm
    if PAD==True:
        max_ang = np.max(np.abs(np.array([angx, angy])))/180.*np.pi
        # Determine the number of pixels to pad the array in order to avoid edges
        # within the tilted array
        LENX_ORIG = np.shape(mag)[3]
        LENY_ORIG = np.shape(mag)[2]
        X_PAD = int((LENX_ORIG/np.cos(max_ang)-LENX_ORIG+1)) 
        M = np.pad(np.copy(mag),pad_width=((0,0),(0,0),(X_PAD//2,X_PAD//2),
                                           (X_PAD//2,X_PAD//2)),
                   mode='edge')
    else:
        M = np.copy(mag)
    PIX=pix
    N_ANG = len(angx)
    #print(N_ANG)
    ANG = zip(angx,angy,angz)
    N_PIX_ORIG_Y = np.shape(M)[2] #pixels
    N_PIX_Y = np.shape(M)[2] #pixels
    N_PIX_ORIG_X = np.shape(M)[3] #pixels
    N_PIX_X = np.shape(M)[3] #pixels
    X_ORIG = np.linspace(0,PIX*N_PIX_X,N_PIX_ORIG_X)
    X = np.linspace(0,PIX*N_PIX_X,N_PIX_X)
    Y_ORIG = np.linspace(0,PIX*N_PIX_Y,N_PIX_ORIG_Y)
    Y = np.linspace(0,PIX*N_PIX_Y,N_PIX_Y)
    BOX = slice(0,N_PIX_ORIG_Y),slice(0,N_PIX_ORIG_X)
    M_RETURN = []
    for n, ANGLE in enumerate(tqdm(ANG)):
        if TEST==True:
            print(n,ANGLE)
        M_TEMP = np.copy(M)
        M_ROT0 = np.empty((np.shape(M)),dtype=np.float64)
        
        #vector rotation
        for j,ANG_IT in enumerate(ANGLE):
            if TEST==True:
                print(j,ANG_IT)
            ANG_IT_rad = ANG_IT/180.*np.pi
            if ANG_IT!=0:
                for i,_ in enumerate(M[0]):
                    FX=sp.interpolate.interp2d(X_ORIG,Y_ORIG,
                                               np.copy(M_TEMP)[0][i][BOX])
                    FY=sp.interpolate.interp2d(X_ORIG,Y_ORIG,
                                               np.copy(M_TEMP)[1][i][BOX])
                    FZ=sp.interpolate.interp2d(X_ORIG,Y_ORIG,
                                               np.copy(M_TEMP)[2][i][BOX])
                    MX=FX(X,Y)
                    MY=FY(X,Y)
                    MZ=FZ(X,Y)

                    MX_NORM=(np.copy(MX))#/np.max(np.abs([MX,MY,MZ])))
                    MY_NORM=(np.copy(MY))#/np.max(np.abs([MX,MY,MZ])))
                    MZ_NORM=(np.copy(MZ))#/np.max(np.abs([MX,MY,MZ])))
                    if j==0:
                        M_ROT0[0,i],M_ROT0[1,i],M_ROT0[2,i]=vt.x_rot_2D(MX_NORM,MY_NORM,
                                                                        MZ_NORM,
                                                                        theta=ANG_IT_rad)
                    elif j==1:
                        M_ROT0[0,i],M_ROT0[1,i],M_ROT0[2,i]=vt.y_rot_2D(MX_NORM,MY_NORM,
                                                                        MZ_NORM,
                                                                        theta=ANG_IT_rad)
                    elif j==2:
                        M_ROT0[0,i],M_ROT0[1,i],M_ROT0[2,i]=vt.z_rot_2D(MX_NORM,MY_NORM,
                                                                        MZ_NORM,
                                                                        theta=ANG_IT_rad)
                    else:
                        print("Improper axis rotation")

                for l,M_ROT in enumerate(M_ROT0):
                    M_TEMP[l] = np.copy(M_ROT)
        #image rotation
        ##for N_m,AX in enumerate(M_TEMP):
        print("ANGLE=",ANGLE)
        if spins_only==True:
            TEMP = np.copy(M_TEMP)
        else:
            TEMP = nd.rotate(nd.rotate(nd.rotate(np.copy(M_TEMP),
                                                 axes=(1,3),
                                                 angle=ANGLE[1]),
                                       axes=(1,2),angle=ANGLE[0]),
                             axes=(2,3),angle=ANGLE[2])
        if TEST==True:
            print(np.shape(TEMP))
        print(np.unique(TEMP),"0")
        print(np.unique(TEMP),"1")
        if TEST==True:
            print("np.shape(TEMP)=",np.shape(TEMP))
        LENY = np.shape(TEMP)[2]
        LENX = np.shape(TEMP)[3]
        TEMP_MIDY = LENY//2
        TEMP_MIDX = LENX//2
        MIDY = np.shape(M)[2]//2
        MIDX = np.shape(M)[3]//2
        if LENY >= np.shape(M)[2]:
            LENY = np.shape(M)[2]
        if LENX >= np.shape(M)[3]:
            LENX = np.shape(M)[3]
        if LENY >= LENY_ORIG:
            LENY = LENY_ORIG 
        if LENX >= LENX_ORIG:
            LENX = LENX_ORIG 
        if TEST==True:
            print("MIDY, MIDX, LENY, LENX", MIDY, MIDX, LENY, LENX)
        EXTY = LENY//2
        EXTX = LENX//2
        BOX_F = slice(MIDY-EXTY,MIDY+EXTY),slice(MIDX-EXTX,MIDX+EXTX)
        BOX_TEMP = slice(TEMP_MIDY-EXTY,
                         TEMP_MIDY+EXTY),slice(TEMP_MIDX-EXTX,
                                               TEMP_MIDX+EXTX)
        M_RETURN.append(np.copy(TEMP)[:,:,BOX_TEMP[0],BOX_TEMP[1]])
        ##M_ROTF = np.copy(M_ROTF)/np.max(np.abs(M_ROTF))
        ##print(np.unique(M_ROTF),"2")
        ##M_RETURN.append(M_ROTF)
    return M_RETURN

def skyrmion_tor(m, gamma, d, pxsize, Npixels, X0=0, Y0=0):
    """
    m=vorticity, gamma=helicity, d=domain width in nm, pxsize = size of a pixel
    in nm, Npixels=number of pixels per dimension. X0 is the x offset and Y0
    is the y offset from center. Returns normalized magnetization.
    """
    X, Y = np.meshgrid(np.linspace(-pxsize*Npixels/2.,pxsize*Npixels/2.,
                                   Npixels),
                       np.linspace(-pxsize*Npixels/2.,pxsize*Npixels/2.,
                                   Npixels))
    Theta = 2. * np.arctan(10.*(np.sqrt((X-X0)**2. + (Y-Y0)**2.)**d))
    Mx = np.sin(Theta) * np.cos(m * np.arctan2((Y-Y0), (X-X0)) + gamma)
    My = np.sin(Theta) * np.sin(m * np.arctan2((Y-Y0), (X-X0)) + gamma)
    Mz = np.cos(Theta)
#     f = 2. * np.arctan(np.sqrt(X**2. + Y**2.)/d)
#     Mx = np.sin(f) * np.cos(m * np.arctan2(Y, X) + gamma)
#     My = np.sin(f) * np.sin(m * np.arctan2(Y, X) + gamma)
    return Mx, My, X, Y, Mz

def skyrmion(m, gamma, r, width, centers, pxsize, Npixels):
    """m=vorticity, gamma=helicity, r=core radius in nm, a sets wall width,
    centers=a list of centers for each skyrmion, pxsize = size of a pixel in nm,
    Npixels=number of pixels per dimension. Returns normalized magnetization"""
    Mxt = np.zeros((Npixels, Npixels), dtype=np.float64)
    Myt = np.zeros((Npixels, Npixels), dtype=np.float64)
    Mzt = np.zeros((Npixels, Npixels), dtype=np.float64)

    for i, c in enumerate(centers):
        if np.shape(m) == ():
            m_a = m
        else:
            m_a = m[i]
        if np.shape(gamma) == ():
            g = gamma
        else:
            g = gamma[i]
        if np.shape(r) == ():
            d = r
        else:
            d = r[i]
        if np.shape(width) == ():
            a = width
        else:
            a = width[i]
        X0, Y0 = c
        X, Y = np.meshgrid(np.linspace(-pxsize*Npixels/2,pxsize*Npixels/2,Npixels),
                           np.linspace(-pxsize*Npixels/2,pxsize*Npixels/2,Npixels))
        if a==1:
            f = (np.sqrt((X-X0)**2. +
                 (Y-Y0)**2.)<d)*(2. * np.arctan((np.sqrt((X-X0)**2. +
                                                (Y-Y0)**2.)/d)**float(a)))
            f += (np.sqrt((X-X0)**2. +
                 (Y-Y0)**2.)>d)*(2. * np.arctan((np.sqrt((X-X0)**2. +
                                                (Y-Y0)**2.)/d)**float(4*a)))
        else:
            f = 2. * np.arctan((np.sqrt((X-X0)**2. + (Y-Y0)**2.)/d)**float(a))
        Mxt += np.sin(f) * np.cos(m_a * np.arctan2((Y-Y0), (X-X0)) + g)
        Myt += np.sin(f) * np.sin(m_a * np.arctan2((Y-Y0), (X-X0)) + g)
        Mzt += np.cos(f)+1
    Mzt-=1
    norm_c=np.max(np.abs([Mxt,Myt,Mzt]))
    Mxt/=norm_c
    Myt/=norm_c
    Mzt/=norm_c
    return(Mxt, Myt, Mzt)

def skyrmion_conic(m, gamma, A, C, width, centers, pxsize, Npixels,
                   r=100,B=0,D=0,E=0):
    """
    m=vorticity, gamma=helicity, r=core radius in nm, a sets wall width,
    centers=a list of centers for each skyrmion, pxsize = size of a pixel in nm,
    Npixels=number of pixels per dimension. A,B,C,D,and E are the coefficients
    of a conic function that forms the skyrmion wall's shape.
    
    Returns normalized magnetization
    """
    
    Mxt = np.zeros((Npixels, Npixels), dtype=np.float64)
    Myt = np.zeros((Npixels, Npixels), dtype=np.float64)
    Mzt = np.zeros((Npixels, Npixels), dtype=np.float64)

    for i, c in enumerate(centers):
        if np.shape(m) == ():
            m_a = m
        else:
            m_a = m[i]
        if np.shape(gamma) == ():
            g = gamma
        else:
            g = gamma[i]
        if np.shape(A) == ():
            d = r
        else:
            d = r[i]
        if np.shape(width) == ():
            a = width
        else:
            a = width[i]
        X0, Y0 = c
        X, Y = np.meshgrid(np.linspace(-pxsize*Npixels/2,pxsize*Npixels/2,Npixels),
                           np.linspace(-pxsize*Npixels/2,pxsize*Npixels/2,Npixels))
        if a==1:
            f = (np.sqrt(A*(X-X0)**2. +
                  B*(X-X0)*(Y-Y0) +
                  C*(Y-Y0)**2. +
                  D*(X-X0) +
                  E*(Y-Y0))<d)*(2. * np.arctan((np.sqrt(A*(X-X0)**2. +
                                                 B*(X-X0)*(Y-Y0) +
                                                 C*(Y-Y0)**2. +
                                                 D*(X-X0) +
                                                 E*(Y-Y0))/d)**float(a)))
            f += (np.sqrt(A*(X-X0)**2. +
                   B*(X-X0)*(Y-Y0) +
                   C*(Y-Y0)**2. +
                   D*(X-X0) +
                   E*(Y-Y0))>d)*(2. * np.arctan((np.sqrt(A*(X-X0)**2. +
                                                  B*(X-X0)*(Y-Y0) +
                                                  C*(Y-Y0)**2. +
                                                  D*(X-X0) +
                                                  E*(Y-Y0))/d)**float(4*a)))
        else:
            f = 2. * np.arctan((np.sqrt(A*(X-X0)**2. +
                                 B*(X-X0)*(Y-Y0) +
                                 C*(Y-Y0)**2. +
                                 D*(X-X0) +
                                 E*(Y-Y0))/d)**float(4*a))
        Mxt += np.sin(f) * np.cos(m_a * np.arctan2((Y-Y0), (X-X0)) + g)
        Myt += np.sin(f) * np.sin(m_a * np.arctan2((Y-Y0), (X-X0)) + g)
        Mzt += np.cos(f)+1
    Mzt-=1
    norm_c=np.max(np.abs([Mxt,Myt,Mzt]))
    Mxt/=norm_c
    Myt/=norm_c
    Mzt/=norm_c
    return(Mxt, Myt, Mzt)

def transfer_function(aperture, pixel_size, defocus, wavelength=0.00196,
                      Cs=0, alpha=None):
    """Cs, wavelength, and pixel_size should be in units of nm."""
    q = SpatialFrequencies(aperture, float(pixel_size))
    if Cs == 0:
        spherical = 0
    else:
        spherical = Cs/4. * (wavelength ** 3) * (q.squared**2)
    if defocus == 0:
        focus = 0
    else:
        focus = defocus/2. * wavelength * q.squared
    t_q = aperture * np.exp(-2 * np.pi * 1.j * (spherical - focus))
    if not alpha:
        return t_q

    else:
        Es_q = np.exp(-((np.pi * alpha / wavelength) ** 2) *
                   (Cs * wavelength ** 3 * q.squared ** (3./2.) +
                    defocus * wavelength * np.sqrt(q.squared)) ** 2)
        return t_q * Es_q
