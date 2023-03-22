"""
This python module stores some functions that are useful for plotting in-plane
magnetic fields.
"""
__author__ = "Fehmi Yasin"
__date__ = "21/07/17"
__version__ = "1.0"
__maintainer__ = "Fehmi Yasin"
__email__ = "fehmi.yasin@riken.jp"

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import smooth_hsv_jjc as smooth_hsv
from matplotlib.colors import colorConverter

def plot_B_3D(b_arr,sv_file_name='',sv_fmt='png',im_w=1,nm_pix=1,arr_w=.01,
              arr_scale=25,m_step=0,arrows=True,im_only=True,normalize=False,
              v_min=-1,v_max=1,annote='',annote_loc=[0.8,0.95],inset_mag='',
              inset=False,inset_loc=[0,0,0.25,0.25],hline_loc=False,
              vline_loc=False,hline_loc_inset=False,vline_loc_inset=False,
              twoD_arr=False,intensity_scale=1,offset_angle=0):
    """
    Plotting a slice of a 3D vector field. B_ARR is the
    vector field array with shape [3, NY_PIX, NX_PIX].
    The first axis represents the x, y and z component of the field.
    NY_PIX and NX_PIX are the number of pixels in the y and x direction,
    respectively. They are usually equal, and its generally a good idea that
    they be factorable by 2^n.
    sv_file_name is the path to the save directory as well as file name:
    "path/to/directory/file_name"
    sv_fmt is the file format you'd like to save the images as
    im_w is the field of view you'd like to save, im_w=0.5 crops the image in
    half from the center.
    nm_pix is the pixel length in nm/pix
    normalize=True will normalize the 2D array for you. Don't use if your data
    is already normalized.
    offset_angle rotates the in-plane vector by some user-defined angle in degrees.
    """
    plt.clf()

    PIX=nm_pix #nm/pix
    
    MX = np.copy(b_arr[0])
    MY = np.copy(b_arr[1])
    if twoD_arr==False:
        MZ = np.copy(b_arr[2])
    else:
        MZ = np.zeros_like(MX)
    MIDX = np.shape(b_arr)[2]//2
    MIDY = np.shape(b_arr)[1]//2
    NX_PIX = np.shape(b_arr)[2]
    NY_PIX = np.shape(b_arr)[1]
    IM_WIDTH_X = int(im_w*NX_PIX//2)
    IM_WIDTH_Y = int(im_w*NY_PIX//2)
    BOX = slice(MIDY-IM_WIDTH_Y,
                MIDY+IM_WIDTH_Y),slice(MIDX-IM_WIDTH_X,
                                       MIDX+IM_WIDTH_X)
    
    if normalize==True:
        MX_NORM=(np.copy(MX)/np.max(np.abs([MX,MY])))[BOX]
        MY_NORM=(np.copy(MY)/np.max(np.abs([MX,MY])))[BOX]
        MZ_NORM=(np.copy(MZ)/np.max(np.abs([MX,MY,MZ])))[BOX]
    else:
        MX_NORM=np.copy(MX)[BOX]
        MY_NORM=np.copy(MY)[BOX]
        MZ_NORM=np.copy(MZ)[BOX]
    
    B_C = np.copy(MY_NORM) + 1.0j*np.copy(MX_NORM)
    B_C_AMP = np.abs(B_C)**1.
    B_C_PHI = np.angle(B_C)+(offset_angle/180.*np.pi)
    B_C = np.copy(B_C_AMP)*np.exp(1.0j*B_C_PHI)
    B_C_PHI2 = np.angle(B_C)+(offset_angle/180.*np.pi)
    B_C2 = np.copy(B_C_AMP)*np.exp(1.0j*B_C_PHI2)
    FTSIZE=35
    TICKSIZE=0.75*FTSIZE
    FIG_SIZE_FACTOR = 13./np.max([np.shape(B_C)[0],np.shape(B_C)[1]])
    FIG_SIZE = (FIG_SIZE_FACTOR*np.shape(B_C)[1],
                FIG_SIZE_FACTOR*np.shape(B_C)[0])
    
    NUM_PTS = 5
    INCREMENT = int(round(PIX*np.shape(B_C)[0]//NUM_PTS,-1)) #nm
    if m_step==0:
        M_STEP=int(np.shape(MX_NORM)[0]//(2**5.5))
    else:
        M_STEP=int(m_step)
    print("M_STEP = "+str(M_STEP))
    TEMPX=np.copy(np.imag(B_C2))[:-1:2*M_STEP,:-1:2*M_STEP]
    TEMPY=np.copy(np.real(B_C2))[:-1:2*M_STEP,:-1:2*M_STEP]
    
    X1 = np.linspace(0, PIX*int(np.shape(B_C)[1]), np.shape(TEMPX)[1])
    Y1 = np.linspace(0, PIX*int(np.shape(B_C)[0]), np.shape(TEMPX)[0])
    
    X, Y = np.meshgrid(X1,Y1)
    
    #X_LABS = np.round(np.arange(0,np.copy(X1)[-1],INCREMENT).astype(int),-1)
    
    fig, ax = plt.subplots(figsize=FIG_SIZE,frameon='False')
    
    if arrows==True:
        #Define where to draw arrows
        SCALE = arr_scale
        WIDTH = arr_w
        ax.quiver(X, Y, (TEMPX), (TEMPY),
                  scale=SCALE,  width=WIDTH,
                  pivot='mid', color='w', alpha=0.7)
    
    EXTENT = np.min(X1), np.max(X1), np.min(Y1), np.max(Y1)
    ax.imshow((smooth_hsv.smooth_hsv(B_C,intensity_scale=intensity_scale)),
               interpolation='none',
               origin='lower', extent=EXTENT, vmin=v_min,vmax=v_max)
    
    if twoD_arr==False:
        # generate the colors for your colormap
        color1 = colorConverter.to_rgba('white')
        color2 = colorConverter.to_rgba('black')
        
        # make the colormaps
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color2,color1],256)
        
        cmap2._init() # create the _lut array, with rgba values
        
        # create your alpha array and fill the colormap with them.
        # here it is progressive, but you can create whathever you want
        alphas = np.linspace(0., 0.5, cmap2.N+3)
        cmap2._lut[:,-1] = alphas
        ax.imshow(MZ_NORM,cmap=cmap2,
                 origin='lower', extent=EXTENT,vmin=v_min,vmax=v_max)
    if annote!='':
        PROPS = dict(boxstyle='round', facecolor='k', alpha=0.5)
        plt.text(annote_loc[0]*np.shape(MX)[1],annote_loc[1]*np.shape(MX)[0],
                 str(annote),
                 color='w',
                 fontsize=FTSIZE, bbox=PROPS)
    if inset==True:
        left, bottom, width, height = inset_loc
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.imshow((smooth_hsv.smooth_hsv(inset_mag[0],
                    norm=np.max(np.abs(inset_mag[0])))),
                   interpolation='none',
                   origin='lower')
        ax2.imshow(inset_mag[1],cmap=cmap2,
                   origin='lower')
        ax2.set_axis_off()
        if vline_loc_inset!=False:
            ax2.axvline(x=vline_loc_inset,color='g',linestyle='--')
        if hline_loc_inset!=False:
            ax2.axhline(y=hline_loc_inset,color='r',linestyle='--')
    if vline_loc!=False:
        ax.axvline(x=vline_loc,color='g',linestyle='--',lw=5)
    if hline_loc!=False:
        ax.axhline(y=hline_loc,color='r',linestyle='--',lw=5)
    # Future code for scalebar insertion 
    # scalebar = ScaleBar(PIX, "nm", location="lower left", color='white',
    # #                     length_fraction=0.5,
    #                     frameon=False, pad=1,
    #                     font_properties={'size':95})
    # plt.gca().add_artist(scalebar)
    
    plt.xlabel('X [nm]', fontsize=0.85*FTSIZE)
#    plt.xticks(X_LABS,labels=X_LABS.astype(str),fontsize=0.85*TICKSIZE)
    plt.ylabel('Y [nm]', fontsize=0.85*FTSIZE)
#    plt.yticks(X_LABS,labels=X_LABS.astype(str),fontsize=0.85*TICKSIZE)
    plt.axis('equal')
    # if you want a title
    #plt.title(TITLE_NAME, fontsize=FTSIZE)
    if im_only==True:
        ax.set_axis_off()
    plt.tight_layout()
    if sv_file_name!='':
        plt.savefig(sv_file_name+
                    '.'+str(sv_fmt),pad_inches=0,
                    dpi=300, transparent=True)
    plt.show()
    plt.clf()

def plot_B_in_plane(angle_arr,b_arr,sv_file_name,sv_fmt='png',im_w=1,nm_pix=1,
                    arrows=True,im_only=True):
    """
    Plotting the magnetization summed along the thickness for all tilt angles
    defined in ANGLE_ARR, which should be of form [ANGX, ANGY, ANGZ], where
    ANGX, ANGY and ANGZ are arrays with tilt angles as elements. B_ARR is the
    array of pre-tilted magnetic field values of shape
    [N_ANG, 3, T_PIX, NY_PIX, NX_PIX].
    N_ANG is number of angles (len(ANGX)=len(ANGY)=len(ANGZ)).
    The second axis represents the x, y and z component of the field.
    T_PIX is the thickness length of the array in pixels, and will be larger
    than the number of T_PIX slices containing non-zero pixels for all but the
    largest tilt angle.
    NY_PIX and NX_PIX are the number of pixels in the y and x direction,
    respectively. They are usually equal, and its generally a good idea that
    they be factorable by 2^n.
    sv_file_name is the path to the save directory as well as file name:
    "path/to/directory/file_name"
    sv_fmt is the file format you'd like to save the images as
    im_w is the field of view you'd like to save, im_w=0.5 crops the image in
    half from the center.
    nm_pix is the pixel length in nm/pix
    """

    PIX=nm_pix #nm/pix

    ANGX = angle_arr[0]
    ANGY = angle_arr[1]
    ANGZ = angle_arr[2]
    ANG=zip(ANGX,ANGY,ANGZ)

    for n_ang, ANGLE in enumerate(ANG):
        MX_ROT = np.sum(b_arr[n_ang,0],axis=0)
        MY_ROT = -np.sum(b_arr[n_ang,1],axis=0)
        MZ_ROT = np.sum(b_arr[n_ang,2],axis=0)
        
        MIDX = np.shape(b_arr)[4]//2
        MIDY = np.shape(b_arr)[3]//2
        NX_PIX = np.shape(b_arr)[4]
        NY_PIX = np.shape(b_arr)[3]
        IM_WIDTH_X = int(im_w*NX_PIX//2)
        IM_WIDTH_Y = int(im_w*NY_PIX//2)
        BOX = slice(MIDY-IM_WIDTH_Y,
                    MIDY+IM_WIDTH_Y),slice(MIDX-IM_WIDTH_X,
                                           MIDX+IM_WIDTH_X)
    
        MX_NORM=(np.copy(MX_ROT)/np.max(np.abs([MX_ROT,MY_ROT,MZ_ROT])))[BOX]
        MY_NORM=(np.copy(MY_ROT)/np.max(np.abs([MX_ROT,MY_ROT,MZ_ROT])))[BOX]
        MZ_NORM=(np.copy(MZ_ROT)/np.max(np.abs([MX_ROT,MY_ROT,MZ_ROT])))[BOX]
    
        B_C = np.copy(MY_NORM) + 1.0j*np.copy(MX_NORM)
        B_C_AMP = np.abs(B_C)
        B_C_PHI = np.angle(B_C)

        FTSIZE=35
        TICKSIZE=0.75*FTSIZE
        FIG_SIZE=(13,13)
    
        NUM_PTS = 5
        INCREMENT = int(round(PIX*np.shape(B_C)[0]//NUM_PTS,-1)) #nm
        M_STEP=int(np.shape(MX_NORM)[0]//(2**5.5))
    
        TEMPX=np.copy(MX_NORM[:-1:2*M_STEP,:-1:2*M_STEP])
        TEMPY=np.copy(MY_NORM[:-1:2*M_STEP,:-1:2*M_STEP])
    
        X1 = np.linspace(0, PIX*int(np.shape(B_C)[1]), np.shape(TEMPX)[1])
        Y1 = np.linspace(0, PIX*int(np.shape(B_C)[0]), np.shape(TEMPX)[0])
    
        X, Y = np.meshgrid(X1,Y1)
    
        #X_LABS = np.round(np.arange(0,np.copy(X1)[-1],INCREMENT).astype(int),-1)
    
        fig, ax = plt.subplots(figsize=FIG_SIZE,frameon='False')
    
        if arrows==True:
            #Define where to draw arrows
            SCALE = 25
    
            ax.quiver(X, Y, (TEMPX)[::-1], (TEMPY)[::-1],
                      scale=SCALE,  width=1/100,
                      pivot='mid', color='w', alpha=0.7)
    
        EXTENT = np.min(X1), np.max(X1), np.min(Y1), np.max(Y1)
        ax.imshow((smooth_hsv.smooth_hsv(B_C,norm=0)), interpolation='none',
                   origin='lower', extent=EXTENT)
    
        # generate the colors for your colormap
        color1 = colorConverter.to_rgba('white')
        color2 = colorConverter.to_rgba('black')
    
        # make the colormaps
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color2,color1],256)
    
        cmap2._init() # create the _lut array, with rgba values
    
        # create your alpha array and fill the colormap with them.
        # here it is progressive, but you can create whathever you want
        alphas = np.linspace(0, 0.5, cmap2.N+3)
        cmap2._lut[:,-1] = alphas
        ax.imshow(MZ_NORM,cmap=cmap2,
                 origin='lower', extent=EXTENT)

        # Future code for scalebar insertion 
        # scalebar = ScaleBar(PIX, "nm", location="lower left", color='white',
        # #                     length_fraction=0.5,
        #                     frameon=False, pad=1,
        #                     font_properties={'size':95})
        # plt.gca().add_artist(scalebar)
    
        plt.xlabel('X [nm]', fontsize=0.85*FTSIZE)
#        plt.xticks(X_LABS,labels=X_LABS.astype(str),fontsize=0.85*TICKSIZE)
        plt.ylabel('Y [nm]', fontsize=0.85*FTSIZE)
#        plt.yticks(X_LABS,labels=X_LABS.astype(str),fontsize=0.85*TICKSIZE)
        plt.axis('equal')
        # if you want a title
        #plt.title(TITLE_NAME, fontsize=FTSIZE)
        if im_only==True:
            ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(sv_file_name+
                    str([str(round(a,0)) for a in ANGLE])+
                    '.'+str(sv_fmt), pad_inches=0, transparent=True)
        plt.show()
        plt.clf()
