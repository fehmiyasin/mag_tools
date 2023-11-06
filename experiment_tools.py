
#!/usr/bin/python2

import numpy as np

def calc_phase_from_fft(fringes,window=(1,-1)):
    """
    This function returns the phase of a sinewave-like signal from the Fourier
    transform of a one-dimensional array with interference fringes. It assumes
    that the signal of the desired frequency is strong enough to emerge as the
    maximum value FFT peak within the defined window.
    """
    WIND = [window[0],window[1]]
    fft_1D = np.fft.fft(fringes)
    freq = np.fft.fftfreq(len(fringes))
    peak0 = fft_1D[0]
    if WIND[1]==-1:
        WIND[1] = len(fft_1D)//2
    k_pix1 = np.where(np.abs(fft_1D[WIND[0]:WIND[1]])==
                      np.max(np.abs(fft_1D[WIND[0]:WIND[1]])))
    
    phase = np.angle(fft_1D[WIND[0]:WIND[1]][k_pix1[0]])
#     print("Phase = ", phase)
    return(phase)

def calc_phase_from_fft2D(fringes,peak1_loc,peak_box_w=3):
    """
    This function returns the phase of a sinewave-like signal from the Fourier
    transform of a two-dimensional array with interference fringes. It assumes
    that the signal of the desired frequency is strong enough to emerge as the
    maximum value FFT peak within the defined peak_box (slice).
    """
    PEAK1X = peak1_loc[0]
    PEAK1Y = peak1_loc[1]
    MIDX = np.shape(fringes)[1]//2
    MIDY = np.shape(fringes)[0]//2
    FRINGES_FFT = np.fft.fftshift(np.fft.fft2(fringes))
    if peak_box_w > 1:
        PEAK_BOX_W = peak_box_w
        PEAK1_BOX = slice(MIDY+PEAK1Y-PEAK_BOX_W//2,
                          MIDY+PEAK1Y+PEAK_BOX_W//2), slice(MIDX+PEAK1X-
                                                            PEAK_BOX_W//2,
                                                            MIDX+PEAK1X+
                                                            PEAK_BOX_W//2)
        PEAK1_LOC = np.where(np.abs(FRINGES_FFT[PEAK1_BOX])==
                             np.max(np.abs(FRINGES_FFT[PEAK1_BOX])))

        return(np.angle(FRINGES_FFT[PEAK1_BOX][PEAK1_LOC]))
    else:
        return(np.angle(FRINGES_FFT[MIDY + PEAK1Y, MIDX + PEAK1X]))

def calc_V_from_fft(fringes, window=(1,-1), freq_max=False):
    """
    This function returns the fringe contrast from a the Fourier transform of
    a one-dimensional array with interference fringes. It assumes that the
    signal of the desired frequency is strong enough to emerge as the maximum
    value FFT peak within the defined window.
    """
    WIND = [window[0],window[1]]
    fft_1D = np.fft.fft(fringes)
    freq = np.fft.fftfreq(len(fringes))
    peak0 = fft_1D[0]
    if WIND[1]==-1:
        WIND[1] = len(fft_1D)//2
    k_pix1 = np.where(np.abs(fft_1D[WIND[0]:WIND[1]])==
                      np.max(np.abs(fft_1D[WIND[0]:WIND[1]])))
    
    V = np.abs(2.*(fft_1D[WIND[0]:WIND[1]][k_pix1[0]])/peak0)
#     print("V = ", V)
    if freq_max == True:
        return(freq[WIND[0]:WIND[1]][k_pix1[0]], V)
    else:
        return(V)

def calc_V_from_fft2D(fringes,peak1_loc,peak_box_w=3,mean_solv=False):
    """
    Calculates the fringe visibility of a 2D image with the desired fringe
    frequency specified in k-space using peak1_loc.
    fringes -> 2D array with interference fringes
    peak1_loc -> 2-element list or array with integer value [x-direction,
    y-direction] steps from the center FFT peak.
    """

    PEAK1X = peak1_loc[0]
    PEAK1Y = peak1_loc[1]
    MIDX = np.shape(fringes)[1]//2
    MIDY = np.shape(fringes)[0]//2
    FRINGES_FFT = np.fft.fftshift(np.fft.fft2(fringes))
    PEAK0 = np.fft.fft2(fringes)[0,0]
    if peak_box_w > 1:
        PEAK_BOX_W = peak_box_w
        PEAK1_BOX = slice(MIDY+PEAK1Y-PEAK_BOX_W//2,
                          MIDY+PEAK1Y+PEAK_BOX_W//2), slice(MIDX+PEAK1X-
                                                            PEAK_BOX_W//2,
                                                            MIDX+PEAK1X+
                                                            PEAK_BOX_W//2)
        PEAK1_LOC = np.where(np.abs(FRINGES_FFT[PEAK1_BOX])==
                             np.max(np.abs(FRINGES_FFT[PEAK1_BOX])))
        if mean_solv==True:
            XTEMP = np.linspace(-10,10,PEAK_BOX_W)
            G = np.exp(-0.1*XTEMP**2)
            G/=np.max(G)
            KERN = G[:,np.newaxis]*G[np.newaxis,:]
            PEAK1Y += PEAK1_LOC[0][0]
            PEAK1X += PEAK1_LOC[1][0]
            if PEAK_BOX_W%2==1:
                PEAK1_BOX = slice(MIDY+PEAK1Y-PEAK_BOX_W//2-1,
                                  MIDY+PEAK1Y+
                                  PEAK_BOX_W//2), slice(MIDX+PEAK1X-
                                                        PEAK_BOX_W//2-1,
                                                        MIDX+PEAK1X+
                                                        PEAK_BOX_W//2)
            elif PEAK_BOX_W%2==2:
                PEAK1_BOX = slice(MIDY+PEAK1Y-PEAK_BOX_W//2,
                                  MIDY+PEAK1Y+
                                  PEAK_BOX_W//2), slice(MIDX+PEAK1X-
                                                        PEAK_BOX_W//2,
                                                        MIDX+PEAK1X+
                                                        PEAK_BOX_W//2)
            WEIGHTED_BOX1 = np.copy(FRINGES_FFT)[PEAK1_BOX]#*KERN
            return([np.abs(2.*np.mean(WEIGHTED_BOX1/PEAK0))])

        return(np.abs(2.*(FRINGES_FFT[PEAK1_BOX][PEAK1_LOC]) / PEAK0))
    else:
        return(np.abs(2.*(FRINGES_FFT[MIDY + PEAK1Y, MIDX + PEAK1X]) / PEAK0))

def create_circular_mask(h, w, center=None, radius=None, keep_inside=False):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image
                       # walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    if keep_inside == True:
        mask = dist_from_center <= radius
    else:
        mask = dist_from_center >= radius
    return mask

def JEM2100_int_conv_B(OLC_int):
    B = round(5.8928+0.0406243*OLC_int,1) #mT
    return B

def JEM2800_hex_conv_B(OLC_hex, linear=True, log=False):
    '''
    This function inputs an objective lens current or series (array) of current values in
    hexadecimal and converts it to a base-ten integer before calculating the approximate
    magnetic field value in mT. For current values less than 500 mT, set the keyword
    linear=True. For values larger than that, you can either say linear=False for a 6th-
    order polynomial fit or set log=True for the logarithmic fit.
    '''
    if type(OLC_hex)==int or type(OLC_hex)==float or type(OLC_hex)==str:
        OLC_hex=[OLC_hex]
    OLC_int = []
    for h in OLC_hex:
        OLC_int.append(int('0x'+str(h),0))
#         print(OLC_int)
    if linear==True:
        B = []
        for n in OLC_int:
            B.append(round(0.06468778*n+2.5984,1))
    elif log==True:
        B = []
        for n in OLC_int:
            B.append(round(620.37*np.log(n)+(-5038.61),1))
    else:
        B = []
        for n in OLC_int:
            B.append(round(16.4605+0.0499*OLC_int+
                           (3.422*(10**(-6)))*OLC_int**2+
                           (-2.548*(10**(-10)))*OLC_int**3+
                           (6.43417*(10**(-15)))*OLC_int**4+
                           (-7.22*(10**(-20)))*OLC_int**5+
                           (3.04*(10**(-25)))*OLC_int**6,1)) #mT
    return B

def JEM2800_B_conv_hex(B_mTa):
    '''
    This function converts an input magnetic field value to an objective lens value
    in hexadecimal. Here I automatically switch between a linear fit for B<500 mT and
    a 6th-order polynomial for B>500 mT.'''
    if type(B_mTa)==int or type(B_mTa)==float or type(B_mTa)==str:
        B_mTa=[B_mTa]
    OLC_int=[]
    
    for B_mT in B_mTa:
        if B_mT>500:
            OLC_int.append(int(round(67.063+12.062755*B_mT+
                                     (2.45918*(10**(-2)))*B_mT**2+
                                     (-6.78*(10**(-5)))*B_mT**3+
                                     (8.46*(10**(-8)))*B_mT**4+
                                     (-5.07*(10**(-11)))*B_mT**5+
                                     (1.33*(10**(-14)))*B_mT**6,2))) #int
        elif B_mT==0:
            OLC_int.append(0)
        else:
            OLC_int.append(int(round(-40.0584+15.4584391934*B_mT))) #int
    OLC_hex=[]
    for n in OLC_int:
        OLC_hex.append(hex(int(n))[2:])
    return OLC_hex
