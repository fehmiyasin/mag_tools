from enum import Enum as __Enum
from threading import Thread as _Thread
from threading import RLock as _RLock
from multiprocessing import cpu_count as __cpu_count
import numpy as _np


class Normalization(__Enum):
    Intensity = 0
    Length = 1
    Log = 2


def smooth_hsv(complex_array, norm=Normalization.Intensity, max_cutoff=None, high_saturation=False,
               conserve_memory=False, intensity_scale=1):
    """
    This function takes a complex input array and return a list of [red, green, blue] arrays
    that correspond to the complex array via color -> complex angle, intensity -> complex magnitude
    :param complex_array: input array to colorize
    :param norm: The normalization to use
    :param max_cutoff: if set, this will truncate the intensity to max out at this value
    :param high_saturation: if set to True, will cause Sin()^4 to be used instead Sin()^2
    :param conserve_memory: if True, will use float16 (this is slow), else use float32
    :return:
    """
    if conserve_memory:
        float_type = _np.float16
    else:
        float_type = _np.float32

    width = complex_array.shape[0]
    slice_width = 16
    while width % slice_width != 0:
        slice_width -= 1

    lock = _RLock()
    shape = list(_np.shape(complex_array))
    colors = _np.zeros(shape + [3], dtype=float_type)
    magnitude = _np.zeros_like(complex_array, dtype=float_type)
    apply_list = list(range(int(width / slice_width)))

    def absolute():
        while True:
            with lock:
                if apply_list:
                    on = apply_list.pop(0)
                else:
                    return
            j = on * slice_width
            k = j + slice_width
            magnitude[j:k, ::] = _np.absolute(complex_array[j:k, ::]).astype(dtype=float_type)

    threads = list()
    for _ in list(range(__cpu_count())):
        threads.append(_Thread(target=absolute))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    n = 2
    if high_saturation:
        n = 4

    max_magnitude = _np.max(magnitude)
    if max_cutoff is not None:
        max_cutoff /= max_magnitude

    apply_list = list(range(int(width / slice_width)))

    def parfunc():
        while True:
            with lock:
                if apply_list:
                    on = apply_list.pop(0)
                else:
                    return
            j = on * slice_width
            k = j + slice_width

            hue = (_np.angle(complex_array[j:k, ::], deg=True) + 90.) / 60.
            val = _np.array(magnitude[j:k, ::])

            if norm is Normalization.Intensity:
                val **= 2
                val /= max_magnitude ** 2
            elif norm is Normalization.Log:
                val = _np.log(val)
                _np.copyto(val, 0, where=val < 0)
                val /= _np.log(max_magnitude)
            elif norm is Normalization.Length:
                val /= max_magnitude
            else:
                #added 211108 by FSY
                val/=norm

            if max_cutoff is not None:
                _np.copyto(val, 1, where=val > max_cutoff)
                val /= max_cutoff

            pi6 = _np.pi / 6.

            colors[j:k, ::, 0] = val * _np.abs(_np.sin((hue - 0) * pi6)) ** n
            colors[j:k, ::, 1] = 0.6 * val * _np.abs(_np.sin((hue - 4) * pi6)) ** n
            colors[j:k, ::, 2] = val * _np.abs(_np.sin((hue - 8) * pi6)) ** n

            colors[j:k, ::, 1] += colors[j:k, ::, 2] * 0.35
            colors[j:k, ::, 1] += colors[j:k, ::, 0] * 0.1

    threads = list()
    for _ in list(range(__cpu_count())):
        threads.append(_Thread(target=parfunc))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return colors*intensity_scale

def smooth_rgba(b_c,b_z,TEST=False):
    """
    Inputs complex, 2D in-plane magnetic field array b_c (b_y+1.j*b_x) and
    b_z. Returns an RGBA color array of the same size as b_c.
    """

    COL_IN_PLANE = []
    COLORS = (_np.random.random((_np.size(b_z), 4))*255).astype(_np.uint8)
    COLORS[:,-1] = 255 # No transparency

    COL_IN_PLANE.append(smooth_hsv(b_c,intensity_scale=1.))
    COL_IN_PLANE = _np.array(COL_IN_PLANE)
    
    for i,_ in enumerate(_np.ravel(b_z)):
        THETA = (_np.arccos(_np.ravel(_np.copy(b_z))[i]))
        COL_IN_PLANE_TEMP = _np.reshape(COL_IN_PLANE,
                                       (_np.shape(COL_IN_PLANE)[0]*
                                        _np.shape(COL_IN_PLANE)[1]*
                                        _np.shape(COL_IN_PLANE)[2],
                                        _np.shape(COL_IN_PLANE)[3]))[i]
    
        TEMP0 = (_np.sin(THETA)*COL_IN_PLANE_TEMP[0]+(_np.cos(THETA)+1)/2)
        TEMP1 = (_np.sin(THETA)*COL_IN_PLANE_TEMP[1]+(_np.cos(THETA)+1)/2)
        TEMP2 = (_np.sin(THETA)*COL_IN_PLANE_TEMP[2]+(_np.cos(THETA)+1)/2)
        if TEMP0>1 or TEMP1>1 or TEMP2>1:
            TEMP0/=_np.max([TEMP0,TEMP1,TEMP2])
            TEMP1/=_np.max([TEMP0,TEMP1,TEMP2])
            TEMP2/=_np.max([TEMP0,TEMP1,TEMP2])
        AMP = _np.abs(_np.ravel(_np.copy(b_c))[i])
        if AMP>1:
            AMP = 1
        if TEST==True:
            print(AMP)
        COLORS[i][0] = 255*TEMP0
        COLORS[i][1] = 255*TEMP1
        COLORS[i][2] = 255*TEMP2 
        COLORS[i][-1] = 255*(1-AMP)
    COLORS2 = _np.zeros((b_z.shape[0],b_z.shape[1],4),dtype=_np.uint8)
    if TEST==True:
        print("np.shape(COLORS) = ", _np.shape(COLORS))
        print("np.shape(COLORS2) = ", _np.shape(COLORS2))
    
    for i,_ in enumerate(COLORS[0]):
        COLORS2[:,:,i] = _np.reshape(COLORS[:,i],_np.shape(b_z))
    return(COLORS2)
