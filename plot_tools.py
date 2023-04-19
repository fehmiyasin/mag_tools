#!/usr/bin/env python
'''
A Module that includes plot functions based on matplotlib.pyplot that are useful for plotting
arrays, 2D or otherwise.
'''
__author__ = "Fehmi Yasin"
__version__ = "1.0.1"
__date__ = "17/07/13"
__maintainer__ = "Fehmi Yasin"
__email__ = "fyasin@uoregon.edu"

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

def add_scalebar(data, pix, pix_unit, coord_system,
                 scalebar_fraction_of_image = 0.3,
                 loc = 'lower right', label_top = True,
                 sep = 5, frameon = False,
                 ftsize = 18, color = 'white'):
    '''
    Function that adds a scalebar to a plt.subplot image.
    '''
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    fontprops = fm.FontProperties(size=ftsize)
    print(np.shape(data), pix, pix_unit)
    SB_PIX_FRAC = (scalebar_fraction_of_image *
                   np.shape(data)[1]) # pix length of desired scalebar
    SB_UNITS = str(pix_unit)
    SB_NM = int(round(SB_PIX_FRAC * pix,-2)) # nm
    SB_PIX = SB_NM / pix # pix
    if np.logical_and(SB_PIX > SB_PIX_FRAC, SB_NM <=100) or SB_NM == 0:
        SB_NM = 50 #nm
        SB_PIX = SB_NM / pix # pix
    print('SB_NM = ', str(SB_NM))
    scalebar = AnchoredSizeBar(coord_system,
                               SB_PIX, str(str(SB_NM) + ' ' +
                                           str(SB_UNITS)),
                               str(loc),
                               sep = sep,
                               label_top = label_top,
                               pad=1,
                               color = str(color),
                               frameon=frameon,
                               size_vertical=5,
                               fontproperties=fontprops)

    return(scalebar)

def mult_line_plots(lines, colors, labels, linewidth=4,
                    pix=1, ftsize=30, fig_size=(10,7), label_ftsize=25,
                    ticksize=20, title="", ylabel="", xlabel="",
                    num_pts=8, savename="", savefmt='png',
                    y_lim=None, xerror=0, yerror=0, legend_ftsize=0,
                    LEG_LOC=0, ALPHA=0.3):
    """
    Plots multiple lines in the same figure with a legend labeling each plot.
    
    lines should be an array with shape (N_lines, N_pix)
    colors and labels should be lists of strings with size (N_lines)"""
    
    N_pix = len(lines[0])
    increment = int(round(pix*N_pix//num_pts,-1)) #nm
    
    x = np.linspace(0,pix*N_pix,N_pix)
    x_labs = np.round(np.arange(0,np.copy(x)[-1],increment).astype(int),-1)

    plt.figure(figsize=fig_size)
    # plt.scatter(x_labs,Mx_scat1,marker='$x$',
    #             s=marker_size,c='blue',label='$m_{x}$')
    # plt.scatter(x_labs,My_scat1,marker='$y$',
    #             s=marker_size,c='orange',label='$m_{y}$')
    # plt.scatter(x_labs,Mz_scat1,marker='$z$',
    #             s=marker_size,c='green',label='$m_{z}$')
    for i in range(np.shape(lines)[0]):
        plt.plot(x,lines[i],c=colors[i],label=labels[i],lw=linewidth)
    
    if type(yerror)!=int:
        for i in range(np.shape(lines)[0]):
            plt.fill_between(x, lines[i]-yerror[i], lines[i]+yerror[i],
                             color=colors[i], alpha=ALPHA)
        
    if type(y_lim)!=int:
        plt.ylim(y_lim[0],y_lim[1])
    plt.ylabel(ylabel, fontsize=label_ftsize)
    plt.xlabel(xlabel, fontsize=label_ftsize)
    plt.xticks(x_labs,labels=x_labs.astype(str),fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    if legend_ftsize!=0:
        plt.legend(fontsize=0.8*legend_ftsize, loc=LEG_LOC)
    plt.title(title, fontsize=ftsize)
    plt.tight_layout()
    plt.savefig(savename,fmt=savefmt)
    plt.show()

def plot_2D(array, Title='Add Title', colorbar="True", vmin=0.0, vmax=None,
            interpolate='None', x_label='Add x_label', y_label='Add y_label', x_lim=None,
            y_lim=None, fig_size=(5, 5), fontsize_title=15, fontsize_label=10, filename=None,
            Save_only=False, Box_x_len=False, Box_y_len=False, Box_Origin=False,
            origin=None, sv_fmt='png', scalebar_units="m", scalebar_dim="si-length",
            pix_len=False, freq=False, cmap=False, freq_x=False, freq_y=False,
            bar_label='', xmin=False, xmax=False, ymin=False, ymax=False,
            scale_fix_val=None, len_frac=0.2, image_only=False,
            annotation=False, annotation_loc=[0.7, 0.95], annotation_size=20,
            sb_lenpix=False, scalebar = False, FTSIZE = 35, pix_unit = 'nm'):
    '''
    Function to plot images with many features such as scale bar, title,
    labels, and annotations.
    '''

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    #mpl.rcParams['savefig.pad_inches'] = -1
    PIX = pix_len
    if xmin == False:
        xmin = 0
        xmax = np.shape(array)[1]
        ymin = 0
        ymax = np.shape(array)[0]
    
    if colorbar == True:
        import matplotlib.gridspec as grd
        fig = plt.figure(1,figsize = (fig_size[0],fig_size[1]-1))
        gs = grd.GridSpec(1,2,figure = fig,width_ratios = [25,1],wspace=0.01)
        ax = plt.subplot(gs[0])
        ax_cb = plt.subplot(gs[1])
    else:
        fig = plt.figure(figsize = fig_size)
        ax = fig.add_subplot(111, aspect = 'equal')
    if Box_x_len == False:
        
        if origin == 'lower':
            if cmap == False:
                im = ax.imshow(array, origin = 'lower',
                               extent = [xmin,xmax,ymax,ymin],
                               cmap = 'gray', interpolation = interpolate,
                               vmin = vmin, vmax = vmax)
            else:
                im = ax.imshow(array, origin = 'lower',
                               extent = [xmin,xmax,ymax,ymin],
                               cmap = str(cmap), interpolation = interpolate,
                               vmin = vmin, vmax = vmax)
        else:
            if cmap == False:
                im = ax.imshow(array, origin = 'upper',
                               extent = [xmin,xmax,ymax,ymin],
                               cmap = 'gray', interpolation = interpolate,
                               vmin = vmin, vmax = vmax)
            else:
                im = ax.imshow(array, origin = 'upper',
                               extent = [xmin,xmax,ymax,ymin],
                               cmap = str(cmap), interpolation = interpolate,
                               vmin = vmin, vmax = vmax)
        if image_only==False:
            ax.set_title(Title, fontsize = fontsize_title)
            ax.set_xlabel(x_label, fontsize = fontsize_label)
            ax.set_ylabel(y_label, fontsize = fontsize_label)
        plt.ylim(y_lim)
        plt.xlim(x_lim)
        #ax.xaxis.set_label_coords(0.5, -0.25)
        if colorbar == True:
            bar = plt.colorbar(im,cax = ax_cb)
            bar.set_label(bar_label, fontsize = fontsize_label, labelpad=-0.5)
            bar.ax.tick_params(labelsize = fontsize_label)
        if sb_lenpix != False:
            leny, lenx = np.shape(array)
##            print(lenx, leny//20, [lenx//20, lenx//20 + sb_lenpix],
##                     [leny - leny//20, leny - leny//20])
            plt.plot([lenx//20, lenx//20 + sb_lenpix],
                     [leny - leny//20, leny - leny//20], lw = 5, c = 'w')
        if scalebar == True:
            scalebar = add_scalebar(array, pix = PIX, 
                                    pix_unit = pix_unit,
                                    coord_system = ax.transData,
                                    scalebar_fraction_of_image = 0.25,
                                    loc = 'lower right', label_top = True,
                                    sep = 5, frameon = False,
                                    ftsize = FTSIZE, color = 'white')
            ax.add_artist(scalebar)
        if image_only == True:
            plt.tick_params(axis = 'both', which = 'both', bottom = 'off',
                            labelbottom = 'off', left = 'off',
                            labelleft = 'off')
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        elif freq != False:
            plt.xticks(freq_x, fontsize = fontsize_label)
            plt.yticks(freq_y, fontsize = fontsize_label)
        else:
            plt.xticks(fontsize = fontsize_label)
            plt.yticks(fontsize = fontsize_label)
        if annotation != False:
            plt.text(s = str(annotation),
                     x = annotation_loc[0] * array.shape[1],
                     y = annotation_loc[1] * array.shape[0], c = 'w',
                     size = annotation_size, backgroundcolor = 'k')
        if filename != None:
            plt.autoscale(tight=True)
            plt.savefig(str(filename) + "."+str(sv_fmt), format = str(sv_fmt),
                        dpi = 300, bbox_inches = 'tight', pad_inches = 0)
        if Save_only == False:
            plt.autoscale(tight=True)
            plt.show()
        else:
            plt.close()
    else:
        from matplotlib import patches
        ax.add_patch(
            patches.Rectangle(
                Box_Origin,
                Box_x_len,
                Box_y_len,
                edgecolor = "red",
                fill = False      # remove background
            ))
        if cmap == False:
            im = ax.imshow(array, cmap = 'gray', interpolation = interpolate,
                           vmin = vmin, vmax = vmax)
        else:
            im = ax.imshow(array, interpolation = interpolate, vmin = vmin, vmax = vmax)
        if image_only==False:
            plt.title(Title, fontsize = fontsize_title)
            ax.set_xlabel(x_label, fontsize = fontsize_label)
            ax.set_ylabel(y_label, fontsize = fontsize_label)
        plt.ylim(y_lim)
        plt.xlim(x_lim)
        #ax.xaxis.set_label_coords(0.5, -0.25)
        if image_only == True:
            plt.tick_params(axis = 'both', which = 'both', bottom = 'off',
                            labelbottom = 'off', left = 'off',
                            labelleft = 'off')
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        elif freq != False:
            plt.xticks(freq_x, fontsize = fontsize_label)
            plt.yticks(freq_y, fontsize = fontsize_label)
        else:
            plt.xticks(fontsize = fontsize_label)
            plt.yticks(fontsize = fontsize_label)
        if sb_lenpix != False:
            leny, lenx = np.shape(array)
##            print(lenx//20, leny//20)
            plt.plot([lenx//20, lenx//20 + sb_lenpix],
                     [leny - leny//20, leny - leny//20], lw = 5, c = 'w')
        if pix_len != False:
            from matplotlib_scalebar.scalebar import ScaleBar
            scalebar = ScaleBar(pix_len, color = 'w', box_color = 'k',
                                location = 'lower left', units=scalebar_units,
                                dimension=scalebar_dim,
                                font_properties = 'DejaVu Sans-40',
                                label = None, length_fraction = len_frac)
            plt.gca().add_artist(scalebar)
        if colorbar == True:
            bar = plt.colorbar(im,cax = ax_cb)
            bar.set_label(bar_label, fontsize = fontsize_label, labelpad=-0.5)
            bar.ax.tick_params(labelsize = fontsize_label)
        if annotation != False:
            plt.text(s = str(annotation),
                     x = annotation_loc[0] * array.shape[1],
                     y = annotation_loc[1] * array.shape[0], c = 'w',
                     size = annotation_size, backgroundcolor = 'k')
        if filename != None:
            plt.autoscale(tight=True)
            plt.savefig(str(filename) + "."+str(sv_fmt), format = str(sv_fmt),
                        dpi = 300, bbox_inches = 'tight', pad_inches = 0)
        if Save_only == False:
            plt.show()
        else:
            plt.close()

def plot_hsv(data, ftsize, Title='', colorbar=True, vmin=0.0, vmax=None,
            bar_label='', interpolate='None', x_label='Add x_label',
            y_label='Add y_label', x_lim=None, y_lim=None, fig_size=(5, 5),
            fontsize_title=15, fontsize_label=10, filename=None, origin=None,
            colorwheel=False, offset=+2*np.pi/4., Save_only=False,
            Box_x_len=False, Box_y_len=False, Box_Origin=False, pix_len=False,
            freq=False, cmap=False, freq_x=False, freq_y=False, len_frac=0.2):

    '''
    This function uses smooth_hsv written by Tyler Harvey to plot a complex
    array with brightness as the amplitude and color as the phase. The color
    repeats itself every 2*pi
    '''

    from smooth_hsv_trh import smooth_hsv as hsv
    
    if colorwheel == True:
        image = hsv(data, o = offset)
    else:
        image = hsv(data)

    fig, ax1 = plt.subplots(figsize = fig_size)
    if origin == 'lower':
        p = ax1.imshow(image.data, origin = 'lower', cmap = image.cmap)
    else:
        p = ax1.imshow(image.data, origin = 'upper', cmap = image.cmap)
    if pix_len != False:
        from matplotlib_scalebar.scalebar import ScaleBar
        scalebar = ScaleBar(pix_len, color = 'w', box_color = 'k',
                            location = 'lower left', units=scalebar_units,
                            dimension=scalebar_dim,
                            font_properties = 'DejaVu Sans-40',
                            length_fraction = len_frac)
        plt.gca().add_artist(scalebar)
    ax1.set_title(str(Title), fontsize = ftsize)

    if freq != False:
        ax1.xticks(freq_x, fontsize = fontsize_label)
        ax1.yticks(freq_y, fontsize = fontsize_label)

    if colorwheel == True:
        display_axes = fig.add_axes([0.85,0.25,0.1,0.1], projection = 'polar')
        display_axes._direction = 2*np.pi ## This is a nasty hack - 
        ## using the hidden field to 
        ## multiply the values such that 1 become 2*pi
        ## this field is supposed to take values 1 or -1 only!!

        norm = mpl.colors.Normalize(0.0, 2*np.pi)

        # Plot the colorbar onto the polar axis
        # note - use orientation horizontal so that the gradient goes around
        # the wheel rather than centre out
        quant_steps = 2056
        cb = mpl.colorbar.ColorbarBase(display_axes, cmap = image.cmap,
#                                        cm.get_cmap('hsv',quant_steps),
                                       norm = norm,
                                       orientation = 'horizontal')

        # aesthetics - get rid of border and axis labels                                   
        cb.outline.set_visible(False)                                 
        display_axes.set_axis_off()
    if colorbar == True:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        ax2 = plt.gca()
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        bar = plt.colorbar(image.fake_data, cax = cax)
        bar.set_label(bar_label, fontsize = fontsize_label, labelpad=-0.5)
        bar.ax.tick_params(labelsize = ftsize*0.75)
    if x_label == '':
        #ax = plt.axes()

        ax1.xaxis.set_major_locator(plt.NullLocator())
    if y_label == '':
        #ax = plt.axes()

        ax1.yaxis.set_major_locator(plt.NullLocator())
    else:
        plt.xlabel(x_label, fontsize = fontsize_label)
        plt.ylabel(y_label, fontsize = fontsize_label)
        plt.ylim(y_lim)
        plt.xlim(x_lim)
        plt.xticks(fontsize = fontsize_label)
        plt.yticks(fontsize = fontsize_label)
    fig.tight_layout()
    if filename != None:
        plt.savefig(str(filename) + ".pdf", format = 'pdf', dpi = 200,
                    pad_inches=-0.5)
        plt.savefig(str(filename) + ".jpg", format = 'jpeg', dpi = 200,
                    pad_inches=-0.5)
    if Save_only == False:
        plt.show()

def std_contrast(img, s_factor):
    """
    This function calculates the vmin and vmax values the standard deviation
    from the mean.
    """
    mean = np.mean(img)
    std = np.std(img)
    vmin = mean-s_factor*std
    vmax = mean+s_factor*std
    return(vmin,vmax)

def z_score_contrast(img):
    """
    This function calculates the vmin and vmax values using the zscore. When
    abs(zscore)>3, the values are set to zero and the remaining min and max
    values are returned as vmin and vmax.
    zscore = x-mean/std
    """
    z_score = stats.zscore(img)
    z_score[np.where(np.abs(z_score)<=3.)]=0
    vmin = img[np.where(z_score==np.min(z_score))]
    vmax = img[np.where(z_score==np.max(z_score))]
    return(vmin,vmax)
