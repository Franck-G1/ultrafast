import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import pyplot as plt, colors as mcolors
from matplotlib.ticker import ScalarFormatter

def read_ta(fname, fname_cal,bkg_lim=10):
    """Read and interpret Transient Absorption data file (file.dat)
    
    Fname:
        Filename or pathname to file.dat

    Fname_cal:
        Filename or pathname to file.csv. This file is used to recover calibration parameters from the calibration function.
        Ideally use the template script "TA_Cal.py" to perform calibration in the same directory as the script you are using. 

    Bkg_lim:
        Number of averaged scans in the negative times that are considered as background an substracted from the data (10 by default).
    """
    
    #Defining variables
    count = 0 
    scans = None 
    t_delay = None 
    px = None

    with open(fname, "r") as file: #Open the file and search for specific characteristics
        for line in file:
            if line.startswith("%"):
                count+=1
            
                if "scans" in line:
                    scans = int(line.split("=")[1].strip())
            
                if "time-delays" in line:
                    t_delay = int(line.split("=")[1].strip())

                if "number-of-pixels" in line:
                    px = int(line.split("=")[1].strip())

                else:
                    continue
            
            else:
                break
        
        file.close()
    
    print(f" Scans: {scans}, Time-delays: {t_delay}, CCD pixels: {px}") #printing header information
    
    #STARTING DATA TREATMENT
    data_matrix = np.loadtxt(fname, skiprows = count) #skiping the header lines and storing openened data
    data_matrix = np.array(data_matrix) #reformating n x m matrix correctly in python

    delta_A = []
    delta_Aerr = []
    t = []

        #Ensuring coherency to perform reformating
    px_data_read = data_matrix.shape[1]/2-1
    if (px_data_read) != px:
        raise ValueError(f"Pixels in header ({px}) and pixels read ({px_data_read}) are not equivalent.")

    for col_index in range(data_matrix.shape[1]):
            #reshaping correctly the time dimension using the number of scans
        if col_index == 0: 
            t.append([row[col_index] for row in data_matrix])
            t = t[0]
            t = t[0:len(t) // scans]
            t = np.array(t)
            t = t*1e12
        
            #reshaping TAS signal 
        if  col_index %2 == 0 and col_index !=0:
            delta_A.append([row[col_index] for row in data_matrix])

            #reshaping TAS signal error 
        if col_index %2 !=0 and col_index !=1:
            delta_Aerr.append([row[col_index] for row in data_matrix])

        else:
            continue

    #Averaging signal value
    signal_avg = [[] for _ in range(len(delta_A))]
    n_px = len(delta_A)

    for pixel in range(n_px):

        tmp = np.array(delta_A[pixel])*1e-6
        dt_length = len(tmp) // scans #scans number to reshape properly 

            #ensuring t-delays consistency
        if len(tmp) %scans != 0:
            raise ValueError("t-delays aren't consistent for each scan.")
    
        AVG = np.zeros(dt_length)
        row_start = 0
        row_stop = dt_length
    
        for n in range(scans):
            data_block = tmp[row_start : row_stop]

            AVG = AVG + data_block #stacking process
            row_start += dt_length
            row_stop += dt_length 
    
        AVG = AVG / scans

        signal_avg[pixel] = AVG
    
    
    #Bkg correction
    for pixel in range(n_px):
        tmp = signal_avg[pixel] #loading the correct column
        bkg = tmp [0:bkg_lim]   #selecting the first negative t-delays
        bkg = np.mean(bkg) #average of this t-delays

        signal_avg[pixel] = signal_avg[pixel] - bkg #destructive substraction operation
    
    signal_avg = np.transpose(signal_avg)*1e3

    #From px to wl & wn
    param = np.loadtxt(fname_cal, delimiter = ",")
    pixels = np.array([i for i in range(px)])
    pixels += 1 

    wl = pixels * param[0] + param[1] #wl in nm
    wn = 1e7 / wl #wn in cm-1

    return t, wl, wn, signal_avg, delta_Aerr

def nm_to_cm1(x):
    return 1e7 / x

def cm1_to_nm(x):
    return 1e7 / x

def plot_ta(wl, t, ta_signal, title, xlim = [350, 800], levels = 20, cmap = "bwr", vmax = 20, force_bkg_to_0 = None):
    """Plot transient asbroption data into a convenient contour plot and save it into the active directory.
    This function should be used with read_ta() for a good result. 

    Wl: 
    Wavelength axis [nm].

    T:
    Time axis [ps]. 

    Ta_signal:
    Absorption difference. 

    Title:
    Title of the plot (just give the compound). 

    Xlim:
    X-axis boundaries [a,b] [nm]. 

    Levels:
    Number of levels for the contour plot (by default 20).

    Cmap:
    Color map used (bwr, Reds or Blues) (by default bwr). 

    Vmax:
    Threshold to maximum signal. 

    Force_bkg_to_0:
    Number of positive signal that is sent to 0 (white) for clarity (None by default). 
    """

    level_list = np.linspace(0, vmax, levels)

    if "bwr" in cmap:
        colormap = plt.cm.bwr(np.linspace(0,1,levels))
    
    if "Reds" in cmap:
        colormap = plt.cm.Reds(np.linspace(0,1,levels))

    if "Blues" in cmap:
        colormap = plt.cm.Blues(np.linspace(0,1,levels))
    
    if force_bkg_to_0 is not None:
        colormap[:force_bkg_to_0] = 0

    fig, ax = plt.subplots(figsize = (8,5))
    cf = ax.contourf(wl, t, ta_signal, levels = level_list, colors = colormap)

    # add secondary x-axis for wavenumber
    def nm_to_cm1(x):
        x = np.array(x)
        x = np.where(x==0, np.nan, x)
        return 1e7 / x

    def cm1_to_nm(x):
        x = np.array(x)
        x = np.where(x==0, np.nan, x)
        return 1e7 / x

    secax = ax.secondary_xaxis('top', functions=(nm_to_cm1, cm1_to_nm))
    secax.set_xlabel(r"Wavenumber, $\tilde{\nu}$ \ cm$^{-1}$")
    secax.tick_params(labelsize = 6)

    # set scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))  # force sci notation
    secax.xaxis.set_major_formatter(formatter)

    ax.set_title(f"TA Contour Plot for {title}", fontweight = "bold")
    ax.set_xlabel(r"Wavelength, $\lambda$ \nm")
    ax.set_ylabel(r"Time, t \ps ")
    ax.set_xlim(xlim)

    ax.tick_params(axis='both', which='major', labelsize=8)

    fig.colorbar(cf, ax = ax, label=r"$\Delta A$ \mOD")

    fig.tight_layout()
    plt.show()


    

def calibrate_ta():
    return print("Not Implemented")

def interpolate_ta():
    return print("Not Implemented")

def fit_Gaussian(t, A, mu, sigma):
    """Try to fit a common Gaussian function to data"""
    dA = A * np.exp(- (t-mu)**2 / (2 * sigma**2)) #common Gaussian function
    return dA

def fit_oke():
    return print("Not Implemented")

def get_abs(fname, cuts=None, normalization=None):

    """Read and analyzes absorption data from a provided .csv file. Return: wavelength, wavenumber, absorption 

    Fname: 
        Filename or pathname to file.csv

    Cuts:
        Boundaries [nm] to take data from (e.g [200,800])

    Normalization:
        Normalize by maximum or no.
    
    The provided file should have at least two columns (wavelength, absorbance) and a two rows header.  
    """

    # load absorption data
    data_abs = np.loadtxt(fname, skiprows=2, delimiter=',', usecols=(0,1))
    
    wl_abs = data_abs[:,0]  # wavelength
    A = data_abs[:,1]       # absorbance
    
    if cuts is not None:
        mask = (wl_abs > cuts[0]) & (wl_abs < cuts[1])
        wl_abs = wl_abs[mask]
        A = A[mask]
    
    wn = (1 / wl_abs) * 1e7  # convert to wavenumber cm^-1
    
    if normalization:
        A = A / np.max(A)
    
    return wl_abs, wn, A


def read_em():
    return print("Not Implemented")
