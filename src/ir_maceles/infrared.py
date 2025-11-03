import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, rfft, rfftfreq
from scipy.signal.windows import gaussian
from scipy.signal import convolve


# from util_fft import *
def fftcrosscorr(x,y,dlen=10000):
    
        # make sure that we have odd number of signals as it makes fft easier
    if (dlen%2 == 0):
        dlen -= 1
        #x.append(x[0])
        #y.append(y[0])
        
    # the fft coecofficents of the crosscorrelation function c_xy(t)
    dt = x[1,0] - x[0,0] # assume the timestep is constant
    window = len(x)//dlen
    omega0 = 2.0*np.pi/(dlen-1)/dt
    cxyomega = np.zeros((dlen,2),dtype=np.complex_)
    cxyomega[0:dlen//2 + 1,0] = np.arange(dlen//2 + 1)*omega0
    cxyomega[dlen//2 + 1:,0] = np.arange(dlen//2,0,-1)*omega0*-1
    
    for i in range(window):
        dx = x[i*dlen:(i+1)*dlen,1] 
        dy = y[i*dlen:(i+1)*dlen,1]
        Ax = np.fft.fft(dx[:] , axis = 0)
        Ay = np.fft.fft(dy[:] , axis = 0)

        cxyomega[:,1] += np.conjugate(Ax[:]) * Ay[:]/dlen*dt
        
    for i in range(window-1):
        dx = x[i*dlen+dlen//2:(i+1)*dlen+dlen//2,1] 
        dy = y[i*dlen+dlen//2:(i+1)*dlen+dlen//2,1]
        Ax = np.fft.fft(dx[:] , axis = 0)
        Ay = np.fft.fft(dy[:] , axis = 0)

        cxyomega[:,1] += np.conjugate(Ax[:]) * Ay[:]/dlen*dt
        
    cxyomega[:,1]/=(window*2-1)
    return cxyomega

def smooth_signal(signal, window_size=51, sigma=7):
    # Gaussian kernel for convolution and normalization
    kernel = gaussian(window_size, std=sigma)
    kernel /= np.sum(kernel)
    smooth_signal = convolve(signal, kernel, mode='same')
    return smooth_signal

def IR_plot(total_dP, dt=0.50, dlen=10000, length=800, window_size=30, sigma=3):
    dPt = np.zeros((len(total_dP), 4))
    dPt[:,0] = np.arange(len(total_dP)) * dt
    dPt[:,1:4] = total_dP
    ft_x = fftcrosscorr(dPt[:, [0, 1]], dPt[:, [0, 1]], dlen=dlen)
    ft_y = fftcrosscorr(dPt[:, [0, 2]], dPt[:, [0, 2]], dlen=dlen)
    ft_z = fftcrosscorr(dPt[:, [0, 3]], dPt[:, [0, 3]], dlen=dlen)

    omega = ft_x[:,0] *1e15/2.99792458e10/(2*np.pi)
    ft_avg = (ft_x[:,1] + ft_y[:,1] + ft_z[:,1])/3
    smooth_inten = smooth_signal(ft_avg, window_size=window_size, sigma=sigma)
    return omega[:length], ft_avg[:length], smooth_inten[:length]

def normalize_area(omega, intensity):
    area = np.trapz(intensity, omega)
    return intensity/area

def pickle_plot(pickle_file, dt=0.50, dlen=10000, length=800, window_size=30, sigma=3):
    with open(pickle_file, 'rb') as f:
        bec_dict = pickle.load(f)
        total_dP = bec_dict['total_dp']
        omega, ft_avg, inten = IR_plot(total_dP, dt, dlen, length, window_size, sigma)
        normalized_omega, normalized_inten = normalize_area(omega, inten)
    return normalized_omega, ft_avg, normalized_inten
       
