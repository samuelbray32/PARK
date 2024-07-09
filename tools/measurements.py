"""
Set of scalar measurements of pulse response that are compatible with bootstrapping
"""
import numpy as np

######################################
#       Trace measurements           #
######################################
# def timeDependentDifference(data,ref,n_boot,conf_interval=95):
#     '''Returns whether data and ref are significantly deifferenct at each timepoint'''
#     '''Runs differently than the other measurements below (aka: not interchangable)'''
#     pop_samp = lambda x: np.median(x[np.random.choice(np.arange(x.shape[0]),x.shape[0])],axis=0)
#     diff = np.array([pop_samp(data) - pop_samp(ref) for i in range(n_boot)])
#     diff = (np.percentile(diff,(100-conf_interval)/2)>0) or
#      (np.percentile(diff,conf_interval+(100-conf_interval)/2)<0)
#     return diff

def adaptationTime(data):
    '''time for population response to come down to 1/2 it's peak value (see Uri's paper)'''
    x = np.median(data,axis=0)
    x -= np.percentile(x,5)#x.min()#
    x_max = x.max()
    loc = np.argmax(x)
    try:
        return np.where(x[loc:]/x.max()<=.5)[0][0]/120.0
    except: return 0

def halfLife(data):
    x = np.median(data,axis=0)
    # x = x-np.median(x[-120:])
    loc = np.argmax(x)
    val = x[loc]
    # print(val)
    try:
        # print((loc + np.where(x[loc:]<=val/2)[0][0])/120.0)
        # return np.where(x>=val/2)[0][-1]/120
        return (loc + np.where(x[loc:]<=val/2)[0][0])/120.0
    except:
        print('none')
        return np.nan()#loc)/120
    

def responseDuration(data,thresh=.3):
    '''time for population traces to go below threshold value'''
    x = np.median(data,axis=0)
    # print(thresh)
    # loc = np.where(x>=thresh)[0]
    # if len(loc)>0:
    #     loc = loc[0]
    # else:
    #     return 1/120
    # x = x[loc:]
    # return np.where(x<=thresh)[0][0]/120
    try:
        return np.where(x<=thresh)[0][0]/120
    except:
        if x.mean()>thresh: #if didn't go below threshold in sample window
            return x.size/120
        else:
            return 1/120 #return small value so don't get a divide by zero in comparison tests

def peakResponse(data):
    '''peak response of population trace'''
    return np.max(np.median(data,axis=0))

def totalResponse_pop(data):
    '''total response of population trace'''
    return np.mean(np.median(data[:,:10*120],axis=0))#np.mean(np.var((data>.5).astype(float),axis=0))#

def totalResponse(data):
    '''total response of individual trace'''
    return np.mean(data,axis=1)

def peakTime(data): 
    '''Calculate the time at which the peak response occurs''' 
    sampling_rate = 120
    median_data = np.median(data, axis=0)
    peak_index = np.argmax(median_data)
    peak_time = peak_index / sampling_rate
    return peak_time

def slidingPeak(data):
    window_size = 20
    sampling_rate = 120
    median_data = np.median(data, axis=0)
    windowed_peak_indices = np.convolve(median_data, np.ones(window_size)/window_size, mode='valid').argmax()
    peak_index = windowed_peak_indices + window_size // 2
    sliding_peak = peak_index / sampling_rate
    return sliding_peak  

######################################
#     Pulse train measurements       #
######################################
from scipy.stats import linregress
def pulsesPop(data,n,isi,integrate):
    y = np.median(data,axis=0)
    return np.array([y[i*isi:i*isi+integrate].mean() for i in range(n)])

def sensitization(data,**kwargs):
    ''' (population peak response- population first response)'''
    y = pulsesPop(data,**kwargs)
    return y[5]-y[0] #(y.max()-y[0])#/np.argmax(y)

def habituation(data,**kwargs):
    ''' (population peak response-population last response)'''
    y = pulsesPop(data,**kwargs)
    return y.max()-y[-1]#y.max()-y[-1]#/(y.size-np.argmax(y))



def sensitizationRate(data):
    ''' (peak response-first response)/t_peak'''
    #shape = (samples, pulses)
    t_peak = np.argmax(data,axis=1)
#     return (data.max(axis=1)-data[:,0])/(t_peak+1)
    values = []
    for i in range(data.shape[0]):
        xx = np.arange(max(t_peak[i],1))
        values.append(linregress(xx,data[i,:xx.size])[0])
    values = np.array(values)
    values[~np.isfinite(values)] = 0
    return values

def habituationRate(data):
    ''' (peak response-final response)/(t_final-t_peak+1)'''
    #shape = (samples, pulses)
    t_peak = np.argmax(data,axis=1)
#     return (data.max(axis=1)/data[:,-1])/(data.shape[1]-t_peak+1)
    values = []
    for i in range(data.shape[0]):
        xx = np.arange(max(data.shape[1]-t_peak[i],1))
        values.append(linregress(xx,data[i,-xx.size:])[0])
    values = -np.array(values)
    values[~np.isfinite(values)] = 0
    return values


# def sensitization(data):
#     ''' (peak response-first response)'''
#     #shape = (samples, pulses)
#     return (data.max(axis=1)-data[:,0])
#
# def habituation(data):
#     ''' (peak response-first response)'''
#     #shape = (samples, pulses)
#     return (data.max(axis=1)-data[:,-1])

def tPeak(data):
    ''' (peak response-first response)'''
    #shape = (samples, pulses)
    return data.argmax(axis=1)/data.shape[1]

'''
Sin Wave measurements
'''

def cross_correlate(x,U,tau):
    xx = np.median(x,axis=0)
    # xx = xx-xx.mean()
    # uu = U-U.mean()
    # return np.array([np.cov(xx[t:],U[:-t])[0,1] for t in tau])
    return np.array([np.cov([xx],np.roll(U,t))[0,1] for t in tau]) # SB-changed 2/09/23 to avoid cropping the range with longer lag times

def cross_correlate_auto(x,tau):
    # if len(x.shape)>1:
    #     x = np.median(x,axis=0)
    # xx = xx-xx.mean()
    # uu = U-U.mean()
    return np.array([np.cov(x[t:],x[:-t])[0,1] for t in tau])

from scipy import signal
def power_spectrum(x, sampling_rate=2, window_size=None):
    if window_size is None:
        window_size = 30*120#x.size//1
    window_filt = signal.windows.hamming(window_size, sym=True)
    noverlap = window_size//2
    frequencies, P = signal.welch(
        x,
        fs = sampling_rate,
        window = window_filt,
        noverlap=noverlap,
        scaling='spectrum',
        nfft=50000
    )
    return frequencies, P/np.max(P)
