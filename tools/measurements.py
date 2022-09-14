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
        return np.where(x[loc:]/x.max()<=.5)[0][0]/120
    except: return 0

def responseDuration(data,thresh=.3):
    '''time for population traces to go below threshold value'''
    x = np.median(data,axis=0)
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
