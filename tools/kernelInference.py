import os
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import matplotlib.pyplot as plt
import pickle
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def data_of_interest(names,interest=[],exclude=[]):
    to_plot = []
    for dat in names:
        if dat in to_plot: continue
        for i in interest:
            if i in dat:
                keep = True
                for ex in exclude:
                    if ex in dat: keep = False
                #check double/triple knockdown
                if dat.count('+')>i.count('+'):
                    # print(dat)
                    keep=False
                if keep: to_plot.append(dat)
    return to_plot

def augment_stimulus_delta(UU):
    UU_d = np.zeros_like(UU)
    UU_d[:,1:] = (UU[:,1:]-UU[:,:-1])
    def relu(x):
        return x * (x>0)
    return np.concatenate([UU,relu(UU_d),relu(-UU_d)],axis=-1)

def prepareData(gene,pulses,steps,exclude=[],
                light_sample=(-9,15),step_off=True,augment=2000,
                kernel_size=0,ramps=[],paired_delays=[]):
    UU = []
    ZZ = []
    '''get pulse data'''
    data_name = 'data/LDS_response_rnai.pickle'
    with open(data_name,'rb') as f:
        result = pickle.load(f)
    tp = result['tau']
    ind_t = np.where((tp>light_sample[0])&(tp<=light_sample[1]))[0]
    t_on = np.argmin(tp**2)
    #loop through pulses
    for j,pulse in enumerate(pulses):
        exclude_this=exclude.copy()
        if not 'p' in gene:
            exclude_this.append('p')
        tag = f'_{pulse}s'
        if pulse==5:
            tag=''
            exclude_this.append('_')
        #find datasets
        yp = []
        search=gene+tag
        to_plot = data_of_interest(result.keys(),[search],exclude_this)
        yp = [result[dat] for dat in to_plot]
        if len(yp)==0:
            print(gene,pulse,'[]',0)
            continue
        yp = np.concatenate(yp)
        print(gene, pulse, to_plot,yp.shape[0])
        #take median and append
        yp = np.median(yp,axis=0)
        u_i = np.zeros((tp.size))
        if pulse*2>=1:
            u_i[t_on:int(t_on+pulse*2)] = 1
        else:
            u_i[t_on:t_on+1] = pulse*2
        u_i = u_i[ind_t]
        if kernel_size:
            ZZ.append(np.append(np.zeros((1,kernel_size-1))*np.nan,yp[None,ind_t],axis=1))
            UU.append(np.append(np.zeros((1,kernel_size-1)),u_i[None,:],axis=1))
        else:
            ZZ.append(yp[None,ind_t])
            UU.append(u_i[None,:])
    # print(ZZ[-1].shape)
    '''get step data'''
    data_name = 'data/LDS_response_LONG.pickle'
    with open(data_name,'rb') as f:
        result_step = pickle.load(f)
    tp = result_step['tau']
    ind_t = np.where((tp>light_sample[0])&(tp<=light_sample[1]))[0]
    t_on = np.argmin(tp**2)
    for power in steps:
        exclude_this=exclude.copy()
        tag = f'_30m2h{power}bp'
        #find datasets
        yp = []
        search=gene+tag
        to_plot = data_of_interest(result_step.keys(),[search],exclude_this)
        yp = [result_step[dat] for dat in to_plot]
        if len(yp)==0:
            print(gene,power,'[]',0)
            continue
        yp = np.concatenate(yp)
        print(gene, power, to_plot,yp.shape[0])
        #take median and append
        yp = np.median(yp,axis=0)
        u_i = np.zeros((tp.size))
        u_i[t_on:t_on+30*120] = power/255
        if kernel_size:
            ZZ.append(np.append(np.zeros((1,kernel_size-1))*np.nan,yp[None,ind_t],axis=1))
            UU.append(np.append(np.zeros((1,kernel_size-1)),u_i[None,ind_t],axis=1))
        else:
            ZZ.append(yp[None,ind_t])
            UU.append(u_i[None,ind_t])
        if step_off:
            ind_t_off = np.where((tp>30+light_sample[0]-kernel_size/120)&(tp<=30+light_sample[1]))[0][1:]
            ZZ.append(yp[None,ind_t_off])
            UU.append(u_i[None,ind_t_off])
            print(UU[-1].shape,UU[-2].shape)
    '''find ramp data'''
    if len(ramps)>0:
        print('WARNING: ramps not implemnented in prepare data SB-101222')
    '''get paired pulse data'''
    #load data
    experiment='uv-uv'
    name = f'data/LDS_response_pairedPulse_{experiment}.pickle'
    with open(name,'rb') as f:
            result = pickle.load(f)
    tp = result['tau']
    ind_t = np.where((tp>light_sample[0])&(tp<=light_sample[1]))[0]
    #loop experiments
    for i,d in enumerate(paired_delays):
        test = [f'{gene}_{5}s{5}s_{d}mDelay']#f'WT_{pulse1}s{pulse2}s_{d}mDelay'
        if d<1:
            test = [f'{gene}_{5}s{5}s_{int(d*60)}sDelay']
        # if control_rnai and cond=='WT':
        #     if d>=1:
        #         test = test + [f'cntrl_{pulse1}s{pulse2}s_{d}mDelay']
        #     else:
        #         test = test + [f'cntrl_{pulse1}s{pulse2}s_{int(d*60)}sDelay']

        test = data_of_interest(result.keys(),test,['_24h']+['081222cntrl_5s5s_1mDelay','081222cntrl_5s5s_2mDelay','081222cntrl_5s5s_3mDelay',])
        print(test)

        yp = np.concatenate([result[test_i]['primary'] for test_i in test])
        yp = np.median(yp,axis=0)
        u_i = 0*tp
        loc = np.argmin(np.abs(tp))
        u_i[loc:loc+10]=1
        loc+= int(10+d*120)
        u_i[loc:loc+10]=1
        if kernel_size:
            ZZ.append(np.append(np.zeros((1,kernel_size-1))*np.nan,yp[None,ind_t],axis=1))
            UU.append(np.append(np.zeros((1,kernel_size-1)),u_i[None,ind_t],axis=1))
        else:
            ZZ.append(yp[None,ind_t])
            UU.append(u_i[None,ind_t])
    '''prep and return data'''
    UU = np.concatenate(UU)[...,None]
    ZZ = np.concatenate(ZZ)

    if augment:
        UU = np.concatenate([UU for _ in range(augment)])
        ZZ = np.concatenate([ZZ for _ in range(augment)])
    ind = np.arange(UU.shape[0])
    np.random.shuffle(ind)
    UU = UU[ind]
    ZZ = ZZ[ind]
    return UU,ZZ

def prepareData_regen(dpa,light_sample=(-9,15),step_off=True,augment=2000,
                kernel_size=0):

    '''Note: currently hard-coded for WT. Ask Sam if need RNAi regen'''

    UU = []
    ZZ = []
    '''get pulse data'''
    data_list = [('standard_30s2h','1002_78hpa_30s2h'),
            ('standard_5s2h','110521WT_72hpa_5s2h')]

    day_list = [(0,3),(0,3)]
    name = 'data/LDS_response_regen.pickle'
    with open(name,'rb') as f:
            result = pickle.load(f)
    tp = result['tau']
    ind_t = np.where((tp>light_sample[0])&(tp<=light_sample[1]))[0]
    t_on = np.argmin(tp**2)

    #loop through pulses
    for j,(data,day) in enumerate(zip(data_list,day_list)):
        #compile data
        yp = []
        for i,dat in enumerate(data):
            if dpa<day[i] or (dpa-day[i])>=len(result[dat]):
                continue
            if result[dat][dpa-day[i]].size==0: continue
            yp.append(result[dat][dpa-day[i]])
        yp = np.concatenate(yp)

        #define stimulus
        if '30s' in data[0]:
            pulse=30
        elif '10s' in data[0]:
            pulse=10
        elif '1s' in data[0]:
            pulse=1
        else:
            pulse=5
        print(dpa, pulse, data,yp.shape[0])
        #take median and append
        yp = np.median(yp,axis=0)
        u_i = np.zeros((tp.size))
        if pulse*2>=1:
            u_i[t_on:int(t_on+pulse*2)] = 1
        else:
            u_i[t_on:t_on+1] = pulse*2
        u_i = u_i[ind_t]
        if kernel_size:
            ZZ.append(np.append(np.zeros((1,kernel_size-1))*np.nan,yp[None,ind_t],axis=1))
            UU.append(np.append(np.zeros((1,kernel_size-1)),u_i[None,:],axis=1))
        else:
            ZZ.append(yp[None,ind_t])
            UU.append(u_i[None,:])
    print(ZZ[-1].shape)
    '''get step data'''
    data_name = 'data/LDS_response_regen_LONG.pickle'
    with open(data_name,'rb') as f:
        result_step = pickle.load(f)
    data_list=[('032222WT_30m2h64bp', '052422WT_8hpa_30m2h64bp')]
    day_list=[(0,0)]
    power_list=[64]

    tp = result_step['tau']
    ind_t = np.where((tp>light_sample[0])&(tp<=light_sample[1]))[0]
    t_on = np.argmin(tp**2)
    for j,(data,day,power) in enumerate(zip(data_list,day_list,power_list)):
        #compile data
        yp = []
        for i,dat in enumerate(data):
            if dpa<day[i] or (dpa-day[i])>=len(result_step[dat]):
                continue
            if result_step[dat][dpa-day[i]].size==0: continue
            yp.append(result_step[dat][dpa-day[i]])
        yp = np.concatenate(yp)


        print(dpa, power, data_list,yp.shape[0])
        #take median and append
        yp = np.median(yp,axis=0)
        u_i = np.zeros((tp.size))
        u_i[t_on:t_on+30*120] = power/255
        if kernel_size:
            ZZ.append(np.append(np.zeros((1,kernel_size-1))*np.nan,yp[None,ind_t],axis=1))
            UU.append(np.append(np.zeros((1,kernel_size-1)),u_i[None,ind_t],axis=1))
        else:
            ZZ.append(yp[None,ind_t])
            UU.append(u_i[None,ind_t])
        if step_off:
            # print(tp.shape,yp.shape,u_i.shape)
            ind_t_off = np.where((tp>30+light_sample[0]-kernel_size/120)&(tp<=30+light_sample[1]))[0][1:]
            # print(tp[-1],30+light_sample[1])
            # print(ind_t_off)
            ZZ.append(yp[None,ind_t_off])
            UU.append(u_i[None,ind_t_off])
            print(UU[-1].shape,UU[-2].shape)
    #prep and return data
    UU = np.concatenate(UU)[...,None]
    ZZ = np.concatenate(ZZ)

    if augment:
        UU = np.concatenate([UU for _ in range(augment)])
        ZZ = np.concatenate([ZZ for _ in range(augment)])
    ind = np.arange(UU.shape[0])
    np.random.shuffle(ind)
    UU = UU[ind]
    ZZ = ZZ[ind]
    return UU,ZZ




def inferKernel(interest, exclude=[], kernel_size = 5*120,
                pulses=[5,30], steps=[64], ramps=[], paired_delays=[], light_sample=(-1,15),control=False,smooth=0,
                augment_delta=False):
    kernels = []
    biases = []
    fig,ax = plt.subplots(figsize=(10,10))
    plt.plot([-kernel_size/120,0],[0,0],c='grey',ls=':')
    for i,gene in enumerate(interest):
        if (gene is 'WT') or control:
            if (len(interest)>1) or control:
                c='grey'
            else:
                c=c=plt.cm.Set1((0)/9)
        elif 'WT' in interest:
            c=plt.cm.Set1((i-1)/9)
        else:
            c=plt.cm.Set1(i/9)
        exclude_this=exclude.copy()
        conditional_exclude = ['+','PreRegen','1F']
        for c_e in conditional_exclude:
            if not c_e in interest:
                exclude_this.append(c_e)
        UU, ZZ = prepareData(gene,pulses,steps,exclude_this,
            light_sample=light_sample, kernel_size=kernel_size,ramps=ramps,
            paired_delays=paired_delays)
        if augment_delta:
            UU = augment_stimulus_delta(UU)
        """Build Model"""
        U = keras.layers.Input((UU.shape[1],UU.shape[-1]))
        conv = keras.layers.Conv1D(filters=1,kernel_size=kernel_size,padding='valid',
            kernel_regularizer='l2',kernel_initializer='glorot_uniform')
        Z = conv(U)
        Z = keras.layers.Lambda(lambda x:K.squeeze(x,axis=-1))(Z)
        model = keras.models.Model(U,Z)
        opt = keras.optimizers.Adam(learning_rate=1e-3)
        early_stop = keras.callbacks.EarlyStopping(monitor='loss',min_delta=1e-4,patience=1)
        model.compile(optimizer=opt,loss='mse')
        model.fit(UU,ZZ[:,-Z.shape.as_list()[1]:],batch_size=32,epochs=10,callbacks=[early_stop])
        #plot
        W = conv.get_weights()
        shift = np.squeeze(W[1])
        kernel = np.squeeze(W[0])
        if smooth:
            kernel = smooth_kernel(kernel,smooth)
        tp = np.arange(-kernel.shape[0],0)/120
        ax.plot(tp,kernel,label=gene,c=c)
        kernels.append(kernel)
        biases.append(shift)
    #details
    ax.legend()
    plt.xlabel('time (min)')
    plt.ylabel('kernel value')
    plt.show()
    return fig, kernels, biases

def inferKernel_regen(dpa, kernel_size = 5*120,light_sample=(-1,15),smooth=0):
    kernels = []
    biases = []
    # fig,ax = plt.subplots(figsize=(10,10))
    # plt.plot([-kernel_size/120,0],[0,0],c='grey',ls=':')
    fig,kernels,biases = inferKernel(['WT'],kernel_size=kernel_size,
                        light_sample=light_sample,control=True,smooth=smooth)
    ax=fig.gca()
    for i,day in enumerate(dpa):
        c = plt.cm.cool(i/len(dpa))
        UU, ZZ = prepareData_regen(day,light_sample=light_sample, kernel_size=kernel_size)
        """Build Model"""
        U = keras.layers.Input((UU.shape[1],UU.shape[-1]))
        conv = keras.layers.Conv1D(filters=1,kernel_size=kernel_size,padding='valid',
            kernel_regularizer='l2',kernel_initializer='glorot_uniform')
        Z = conv(U)
        Z = keras.layers.Lambda(lambda x:K.squeeze(x,axis=-1))(Z)
        model = keras.models.Model(U,Z)
        opt = keras.optimizers.Adam(learning_rate=1e-3)
        early_stop = keras.callbacks.EarlyStopping(monitor='loss',min_delta=1e-4,patience=1)
        model.compile(optimizer=opt,loss='mse')
        model.fit(UU,ZZ[:,-Z.shape.as_list()[1]:],batch_size=32,epochs=10,callbacks=[early_stop])
        #plot
        W = conv.get_weights()
        shift = np.squeeze(W[1])
        kernel = np.squeeze(W[0])
        if smooth:
            kernel = smooth_kernel(kernel,smooth)
        tp = np.arange(-kernel.shape[0],0)/120
        ax.plot(tp,kernel,label=day,c=c)
        kernels.append(kernel)
        biases.append(shift)
    #details
    ax.legend()
    plt.xlabel('time (min)')
    plt.ylabel('kernel value')
    plt.show()
    return fig, kernels, biases

def smooth_kernel(kernel,filter):
    buffer = np.ones(filter)
    k = np.append(buffer*kernel[0],kernel)
    k = np.append(k,buffer*kernel[-1])
    a = np.append(buffer,buffer)
    return np.convolve(k, a, mode='valid')/(2*filter)

"""
def inferKernel_2mem(interest, exclude=[], kernel_size_short = 5*120,kernel_size_long = 5*120,
                pulses=[5,30], steps=[64], light_sample=(-1,15)):
    kernels = []
    biases = []
    fig,ax = plt.subplots(figsize=(10,10),ncols=2,sharey=True)
    ax[-1].plot([-kernel_size_long/120,0],[0,0],c='grey',ls=':')
    ax[0].plot([-kernel_size_short/120,0],[0,0],c='grey',ls=':')
    for gene in interest:
        exclude_this=exclude.copy()
        conditional_exclude = ['+','PreRegen','1F']
        for c_e in conditional_exclude:
            if not c_e in interest:
                exclude_this.append(c_e)
        UU, ZZ = prepareData(gene,pulses,steps,exclude_this,
        light_sample=light_sample, kernel_size=kernel_size_long)
        # if True:
        #     print(UU.shape,ZZ.shape)
        #     U_append = np.zeros((UU.shape[0],kernel_size-1,1))
        #     Z_append = U_append[:,:,0].copy()*np.nan
        #     UU=np.append(U_append,UU,axis=1)
        #     ZZ = np.append(Z_append,ZZ,axis=1)
        #     print(UU.shape,ZZ.shape)
        ""Build Model"
        # frame_rate=120
        #import keras.backend as K
        U = keras.layers.Input((UU.shape[1],1))
        conv_long = keras.layers.Conv1D(filters=1,kernel_size=kernel_size_long,padding='valid',
            kernel_regularizer='l2',kernel_initializer='glorot_uniform')
        conv_short = keras.layers.Conv1D(filters=1,kernel_size=kernel_size_short,padding='valid',
            kernel_regularizer='l2',kernel_initializer='glorot_uniform')

        Z_long = conv_long(U)
        Z_long = keras.layers.Lambda(lambda x: K.exp(x))(Z_long)
        Z_short = conv_short(U)
        Z_short = keras.layers.Lambda(lambda x: x[:,-Z_long.shape.as_list()[1]:])(Z_short)
        Z = keras.layers.Lambda(lambda x: x[0]*x[1])([Z_long,Z_short])
        Z = keras.layers.Lambda(lambda x:K.squeeze(x,axis=-1))(Z)
        model = keras.models.Model(U,Z)
        opt = keras.optimizers.Adam(learning_rate=1e-3)
        early_stop = keras.callbacks.EarlyStopping(monitor='loss',min_delta=1e-3,patience=1)
        model.compile(optimizer=opt,loss='mse')
        model.fit(UU,ZZ[:,-Z.shape.as_list()[1]:],batch_size=32,epochs=5,callbacks=[early_stop])
        #plot
        W = conv_long.get_weights()
        norm = np.sign(W[0][-1][0])
        shift_long = np.squeeze(W[1])*norm
        kernel_long = np.squeeze(W[0])*norm
        W = conv_short.get_weights()
        shift_short = np.squeeze(W[1])*norm
        kernel_short = np.squeeze(W[0])*norm
        tp = np.arange(-kernel_long.shape[0],0)/120
        ax[1].plot(tp,kernel_long,label=gene)
        ax[0].plot(tp[-kernel_size_short:],kernel_short,label=gene)
        kernels.append((kernel_short,kernel_long))
        biases.append((shift_short,shift_long))
    #details
    ax[1].legend()
    plt.xlabel('time (min)')
    plt.ylabel('kernel value')
    plt.show()
    return fig, kernels, biases


def inferKernel_2mem_layered(interest, exclude=[], kernel_size_1 = 5*120,kernel_size_2 = 5*120,
                pulses=[5,30], steps=[64], light_sample=(-1,15)):
    kernels = []
    biases = []
    fig,ax = plt.subplots(figsize=(10,10),ncols=2,sharey=True)
    ax[-1].plot([-kernel_size_1/120,0],[0,0],c='grey',ls=':')
    ax[0].plot([-kernel_size_2/120,0],[0,0],c='grey',ls=':')
    for gene in interest:
        exclude_this=exclude.copy()
        conditional_exclude = ['+','PreRegen','1F']
        for c_e in conditional_exclude:
            if not c_e in interest:
                exclude_this.append(c_e)
        UU, ZZ = prepareData(gene,pulses,steps,exclude_this,
        light_sample=light_sample, kernel_size=kernel_size_1+kernel_size_2-1)
        # if True:
        #     print(UU.shape,ZZ.shape)
        #     U_append = np.zeros((UU.shape[0],kernel_size-1,1))
        #     Z_append = U_append[:,:,0].copy()*np.nan
        #     UU=np.append(U_append,UU,axis=1)
        #     ZZ = np.append(Z_append,ZZ,axis=1)
        #     print(UU.shape,ZZ.shape)
        ""Build Model"
        # frame_rate=120
        #import keras.backend as K
        U = keras.layers.Input((UU.shape[1],1))
        conv_1 = keras.layers.Conv1D(filters=1,kernel_size=kernel_size_1,padding='valid',
            kernel_regularizer='l2',kernel_initializer='glorot_uniform')
        conv_2 = keras.layers.Conv1D(filters=1,kernel_size=kernel_size_2,padding='valid',
            kernel_regularizer='l2',kernel_initializer='glorot_uniform')

        Z_1 = conv_1(U)
        Z_1 = keras.layers.Lambda(lambda x:x[0]*x[1][:,-x[0].shape.as_list()[1]:])([Z_1,U])
        Z = conv_2(Z_1)
        print(U.shape,Z_1.shape,Z.shape)
        Z = keras.layers.Lambda(lambda x:K.squeeze(x,axis=-1))(Z)
        model = keras.models.Model(U,Z)
        opt = keras.optimizers.Adam(learning_rate=1e-3)
        early_stop = keras.callbacks.EarlyStopping(monitor='loss',min_delta=1e-3,patience=1)
        model.compile(optimizer=opt,loss='mse')
        model.fit(UU,ZZ[:,-Z.shape.as_list()[1]:],batch_size=32,epochs=5,callbacks=[early_stop])
        #plot
        W = conv_1.get_weights()
        shift_1 = np.squeeze(W[1])
        kernel_1 = np.squeeze(W[0])
        W = conv_2.get_weights()
        shift_2 = np.squeeze(W[1])
        kernel_2 = np.squeeze(W[0])
        t1 = np.arange(-kernel_1.shape[0],0)/120
        t2 = np.arange(-kernel_2.shape[0],0)/120
        ax[0].plot(t1,kernel_1,label=gene)
        ax[1].plot(t2,kernel_2,label=gene)
        kernels.append((kernel_2,kernel_2))
        biases.append((shift_1,shift_2))
    #details
    ax[1].legend()
    plt.xlabel('time (min)')
    plt.ylabel('kernel value')
    plt.show()
    return fig, kernels, biases
"""

def predictResponse(kernel,bias,stimulus,):
    Z = np.convolve(stimulus,np.flip(kernel),mode='valid')+bias
    buffer = np.zeros(kernel.size-1)*np.nan
    return np.append(buffer,Z)
