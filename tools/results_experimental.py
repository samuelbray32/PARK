import numpy as np
import matplotlib.pyplot as plt
import pickle
from .bootstrapTest import bootstrap_traces, bootstrapTest, bootstrap
from .bootstrapTest import bootstrap_diff, bootstrap_relative
from tools.measurements import pulsesPop, sensitization, habituation
###############################################################################################
def data_of_interest_pulseTrain(result,pulse,isi,iti,n,interest=[],exclude=[],framerate=2):
    to_plot = []
    names=result.keys()
    for dat in names:
        if dat in to_plot: continue
        for i in interest:
            if i in dat:
                keep = True
                for ex in exclude:
                    if ex in dat: keep = False
                if not result[dat]['pulse']==pulse*framerate: keep=False
                if not result[dat]['delay']==(isi-pulse)*framerate: keep=False
                if not result[dat]['n']==n: keep=False
                if keep: to_plot.append(dat)
    return to_plot

###############################################################################################
def pulseTrain_WT(**kwargs):
    '''deprecated version of pulse train. should be able to remove...'''
    return pulseTrain(**kwargs)

def pulseTrain(pulse=[1,], isi=[30,], iti=[2,], n=20, n_boot=1e3, statistic=np.median, integrate=30,
               trial_rng=(0,12), color_scheme=plt.cm.viridis, interest=['WT'], exclude=['regen'],
               norm_time=False, ax=None, color=None, fig=None, shift=0,
               measurements=[sensitization, habituation, ]):
    '''Multifunctional tool. can sweep through pulse isi or iti for given gene condition.
    Alternatively, can return the compiled result for a given gene condition (see: call in pulseTrain_rnai)'''
    #load data
    name = 'data/LDS_response_pulseTrain.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    #check that variables provided are list
    if not type(pulse) is list: pulse=[pulse]
    if not type(isi) is list: isi=[isi]
    if not type(iti) is list: iti=[iti]
    #manage which variable is being scanned
    sweep = None
    sweep_ind = None
    error1 = 'pulseTrain only accepts 1 swept variable at a time. please check that at most one variable list is >len(1)'
    if len(pulse)>1:
        sweep = pulse.copy()
        sweep_ind=0
    elif len(isi)>1:
        if sweep is None:
            sweep = isi.copy()
            sweep_ind=1
        else:
            print(error1)
            return
    elif len(iti)>1:
        if sweep is None:
            sweep = iti.copy()
            sweep_ind=2
        else:
            print(error1)
            return
    else:
        sweep = pulse.copy()
        sweep_ind=0
    condition = [pulse[0],isi[0],iti[0]]

    #Plot
    if ax is None:
        fig,ax=plt.subplots(ncols=3,sharey=False,figsize=(16,8))
        ax[0].get_shared_y_axes().join(ax[0], ax[1])
    for i,var in enumerate(sweep):
        condition[sweep_ind]=var
        if color is None:
            c = color_scheme(i/len(sweep))
        else:
            c = color
        #find matching datasets
        to_plot = data_of_interest_pulseTrain(result,*condition,n,interest=interest,exclude=exclude,framerate=2)
        if len(to_plot)==0:continue
        n_i=n
        if condition[0]==condition[1]: #TODO: resolve better?
            to_plot=['101921WT_1s1s_n300_2hITI']
            n_i=300
        yp = result[to_plot[0]]['data']
        for dat in to_plot[1:]:
            for j,val in enumerate(result[dat]['data']):
                if len(yp)>j:
                    yp[j]=np.append(yp[j],val,axis=0)
                else:
                    yp.append(val)
        yp = np.concatenate(yp[trial_rng[0]:trial_rng[1]])
        print(to_plot,yp.shape)
        #plot traces
        y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)
        tp=result['tau'].copy()
        if norm_time:
            tp = tp/condition[1]*60
        ax[0].plot(tp,y,c=c,label=f'{interest[0]}_{condition[0]}s{condition[1]}s_n{n}_{condition[2]}hiti, ({yp.shape[0]})',lw=1)
        ax[0].fill_between(tp,*rng,alpha=.1,color=c,lw=0,edgecolor='None')
        #plot integrated
        loc = np.where(tp==0)[0][0]
        y, (lo, hi) = bootstrap(yp[:,loc:],n_boot=n_boot,statistic=pulsesPop,n=n_i,isi=int(condition[1]*2),integrate=integrate*2)
        xp = np.arange(len(y))
        ax[1].scatter(xp,y,color=c)
        ax[1].plot(xp,y,color=c,ls=':')
        ax[1].fill_between(xp,lo,hi,alpha=.1,facecolor=c,zorder=-1)
        #do scalar measurements
        for j,M in enumerate(measurements):
            yy, rng = bootstrap(yp[:,loc:],n_boot=n_boot,statistic=M,n=n_i,isi=int(condition[1]*2),integrate=integrate*2)
            xx = j+i/len(sweep)/2+shift
            ax[2].bar([xx],[yy],color=c,alpha=.2,width=.1)
            ax[2].plot([xx,xx],rng,c=c)
        if shift==0:
            ax[2].set_xticks(np.arange(len(measurements)))
            ax[2].set_xticklabels([M.__name__ for M in measurements],rotation=-45)
    if norm_time:
        ax[0].set_xlim(-2,1.25*n)
        ax[0].set_xlabel('time/isi')
    else:
        ax[0].set_xlim(-2,n*np.max(isi)/60*1.25)
        ax[0].set_xlabel('time (min)')
    ax[1].set_xlabel('pulse number')
    ax[0].set_ylabel('Activity')
    ax[1].set_ylabel(f'total response (0-{integrate}s post pulse')
    ax[0].set_ylim(0,1.5)
    ax[0].legend()
    sweep_names = ['pulse','ISI','ITI']
    if not fig is None:
        fig.suptitle(f'variable {sweep_names[sweep_ind]}')
    return fig, ax
###############################################################################################
def pulseTrain_rnai(interest=[], exclude=['regen'],pulse=1, isi=30, iti=2, n=20, n_boot=1e3,
                    statistic=np.median, integrate=30, trial_rng=(0,100), norm_time=False):
    '''plots knockdown pulse train overlayed on WT'''
    #load data
    name = 'data/LDS_response_pulseTrain.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    #plot WT
    fig, ax = pulseTrain(pulse=pulse, isi=isi, n_boot=n_boot, norm_time=norm_time, trial_rng=trial_rng, n=n,
               integrate=integrate, exclude=['regen','vibe'], color='grey')
    #plot each gene
    for i,name in enumerate(interest):
        exclude_i = exclude
        if not '+' in name:
            exclude_i.append('+')
        pulseTrain(interest=[name], pulse=pulse, isi=isi, n_boot=n_boot, norm_time=norm_time, trial_rng=trial_rng, n=n,
                   integrate=integrate, exclude=exclude_i, ax=ax, color=plt.cm.Set1(i/9),shift=(i+1)/len(interest)/2)
    #plot details
    fig.suptitle('')
    return fig,ax
