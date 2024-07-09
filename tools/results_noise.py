import numpy as np
import matplotlib.pyplot as plt
import pickle
from .bootstrapTest import bootstrap_traces, bootstrapTest, bootstrap, timeDependentDifference
from .bootstrapTest import bootstrap_diff, bootstrap_relative

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
                    print(dat)
                    keep=False
                if keep: to_plot.append(dat)
    return to_plot


def autocovariance(interest_list=['WT'],exclude=[],tau=np.arange(1,120*10,10),
                    sample_range=(15,30),duration=30,mean=128,amp=64,
                    n_boot=1e3,statistic=np.median,
                    measure_compare=None,ind_measure=[],
                    pop_measure=[],baseline=128,
                    conf_interval=95, stat_testing=True,correlation=False):
    '''
    Haphazard version to get single plot for my defense. Promise to make it better -SB 101922
    Making it better -SB 120222
    '''
    fig,ax = plt.subplots(ncols=1,figsize=(6,6))
    from tools.measurements import cross_correlate_auto
    name = 'data/LDS_response_Noise.pickle'
    if duration>60:
        name = 'data/LDS_response_Noise_LONG.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    # to_plot = ['101422WT_30m_128Mean_64Amp_100msRefresh']
    # to_plot = ['102122WT_60m_128Mean_64Amp_100msRefresh']
    # print(result.keys())
    print(interest_list)
    for i,interest in enumerate(interest_list):
        #pull the data
        to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{mean}Mean_{amp}Amp_100msRefresh'],exclude)
        print(f'{interest}_30m_128Mean_64Amp_100msRefresh',to_plot)# print(interest,to_plot)
        if len(to_plot)==0:
            continue
        xp = result['tau']
        yp = np.concatenate([result[dat] for dat in to_plot])
        ind_t = np.where((xp>sample_range[0])&(xp<sample_range[1]))[0]
        C = np.array([cross_correlate_auto(yy,tau) for yy in yp[:,ind_t]])
        if correlation:
            C=C/C[:,0][:,None]
        def med(x):
            return np.mean(x,axis=0)
        y,rng,dist=bootstrap(C,statistic=med,conf_interval=conf_interval,n_boot=n_boot,
            return_samples=True,)
        # y,rng,dist=bootstrap(yp[:,ind_t],statistic=cross_correlate_auto,conf_interval=conf_interval,n_boot=n_boot,
        #     return_samples=True,tau=tau)
        if interest=='WT' and len(interest_list)>1:
            c='grey'
        else:
            c = plt.cm.Set1(i/9)
        ax.plot(tau/120,y,c=c,label=interest)
        ax.fill_between(tau/120,*rng,facecolor=c,alpha=.2)
        ax.plot(tau/120,tau*0,c='grey',zorder=-10)
        ax.set_xlabel('$\\tau$')
        ax.set_ylabel('Autocovariance')
    plt.legend()
    return fig,ax


def autocovariance_regen(interest_list=['WT'],exclude=[],tau=np.arange(1,120*10,10),
                    sample_range=(15,30),duration=30,mean=128,amp=64,
                    n_boot=1e3,statistic=np.median,
                    measure_compare=None,ind_measure=[],
                    pop_measure=[],baseline=128,
                    conf_interval=95, stat_testing=True,correlation=False,
                    whole_worm_ref=True):
    '''
    Same as above but for regenerating data
    '''
    if whole_worm_ref:
        fig, ax = autocovariance(interest_list,exclude,tau,
                            sample_range,duration,mean,amp,
                            n_boot,statistic,
                            measure_compare,ind_measure,
                            pop_measure,baseline,
                            conf_interval, stat_testing,correlation)
    else:
        fig,ax = plt.subplots(ncols=1,figsize=(6,6))

    from tools.measurements import cross_correlate_auto
    name = 'data/LDS_response_Noise_regen.pickle'
    # if duration>60:
    #     name = 'data/LDS_response_Noise_LONG.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    print(interest_list)
    for i,interest in enumerate(interest_list):
        #pull the data
        to_plot = data_of_interest(result.keys(),[f'{interest}_regen_{duration}m_{mean}Mean_{amp}Amp_100msRefresh'],exclude=exclude)
        print(f'{interest}_30m_128Mean_64Amp_100msRefresh',to_plot)# print(interest,to_plot)
        if len(to_plot)==0:
            continue
        xp = result['tau']

        from tqdm import tqdm
        for day in tqdm(np.arange(7)): #ToDO: make what days you plot more flexible
            yp = np.concatenate([result[dat][day] for dat in to_plot])
            ind_t = np.where((xp>sample_range[0])&(xp<sample_range[1]))[0]
            C = np.array([cross_correlate_auto(yy,tau) for yy in yp[:,ind_t]])
            if correlation:
                C=C/C[:,0][:,None]
            def med(x):
                return np.mean(x,axis=0)
            y,rng,dist=bootstrap(C,statistic=med,conf_interval=conf_interval,n_boot=n_boot,
                return_samples=True,)


            c = plt.cm.cool(day/7) #change if flexible days
            ax.plot(tau/120,y,c=c,label=f'{interest} {day}dpa')
            ax.fill_between(tau/120,*rng,facecolor=c,alpha=.2)
            ax.plot(tau/120,tau*0,c='grey',zorder=-10)
        ax.set_xlabel('$\\tau$')
        ax.set_ylabel('Autocovariance')
    plt.legend()
    return fig,ax

def autocovariance_step(interest_list,exclude=[],powers =[64,],
                    tau=np.arange(1,120*10,10),
                    sample_range=(15,30),
                    n_boot=1e3,statistic=np.median,
                    measure_compare=None,ind_measure=[],
                    pop_measure=[],conf_interval=95, stat_testing=True,):
    '''
    Haphazard version to get single plot for my defense. Promise to make it better -SB 101922
    #adapted to look at changes in a step function
    '''
    #loop the sin periods
    fig,ax = plt.subplots(ncols=1,figsize=(6,6))
    from tools.measurements import cross_correlate_auto
    name = 'data/LDS_response_LONG.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)

    for i,interest in enumerate(interest_list):
        print(interest)
        c=plt.cm.Set1(i/9)
        for num,power in enumerate(powers):
            exclude_this = exclude.copy()
            if not '+' in interest:
                exclude_this.append('+')
            to_plot = data_of_interest(result.keys(),[f'{interest}_30m2h{power}bp'],exclude_this)
            print(f'{interest}_30m2h{power}bp',to_plot)
            if len(to_plot)==0: continue
            yp=[]
            for dat in to_plot:
                yp.extend(result[dat])
            yp = np.array(yp)
            print(to_plot)
            xp = result['tau']
            ind_t = np.where((xp>sample_range[0])&(xp<sample_range[1]))[0]
            C = np.array([cross_correlate_auto(yy,tau) for yy in yp[:,ind_t]])
            def med(x):
                return np.mean(x,axis=0)
            y,rng,dist=bootstrap(C,statistic=med,conf_interval=conf_interval,n_boot=n_boot,
                return_samples=True,)
            # y,rng,dist=bootstrap(yp[:,ind_t],statistic=cross_correlate_auto,conf_interval=conf_interval,n_boot=n_boot,
            #     return_samples=True,tau=tau)
            # c='cornflowerblue'
            ax.plot(tau/120,y,c=c,label=interest)
            ax.fill_between(tau/120,*rng,facecolor=c,alpha=.2)
            ax.plot(tau/120,tau*0,c='grey',zorder=-10)
    ax.set_xlabel('$\\tau$')
    ax.set_ylabel('Autocovariance')
    ax.legend()
    return fig,ax
