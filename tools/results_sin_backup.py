##This backup was created June 01. 2023, the code should plot all sine waves and individual trials

import numpy as np
import matplotlib.pyplot as plt
import pickle
from .bootstrapTest import bootstrap_traces, bootstrapTest, bootstrap, timeDependentDifference
from .bootstrapTest import bootstrap_diff, bootstrap_relative

def data_of_interest(names,interest=[],exclude=[]):
    to_plot = []
    full_amp = True
    for i in interest:
        if 'amp' in i and (not '127amp' in i):
            print('hi')
            full_amp=False
    if full_amp:
        exclude.append('amp')
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

def calc_delta(x,w=2):
    if w%2==0:
        w+=1
    filt = np.ones(w)
    filt[:w//2]=-1
    filt[w//2]=0
    return np.convolve(x,filt,mode='same')/w

def response_sin(interest_list,exclude,periods =[.5,1,2,3,4,5,10],duration=30,
                          n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],baseline=128,
                          conf_interval=95, stat_testing=True, derivative=False,
                          visible=False,t_samp=(-3,40),plot_indiv_trial=False,):  # trial_range=None


    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey'):
      ax.scatter([loc,],12,marker='*',color=c)
    #give extra n_boot to measurements
    n_boot_meas = max(n_boot, 3e2)

    fig, ax_all=plt.subplots(nrows=len(periods), ncols=1, sharex=True, sharey=True,figsize=(10,2*len(periods)))


    if len(periods)==1: ax_all = np.array([ax_all])#ax_all[None,:]
    if True: ax_all=ax_all[:,None]
    ax = ax_all

    for ii,interest in enumerate(interest_list):
        if ii==0:
            #step function reference
            #for trial in trial_subset: #so here i want take only the relevant trials then loop through
            name = 'data/LDS_response_LONG.pickle'
            with open(name,'rb') as f:
                result = pickle.load(f)
            # print(result['tau'])
            xp = result['tau']
            ind_t = np.where((xp>=t_samp[0])&(xp<=t_samp[1]))[0] #extracting a subset of data that falls within the specified time range
            print("keys",result.keys())
            to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m2h{128}bp'],exclude=exclude)
            print("to_plot",to_plot)
            if len(to_plot)==0:
                yp_ref=np.zeros((1,xp.size))
            else:
                yp_ref = np.concatenate([result[dat] for dat in to_plot])
                loc = np.argmin(xp**2)
                #for trial in trial_subset: #loop through the subset
                y,rng = bootstrap_traces(yp_ref[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
                for a in ax[:,0]:
                    c='grey'
                    a.plot(xp[ind_t],y,lw=1,color=c,zorder=-2,)
                    print("plot", len(xp[ind_t]), len(y))
                    a.fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
            ind_t_ref = ind_t.copy()
            xp_ref=xp.copy()
        else:
            continue

        #loop the sin periods
        name = 'data/LDS_response_sinFunc.pickle'
        if plot_indiv_trial:
            name = 'data/LDS_response_sinFunc_individual.pickle'
        if visible:
            print('visible') # no print
            name = 'data/LDS_response_Vis_sinFunc.pickle'
        with open(name,'rb') as f:
            result = pickle.load(f)
            #print(result.keys())

        for i,p in enumerate(periods):
            xp = result['tau']
            ind_t = np.where((xp>=t_samp[0]-1e-5)&(xp<=t_samp[1]))[0] #SB 03.01.23, added the -1e-5 due to weird float precision issue
         # Start with minutes
            p_name = f'{p}m'
            to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],exclude=exclude)

            # If no results, try seconds
            if len(to_plot) == 0 and (p<1 or p%1>0):
                p_name = f'{int(p*60)}s'
                to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],exclude=exclude)

            if len(to_plot) == 0:
                continue


            # p_name = f'{p}m'
            # if p<1 or p%1>0:
            #     p_name = f'{int(p*60)}s'
            #
            # print(f'{interest}_{duration}m_{p_name}')
            # to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],exclude=exclude)
            # if len(to_plot)==0:
            #     continue
            #xp = result['tau']

            print(len(xp), to_plot)
            loc = np.argmin(xp**2)
            #return 0 #result[to_plot[0]]
            if plot_indiv_trial:
                #concatenate sets of worms from diff. experiments
                yp = []
                for dat in to_plot:
                    yp.extend(result[dat]['data'])

                trial_plot = plot_indiv_trial
                for trial in trial_plot:
                    # make list of all worms from a given trial
                    yp_trial = []
                    for y in yp:
                        try:
                            yp_trial.append(y[trial])
                        except:
                            continue
                    yp_trial = np.array(yp_trial)
                    print(yp_trial.shape)
                    # calculate and plot trial results
                    y, rng = bootstrap_traces(yp_trial[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
                    c = plt.cm.plasma(trial/np.max(trial_plot))
                    ax[i,0].plot(xp[ind_t],y,lw=1,color=c,zorder=-1,label=f'{interest} {p}m, trial:{trial}, ({yp_trial.shape[0]})',)
                    ax[i,0].fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
                    # continue

            else:
                yp = np.concatenate([result[dat]['data'] for dat in to_plot])
                y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
            # xp = np.arange(np.linspace(-30, 90, len(yp)))
            # if derivative:
            #     print(yp[0].shape)
            #     yp = np.array([calc_delta(yy,derivative) for yy in yp])
            #     print(yp.shape)


            # if plot_individual_worm:
            #     for iii,yp_i in enumerate(yp):
            #         c=plt.cm.Set1(iii/9)
            #         y,rng = bootstrap_traces(yp_i[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
            #         ax[i,0].plot(xp[ind_t],y,lw=1,color=c,zorder=-1,label=f'{interest} {p}m, ({yp.shape[0]})',)
            #         ax[i,0].fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
            #     ax[i,0].plot(xp,result[to_plot[0]]['stim'],c='thistle',zorder=-10)
            #
            #     print(yp)
            #
            #     num_samples = xp.size
            #     print(f"Number of samples in xp: {num_samples}")


                if derivative:
                    True #not interesting results :(
                        # y = np.array([calc_delta(yy,derivative) for yy in y])
                        # rng = np.percentile
                        # def delta_meas(x,axis=0):
                        #     xx = np.median(x,axis=0)
                        #     return calc_delta(xx,derivative)
                        # y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=delta_meas,conf_interval=conf_interval)

                    c=plt.cm.Set1(ii/9)#'cornflowerblue'
                # if plot_indiv_trial:
                #     print(len(yp))
                #     for yp_i in yp:
                #         ax[i,0].plot(xp[ind_t],yp_i[ind_t],alpha=.05,c=c)
                        # print(result[to_plot[0]])
                ax[i,0].plot(xp[ind_t],y,lw=1,color=c,zorder=-1,label=f'{interest} {p}m, ({yp.shape[0]})',)
                ax[i,0].fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
                ax[i,0].plot(xp,result[to_plot[0]]['stim'],c='thistle',zorder=-10)
                time_sig = timeDependentDifference(yp[:,ind_t],yp_ref[:,ind_t_ref],n_boot=n_boot,conf_interval=conf_interval)
                x_sig = xp[ind_t]#np.arange(0,10,1/120)
                y_sig = .1
                    #                         if len(powers)-1>1:
                    #                             y_sig=2*y_sig/len(interest_list)
                j=0
                bott = np.zeros_like(x_sig)+1.5 - (j-1)*y_sig
                ax[i,0].fill_between(x_sig,bott,bott-y_sig*time_sig,
                    facecolor=c,alpha=.4)
                box_keys={'lw':1, 'c':'k'}
                ax[i,0].plot(x_sig,bott,**box_keys)
                ax[i,0].plot(x_sig,bott-y_sig,**box_keys)
            ax[i,0].legend()
                # ax[i,0].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
                # ax[i,0].plot([ind_sig[1],ind_sig[1]],[bott[0],bott[0]-y_sig],**box_keys)
                # ax[i,0].set_title(f'{dur} min')
        ax[i,0].set_xlim(-3,40)
        ax[i,0].set_ylim(0,1.6)

    ax[-1,0].set_xlabel('time (min)')
    return fig,ax

def response_sin_amp(interest_list,exclude,periods =[.5,1,2,3,4,5,10],duration=30,
                          n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],baseline=128,
                          conf_interval=95, stat_testing=True,t_samp=(20,30)):

    ''' shows response amplitude and phase relative to single sinusoidal driver'''
    #loop the sin periods
    if type(interest_list) is str:
        interest_list = [interest_list,]
    fig,ax = plt.subplots(ncols=3,figsize=(18,6))

    from tools.measurements import cross_correlate
    name = 'data/LDS_response_sinFunc.pickle'
    with open(name,'rb') as f:
      result = pickle.load(f)
    for ii,interest in enumerate(interest_list):
        c = plt.cm.Set1(ii/9)
        if interest is 'WT':
            c='grey'
        labeled = False #whether we've but a legend label on this gene yet
        for i,p in enumerate(periods):
            # if len(interest_list)==1:
            #     c = plt.cm.magma(i/len(periods))
            xp = result['tau']
            ind_t = np.where((xp>=t_samp[0])&(xp<=t_samp[1]))[0]
            p_name = f'{p}m'
            if p<1 or p%1>0:
              p_name = f'{int(p*60)}s'
            print(f'{interest}_{duration}m_{p_name}')
            to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],exclude=exclude)
            if len(to_plot)==0:
                continue
            print(to_plot)
            xp = result['tau']
            yp = np.concatenate([result[dat]['data'] for dat in to_plot])
            tau = np.arange(1,int(p*120))
            # print(yp.shape,result[to_plot[0]]['stim'][ind_t].shape)
            y,rng,dist=bootstrap(yp[:,ind_t],statistic=cross_correlate,conf_interval=conf_interval,n_boot=n_boot,
                return_samples=True,U=result[to_plot[0]]['stim'][ind_t],tau=tau)
            print(np.array(y).shape,np.array(dist).shape)
            xx=np.linspace(0,2,np.array(y).size)
            ax[0].plot(xx,y,c=c)
            ax[0].fill_between(xx,*rng,facecolor=c,alpha=.2)

            #amplitudes
            amp = np.nanmax(dist,axis=1)
            ax[1].scatter(p,amp.mean(),color=c,label=interest)
            v = ax[1].violinplot([amp],positions=[p],vert=True,widths=[.5],showextrema=False,)
            for pc in v['bodies']:
                pc.set_facecolor(c)
                pc.set_alpha(.2)
            #phases
            amp = xx[np.nanargmax(dist,axis=1)]
            #rotate if near phi=0 boundary
            # if amp.mean()<.5 or amp.mean()>1.5:
            #     ind_rot = np.where(amp>1)
            #     amp[ind_rot] = amp[ind_rot]-2
            amp_alt = amp.copy()
            ind_rot = np.where(amp>1)
            amp_alt[ind_rot] = amp[ind_rot]-2
            if np.var(amp_alt)<np.var(amp):
                amp=amp_alt.copy()
            label=None
            if not labeled:
                label=interest
                labeled=True
            ax[2].scatter(p,amp.mean(),color=c,label=label)
            v = ax[2].violinplot([amp],positions=[p],vert=True,widths=[.5],showextrema=False)
            for pc in v['bodies']:
                pc.set_facecolor(c)
                pc.set_alpha(.2)

    ax[0].set_xlabel('phase angle ($\pi$ rad)')
    ax[0].set_ylabel('signal-stimulus covariance')
    ax[1].set_xlabel('period (min)')
    ax[1].set_ylabel('amplitude')
    ax[2].set_xlabel('period (min)')
    ax[2].set_ylabel('phase shift ($\pi$ rad)')
    ax[2].legend()
    return fig, ax

def response_sin_amp_individual(interest_list,exclude,periods =[.5,1,2,3,4,5,10],duration=30,
                          n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],baseline=128,
                          conf_interval=95, stat_testing=True,t_samp=(20,30),fig=None,ax=None,position_offset=0,
                          phase_calculation='bootstrap_mean'):

    ''' shows response amplitude and phase relative to single sinusoidal driver
    KEY DIFFERENCE: covariance calculated by individual trace.'''
    #loop the sin periods
    if type(interest_list) is str:
        interest_list = [interest_list,]
    if fig is None:
        fig,ax = plt.subplots(ncols=3,figsize=(18,6))
        default_color='grey'
    else:
        default_color='firebrick'

    from tools.measurements import cross_correlate
    name = 'data/LDS_response_sinFunc.pickle'
    with open(name,'rb') as f:
      result = pickle.load(f)
    for ii,interest in enumerate(interest_list):
        c = plt.cm.Set1(ii/9)
        if interest is 'WT':
            c=default_color#'grey'
        labeled = False #whether we've but a legend label on this gene yet
        for i,p in enumerate(periods):
            # if len(interest_list)==1:
            #     c = plt.cm.magma(i/len(periods))
            xp = result['tau']
            ind_t = np.where((xp>=t_samp[0])&(xp<=t_samp[1]))[0]
            p_name = f'{p}m'
            if p<1 or p%1>0:
              p_name = f'{int(p*60)}s'
            print(f'{interest}_{duration}m_{p_name}')
            to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],exclude=exclude)
            if len(to_plot)==0:
                continue
            print(to_plot)
            xp = result['tau']
            yp = np.concatenate([result[dat]['data'] for dat in to_plot])
            tau = np.arange(1,int(p*120))
            # print(yp.shape,result[to_plot[0]]['stim'][ind_t].shape)
            dist = []
            for yp_i in yp:
                c_i = cross_correlate(yp_i[None,ind_t],U=result[to_plot[0]]['stim'][ind_t],tau=tau)
                dist.append(c_i)
            dist = np.array(dist)

            # y,rng,dist=bootstrap(yp[:,ind_t],statistic=cross_correlate,conf_interval=conf_interval,n_boot=n_boot,
            #     return_samples=True,U=result[to_plot[0]]['stim'][ind_t],tau=tau)
            # print(np.array(y).shape,np.array(dist).shape)
            xx=np.linspace(0,2,dist.shape[1])
            # ax[0].plot(xx,y,c=c)
            # ax[0].fill_between(xx,*rng,facecolor=c,alpha=.2)

            #amplitudes
            amp = np.nanmax(dist,axis=1)

            amp,rng_amp,dist_amp = bootstrap(amp,statistic=np.mean,conf_interval=conf_interval,n_boot=n_boot,
                return_samples=True,)

            ax[1].scatter(p+position_offset,amp,color=c,label=interest)
            v = ax[1].violinplot([dist_amp],positions=[p+position_offset],vert=True,widths=[.5],showextrema=False,)
            for pc in v['bodies']:
                pc.set_facecolor(c)
                pc.set_alpha(.2)
            #phases
            phase = xx[np.nanargmax(dist,axis=1)]
            #rotate if near phi=0 boundary
            # if amp.mean()<.5 or amp.mean()>1.5:
            #     ind_rot = np.where(amp>1)
            #     amp[ind_rot] = amp[ind_rot]-2
            phase_alt = phase.copy()
            ind_rot = np.where(phase>1)
            phase_alt[ind_rot] = phase[ind_rot]-2
            if np.var(phase_alt)<np.var(phase):
                phase=phase_alt.copy()
            label=None
            if not labeled:
                label=interest
                labeled=True
            if phase_calculation=='bootstrap_mean':
                # bootstrap distribution of individual trace phase
                phase,rng_phase,dist_phase = bootstrap(phase,statistic=np.mean,conf_interval=conf_interval,n_boot=n_boot,
                    return_samples=True,)
                ax[2].set_title('bootstrap mean phase ($\pi$ rad)')
            elif phase_calculation == 'values':
                # distribution of individual trace phase
                dist_phase = phase.copy()
                phase = np.mean(phase)
                ax[2].set_title('mean phase ($\pi$ rad)')
            #bootstrap dist of variance of phase
            elif phase_calculation == 'bootstrap_var':
                phase,rng_phase,dist_phase = bootstrap(phase,statistic=np.var,conf_interval=conf_interval,n_boot=n_boot,return_samples=True)
                ax[2].set_ylabel('bootstrap var phase ($\pi$ rad)')
            else:
                print("ERROR: options fopr phase_calculation are: ['bootstrap_mean','values','bootstrap_var']")
                return
            ax[2].scatter(p+position_offset,phase,color=c,label=label)
            v = ax[2].violinplot([dist_phase],positions=[p+position_offset],vert=True,widths=[.5],showextrema=False)
            for pc in v['bodies']:
                pc.set_facecolor(c)
                pc.set_alpha(.2)

    ax[0].set_xlabel('phase angle ($\pi$ rad)')
    ax[0].set_ylabel('signal-stimulus covariance')
    ax[1].set_xlabel('period (min)')
    ax[1].set_ylabel('amplitude')
    ax[2].set_xlabel('period (min)')
    # ax[2].set_ylabel('phase shift ($\pi$ rad)')
    ax[2].legend()
    return fig, ax

def response_sin_amp_vDriveamp(interest,exclude,periods =[2,3],drive_amp=[32,64,127],duration=30,
                          n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],baseline=128,
                          conf_interval=95, stat_testing=True,):

    '''compares response amplitude vs driving amplitude at each frequency'''
    #loop the sin periods
    fig,ax = plt.subplots(ncols=3,figsize=(18,6))

    from tools.measurements import cross_correlate
    name = 'data/LDS_response_sinFunc.pickle'
    with open(name,'rb') as f:
      result = pickle.load(f)

    for i,p in enumerate(periods):
        c = plt.cm.Set1(i/9)
        for j,amp_j in enumerate(drive_amp):

            xp = result['tau']
            ind_t = np.where((xp>=20)&(xp<=30))[0]
            p_name = f'{p}m'
            if p<1:
              p_name = f'{int(p*60)}s'
            # print(f'{interest}_{duration}m_{p_name}')
            if amp_j<127:
                to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}Period_amp{amp_j}bp'],)
            else:
                to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}Period'],['amp'])
            print(f'{interest}_{duration}m_{p_name}Period_amp{amp_j}bp',to_plot)
            if len(to_plot)==0: continue
            xp = result['tau']
            yp = np.concatenate([result[dat]['data'] for dat in to_plot])
            tau = np.arange(1,int(p*120))
            # print(yp.shape,result[to_plot[0]]['stim'][ind_t].shape)
            u=result[to_plot[0]]['stim'][ind_t]
            u -= u.min()#normalize the oscillations to the same range
            u /= u.max()
            y,rng,dist=bootstrap(yp[:,ind_t],statistic=cross_correlate,conf_interval=conf_interval,n_boot=n_boot,
                return_samples=True,U=u,tau=tau)
            print(np.array(y).shape,np.array(dist).shape)
            xx=np.linspace(0,2,np.array(y).size)
            ax[0].plot(xx,y,c=c)
            ax[0].fill_between(xx,*rng,facecolor=c,alpha=.2)

            #amplitudes
            plot_x = j + i/len(drive_amp)/2
            plot_w = 1/len(drive_amp)/2
            amp = np.max(dist,axis=1)
            label=None
            if j==0:
                label=p_name
            ax[1].scatter(plot_x,np.nanmean(amp),color=c,label=label)
            v = ax[1].violinplot([amp],positions=[plot_x],vert=True,widths=[plot_w],showextrema=False)
            for pc in v['bodies']:
                pc.set_facecolor(c)
                pc.set_alpha(.2)
            #phases
            amp = xx[np.argmax(dist,axis=1)]
            ax[2].scatter(plot_x,amp.mean(),color=c,label=label)
            v = ax[2].violinplot([amp],positions=[plot_x],vert=True,widths=[plot_w],showextrema=False)
            for pc in v['bodies']:
                pc.set_facecolor(c)
                pc.set_alpha(.2)

    ax[0].set_xlabel('phase angle ($\pi$ rad)')
    ax[0].set_ylabel('signal-stimulus covariance')
    for a in ax[1:]:
        a.set_xlabel('drive amplitude')
        a.set_xticks(np.arange(len(drive_amp)))
        a.set_xticklabels(drive_amp)
        a.legend(loc=2)
    ax[1].set_ylabel('response amplitude')
    ax[2].set_ylabel('phase shift ($\pi$ rad)')
    return fig, ax


def response_sin_autocov(interest,exclude,periods =[.5,1,2,3,4,5,10],duration=30,
                          n_boot=1e3,tau=np.arange(1,120*10,10),statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],baseline=128,
                          conf_interval=95, stat_testing=True,):


    #loop the sin periods
    fig,ax = plt.subplots(ncols=1,figsize=(6,6))

    from tools.measurements import cross_correlate
    name = 'data/LDS_response_sinFunc.pickle'
    with open(name,'rb') as f:
      result = pickle.load(f)
    for i,p in enumerate(periods):
        c = plt.cm.magma(i/len(periods))
        xp = result['tau']
        ind_t = np.where((xp>=15)&(xp<=30))[0]
        p_name = f'{p}m'
        if p<1:
          p_name = f'{int(p*60)}s'
        print(f'{interest}_{duration}m_{p_name}')
        to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],)
        print(to_plot)
        xp = result['tau']
        yp = np.concatenate([result[dat]['data'] for dat in to_plot])
        # tau = np.arange(1,int(p*120))
        # print(yp.shape,result[to_plot[0]]['stim'][ind_t].shape)
        from tools.measurements import cross_correlate_auto
        C = np.array([cross_correlate_auto(yy,tau) for yy in yp[:,ind_t]])
        def med(x):
            return np.mean(x,axis=0)
        y,rng,dist=bootstrap(C,statistic=med,conf_interval=conf_interval,n_boot=n_boot,
            return_samples=True,)
        print(np.array(y).shape,np.array(dist).shape)
        xx=tau/120
        ax.plot(xx,y,c=c,label=p)
        ax.fill_between(xx,*rng,facecolor=c,alpha=.2)
    ax.legend()
    # ax[1].set_xlabel('period (min)')
    # ax[1].set_ylabel('amplitude')
    return fig, ax


def response_dualSin_amp(interest_list,exclude,pair_periods =[(3,5)],duration=30,
                          n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],baseline=128,
                          conf_interval=95, stat_testing=True,t_samp=(20,30)):
    """
    Analyzes the results of 2 superimposed UV sin waves
    """
    #loop the sin periods
    if type(interest_list) is str:
        interest_list = [interest_list,]
    fig,ax = plt.subplots(ncols=3,figsize=(18,6))

    from tools.measurements import cross_correlate
    name = 'data/LDS_response_sinFunc.pickle'
    with open(name,'rb') as f:
      result = pickle.load(f)
    for ii,interest in enumerate(interest_list):
        c = plt.cm.Set1(ii/9)
        if interest is 'WT':
            c='grey'

            for periods in pair_periods:
                #get double data
                xp = result['tau']
                ind_t = np.where((xp>=t_samp[0])&(xp<=t_samp[1]))[0]
                p = periods[0]
                p_name1 = f'{p}m'
                if p<1 or p%1>0:
                    p_name1 = f'{int(p*60)}s'
                p = periods[1]
                p_name2 = f'{p}m'
                if p<1 or p%1>0:
                    p_name2 = f'{int(p*60)}s'
                print(f'{interest}_{duration}m_{p_name1}{p_name2}')
                to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name1}{p_name2}'],)
                if len(to_plot)==0:
                    continue
                print(to_plot)
                yp_dual = np.concatenate([result[dat]['data'] for dat in to_plot])
                c_dual = 'cornflowerblue'
                #tests on each component period
                for i,p in enumerate(periods):
                    # if len(interest_list)==1:
                    #     c = plt.cm.magma(i/len(periods))
                    xp = result['tau']
                    ind_t = np.where((xp>=t_samp[0])&(xp<=t_samp[1]))[0]
                    p_name = f'{p}m'
                    if p<1 or p%1>0:
                      p_name = f'{int(p*60)}s'
                    print(f'{interest}_{duration}m_{p_name}Period_amp{128//len(periods)}bp')
                    to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}Period_{128//len(periods)}amp'],[])
                    if len(to_plot)==0:
                        continue
                    print(to_plot)
                    xp = result['tau']
                    yp = np.concatenate([result[dat]['data'] for dat in to_plot])
                    tau = np.arange(1,int(p*120))
                    # print(yp.shape,result[to_plot[0]]['stim'][ind_t].shape)
                    y,rng,dist=bootstrap(yp[:,ind_t],statistic=cross_correlate,conf_interval=conf_interval,n_boot=n_boot,
                        return_samples=True,U=result[to_plot[0]]['stim'][ind_t],tau=tau)
                    y_dual,rng_dual,dist_dual=bootstrap(yp_dual[:,ind_t],statistic=cross_correlate,conf_interval=conf_interval,n_boot=n_boot,
                        return_samples=True,U=result[to_plot[0]]['stim'][ind_t],tau=tau)
                    print(np.array(y).shape,np.array(dist).shape)
                    xx=np.linspace(0,2,np.array(y).size)
                    ax[0].plot(xx,y,c=c)
                    ax[0].fill_between(xx,*rng,facecolor=c,alpha=.2)

                    #amplitudes
                    amp = np.nanmax(dist,axis=1)
                    ax[1].scatter(p-.2,amp.mean(),color=c,label=interest+'single freq')
                    v = ax[1].violinplot([amp],positions=[p-.2],vert=True,widths=[.5],showextrema=False,)
                    for pc in v['bodies']:
                        pc.set_facecolor(c)
                        pc.set_alpha(.2)
                    amp_dual = np.nanmax(dist_dual,axis=1)
                    ax[1].scatter(p+.2,amp_dual.mean(),color=c_dual,label=interest+'dual freq')
                    v = ax[1].violinplot([amp_dual],positions=[p+.2],vert=True,widths=[.5],showextrema=False,)
                    for pc in v['bodies']:
                        pc.set_facecolor(c_dual)
                        pc.set_alpha(.2)

                    #phases
                    amp = xx[np.nanargmax(dist,axis=1)]
                    #rotate if near phi=0 boundary
                    # if amp.mean()<.5 or amp.mean()>1.5:
                    #     ind_rot = np.where(amp>1)
                    #     amp[ind_rot] = amp[ind_rot]-2
                    amp_alt = amp.copy()
                    ind_rot = np.where(amp>1)
                    amp_alt[ind_rot] = amp[ind_rot]-2
                    if np.var(amp_alt)<np.var(amp):
                        amp=amp_alt.copy()
                    label=None
                    if i==0:
                        label=interest
                    ax[2].scatter(p-.2,amp.mean(),color=c,label=label)
                    v = ax[2].violinplot([amp],positions=[p-.2],vert=True,widths=[.5],showextrema=False)
                    for pc in v['bodies']:
                        pc.set_facecolor(c)
                        pc.set_alpha(.2)

                    amp_dual = xx[np.nanargmax(dist_dual,axis=1)]
                    #rotate if near phi=0 boundary
                    # if amp.mean()<.5 or amp.mean()>1.5:
                    #     ind_rot = np.where(amp>1)
                    #     amp[ind_rot] = amp[ind_rot]-2
                    amp_alt_dual = amp_dual.copy()
                    ind_rot_dual = np.where(amp_dual>1)
                    amp_alt_dual[ind_rot_dual] = amp_dual[ind_rot_dual]-2
                    if np.var(amp_alt_dual)<np.var(amp_dual):
                        amp_dual=amp_alt_dual.copy()
                    label=None
                    if i==0:
                        label=interest
                    ax[2].scatter(p+.2,amp_dual.mean(),color=c_dual,label=label)
                    v = ax[2].violinplot([amp_dual],positions=[p+.2],vert=True,widths=[.5],showextrema=False)
                    for pc in v['bodies']:
                        pc.set_facecolor(c_dual)
                        pc.set_alpha(.2)



            # #the double one
            # xp = result['tau']
            # ind_t = np.where((xp>=t_samp[0])&(xp<=t_samp[1]))[0]
            # p = periods[0]
            # p_name1 = f'{p}m'
            # if p<1 or p%1>0:
            #     p_name1 = f'{int(p*60)}s'
            # p = periods[1]
            # p_name2 = f'{p}m'
            # if p<1 or p%1>0:
            #     p_name2 = f'{int(p*60)}s'
            #
            # print(f'{interest}_{duration}m_{p_name}')
            # to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name1}{p_name2}'],)
            # if len(to_plot)==0:
            #     continue
            # print(to_plot)
            # xp = result['tau']
            # yp = np.concatenate([result[dat]['data'] for dat in to_plot])
            # tau = np.arange(1,int(p*120))
            # # print(yp.shape,result[to_plot[0]]['stim'][ind_t].shape)
            # y,rng,dist=bootstrap(yp[:,ind_t],statistic=cross_correlate,conf_interval=conf_interval,n_boot=n_boot,
            #     return_samples=True,U=result[to_plot[0]]['stim'][ind_t],tau=tau)
            # print(np.array(y).shape,np.array(dist).shape)
            # xx=np.linspace(0,2,np.array(y).size)
            # ax[0].plot(xx,y,c=c)
            # ax[0].fill_between(xx,*rng,facecolor=c,alpha=.2)
            #
            # #amplitudes
            # amp = np.nanmax(dist,axis=1)
            # ax[1].scatter(p,amp.mean(),color=c,label=interest)
            # v = ax[1].violinplot([amp],positions=[p],vert=True,widths=[.5],showextrema=False,)
            # for pc in v['bodies']:
            #     pc.set_facecolor(c)
            #     pc.set_alpha(.2)
            # #phases
            # amp = xx[np.nanargmax(dist,axis=1)]
            # #rotate if near phi=0 boundary
            # # if amp.mean()<.5 or amp.mean()>1.5:
            # #     ind_rot = np.where(amp>1)
            # #     amp[ind_rot] = amp[ind_rot]-2
            # amp_alt = amp.copy()
            # ind_rot = np.where(amp>1)
            # amp_alt[ind_rot] = amp[ind_rot]-2
            # if np.var(amp_alt)<np.var(amp):
            #     amp=amp_alt.copy()
            # label=None
            # if i==0:
            #     label=interest
            # ax[2].scatter(p,amp.mean(),color=c,label=label)
            # v = ax[2].violinplot([amp],positions=[p],vert=True,widths=[.5],showextrema=False)
            # for pc in v['bodies']:
            #     pc.set_facecolor(c)
            #     pc.set_alpha(.2)

    ax[0].set_xlabel('phase angle ($\pi$ rad)')
    ax[0].set_ylabel('signal-stimulus covariance')
    ax[1].set_xlabel('period (min)')
    ax[1].set_ylabel('amplitude')
    ax[2].set_xlabel('period (min)')
    ax[2].set_ylabel('phase shift ($\pi$ rad)')
    ax[2].legend()
    return fig, ax

def response_sin_amp_visUV(interest,exclude=[],phase_shift=[0,1],period=(3,3),
                            duration=30,amplitude=(127,127),n_boot=1e3,statistic=np.median,
                            baseline=(128,128),measure_compare=None,ind_measure=[],pop_measure=[],
                            conf_interval=95,stat_testing=True,t_samp=(20,30)):
    ''' shows response amplitude and phase relative to single sinusoidal driver'''
    fig,ax = plt.subplots(ncols=3,figsize=(18,6))

    from tools.measurements import cross_correlate
    name = 'data/LDS_response_Vis_sinFunc.pickle'
    with open(name,'rb') as f:
      result = pickle.load(f)


    for i,phase in enumerate(phase_shift):
        c = 'cornflowerblue'
        xp = result['tau']
        ind_t = np.where((xp>=t_samp[0])&(xp<=t_samp[1]))[0]

        phase_name = str(phase)
        if np.abs(phase)<1 and (not phase==0) :
            if phase>0:
                phase_name = phase_name[1:]
            else:
                phase_name = '-'+phase_name[2:]
        dat_name = f'{interest}_{duration}m'
        q = 0
        dat_name = f'{dat_name}_UV_{period[q]}mPeriod_{amplitude[q]}Amp_{baseline[q]}Mean'
        q=1
        dat_name = f'{dat_name}_Vis_{period[q]}mPeriod_{amplitude[q]}Amp_{baseline[q]}Mean_{phase_name}piPhase'
        to_plot = data_of_interest(result.keys(),[dat_name],exclude=exclude)
        print(dat_name,to_plot)
        if len(to_plot)==0:
            continue

        xp = result['tau']
        yp = np.concatenate([result[dat]['data'] for dat in to_plot])
        tau = np.arange(1,int(period[0]*120))
        print(yp.shape,)
        y,rng,dist=bootstrap(yp[:,ind_t],statistic=cross_correlate,conf_interval=conf_interval,n_boot=n_boot,
            return_samples=True,U=result[to_plot[0]]['stim'][0,ind_t],tau=tau)
        print(np.array(y).shape,np.array(dist).shape)
        xx=np.linspace(0,2,np.array(y).size)
        ax[0].plot(xx,y,c=c)
        ax[0].fill_between(xx,*rng,facecolor=c,alpha=.2)

        #amplitudes
        amp = np.nanmax(dist,axis=1)
        ax[1].scatter(phase,amp.mean(),color=c,label=interest)
        v = ax[1].violinplot([amp],positions=[phase],vert=True,widths=[.5],showextrema=False,)
        for pc in v['bodies']:
            pc.set_facecolor(c)
            pc.set_alpha(.2)
        #phases
        amp = xx[np.nanargmax(dist,axis=1)]
        #rotate if near phi=0 boundary
        # if amp.mean()<.5 or amp.mean()>1.5:
        #     ind_rot = np.where(amp>1)
        #     amp[ind_rot] = amp[ind_rot]-2
        amp_alt = amp.copy()
        ind_rot = np.where(amp>1)
        amp_alt[ind_rot] = amp[ind_rot]-2
        if np.var(amp_alt)<np.var(amp):
            amp=amp_alt.copy()
        label=None
        # if not labeled:
        #     label=interest
        #     labeled=True
        ax[2].scatter(phase,amp.mean(),color=c,label=label)
        v = ax[2].violinplot([amp],positions=[phase],vert=True,widths=[.5],showextrema=False)
        for pc in v['bodies']:
            pc.set_facecolor(c)
            pc.set_alpha(.2)

    ax[0].set_xlabel('phase angle ($\pi$ rad)')
    ax[0].set_ylabel('signal-stimulus covariance')
    ax[1].set_xlabel('period (min)')
    ax[1].set_ylabel('amplitude')
    ax[2].set_xlabel('period (min)')
    ax[2].set_ylabel('phase shift ($\pi$ rad)')
    ax[2].legend()
    return fig, ax


def response_sin_visUV(interest,exclude=[],phase_shift=[0,],period=(0,3),
                            duration=30,amplitude=(0,127),n_boot=1e3,statistic=np.median,
                            baseline=(128,128),measure_compare=None,ind_measure=[],pop_measure=[],
                            conf_interval=95,stat_testing=True,t_samp=(-10,40)):
    ''' shows response curves to combined uv vis input'''
    '''Note:
    order for amp+the other variables: (uv,vis)
    for constant UV use period 0, amplitude 0 (in the uv slot)'''
    fig,ax = plt.subplots(ncols=1,figsize=(18,6))

    from tools.measurements import cross_correlate
    name = 'data/LDS_response_Vis_sinFunc.pickle'
    with open(name,'rb') as f:
      result = pickle.load(f)
    print(result.keys())


    for i,phase in enumerate(phase_shift):


        #step function reference
        name = 'data/LDS_response_LONG.pickle'
        with open(name,'rb') as f:
            result = pickle.load(f)
        xp = result['tau']
        ind_t = np.where((xp>=t_samp[0])&(xp<=t_samp[1]))[0]
        to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m2h{baseline[0]}bp'],exclude=exclude)
        if len(to_plot)==0:
            yp_ref=np.zeros((1,xp.size))
        else:
            yp_ref = np.concatenate([result[dat] for dat in to_plot])
            loc = np.argmin(xp**2)
            y,rng = bootstrap_traces(yp_ref[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
            # for a in ax[:,0]:
            c='grey'
            ax.plot(xp[ind_t],y,lw=1,color=c,zorder=-2,)
            ax.fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
        ind_t_ref = ind_t.copy()


        #analyze sin's
        name = 'data/LDS_response_Vis_sinFunc.pickle'
        with open(name,'rb') as f:
          result = pickle.load(f)
        print(result.keys())
        c = 'cornflowerblue'
        xp = result['tau']
        ind_t = np.where((xp>=t_samp[0])&(xp<=t_samp[1]))[0]

        phase_name = str(phase)
        if np.abs(phase)<1 and (not phase==0) :
            if phase>0:
                phase_name = phase_name[1:]
            else:
                phase_name = '-'+phase_name[2:]
        phase_name = '_'+phase_name+'piPhase'
        if period[0]==0:
            phase_name = ''
        dat_name = f'{interest}_{duration}m'
        q = 0
        dat_name = f'{dat_name}_UV_{period[q]}mPeriod_{amplitude[q]}Amp_{baseline[q]}Mean'
        q=1
        dat_name = f'{dat_name}_Vis_{period[q]}mPeriod_{amplitude[q]}Amp_{baseline[q]}Mean{phase_name}'
        to_plot = data_of_interest(result.keys(),[dat_name],exclude=exclude)
        print(dat_name,to_plot)
        if len(to_plot)==0:
            continue

        xp = result['tau']
        yp = np.concatenate([result[dat]['data'] for dat in to_plot])
        tau = np.arange(1,int(period[0]*120))
        #plot the results_sin
        loc = np.argmin(xp**2)
        y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)

        c='cornflowerblue'
        plt.title(f'{dat_name}, ({yp.shape[0]})')
        ax.plot(xp[ind_t],y,lw=1,color=c,zorder=-1,)
        ax.fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
        # ax.plot(xp,result[to_plot[0]]['stim'],c='thistle',zorder=-10)
        time_sig = timeDependentDifference(yp[:,ind_t],yp_ref[:,ind_t_ref],n_boot=n_boot,conf_interval=conf_interval)
        x_sig = xp[ind_t]#np.arange(0,10,1/120)
        y_sig = .1
    #                         if len(powers)-1>1:
    #                             y_sig=2*y_sig/len(interest_list)
        j=0
        bott = np.zeros_like(x_sig)+1.5 - (j-1)*y_sig
        ax.fill_between(x_sig,bott,bott-y_sig*time_sig,
            facecolor=c,alpha=.4)
        box_keys={'lw':1, 'c':'k'}
        ax.plot(x_sig,bott,**box_keys)
        ax.plot(x_sig,bott-y_sig,**box_keys)
        # ax[i,0].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
        # ax[i,0].plot([ind_sig[1],ind_sig[1]],[bott[0],bott[0]-y_sig],**box_keys)
        # ax[i,0].set_title(f'{dur} min')
        ax.set_xlim(-3,40)
        ax.set_ylim(0,1.6)
        ax.legend()
        # plt.plot(xp,np.median(yp,axis=0))

        plt.plot(xp,result[to_plot[0]]['stim'][0],c='thistle')
        plt.plot(xp,result[to_plot[0]]['stim'][1],c='green')


    return fig, ax


def response_ind_traces(interest_list,exclude,periods =[.5,1,2,3,4,5,10],duration=30,
                          n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],baseline=128,
                          conf_interval=95, stat_testing=True,derivative=False,
                          visible=False):
    ''' LW: trying to show individual traces from the sine waves  '''
    '''Note:
    order for amp+the other variables: (uv,vis)
    for constant UV use period 0, amplitude 0 (in the uv slot)'''


    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey'):
      ax.scatter([loc,],12,marker='*',color=c)
    #give extra n_boot to measurements
    n_boot_meas = max(n_boot, 3e2)

    fig, ax_all=plt.subplots(nrows=len(periods), ncols=1, sharex=True, sharey=True,figsize=(10,2*len(periods)))
    if len(periods)==1: ax_all = ax_all[None,:]
    if True: ax_all=ax_all[:,None]
    ax = ax_all

    for ii,interest in enumerate(interest_list):
        if ii==0:
            #step function reference
            name = 'data/LDS_response_LONG.pickle'
            with open(name,'rb') as f:
                result = pickle.load(f)
            xp = result['tau']
            ind_t = np.where((xp>=-3)&(xp<=40))[0]
            to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m2h{128}bp'],exclude=exclude)
            if len(to_plot)==0:
                yp_ref=np.zeros((1,xp.size))
            else:
                yp_ref = np.concatenate([result[dat] for dat in to_plot])
                loc = np.argmin(xp**2)
                y,rng = bootstrap_traces(yp_ref[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
                for a in ax[:,0]:
                    c='grey'
                    a.plot(xp[ind_t],y,lw=1,color=c,zorder=-2,)
                    a.fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
            ind_t_ref = ind_t.copy()

        #loop the sin periods
        name = 'data/LDS_response_sinFunc.pickle'
        if visible:
            name = 'data/LDS_response_Vis_sinFunc.pickle'
        with open(name,'rb') as f:
            result = pickle.load(f)
        # print(result.keys())
        for i,p in enumerate(periods):
            xp = result['tau']
            ind_t = np.where((xp>=-3)&(xp<=40))[0]
            p_name = f'{p}m'
            if p<1 or p%1>0:
                p_name = f'{int(p*60)}s'

            print(f'{interest}_{duration}m_{p_name}')
            to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],exclude=exclude)
            if len(to_plot)==0:
                continue
            print(to_plot)
            xp = result['tau']
            yp = np.concatenate([result[dat]['data'] for dat in to_plot])
            # if derivative:
            #     print(yp[0].shape)
            #     yp = np.array([calc_delta(yy,derivative) for yy in yp])
            #     print(yp.shape)

            loc = np.argmin(xp**2)
            y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
            if derivative:
                True #not interesting results :(
                # y = np.array([calc_delta(yy,derivative) for yy in y])
                # rng = np.percentile
                # def delta_meas(x,axis=0):
                #     xx = np.median(x,axis=0)
                #     return calc_delta(xx,derivative)
                # y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=delta_meas,conf_interval=conf_interval)

            c=plt.cm.Set1(ii/9)#'cornflowerblue'
            ax[i,0].plot(xp[ind_t],y,lw=1,color=c,zorder=-1,label=f'{interest} {p}m, ({yp.shape[0]})',)
            ax[i,0].fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
            ax[i,0].plot(xp,result[to_plot[0]]['stim'],c='thistle',zorder=-10)
            time_sig = timeDependentDifference(yp[:,ind_t],yp_ref[:,ind_t_ref],n_boot=n_boot,conf_interval=conf_interval)
            x_sig = xp[ind_t]#np.arange(0,10,1/120)
            y_sig = .1
        #                         if len(powers)-1>1:
        #                             y_sig=2*y_sig/len(interest_list)
            j=0
            bott = np.zeros_like(x_sig)+1.5 - (j-1)*y_sig
            ax[i,0].fill_between(x_sig,bott,bott-y_sig*time_sig,
                facecolor=c,alpha=.4)
            box_keys={'lw':1, 'c':'k'}
            ax[i,0].plot(x_sig,bott,**box_keys)
            ax[i,0].plot(x_sig,bott-y_sig,**box_keys)
            # ax[i,0].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
            # ax[i,0].plot([ind_sig[1],ind_sig[1]],[bott[0],bott[0]-y_sig],**box_keys)
            # ax[i,0].set_title(f'{dur} min')
            ax[i,0].set_xlim(-3,40)
            ax[i,0].set_ylim(0,1.6)
            ax[i,0].legend()
    ax[-1,0].set_xlabel('time (min)')
    return fig,ax



# def response_dualSin(interest_list,exclude,pair_periods =[(3,5)],duration=30,
#                           n_boot=1e3,statistic=np.median,
#                           measure_compare=None,ind_measure=[],
#                           pop_measure=[],baseline=128,
#                           conf_interval=95, stat_testing=True,derivative=False,
#                           visible=False):
#
#     #keys for measure_compare
#     DIFFERENCE = ['diff','difference','subtract']
#     RELATIVE = ['relative','rel','divide']
#     #significance marker
#     def mark_sig(ax,loc,c='grey'):
#       ax.scatter([loc,],12,marker='*',color=c)
#     #give extra n_boot to measurements
#     n_boot_meas = max(n_boot, 3e2)
#
#     fig, ax_all=plt.subplots(nrows=len(periods), ncols=1, sharex=True, sharey=True,figsize=(10,2*len(periods)))
#     if len(periods)==1: ax_all = ax_all[None,:]
#     if True: ax_all=ax_all[:,None]
#     ax = ax_all
#
#     for ii,interest in enumerate(interest_list):
#         if ii==0:
#             #step function reference
#             name = 'data/LDS_response_LONG.pickle'
#             with open(name,'rb') as f:
#                 result = pickle.load(f)
#             xp = result['tau']
#             ind_t = np.where((xp>=-3)&(xp<=40))[0]
#             to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m2h{baseline}bp'],)
#             if len(to_plot)==0:
#                 yp_ref=np.zeros((1,xp.size))
#             else:
#                 yp_ref = np.concatenate([result[dat] for dat in to_plot])
#                 loc = np.argmin(xp**2)
#                 y,rng = bootstrap_traces(yp_ref[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
#                 for a in ax[:,0]:
#                     c='grey'
#                     a.plot(xp[ind_t],y,lw=1,color=c,zorder=-2,)
#                     a.fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
#             ind_t_ref = ind_t.copy()
#
#         #loop the sin periods
#         name = 'data/LDS_response_sinFunc.pickle'
#         if visible:
#             name = 'data/LDS_response_Vis_sinFunc.pickle'
#         with open(name,'rb') as f:
#             result = pickle.load(f)
#         # print(result.keys())
#
#
#         for ii,periods in enumerate(pair_periods):
#             xp = result['tau']
#             ind_t = np.where((xp>=-3)&(xp<=40))[0]
#             #single osacillation control
#             for i,p in enumerate(periods):
#
#                 p_name = f'{p}m'
#                 if p<1 or p%1>0:
#                     p_name = f'{int(p*60)}s'
#                 print(f'{interest}_{duration}m_{p_name}')
#                 to_plot = data_of_interest(result.keys(),[f'{interest}_{duration}m_{p_name}'],)
#                 if len(to_plot)==0:
#                     continue
#                 print(to_plot)
#                 xp = result['tau']
#                 yp = np.concatenate([result[dat]['data'] for dat in to_plot])
#                 # if derivative:
#                 #     print(yp[0].shape)
#                 #     yp = np.array([calc_delta(yy,derivative) for yy in yp])
#                 #     print(yp.shape)
#
#                 loc = np.argmin(xp**2)
#                 y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
#                 if derivative:
#                     True #not interesting results :(
#                     # y = np.array([calc_delta(yy,derivative) for yy in y])
#                     # rng = np.percentile
#                     # def delta_meas(x,axis=0):
#                     #     xx = np.median(x,axis=0)
#                     #     return calc_delta(xx,derivative)
#                     # y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=delta_meas,conf_interval=conf_interval)
#
#                 c=plt.cm.Set1(ii/9)#'cornflowerblue'
#                 ax[i,0].plot(xp[ind_t],y,lw=1,color=c,zorder=-1,label=f'{interest} {p}m, ({yp.shape[0]})',)
#                 ax[i,0].fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
#                 ax[i,0].plot(xp,result[to_plot[0]]['stim'],c='thistle',zorder=-10)
#                 time_sig = timeDependentDifference(yp[:,ind_t],yp_ref[:,ind_t_ref],n_boot=n_boot,conf_interval=conf_interval)
#                 x_sig = xp[ind_t]#np.arange(0,10,1/120)
#                 y_sig = .1
#             #                         if len(powers)-1>1:
#             #                             y_sig=2*y_sig/len(interest_list)
#                 j=0
#                 bott = np.zeros_like(x_sig)+1.5 - (j-1)*y_sig
#                 ax[i,0].fill_between(x_sig,bott,bott-y_sig*time_sig,
#                     facecolor=c,alpha=.4)
#                 box_keys={'lw':1, 'c':'k'}
#                 ax[i,0].plot(x_sig,bott,**box_keys)
#                 ax[i,0].plot(x_sig,bott-y_sig,**box_keys)
#                 # ax[i,0].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
#                 # ax[i,0].plot([ind_sig[1],ind_sig[1]],[bott[0],bott[0]-y_sig],**box_keys)
#                 # ax[i,0].set_title(f'{dur} min')
#                 ax[i,0].set_xlim(-3,40)
#                 ax[i,0].set_ylim(0,1.6)
#                 ax[i,0].legend()
#     ax[-1,0].set_xlabel('time (min)')
#     return fig,ax
