import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype']='none'
import pickle
from .bootstrapTest import bootstrap_traces, bootstrapTest, bootstrap, timeDependentDifference
from .bootstrapTest import bootstrap_diff, bootstrap_relative
from .measurements import adaptationTime, peakResponse, totalResponse, totalResponse_pop, responseDuration
from .measurements import sensitization, habituation,sensitizationRate,habituationRate, tPeak
from tools.measurements import pulsesPop, sensitization, habituation
def data_of_interest(names,interest=[],exclude=[]):
    to_plot = []
    for dat in names:
        if dat in to_plot: continue
        for i in interest:
            if i in dat:
                keep = True
                for ex in exclude:
                    if ex in dat: keep = False
                if keep: to_plot.append(dat)
    return to_plot

##############################################################################################################

def rnai_response_regen(interest,exclude,n_boot=1e3,statistic=np.median,whole_ref=False,
                       ylim=(0,2),stat_testing=True,conf_interval=99):
    name = 'data/LDS_response_regen.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    data = data_of_interest(result.keys(),interest,exclude)
    result_ref = result
    ref = []
    for d in data:
        if '30s' in d:
            ref.append('standard_30s2h')
        else:
            ref.append('standard_5s2h')
    if whole_ref:
        ref = []
        name = 'data/LDS_response_rnai.pickle'
        for d in data:
            if '30s' in d:
                if 'vibe' in d:
                    name = 'data/LDS_response_vibration.pickle'
                    ref.append('WT_30s75p')
                else:
                    ref.append('WT_30s')
            elif '10s' in d:
                name = 'data/LDS_response_uvRange.pickle'
                ref.append('WT_10s')
            elif '1s' in d:
                name = 'data/LDS_response_uvRange.pickle'
                ref.append('WT_1s')
            else:
                ref.append('WT')
        with open(name,'rb') as f:
            result_ref = pickle.load(f)

    if not(type(ylim[0]) is tuple):
        ylim = tuple([ylim for _ in range(len(data))])

    day_shift = [0 for d in data]
    fig, ax = plt.subplots(ncols=len(data),nrows=len(result[(data[0])]),sharex=True,sharey='col',figsize=(4*len(data),12))
    if len(data)==1:
        ax=ax[:,None]
    if len (ref)==1:
        ref = [ref[0] for r in data]

    for i in range(len(data)):
        ax[0,i].set_title(data[i])
        print(ylim[i])
        ax[0,i].set_ylim(ylim[i])
        if whole_ref:
            print(ref[i])
            yp = result_ref[ref[i]]
            y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)

        for j in range(len (result[(data[i])])):
            sh=0
            if '30s' in data[i]: sh=.5
            if '5s' in data[i]: sh=5/60
            xp = result['tau']-sh
            yp_=(result[data[i]][j])
            if yp_.size==0:
                continue
            y_,rng_ = bootstrap_traces(yp_,n_boot=n_boot,statistic=statistic)
            ax[j+day_shift[i],i].plot(xp,y_,label=j, color=plt.cm.cool(j/7))
            ax[j+day_shift[i],i].fill_between(xp,*rng_,alpha=.3,edgecolor=None,facecolor=plt.cm.cool(j/7))
#         for j in range(len(result[ref[i]])):
            if not whole_ref:
                if len(result[ref[i]])<j: continue
                yp=(result_ref[ref[i]][j])
                y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)
            xp_ref = result_ref['tau']-sh
            ax[j,i].plot(xp_ref,y,color='grey',zorder=-1)
            ax[j,i].fill_between(xp_ref,*rng,alpha=.3,edgecolor=None,facecolor='grey',zorder=-2)

            #Do a time-dependent test on the reference and current condition
            loc=np.argmin(xp**2)
            ind_sig = np.arange(-5*120,10*120)+loc
            loc_ref=np.argmin(xp_ref**2)
            ind_sig_ref = np.arange(-5*120,10*120)+loc_ref
            if stat_testing:
                time_sig = timeDependentDifference(yp_[:,ind_sig],yp[:,ind_sig_ref],
                                                   n_boot=n_boot,conf_interval=conf_interval)
                x_sig = xp[ind_sig]
                y_sig = .1*(ylim[i][1]-ylim[i][0])
                bott = np.zeros_like(x_sig)+ylim[i][1]
                ax[j,i].fill_between(x_sig,bott,bott-y_sig*time_sig,
                    facecolor=plt.cm.cool(j/7),alpha=.4)
                box_keys={'lw':1, 'c':'k'}
                ax[j,i].plot(x_sig,bott,**box_keys)
                ax[j,i].plot(x_sig,bott-y_sig,**box_keys)
                ax[j,i].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
                ax[j,i].plot([10,10],[bott[0],bott[0]-y_sig],**box_keys)

            ax[j,i].fill_between([-sh,0,],[-1,-1],[10,10],facecolor='thistle',alpha=.3,zorder=-20)
            ax[j,i].spines['top'].set_visible(False)
            ax[j,i].spines['right'].set_visible(False)
            if i>0:
                ax[j,i].spines['left'].set_visible(False)

    fig.suptitle('regen control')
    #    ax[j].legend()
    plt.xlim(-3,20)
    #plt.yscale('log')
    ax[len(ax)//2,0].set_ylabel('Z')
    ax[-1,ax.shape[1]//2].set_xlabel('time (min)')

#     for a in ax:
#         a[0].set_yticks([0,1,2])

    return fig,ax

##############################################################################################################


def merge_regenerations(data,ref=None,day=None,conf_interval=99,n_boot=1e3,stat_testing=True,
                       ylim=(0,2)):
    if day is None:
        day = [0 for _ in data]
    if ref is None:
        if '30s' in data[0]:
            ref = 'WT_30s'
        elif '10s' in data[0]:
            ref = 'WT_10s'
        elif '1s' in data[0]:
            ref = 'WT_1s'
        else:
            ref = 'WT'
    print(data[0],ref)
    if '30s' in data[0]:
        sh=.5
    elif '10s' in data[0]:
        sh=10/60
    elif '1s' in data[0]:
        sh=1/60
    else:
        sh=5/60

    pl=25
    ph=75

    name = 'data/LDS_response_regen.pickle'
    with open(name,'rb') as f:
            result = pickle.load(f)
    ref_name = 'data/LDS_response_uvRange.pickle'
#     ref_name = 'data/LDS_response_rnai.pickle'
    with open(ref_name,'rb') as f:
            result_ref = pickle.load(f)

    # print(result_ref.keys())
    fig, ax = plt.subplots(nrows=9,sharex=True,sharey=True,figsize=(4,16))
    xx_ref=(result_ref[ref])
    print(ref_name,xx_ref.shape[0])
    if n_boot>1:
        yy_ref,rng_ref = bootstrap_traces(xx_ref,statistic=np.median,n_boot=n_boot,
                                          conf_interval=conf_interval)
    else:
        yy_ref=np.median(xx_ref,axis=0)
        rng_ref = (np.percentile(xx_ref,pl,axis=0),np.percentile(xx_ref,ph,axis=0))
    d=day.copy()
    for j in range(len(ax)):
        c = plt.cm.cool(j/len(ax))
        xx=[]
        for i,dat in enumerate(data):
            if d[i]>j or (j-d[i])>=len(result[dat]):
                continue
            if result[dat][j-d[i]].size==0: continue
            xx.append((result[dat][j-d[i]]))
        if len(xx)==0: continue
        xx = np.concatenate(xx)
        print(j,xx.shape[0])
        if xx.size==0: continue
        if n_boot>1:
            yy,rng = bootstrap_traces(xx,statistic=np.median,n_boot=n_boot,
                                      conf_interval=conf_interval)
            ax[j].plot(result['tau']-sh,yy,label=j, color=c)
            ax[j].fill_between(result['tau']-sh,*rng,alpha=.3,edgecolor=None,facecolor=c)

        else:
            ax[j].plot(result['tau']-sh,np.median(xx,axis=0),label=j, color=c)
            ax[j].fill_between(result['tau']-sh,np.percentile(xx,pl,axis=0),np.percentile(xx,ph,axis=0),alpha=.3,edgecolor=None,facecolor=c)
            xx_dat=xx.copy()

        ax[j].plot(result_ref['tau']-sh,yy_ref,color='grey',zorder=-1)
        ax[j].fill_between(result_ref['tau']-sh,*rng_ref,alpha=.3,edgecolor=None,facecolor='grey',zorder=-2)
        ax[j].fill_between([-sh,0,],[-1,-1],[10,10],facecolor='thistle',alpha=.3,zorder=-20)
        ax[j].set_yticks([0,1,2])

        #Do a time-dependent test on the reference and current condition
        xp_ref = result_ref['tau']
        xp_ = result['tau']
        loc=np.argmin(xp_**2)
        ind_sig = np.arange(-5*120,10*120)+loc
        loc_ref=np.argmin(xp_ref**2)
        ind_sig_ref = np.arange(-5*120,10*120)+loc_ref
        if stat_testing:
                time_sig = timeDependentDifference(xx[:,ind_sig],xx_ref[:,ind_sig_ref],
                                                   n_boot=n_boot,conf_interval=conf_interval)
                x_sig = xp_[ind_sig]
                y_sig = .1*(ylim[1]-ylim[0])
                bott = np.zeros_like(x_sig)+ylim[1]
                ax[j].fill_between(x_sig,bott,bott-y_sig*time_sig,
                    facecolor=plt.cm.cool(j/7),alpha=.4)
                box_keys={'lw':1, 'c':'k'}
                ax[j].plot(x_sig,bott,**box_keys)
                ax[j].plot(x_sig,bott-y_sig,**box_keys)
                ax[j].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
                ax[j].plot([10,10],[bott[0],bott[0]-y_sig],**box_keys)


    #    ax[j].legend()


    #plt.xlim(-20,60)
    #plt.yscale('log')
    #ax[-1].set_ylim(0,.5)
    ax[-1].set_xlim(-1,7)
    ax[-1].set_ylim(ylim)
    ax[len(ax)//2].set_ylabel('Z')
    plt.xlabel('time (min)')
    #ax[0].set_title(data + ': Whole Worm control')
    return fig, ax

###############################################################################################################

def regen_single_day(data,ref=None,dpa=0,n_boot=1e3,statistic=np.median,
                      ylim=(0,2),stat_testing=True,conf_interval=99,day_shift=None,
                      measure_compare='diff',pop_measure=[responseDuration,totalResponse_pop,peakResponse],
                      plot_comparison=False,):
    #keys for measure_compare
    n_boot_meas=n_boot
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey',yy=12):
        ax.scatter([loc,],yy,marker='*',color=c)

    if ref is None:
        if '30s' in data[0]:
            ref = 'WT_30s'
        elif '10s' in data[0]:
            ref = 'WT_10s'
        elif '1s' in data[0]:
            ref = 'WT_1s'
        else:
            ref = 'WT'
    if '30s' in data[0]:
        sh=.5
    elif '10s' in data[0]:
        sh=10/60
    elif '1s' in data[0]:
        sh=1/60
    else:
        sh=5/60

    pl=25
    ph=75
    name = 'data/LDS_response_regen.pickle'
    with open(name,'rb') as f:
            result = pickle.load(f)
    ref_name = 'data/LDS_response_uvRange.pickle'
    ref_name = 'data/LDS_response_rnai.pickle'
    with open(ref_name,'rb') as f:
        result_ref = pickle.load(f)

    fig, ax = plt.subplots(nrows=1,ncols=1+len(pop_measure),figsize=(15,10),
                             gridspec_kw={'width_ratios': [4]+[1 for _ in range(len(pop_measure))]})
    xx_ref=(result_ref[ref])
    yp_ref = xx_ref.copy()
    if n_boot>1:
        yy_ref,rng_ref = bootstrap_traces(xx_ref,statistic=np.median,n_boot=n_boot,
                                          conf_interval=conf_interval)
    else:
        yy_ref=np.median(xx_ref,axis=0)
        rng_ref = (np.percentile(xx_ref,pl,axis=0),np.percentile(xx_ref,ph,axis=0))
    xp_ref = result_ref['tau']-sh
    ax[0].plot(xp_ref,yy_ref,color='grey',zorder=-1)
    ax[0].fill_between(xp_ref,*rng_ref,alpha=.3,edgecolor=None,facecolor='grey',zorder=-2)

    #calculate reference population statistics
    ax2 = ax[1:]
    for n_m, M in enumerate(pop_measure):
        loc=np.argmin(xp_ref**2)
        bott = 0
        if measure_compare in DIFFERENCE:
            y,rng,significant = bootstrap_diff(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
                                       ,n_boot=n_boot_meas,measurement=M,conf_interval=conf_interval)
        elif measure_compare in RELATIVE:
            bott = 1
            y,rng,significant = bootstrap_relative(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
                                       ,n_boot=n_boot_meas,measurement=M,conf_interval=conf_interval)
        else:
            y,rng = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot_meas,statistic=M,conf_interval=conf_interval)
        x_loc = 0#n_m*(len(interest_list)+1)
        if not plot_comparison:
            y,rng,dist = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot_meas,statistic=M,
                                   conf_interval=conf_interval, return_samples=True)
            v = ax2[n_m].violinplot([dist],positions=[0],vert=True,widths=[.07],showextrema=False)
            ax2[n_m].scatter([x_loc],[y-bott],color='grey')
            for pc in v['bodies']:
                pc.set_facecolor('grey')
        else:
            ax2[n_m].bar([x_loc],[y-bott],color='grey',width=.07,bottom=[bott],alpha=.2)
            ax2[n_m].plot([x_loc,x_loc],rng,color='grey')


    #loop through datagroups
    yy_comp = []
    xp = result['tau']-sh
    for i in range(len(data)):
        print(i)
        c = plt.cm.Set1(i/9)

        if type(data[i]) is str:
            yp_=(result[data[i]][dpa-day_shift[i]])
            label_name = data[i]
        else:
            yp_ = []
            for j in range(len(data[i])):
                if day_shift is None:
                    d_sh=0
                else:
#                     print(i,day_shift,day_shift[i])
                    d_sh = day_shift[i][j]
#                     print(d_sh)
                yp_.append(result[data[i][j]][dpa-d_sh])
            yp_ = np.concatenate(yp_,axis=0)
            label_name=data[i][0]
        if yp_.size==0:
            continue
        y_,rng_ = bootstrap_traces(yp_,n_boot=n_boot,statistic=statistic)
        ax[0].plot(xp,y_,label=f'{label_name} ({yp_.shape[0]})', color=c)
        ax[0].fill_between(xp,*rng_,alpha=.3,edgecolor=None,facecolor=c)
        #Do a time-dependent test on the reference and current condition
        loc=np.argmin(xp**2)
        ind_sig = np.arange(-5*120,10*120)+loc
        loc_ref=np.argmin(xp_ref**2)
        ind_sig_ref = np.arange(-5*120,10*120)+loc_ref
        if stat_testing:
            time_sig = timeDependentDifference(yp_[:,ind_sig],xx_ref[:,ind_sig_ref],
                                               n_boot=n_boot,conf_interval=conf_interval)
            x_sig = xp[ind_sig]
            y_sig = .1*(ylim[1]-ylim[0])/len(data)
            bott = np.zeros_like(x_sig)+ylim[1]-i*y_sig
            ax[0].fill_between(x_sig,bott,bott-y_sig*time_sig,
                facecolor=c,alpha=.4)
            box_keys={'lw':1, 'c':'k'}
            ax[0].plot(x_sig,bott,**box_keys)
            ax[0].plot(x_sig,bott-y_sig,**box_keys)
            ax[0].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
            ax[0].plot([10,10],[bott[0],bott[0]-y_sig],**box_keys)
            yy_comp.append(yp_[:,ind_sig])
        sh=0
        if '30s' in data[i]: sh=.5
        if '5s' in data[i]: sh=5/60
        ax[0].fill_between([-sh,0,],[-1,-1],[10,10],facecolor='thistle',alpha=.3,zorder=-20)
        ax[0].spines['right'].set_visible(False)
        #calculate population based measures
        for n_m, M in enumerate(pop_measure):
            loc=np.argmin(xp**2)
            loc_ref=np.argmin(xp_ref**2)
            bott = 0
            if measure_compare in DIFFERENCE:
                y,rng,significant = bootstrap_diff(yp_[:,loc:loc+15*120],yp_ref[:,loc_ref:loc_ref+15*120]
                                           ,n_boot=n_boot_meas,measurement=M,conf_interval=conf_interval)
            elif measure_compare in RELATIVE:
                bott=1
                y,rng,significant = bootstrap_relative(yp_[:,loc:loc+15*120],yp_ref[:,loc_ref:loc_ref+15*120]
                                           ,n_boot=n_boot_meas,measurement=M,conf_interval=conf_interval)
            else:
                y,rng = bootstrap(yp_[:,loc:loc+15*120],n_boot=n_boot_meas,statistic=M,conf_interval=conf_interval)
                significant=False
            x_loc= (i+1)*.07# n_m + (i+1)*.07   #n_m*(len(interest_list)+1)+i+1
            if not plot_comparison:
                y,rng,dist = bootstrap(yp_[:,loc:loc+15*120],n_boot=n_boot_meas,statistic=M,
                                       conf_interval=conf_interval,return_samples=True)
                v = ax2[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[.07],
                                            showmeans=False,showextrema=False)
                ax2[n_m].scatter([x_loc],[y],color=c)
                for pc in v['bodies']:
                    pc.set_facecolor(c)

            else:
                ax2[n_m].bar([x_loc],[y-bott],color=c,width=.07,bottom=[bott],alpha=.2)
                ax2[n_m].plot([x_loc,x_loc],rng,color=c)
            if significant:
                mark_sig(ax2[n_m],x_loc,c=c,yy=rng[1]*1.1)


    #stat test betweeen the 2 regenerations
    if stat_testing and len(data)==2:
            time_sig = timeDependentDifference(yy_comp[0],yy_comp[1],
                                               n_boot=n_boot,conf_interval=conf_interval)
            x_sig = xp[ind_sig]
            y_sig = .1*(ylim[1]-ylim[0])
            bott = np.zeros_like(x_sig)+ylim[1]-y_sig
            c='darkgoldenrod'
            ax[0].fill_between(x_sig,bott,bott-y_sig*time_sig,
                facecolor=c,alpha=.4)
            box_keys={'lw':1, 'c':'k'}
            ax[0].plot(x_sig,bott,**box_keys)
            ax[0].plot(x_sig,bott-y_sig,**box_keys)
            ax[0].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
            ax[0].plot([10,10],[bott[0],bott[0]-y_sig],**box_keys)



    ax[0].set_ylim(ylim)
    ax[0].set_xlim(-3,10)
    ax[0].legend()
    ax[0].set_title(f'Day {dpa}')
    ax[0].set_ylabel('Z')
    ax[0].set_xlabel('time (min)')
    for a,M in zip(ax2,pop_measure):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_title(M.__name__)
    return fig,ax





#############################################################################################################
def excess_activity_regeneration(n_boot=1e3,statistic=np.median,
                               integrate=10*120,pool=12,color_scheme=None,
                                conf_interval=99):

    interest=[]
    exclude=[]
    name = 'data/LDS_response_regen_indPulse.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    to_plot = ['WT_8hpa_30s2h','WT_8hpa_10s2h','WT_8hpa_5s2h','WT_8hpa_1s2h']
    to_plot = [('WT_8hpa_30s2h','WT_79hpa_30s2h'),
               ('WT_8hpa_10s2h','070622WT_72hpa_10s2h'),
               ('WT_8hpa_5s2h','WT_72hpa_5s2h'),
                ('WT_8hpa_1s2h',)]#'WT_72hpa_1s2h')]
#     to_plot = [('WT_8hpa_5s2h',),#'WT_72hpa_5s2h'),
#                ('WT_8hpa_5s2h','WT_72hpa_5s2h'),]
    to_plot.reverse()
    to_plot.extend(data_of_interest(result.keys(),interest,exclude))

    subtract_ref=True #has to be for this analysis
    reference = []
    if subtract_ref:
        ref_name = 'data/LDS_response_uvRange.pickle'
        with open(ref_name,'rb') as f:
            result_ref = pickle.load(f)

    max_t = 200
    isi = integrate
    integrate = isi
    pulse_list = [0,]
    sh = []

    n_pulse=len(pulse_list)
    fig, ax = plt.subplots(nrows=n_pulse,sharex=True,figsize=(12,8))
    if n_pulse==1:
        ax=[ax]
    x_names = []
    for ii,cond in enumerate(to_plot):
        #print(cond)
        excess_start = []
        excess_end = []
        y_all = []
        from tqdm import tqdm
#         #define reference
#         if '5s2h' in cond:
#             response = result_ref['WT']
#             x_names.append('5')
#         else:
#             if 'hpa_' in cond:
#                 x_st = cond.find('hpa_')
#             else:
#                 x_st = cond.find('hpi_')
#             x_en = cond[x_st:].find('s')
#             response = result_ref['WT_'+cond[x_st+4:x_st+x_en+1]]
#             x_names.append(cond[x_st+4:x_st+x_en+1])
# #                 response = result_ref[reference[ii]]

        #define reference
        if '5s2h' in cond[0]:
            response = result_ref['WT']
            x_names.append('5')
        else:
            if 'hpa_' in cond[0]:
                x_st = cond[0].find('hpa_')
            else:
                x_st = cond[0].find('hpi_')
            x_en = cond[0][x_st:].find('s')
            response = result_ref['WT_'+cond[0][x_st+4:x_st+x_en+1]]
            x_names.append(cond[0][x_st+4:x_st+x_en+1])
#                 response = result_ref[reference[ii]]

        tau = result_ref['tau'].copy()
        if sh:
            tau -= sh[ii]
        pulse=0
        loc = np.argmin(result_ref['tau']**2)
        from .bootstrapTest import bootstrap
        from .measurements import totalResponse_pop as M
        y,rng,dist = bootstrap(response[:,loc:loc+10*120],n_boot=1e3,statistic=M,
                                               conf_interval=conf_interval,return_samples=True)
#         data = bootstrapTest('reference', [response,], tau=tau,
#                               n_pulse=max(pulse_list)+1, isi=isi,
#                               integrate_time=integrate,experiment_list=None)
#         y,rng = data.canonical_estimate([0,pulse],sample_size=None,
#                                         statistic=statistic, n_boot=n_boot)
        y_ref = y.copy()
        #each bootstrap is a set of samples from all timepoints
        for count in tqdm(range(int(n_boot))):
            for n_pulse, pulse in enumerate(pulse_list):
                y = []
                t_plot = []
#                 response = result[cond]['data']
#                 response_pool = [np.concatenate(response[i:i+pool]) for i in range(len(response))]
                response = result[cond[0]]['data']
                if len(cond)>1:
                    init = int((result[cond[1]]['initial']-result[cond[0]]['initial'])/2)
                    #print(result[cond[1]]['initial'],result[cond[0]]['initial'], init)
                    for i,dat in enumerate(result[cond[1]]['data']):
                        if i+init>=len(response):
                            response.append(dat)
                        else:
                            response[init+i] = np.append(response[init+i],dat,axis=0)

                response_pool = [np.concatenate(response[i:i+pool]) for i in range(len(response))]


                tau = result['tau'].copy()
                if sh:
                    tau -= sh[ii]
#                 data = bootstrapTest('data', response_pool, tau=tau,
#                                       n_pulse=max(pulse_list)+1, isi=isi,
#                                       integrate_time=integrate,experiment_list=None)
#                 for t in range(min(max_t,len(response))):
#                     y_t, rng = data.canonical_estimate([t,pulse],sample_size=None,
#                                                        statistic=statistic,n_boot=1)
#                     y.append(y_t)
#                     t_plot.append(t)
#                 t_plot = np.array(t_plot) * result[cond[0]]['spacing'] + result[cond[0]]['initial']
                t0 = np.argmin(tau)
                for r in response_pool:
                   y.append(np.median(r[np.random.choice(np.arange(r.shape[0]),r.shape[0],replace=True),
                                        t0:t0+integrate],axis=0).mean())
                t_plot = np.arange(len(y)) * result[cond[0]]['spacing'] + result[cond[0]]['initial']

                y=np.array(y)
                y_all.append(y)
                if subtract_ref:
                    y -= y_ref
                #for this bootstrap, note the first and last timepoints the response is above reference
                try:
                    excess_start.append(t_plot[np.where(y>0)[0][0]])
                except:
                    continue
#                     excess_start.append(t_plot[-1])
                excess_end.append(t_plot[np.where(y>0)[0][-1]])
                peak = np.argmax(y)
#                 try:
#                     excess_end.append(t_plot[peak:][np.where(y[peak:]<.1*y[peak])[0][0]])
#                 except:
#                     excess_end.append(t_plot[peak])
#                 try:
#                     excess_end.append(t_plot[peak:][np.where(y[peak:]<0)[0][0]])
#                 except:
#                     excess_end.append(t_plot[-1])

#         plt.plot(t_plot,np.mean(y_all,axis=0),label=cond[0])
#     plt.legend()
        #calculate the mean and confidence range
        excess_start = np.array(excess_start)
        excess_end = np.array(excess_end)
        if ii==0:
            plt.scatter([ii],[excess_start.mean()],c='cornflowerblue',label='excess activity begins')
            plt.scatter([ii],[excess_end.mean()],c='firebrick',label='excess activity ends')
        else:
            plt.scatter([ii],[excess_start.mean()],c='cornflowerblue')
            plt.scatter([ii],[excess_end.mean()],c='firebrick')
        start_lo = np.percentile(excess_start,(100-conf_interval)/2)
        start_hi = np.percentile(excess_start,conf_interval+(100-conf_interval)/2)
        plt.plot([ii,ii],[start_lo,start_hi],c='cornflowerblue')
        end_lo = np.percentile(excess_end,(100-conf_interval)/2)
        end_hi = np.percentile(excess_end,conf_interval+(100-conf_interval)/2)
        plt.plot([ii,ii],[end_lo,end_hi],c='firebrick')

    plt.legend()
    plt.xlabel('UV dose')
    plt.ylabel('hpa')
    plt.xticks(np.arange(ii+1),labels=x_names)
    plt.legend()

    return fig, ax


#####################################################################################################






def excess_activity_regeneration_v2(n_boot=1e3,statistic=np.median,
                               integrate=10*120,pool=12,color_scheme=None,
                                conf_interval=99):

    interest=[]
    exclude=[]
    name = 'data/LDS_response_regen_indPulse.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    to_plot = ['WT_8hpa_30s2h','WT_8hpa_10s2h','WT_8hpa_5s2h','WT_8hpa_1s2h']
    to_plot = [('WT_8hpa_30s2h','WT_79hpa_30s2h'),
               ('WT_8hpa_10s2h',),
               ('WT_8hpa_5s2h','WT_72hpa_5s2h'),
                ('WT_8hpa_1s2h',)]#'WT_72hpa_1s2h')]
#     to_plot = [('WT_8hpa_5s2h',),#'WT_72hpa_5s2h'),
#                ('WT_8hpa_5s2h','WT_72hpa_5s2h'),]
    to_plot.reverse()
    to_plot.extend(data_of_interest(result.keys(),interest,exclude))

    subtract_ref=True #has to be for this analysis
    reference = []
    if subtract_ref:
        ref_name = 'data/LDS_response_uvRange.pickle'
        with open(ref_name,'rb') as f:
            result_ref = pickle.load(f)

    max_t = 200
    isi = integrate
    integrate = isi
    pulse_list = [0,]
    sh = []

    n_pulse=len(pulse_list)
    fig, ax = plt.subplots(nrows=n_pulse,sharex=True,figsize=(12,8))
    if n_pulse==1:
        ax=[ax]
    x_names = []
    for ii,cond in enumerate(to_plot):
        #print(cond)
        excess_start = []
        excess_end = []
        y_all = []
        from tqdm import tqdm
        #define reference
        if '5s2h' in cond[0]:
            response = result_ref['WT']
            x_names.append('5')
        else:
            if 'hpa_' in cond[0]:
                x_st = cond[0].find('hpa_')
            else:
                x_st = cond[0].find('hpi_')
            x_en = cond[0][x_st:].find('s')
            response = result_ref['WT_'+cond[0][x_st+4:x_st+x_en+1]]
            x_names.append(cond[0][x_st+4:x_st+x_en+1])
#                 response = result_ref[reference[ii]]

        tau = result_ref['tau'].copy()
        if sh:
            tau -= sh[ii]
        pulse=0
        loc = np.argmin(result_ref['tau']**2)
        from .bootstrapTest import bootstrap
        from .measurements import totalResponse_pop as M
        y,rng,dist = bootstrap(response[:,loc:loc+10*120],n_boot=1e3,statistic=M,
                                               conf_interval=conf_interval,return_samples=True)

        y_ref = y.copy()


        dist_list = []
        t_plot = []
        for n_pulse, pulse in enumerate(pulse_list):
            y = []
            response = result[cond[0]]['data']
            if len(cond)>1:
                init = int((result[cond[1]]['initial']-result[cond[0]]['initial'])/2)
                #print(result[cond[1]]['initial'],result[cond[0]]['initial'], init)
                for i,dat in enumerate(result[cond[1]]['data']):
                    if i+init>=len(response):
                        response.append(dat)
                    else:
                        response[init+i] = np.append(response[init+i],dat,axis=0)

            response_pool = [np.concatenate(response[i:i+pool]) for i in range(len(response))]
            print('r_pool',response_pool[0].shape)

            tau = result['tau'].copy()
            if sh:
                tau -= sh[ii]
#                 data = bootstrapTest('data', response_pool, tau=tau,
#                                       n_pulse=max(pulse_list)+1, isi=isi,
#                                       integrate_time=integrate,experiment_list=None)
#                 for t in range(min(max_t,len(response))):
#                     y_t, rng = data.canonical_estimate([t,pulse],sample_size=None,
#                                                        statistic=statistic,n_boot=1)
#                     y.append(y_t)
#                     t_plot.append(t)
#                 t_plot = np.array(t_plot) * result[cond[0]]['spacing'] + result[cond[0]]['initial']
            t0 = np.argmin(tau)
            for r in response_pool:
                y_i, rng_i,dist = bootstrap(r,statistic=M,n_boot=n_boot,return_samples=True)
                dist_list.append(dist)




                if subtract_ref:
                    y -= y_ref
                #for this bootstrap, note the first and last timepoints the response is above reference
                try:
                    excess_start.append(t_plot[np.where(y>0)[0][0]])
                except:
                    continue
#                     excess_start.append(t_plot[-1])
                excess_end.append(t_plot[np.where(y>0)[0][-1]])
                peak = np.argmax(y)
#                 try:
#                     excess_end.append(t_plot[peak:][np.where(y[peak:]<.1*y[peak])[0][0]])
#                 except:
#                     excess_end.append(t_plot[peak])
#                 try:
#                     excess_end.append(t_plot[peak:][np.where(y[peak:]<0)[0][0]])
#                 except:
#                     excess_end.append(t_plot[-1])

#         plt.plot(t_plot,np.mean(y_all,axis=0),label=cond[0])
#     plt.legend()
        #calculate the mean and confidence range
        excess_start = np.array(excess_start)
        excess_end = np.array(excess_end)
        if ii==0:
            plt.scatter([ii],[excess_start.mean()],c='cornflowerblue',label='excess activity begins')
            plt.scatter([ii],[excess_end.mean()],c='firebrick',label='excess activity ends')
        else:
            plt.scatter([ii],[excess_start.mean()],c='cornflowerblue')
            plt.scatter([ii],[excess_end.mean()],c='firebrick')
        start_lo = np.percentile(excess_start,(100-conf_interval)/2)
        start_hi = np.percentile(excess_start,conf_interval+(100-conf_interval)/2)
        plt.plot([ii,ii],[start_lo,start_hi],c='cornflowerblue')
        end_lo = np.percentile(excess_end,(100-conf_interval)/2)
        end_hi = np.percentile(excess_end,conf_interval+(100-conf_interval)/2)
        plt.plot([ii,ii],[end_lo,end_hi],c='firebrick')

    plt.legend()
    plt.xlabel('UV dose')
    plt.ylabel('hpa')
    plt.xticks(np.arange(ii+1),labels=x_names)
    plt.legend()

    return fig, ax


##############################################################################################################
from .measurements import adaptationTime, peakResponse, totalResponse, totalResponse_pop, responseDuration
from .measurements import sensitization, habituation,sensitizationRate,habituationRate, tPeak

def rnai_regen_dailyStats(interest,exclude,n_boot=1e3,statistic=np.median,whole_ref=True,
                       stat_testing=True,conf_interval=99, pop_measure=[responseDuration,totalResponse_pop,peakResponse],
                         measure_compare='diff'):

    fig,ax = plt.subplots(nrows=len(pop_measure))
    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey',yy=12):
        ax.scatter([loc,],yy,marker='*',color=c)

    name = 'data/LDS_response_regen.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    data = data_of_interest(result.keys(),interest,exclude)[0]
    if (type(data) is list) and len(data)>1:
        print('WARNING: only analyzing first returned result from interest. Complete interest list is:')
        print(data)
        data = data[0]
    result_ref = result
    ref = []
    if whole_ref:
        ref = []
        name = 'data/LDS_response_rnai.pickle'
        d=data
        if '30s' in d:
            if 'vibe' in d:
                name = 'data/LDS_response_vibration.pickle'
                ref.append('WT_30s75p')
            else:
                ref.append('WT_30s')
        elif '10s' in d:
            name = 'data/LDS_response_uvRange.pickle'
            ref.append('WT_10s')
        elif '1s' in d:
            name = 'data/LDS_response_uvRange.pickle'
            ref.append('WT_1s')
        else:
            ref.append('WT')
        with open(name,'rb') as f:
            result_ref = pickle.load(f)
        yp_ref = result_ref[ref[0]]
        loc_ref = np.argmin(result_ref['tau']**2)
        for i,M in enumerate(pop_measure):
            if measure_compare in DIFFERENCE:
                y,rng,significant = bootstrap_diff(yp_ref[:,loc_ref:loc_ref+15*120],yp_ref[:,loc_ref:loc_ref+15*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
            elif measure_compare in RELATIVE:
                bott = 1
                y,rng,significant = bootstrap_relative(yp_ref[:,loc_ref:loc_ref+15*120],yp_ref[:,loc_ref:loc_ref+15*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
            else:
                y,rng = bootstrap(yp_ref[:,loc_ref:loc_ref+15*120],n_boot=n_boot,statistic=M,conf_interval=conf_interval)

            y,rng,dist = bootstrap(yp_ref[:,loc_ref:loc_ref+15*120],n_boot=n_boot,statistic=M,
                                   conf_interval=conf_interval, return_samples=True)
            v = ax[i].violinplot([dist],positions=[-1],vert=True,widths=[1],showextrema=False)
            ax[i].scatter([-1],[y],color='grey')
            for pc in v['bodies']:
                pc.set_facecolor('grey')

    loc = np.argmin(result['tau']**2)
    for j in range(len (result[data])):
        xp = result['tau']
        yp=result[data][j]
        if yp.size==0:
            continue
        #pull out first day as ref if not whole ref
        if (not whole_ref) and (j==0):
            yp_ref=result[data][0]
            loc_ref = loc.copy()
        for i,M in enumerate(pop_measure):
            if measure_compare in DIFFERENCE:
                y,rng,significant = bootstrap_diff(yp[:,loc:loc+15*120],yp_ref[:,loc_ref:loc_ref+15*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
            elif measure_compare in RELATIVE:
                bott = 1
                y,rng,significant = bootstrap_relative(yp[:,loc:loc+15*120],yp_ref[:,loc_ref:loc_ref+15*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
            else:
                y,rng = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,conf_interval=conf_interval)

            y,rng,dist = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
                                   conf_interval=conf_interval, return_samples=True)
            v = ax[i].violinplot([dist],positions=[j],vert=True,widths=[1],showextrema=False)
            c=plt.cm.cool(j/7)
            ax[i].scatter([j],[y],color=c)
            for pc in v['bodies']:
                pc.set_facecolor(c)
            if significant:
                mark_sig(ax[i],j,c=c,yy=rng[1]*1.1)

    for a,M in zip(ax,pop_measure):
        a.set_ylabel(M.__name__)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    for a in ax[:-1]:
        a.set_xticks([])
        a.spines['bottom'].set_visible(False)
    ax[-1].set_xlabel('day')


#         #Do a time-dependent test on the reference and current condition
#         loc=np.argmin(xp**2)
#         ind_sig = np.arange(-5*120,10*120)+loc
#         loc_ref=np.argmin(xp_ref**2)
#         ind_sig_ref = np.arange(-5*120,10*120)+loc_ref
#         if stat_testing:
#             time_sig = timeDependentDifference(yp_[:,ind_sig],yp[:,ind_sig_ref],
#                                                n_boot=n_boot,conf_interval=conf_interval)
#             x_sig = xp[ind_sig]
#             y_sig = .1
#             bott = np.zeros_like(x_sig)+ylim[i][1]
#             ax[j,i].fill_between(x_sig,bott,bott-y_sig*time_sig,
#                 facecolor=plt.cm.cool(j/7),alpha=.4)
#             box_keys={'lw':1, 'c':'k'}
#             ax[j,i].plot(x_sig,bott,**box_keys)
#             ax[j,i].plot(x_sig,bott-y_sig,**box_keys)
#             ax[j,i].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
#             ax[j,i].plot([10,10],[bott[0],bott[0]-y_sig],**box_keys)
#         sh=0
#         if '30s' in data[i]: sh=.5
#         if '5s' in data[i]: sh=5/60
#         ax[j,i].fill_between([-sh,0,],[-1,-1],[10,10],facecolor='thistle',alpha=.3,zorder=-20)
#         ax[j,i].spines['top'].set_visible(False)
#         ax[j,i].spines['right'].set_visible(False)
#         if i>0:
#             ax[j,i].spines['left'].set_visible(False)

#     fig.suptitle('regen control')
#     #    ax[j].legend()
#     plt.xlim(-3,20)
#     #plt.yscale('log')
#     ax[len(ax)//2,0].set_ylabel('Z')
#     ax[-1,ax.shape[1]//2].set_xlabel('time (min)')

#     for a in ax:
#         a[0].set_yticks([0,1,2])

    return fig,ax

##############################################################################
def excess_activity_regeneration_v072722(interest=[],exclude=[],n_boot=1e3,statistic=np.median,
                               integrate=10*120,pool=12,color_scheme=None,
                               subtract_ref=False):
    name = 'data/LDS_response_regen_indPulse.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    to_plot = ['WT_8hpa_30s2h','WT_8hpa_10s2h','WT_8hpa_5s2h','WT_8hpa_1s2h']
    to_plot.reverse()
    to_plot.extend(data_of_interest(result.keys(),interest,exclude))

    reference = []

    ref_name = 'data/LDS_response_uvRange.pickle'
    with open(ref_name,'rb') as f:
        result_ref = pickle.load(f)

    max_t = 200
    isi = integrate
    integrate = isi
    pulse_list = [0,]
    sh = []

    n_pulse=len(pulse_list)
    fig, ax = plt.subplots(nrows=n_pulse,sharex=True,figsize=(12,8))
    if n_pulse==1:
        ax=[ax]
    for ii,cond in enumerate(to_plot):
        print(cond)
        for n_pulse, pulse in enumerate(pulse_list):

            if '5s2h' in cond:
                response = result_ref['WT']
            else:
                x_st = cond.find('hpa_')
                x_en = cond[x_st:].find('s')
                response = result_ref['WT_'+cond[x_st+4:x_st+x_en+1]]

#                 response = result_ref[reference[ii]]
            tau = result_ref['tau'].copy()
            if sh:
                tau -= sh[ii]
            data = bootstrapTest('reference', [response,], tau=tau,
                                  n_pulse=max(pulse_list)+1, isi=isi,
                                  integrate_time=integrate,experiment_list=None)
            y,rng = data.canonical_estimate([0,pulse],sample_size=None,
                                            statistic=statistic, n_boot=n_boot)
            y_ref = y.copy()

            y = []
            lo = []
            hi = []
            t_plot = []
            response = result[cond]['data']
            response_pool = [np.concatenate(response[i:i+pool]) for i in range(len(response))]
            tau = result['tau'].copy()
            if sh:
                tau -= sh[ii]
            data = bootstrapTest('data', response_pool, tau=tau,
                                  n_pulse=max(pulse_list)+1, isi=isi,
                                  integrate_time=integrate,experiment_list=None)
            for t in range(min(max_t,len(response))):
                y_t, rng = data.canonical_estimate([t,pulse],sample_size=None,
                                                   statistic=statistic,n_boot=n_boot)
                y.append(y_t)
                lo.append(rng[0])
                hi.append(rng[1])
                t_plot.append(t)
            t_plot = np.array(t_plot) * result[cond]['spacing'] + result[cond]['initial']
            if color_scheme is None:
                if 'WT' in cond and not ('vibe' in cond or 'Pharynx' in cond):
                    x_st = cond.find('hpa_')
                    x_en = cond[x_st:].find('s')
                    c = plt.cm.gray_r((float(cond[x_st+4:x_st+x_en])+5)/40)

                else:
                    c=plt.cm.rainbow((ii)/10)
            else:
                c =color_scheme(1-((ii+1)/len(to_plot)))


            y -= y_ref
            lo -= y_ref
            hi -= y_ref
            y /= y_ref
            lo /= y_ref
            hi /= y_ref
                # y = np.log10(y/y_ref)
                # lo = np.log10(lo/y_ref)
                # hi = np.log10(hi/y_ref)
            lo=np.array(lo)
            # try:
            #     start = np.where(lo>0)[0][0]
            # except:
            #     start=np.nan
            # print(start)
            # try:
            #     end =start + np.where(lo[start:]>0)[0][-1]
            # except:
            #     end=np.nan

            try:
                #first point not significantly below zero
                start = np.where(hi>=0)[0][0]
            except:
                start=np.nan
            print(start)
            try:
                # #last point significantly above 0
                # end =start + np.where(lo[start:]>0)[0][-1]
                #first point excess activity ends
                start2 = np.where(lo>0)[0][0]
                end =start2 + np.where(lo[start2:]<=0)[0][0]
            except:
                end=np.nan
            if ii==0:
                ax[n_pulse].bar([ii-.2],[8+start*2],color='cornflowerblue',
                                width=.4,label='end of reduced activity')
                ax[n_pulse].bar([ii+.2],[8+end*2],color='firebrick',
                                width=.4,label='end of excess activity')
            else:
                ax[n_pulse].bar([ii-.2],[8+start*2],color='cornflowerblue',width=.4)
                ax[n_pulse].bar([ii+.2],[8+end*2],color='firebrick',width=.4)
    plt.xticks(np.arange(len(to_plot)),labels=to_plot)
    plt.ylabel('hpa')
    plt.legend()
    # plt.ylabel(f'log10 [integrated activity 0-{isi/120} min] / [wholeworm value]')

    return fig, ax


##############################################################################
def excess_activity_regeneration_v081022(n_boot=1e3,statistic=np.median,
                               integrate=10*120,pool=12,color_scheme=None,
                                conf_interval=99):
    fig,ax=plt.subplots(ncols=2,figsize=(12,8), gridspec_kw={'width_ratios': [3,1]})
    interest=[]
    exclude=[]
    name = 'data/LDS_response_regen_indPulse.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    to_plot = ['WT_8hpa_30s2h','WT_8hpa_10s2h','WT_8hpa_5s2h','WT_8hpa_1s2h']
    to_plot = [('WT_8hpa_30s2h','100220v2022_78hpa_30s2h'),
               ('WT_8hpa_10s2h','070622WT_72hpa_10s2h'),
               ('WT_8hpa_5s2h','110521v2022_72hpa_5s2h'),#'WT_72hpa_5s2h'),
                ('WT_8hpa_1s2h','WT_72hpa_1s2h')]
#     to_plot = [('WT_8hpa_5s2h',),#'WT_72hpa_5s2h'),
#                ('WT_8hpa_5s2h','WT_72hpa_5s2h'),]
    to_plot.reverse()
    to_plot.extend(data_of_interest(result.keys(),interest,exclude))

    subtract_ref=True #has to be for this analysis
    reference = []
    if subtract_ref:
        ref_name = 'data/LDS_response_uvRange.pickle'
        with open(ref_name,'rb') as f:
            result_ref = pickle.load(f)

    max_t = 200
    isi = integrate
    integrate = isi
    pulse_list = [0,]
    sh = []

    n_pulse=len(pulse_list)
    # fig, ax = plt.subplots(nrows=n_pulse,sharex=True,)
    # if n_pulse==1:
    #     ax=[ax]
    x_names = []
    for ii,cond in enumerate(to_plot):
        #print(cond)
        excess_start = []
        excess_end = []
        y_all = []
        from tqdm import tqdm
        #define reference
        if '5s2h' in cond[0]:
            response = result_ref['WT']
            x_names.append('5')
        else:
            if 'hpa_' in cond[0]:
                x_st = cond[0].find('hpa_')
            else:
                x_st = cond[0].find('hpi_')
            x_en = cond[0][x_st:].find('s')
            response = result_ref['WT_'+cond[0][x_st+4:x_st+x_en+1]]
            x_names.append(cond[0][x_st+4:x_st+x_en+1])
#                 response = result_ref[reference[ii]]

        tau = result_ref['tau'].copy()
        if sh:
            tau -= sh[ii]
        pulse=0
        loc = np.argmin(result_ref['tau']**2)
        from .bootstrapTest import bootstrap
        from .measurements import totalResponse_pop as M
        # y,rng,sig,dist = bootstrap_diff(response[:,loc:loc+10*120],response[:,loc:loc+10*120],n_boot=1e3,statistic=M,
        #                                        conf_interval=conf_interval,return_samples=True)
        # plt.plot([0,210],[y,y],c='grey')
        # plt.fill_between([0,210],[rng[0],rng[0]],[rng[1],rng[1]],facecolor='grey',alpha=.3)
        response_ref = response.copy()
        loc_ref = loc.copy()
        # y_ref = y.copy()


        dist_list = []
        t_plot = []

        y = []
        lo = []
        hi = []
        response = result[cond[0]]['data']
        if len(cond)>1:
            init = int((result[cond[1]]['initial']-result[cond[0]]['initial'])/2)
            print('init',init)
            #print(result[cond[1]]['initial'],result[cond[0]]['initial'], init)
            for i,dat in enumerate(result[cond[1]]['data']):
                if i+init>=len(response):
                    response.append(dat)
                else:
                    response[init+i] = np.append(response[init+i],dat,axis=0)

        response_pool = [np.concatenate(response[i:i+pool]) for i in range(len(response))]
        print('r_pool',response_pool[0].shape)

        tau = result['tau'].copy()
        loc= np.argmin(tau**2)
        if sh:
            tau -= sh[ii]
#                 data = bootstrapTest('data', response_pool, tau=tau,
#                                       n_pulse=max(pulse_list)+1, isi=isi,
#                                       integrate_time=integrate,experiment_list=None)
#                 for t in range(min(max_t,len(response))):
#                     y_t, rng = data.canonical_estimate([t,pulse],sample_size=None,
#                                                        statistic=statistic,n_boot=1)
#                     y.append(y_t)
#                     t_plot.append(t)
#                 t_plot = np.array(t_plot) * result[cond[0]]['spacing'] + result[cond[0]]['initial']
        t0 = np.argmin(tau)
        print(cond)
        def relative_diff(x,y):
            return np.subtract(x,y)/y
        from tools.bootstrapTest import bootstrap_compare
        for iii,r in enumerate(response_pool[:-pool]):
            t_plot.append(result[cond[0]]['initial']+iii*result[cond[0]]['spacing'])
            # y_i, rng_i,significant,dist = bootstrap_diff(r[:,loc:loc+120*15],response_ref[:,loc_ref:loc_ref+15*120],statistic=M,n_boot=n_boot,return_samples=True)
            y_i, rng_i, dist = bootstrap_compare(r[:,loc:loc+120*15],response_ref[:,loc_ref:loc_ref+15*120],statistic=M,
                                            n_boot=n_boot,return_samples=True,conf_interval=99,operator=relative_diff)
            dist_list.append(dist)
            lo.append(rng_i[0])
            hi.append(rng_i[1])
            y.append(y_i)
        c = plt.cm.viridis(ii/len(to_plot))
        ax[0].plot(t_plot,y,c=c)
        ax[0].fill_between(t_plot,lo,hi,alpha=.2,facecolor=c)
        # plt.fill
        # break
        hi=np.array(hi)
        lo=np.array(lo)
        try:
            #first point not significantly below zero
            start = np.where(hi>=0)[0][0]
        except:
            start=np.nan
        print(start)
        try:
            # #last point significantly above 0
            # end =start + np.where(lo[start:]>0)[0][-1]
            #first point excess activity ends
            start2 = np.where(lo>0)[0][0]
            end =start2 + np.where(lo[start2:]<=0)[0][0]
        except:
            end=np.nan
        if ii==0:
            ax[1].bar([ii-.2],[8+start*2],color='cornflowerblue',
                            width=.4,label='end of reduced activity')
            ax[1].bar([ii+.2],[8+end*2],color='firebrick',
                            width=.4,label='end of excess activity')
        else:
            ax[1].bar([ii-.2],[8+start*2],color='cornflowerblue',width=.4)
            ax[1].bar([ii+.2],[8+end*2],color='firebrick',width=.4)
    #labels for time plot
    ax[0].set_xlabel('hpa')
    ax[0].set_ylabel('delta response (%)')

    #labels for bar plot
    plt.xticks(np.arange(len(to_plot)),labels=[cond[0] for cond in to_plot])
    plt.ylabel('hpa')
    plt.legend()

    return fig, ax
