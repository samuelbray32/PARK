import numpy as np
import matplotlib.pyplot as plt
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
                #check double/triple knockdown
                if dat.count('+')>i.count('+'):
                    keep=False
                if keep: to_plot.append(dat)
    return to_plot

def rnai_response(interest,exclude,n_boot=1e3,statistic=np.median,regeneration=False, drugs=False,):
    if regeneration:
        from .results_regeneration import rnai_response_regen
        return rnai_response_regen(interest,exclude,n_boot,statistic)
    name = 'data/LDS_response_rnai.pickle'
    if drugs:
        name = 'data/LDS_response_drugs.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    to_plot = data_of_interest(result.keys(),interest,exclude)
    print(to_plot)
    tot=len(to_plot)
    fig, ax=plt.subplots(nrows=tot, sharex=True, sharey=True,figsize=(8,4*len(to_plot)))
    if tot==1:
        ax=[ax]
    num=0
    for i in to_plot:
        #plot control
        sh =0
        if i[-3:]=='30s':
            yp=result['WT_30s']
            xp=result['tau']-.5
            sh = .5
        elif ('_1s' in i):# or i=='oct 20mM':
            print(i)
            yp=result['WT_1s']
            xp=result['tau']-1/60
            sh=1/60
        else:
            yp=result['WT']
            xp=result['tau']-5/60
            sh=5/60
        y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)
        ax[num].plot(xp,y,lw=1,color='grey',zorder=-2,alpha=.6)
        ax[num].fill_between(xp,*rng,alpha=.2,color='grey',lw=0,edgecolor='None', zorder=-1)
        ax[num].set_ylim(0,2)
        ax[num].set_yticks([0,1,2])
        ax[num].plot([0,0],[0,5],c='k',ls=':')
        yp=result[i]
        y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)
        ax[num].plot(xp,y,label=i,lw=1,color='r',)
        ax[num].fill_between(xp,*rng,alpha=.1,color='r',lw=0,edgecolor='None')
        ax[num].legend()
        num+=1
    plt.xlabel('time (min)')
    ax[len(ax)//2].set_ylabel('Z')
    plt.xlim(-3,10)
    return fig, ax

################################################################################
def rnai_response_layered(interest_list,exclude,n_boot=1e3,statistic=np.median,drugs=False,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[responseDuration,totalResponse_pop,peakResponse],
                          pulseTimes=[5,30],conf_interval=99, stat_testing=True,
                         plot_comparison=True, ylim=(0,2),control_rnai=False): #peakResponse,
    '''compiles 5s and 30s data for given genes of interest and layers on plot'''
    name = 'data/LDS_response_rnai.pickle'
    if drugs:
        name = 'data/LDS_response_drugs.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey',yy=12):
        ax.scatter([loc,],yy,marker='*',color=c)
    #give extra n_boot to measurements
    n_boot_meas = max(n_boot, 3e2)

    if not(type(ylim[0]) is tuple):
        ylim = (ylim,ylim)
    fig, ax_all=plt.subplots(nrows=len(pulseTimes), ncols=1+len(pop_measure)+len(ind_measure),
                             sharex='col', sharey=False,figsize=(15,10),
                             gridspec_kw={'width_ratios': [4]+[1 for _ in range(len(pop_measure)+len(ind_measure))]})
    ax = ax_all[:,0]
    ax2 = ax_all[:,1:]
    tic_loc = []
    tic_name=[]
    #reference
    Y_REF = []
    for num,pulse in enumerate(pulseTimes):
        xp=result['tau']-pulse/60
        exclude_this=['35mm','p']
        # if pulse==5:
        #     yp_ref = result['WT']
        # else:
        #     yp_ref = result[f'WT_{pulse}s']

        if pulse==5:
            exclude_this.extend(['_30s','_1s','_'])
            interest_i=['WT']
            if control_rnai:
                interest_i.append('cntrl')
        else:
            interest_i = [f'WT_{pulse}s']
            if control_rnai:
                interest_i.append(f'cntrl_{pulse}s')
        to_plot = data_of_interest(result.keys(),interest_i,exclude_this)
        if len(to_plot)==0: continue
        print(to_plot)
        yp_ref=[]
        for dat in to_plot:
            yp_ref.extend(result[dat])
        yp_ref = np.array(yp_ref)

        Y_REF.append(yp_ref)
        ind_t = np.where((xp>=-4)&(xp<=11))[0]
        y,rng = bootstrap_traces(yp_ref[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
        ax[num].plot(xp[ind_t],y,lw=1,color='grey',zorder=20,alpha=.6,label=f'WT ({yp_ref.shape[0]})')
        ax[num].fill_between(xp[ind_t],*rng,alpha=.2,color='grey',lw=0,edgecolor='None', zorder=-1)
        ax[num].set_ylim(*(ylim[num]))
        ax[num].set_yticks([0,1,2])
        # ax[num].plot([0,0],[0,5],c='k',ls=':')
        ax[num].fill_between([-pulse/60,0,],[-1,-1],[10,10],facecolor='thistle',alpha=.3,zorder=-20)
        #calculate reference population statistics
        for n_m, M in enumerate(pop_measure):
            bott=0
            if not stat_testing:
                continue
            loc=np.argmin(xp**2)
#             y,rng = bootstrap(yp_ref[:,loc:loc+10*120],n_boot=n_boot,statistic=M)
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
                v = ax2[num,n_m].violinplot([dist],positions=[0],vert=True,widths=[.07],showextrema=False)
                ax2[num,n_m].scatter([x_loc],[y-bott],color='grey')
                for pc in v['bodies']:
                    pc.set_facecolor('grey')
            else:
                ax2[num,n_m].bar([x_loc],[y-bott],color='grey',width=.07,bottom=[bott],alpha=.2)
                ax2[num,n_m].plot([x_loc,x_loc],rng,color='grey')
            tic_loc.append(x_loc)
            tic_name.append(M.__name__)

        #calculate reference individual statistics
        for n_m2, M in enumerate(ind_measure):
            if not stat_testing: continue
            loc=np.argmin(xp**2)
            value = M(yp_ref[:,loc:loc+15*120])
            if measure_compare in DIFFERENCE:
                y,rng,significant = bootstrap_diff(value,value,n_boot=n_boot_meas,measurement=None,conf_interval=conf_interval)
            elif measure_compare in RELATIVE:
                y,rng,significant = bootstrap_relative(value,value,n_boot=n_boot_meas,measurement=None,conf_interval=conf_interval)
            else:
                y,rng = bootstrap(value,n_boot=n_boot_meas,conf_interval=conf_interval)
            x_loc2 = 0#n_m2+x_loc+1#n_m2*(len(interest_list)+1) + x_loc +1
            if not plot_comparison:
                y,rng = bootstrap(value,n_boot=n_boot_meas,conf_interval=conf_interval)
                v = ax2[num,n_m+len(pop_measure)].violinplot([dist],positions=[0],vert=True,widths=[.07],showextrema=False)
                for pc in v['bodies']:
                    pc.set_facecolor('grey')
            else:
                ax2[num,n_m2+len(pop_measure)].scatter([x_loc2],[y],color='grey')
                ax2[num,n_m2+len(pop_measure)].plot([x_loc2,x_loc2],rng,color='grey')
            tic_loc.append(x_loc2)
            tic_name.append(M.__name__)

    ''' Loop through genes'''
    for i,interest in enumerate(interest_list):
        c=plt.cm.Set1(i/9)
        for num,pulse in enumerate(pulseTimes):
            exclude_this = exclude.copy()
            if pulse==5:
                exclude_this.extend(['_30s','_1s','_','_40s','_10s'])
                xp=result['tau']-5/60
                interest_i=interest
                # yp_ref = result['WT']
            else:
                # exclude_this.extend(['_1s'])
                interest_i = interest+f'_{pulse}s'
                xp=result['tau']-pulse/60
                # yp_ref=result['WT_30s']
            conditional_exclude = ['+','PreRegen','1F']
            for c_e in conditional_exclude:
                if not c_e in interest:
                    exclude_this.append(c_e)
            # if not '+' in interest:
            #     exclude_this.append('+')
            # if not 'PreRegen' in interest:
            #     exclude_this.append('PreRegen')
            to_plot = data_of_interest(result.keys(),[interest_i],exclude_this)
            if len(to_plot)==0: continue
            print(to_plot)

            yp=[]
            for dat in to_plot:
                yp.extend(result[dat])
            yp = np.array(yp)
            y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
            ax[num].plot(xp[ind_t],y,label=f'{interest} ({yp.shape[0]})',lw=1,color=c,zorder=-1)
            ax[num].fill_between(xp[ind_t],*rng,alpha=.1,color=c,lw=0,edgecolor='None',zorder=-2)
            #Do a time-dependent test on the reference and current knockdown
            loc=np.argmin(xp**2)
            ind_sig = np.arange(-5*120,10*120)+loc
            if stat_testing:
                time_sig = timeDependentDifference(yp[:,ind_sig],Y_REF[num][:,ind_sig],n_boot=n_boot,conf_interval=conf_interval)
                x_sig = xp[ind_sig]#np.arange(0,10,1/120)
                y_sig = .1*(ylim[num][1]-ylim[num][0])
                if len(interest_list)>1:
                    y_sig=2*y_sig/len(interest_list)
                bott = np.zeros_like(x_sig)+ylim[num][1] - i*y_sig
                ax[num].fill_between(x_sig,bott,bott-y_sig*time_sig,
                    facecolor=c,alpha=.4)
                box_keys={'lw':1, 'c':'k'}
                ax[num].plot(x_sig,bott,**box_keys)
                ax[num].plot(x_sig,bott-y_sig,**box_keys)
                ax[num].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
                ax[num].plot([10,10],[bott[0],bott[0]-y_sig],**box_keys)
#                 ax[num].set_ylim(bott[0]-y_sig,2)

                #calculate population based measures
                for n_m, M in enumerate(pop_measure):
                    loc=np.argmin(xp**2)
                    bott = 0
                    if measure_compare in DIFFERENCE:
                        y,rng,significant = bootstrap_diff(yp[:,loc:loc+15*120],Y_REF[num][:,loc:loc+15*120]
                                                   ,n_boot=n_boot_meas,measurement=M,conf_interval=conf_interval)
                    elif measure_compare in RELATIVE:
                        bott=1
                        y,rng,significant = bootstrap_relative(yp[:,loc:loc+15*120],Y_REF[num][:,loc:loc+15*120]
                                                   ,n_boot=n_boot_meas,measurement=M,conf_interval=conf_interval)
                    else:
                        y,rng = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot_meas,statistic=M,conf_interval=conf_interval)
                        significant=False
                    x_loc= (i+1)*.07# n_m + (i+1)*.07   #n_m*(len(interest_list)+1)+i+1
                    if not plot_comparison:
                        y,rng,dist = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot_meas,statistic=M,
                                               conf_interval=conf_interval,return_samples=True)
                        v = ax2[num,n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[.07],
                                                    showmeans=False,showextrema=False)
                        ax2[num,n_m].scatter([x_loc],[y],color=c)
                        for pc in v['bodies']:
                            pc.set_facecolor(c)

                    else:
                        ax2[num,n_m].bar([x_loc],[y-bott],color=c,width=.07,bottom=[bott],alpha=.2)
                        ax2[num,n_m].plot([x_loc,x_loc],rng,color=c)
                    if significant:
                        mark_sig(ax2[num,n_m],x_loc,c=c,yy=rng[1]*1.1)
#                     ax2[num,n_m].set_ylim(ymin=0)
                #calculate individual based
                for n_m2, M in enumerate(ind_measure):
                    loc=np.argmin(xp**2)
                    value = M(yp[:,loc:loc+15*120])
                    value_ref = M(Y_REF[num][:,loc:loc+15*120])
                    if measure_compare in DIFFERENCE:
                        y,rng,significant = bootstrap_diff(value,value_ref,n_boot=n_boot_meas,measurement=None,conf_interval=conf_interval)
                    elif measure_compare in RELATIVE:
                        y,rng,significant = bootstrap_relative(value,value_ref,n_boot=n_boot_meas,measurement=None,conf_interval=conf_interval)
                    else:
                        y,rng = bootstrap(value,n_boot=n_boot_meas,conf_interval=conf_interval)
                        significant=False
                    x_loc2 = (i+1)*.07#len(pop_measure) + n_m2 + (i+1)*.07  #n_m2*(len(interest_list)+1)+i+1 + x_loc +1
                    if not plot_comparison:
                        y,rng,dist = bootstrap(value,n_boot=n_boot_meas,conf_interval=conf_interval,return_samples=True)
                        v = ax2[num,n_m].violinplot([dist],positions=[x_loc2],vert=True,widths=[.07],
                                                    showmeans=False,showextrema=False)
                        ax2[num,n_m].scatter([x_loc2],[y],color=c)
                        for pc in v['bodies']:
                            pc.set_facecolor(c)
#                         ax2[num,n_m].set_ylim(bottom=0)
                    else:
                        ax2[num,n_m2+len(pop_measure)].bar([x_loc2],[y-1],color=c,width=.07,bottom=[1],alpha=.2)
                        ax2[num,n_m2+len(pop_measure)].plot([x_loc2,x_loc2],rng,color=c)
                    if significant:
                        mark_sig(ax2[num,n_m2+len(pop_measure)],x_loc,c=c,yy=rng[1]*1.1)
        ax[0].legend()
        ax[num].legend()
    if stat_testing:
        for a in np.ravel(ax2):
            a.plot([-.1,1],[bott,bott],c='k',lw=.5)
            a.set_xticks([])
            a.set_xlim(-.07,x_loc+.07)
            for q in ['top','right',]:#'bottom']:
                a.spines[q].set_visible(False)
    for label,a in zip(tic_name,ax2.T):
        a[0].set_title(label)
#     ax2[-1,-1].set_xticks(tic_loc)
#     ax2[-1,-1].set_xticklabels(tic_name)
    ax[-1].set_xlabel('time (min)')
    ax[len(ax)//2].set_ylabel('Z')
    ax[0].set_xlim(-2,10)
    for a,lim in zip(ax,ylim):
        print(lim)
        a.set_ylim(lim)
    for a in np.ravel(ax2):
        a.set_ylim(ymin=0)
#     if measure_compare in RELATIVE:
#         ax2[-1].set_ylabel('<M(RNAi)/M(WT)>')
#         ax2[0].set_yscale('log')
#     elif measure_compare in DIFFERENCE:
#         ax2[-1].set_ylabel('<M(RNAi) - M(WT)>')
#     else:
#         ax2[-1].set_ylabel('M(gene)')
#     ax2[-1].set_xlabel('M')
    return fig, ax


def total_response_regeneration(interest,exclude,n_boot=1e3,statistic=np.median,
                               integrate=10*120,pool=12,color_scheme=None,
                               subtract_ref=False):
    name = 'data/LDS_response_regen_indPulse.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    to_plot = ['WT_8hpa_30s2h','WT_8hpa_10s2h','WT_8hpa_5s2h','WT_8hpa_1s2h']
    to_plot.reverse()
    to_plot.extend(data_of_interest(result.keys(),interest,exclude))

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
    for ii,cond in enumerate(to_plot):
        print(cond)
        for n_pulse, pulse in enumerate(pulse_list):
            if subtract_ref:
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
                if not subtract_ref:
                    plt.plot(t_plot,np.ones_like(t_plot)*y,ls='--',zorder=-1,alpha=.5,c=c)
                    # plt.fill_between(t_plot, np.ones_like(t_plot)*rng[0], np.ones_like(t_plot)*rng[1],
                    #                  facecolor="grey",edgecolor=c,hatch='/////',lw=0,zorder=-1)
                    plt.fill_between(t_plot, np.ones_like(t_plot)*rng[0], np.ones_like(t_plot)*rng[1],
                                      facecolor="grey",alpha=.1,zorder=-1)
                    plt.fill_between(t_plot, np.ones_like(t_plot)*rng[0], np.ones_like(t_plot)*rng[1],
                                      facecolor=c,alpha=.1,zorder=-1)
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

            if subtract_ref:
                y -= y_ref
                lo -= y_ref
                hi -= y_ref
                y /= y_ref
                lo /= y_ref
                hi /= y_ref
                # y = np.log10(y/y_ref)
                # lo = np.log10(lo/y_ref)
                # hi = np.log10(hi/y_ref)

            ax[n_pulse].plot(t_plot,y,color=c)
            ax[n_pulse].scatter(t_plot,y,color=c,label=f'{cond}, n={data.canonical_set_traces[0].shape[0]}')
            ax[n_pulse].fill_between(t_plot,lo,hi,facecolor=c,alpha=.2)

    if subtract_ref:
        plt.plot(t_plot,np.ones_like(t_plot)*0,c='k',alpha=.5,ls='-.' )
    plt.legend()
    plt.xlabel('hours')
    plt.ylabel(f'integrated activity 0-{isi/120} min')
    # plt.ylabel(f'log10 [integrated activity 0-{isi/120} min] / [wholeworm value]')
    plt.xlim(0,175)
    return fig, ax
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
###############################################################################################
def pulseTrain_trialDependent(interest=['WT_regen'],exclude=[],n_boot=1e3,statistic=np.median,integrate=30,
                              trial_rng=(0,100),pool=6,color_scheme=plt.cm.cool, norm_time=False,
                             measurements=[sensitization, sensitizationRate, habituation, habituationRate, tPeak],):
    '''plots pooled, time-dependent pulse train on whole worm WT control'''
    #load data
    name = 'data/LDS_response_pulseTrain.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    to_plot = data_of_interest(result.keys(),interest,exclude)
    #plot it
    fig, ax = plt.subplots(nrows=len(to_plot),ncols=3,sharey=False,figsize=(12,6*len(to_plot)))
    print(ax.shape)
    if len(to_plot)==1:
        ax=ax[None,:]
    measured_scalars = [[] for i in measurements]
    measured_scalars_lo = [[] for i in measurements]
    measured_scalars_hi = [[] for i in measurements]
    for i,(dat,a) in enumerate(zip(to_plot,ax)):
        a[0].get_shared_y_axes().join(a[0], a[1])
        data = result[dat]['data']
        # put in control
        pulseTrain(pulse=result[dat]['pulse']/2,
                      isi=(result[dat]['delay']+result[dat]['pulse'])/2,
                      n_boot=n_boot,norm_time=norm_time,trial_rng=trial_rng,n=result[dat]['n'],
                      integrate=30,exclude=['regen','vibe'],ax=a,color='grey') #TODO-don't hardcode integrate?
        for j in np.arange(*trial_rng,pool):
            c = color_scheme(j/trial_rng[1])
            if j>=len(data): break
            yp = np.concatenate(data[j:j+pool])
            y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)
            tp=result['tau'].copy()
            if norm_time:
                tp = tp/condition[1]*60
            a[0].plot(tp,y,c=c,label=j,lw=1)
            a[0].fill_between(tp,*rng,alpha=.1,color=c,lw=0,edgecolor='None')
            a[0].set_xlim(-5,15)
            a[0].set_title(dat)

            #plot integrated
            n_i = result[dat]['n']
            loc = np.where(tp==0)[0][0]
            y = []
            lo = []
            hi = []
            Values=[]
            for n_i in range(n_i):
                val = yp[:,loc:loc+integrate*2].mean(axis=1)
                Values.append(val)
                y_i, rng = bootstrap(val,n_boot=n_boot*10)
                y.append(y_i)
                lo.append(rng[0])
                hi.append(rng[1])
                loc += 60 #todo: dont hardcode (sorry its frday at 5:30)
            xp = np.arange(len(y))
            a[1].scatter(xp,y,color=c,label=j//pool)
            a[1].plot(xp,y,color=c,ls=':')
            a[1].fill_between(xp,lo,hi,alpha=.1,facecolor=c,zorder=-1)
            a[0].set_ylim(0,1.5)
#             #do scalar measurements
#             Values = np.array(Values).T
#             for k,M in enumerate(measurements):
#                 try:
#                     v = M(Values)
#                 except:
#                     print(result[dat].keys())
#                     isi = result[dat]['delay']
#                     v = M(Values,n_i,isi,isi)
#                 yy, rng = bootstrap(v,statistic=np.mean,n_boot=10*n_boot)
#                 measured_scalars[k].append(yy)
#                 measured_scalars_lo[k].append(rng[0])
#                 measured_scalars_hi[k].append(rng[1])
#                 xx = k+(j-trial_rng[0])/(trial_rng[1]-trial_rng[0])/2
#                 a[2].scatter([xx],[yy],color=c)
#                 a[2].plot([xx,xx],rng,c=c)


            if i<len(to_plot)-1:
                a[0].set_xlabel('')
                a[1].set_xlabel('')
                a[0].set_ylabel('')
                a[1].set_ylabel('')
    return fig,ax

#
