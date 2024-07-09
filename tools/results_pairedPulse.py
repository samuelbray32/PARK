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
                #check double/triple knockdown
                if dat.count('+')>i.count('+'):
                    keep=False
                if keep: to_plot.append(dat)
    return to_plot
#################################################################################

def plotTrace(conditions='WT',pulse1=5, pulse2=5, delay=3,
                n_boot=1e2,conf_interval=99,experiment='uv-uv',
                statistic=np.median, ylim=(0,1.3)):
    fig = plt.figure()
    ax = fig.gca()
    cond=conditions
    #load data
    name = f'data/LDS_response_pairedPulse_{experiment}.pickle'
    with open(name,'rb') as f:
            result = pickle.load(f)
    ref_name = 'data/LDS_response_rnai.pickle'
    with open(ref_name,'rb') as f:
        result_ref = pickle.load(f)
    if pulse2==5:
        ref = ['022421pc2']# data_of_interest(result_ref.keys(),[cond],['+','_','amp','Eye'])#'WT'#'022421pc2'#
        ref = data_of_interest(result_ref.keys(),[cond],['+','_','amp','Eye','1F'])
        print(ref)
    else:
        ref = data_of_interest(result_ref.keys(),[f'{cond}_{pulse2}s'],['+'])
    #plot single pulse reference
    xp_ref = result_ref['tau']
    yp_ref = np.concatenate([result_ref[ref_i] for ref_i in ref])
    y,rng = bootstrap_traces(yp_ref,n_boot=n_boot,statistic=statistic)
    ax.plot(xp_ref,y,c='grey')
    ax.fill_between(xp_ref,*rng,alpha=.2,color='grey',lw=0,
                    edgecolor='None', zorder=-1)

    test = f'{cond}_{pulse1}s{pulse2}s_{delay}mDelay'#f'WT_{pulse1}s{pulse2}s_{d}mDelay'
    if delay<1:
        test = f'{cond}_{pulse1}s{pulse2}s_{int(d*60)}sDelay'
    test = data_of_interest(result.keys(),[test],['_24h'])
    print(test)
    xp = result['tau']
    yp = np.concatenate([result[test_i]['secondary'] for test_i in test])
    y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)
    c='cornflowerblue'
    ax.plot(xp,y,c=c)
    ax.fill_between(xp,*rng,alpha=.2,color=c,lw=0,
                    edgecolor='None', zorder=-1)

    #Do a time-dependent test on the reference and current condition
    xp_ref = result_ref['tau']
    xp_ = result['tau']
    loc=np.argmin(xp_**2)
    ind_sig = np.arange(0*120,10*120)+loc
    loc_ref=np.argmin(xp_ref**2)
    ind_sig_ref = np.arange(0*120,10*120)+loc_ref
    time_sig = timeDependentDifference(yp[:,ind_sig],yp_ref[:,ind_sig_ref],
                                       n_boot=n_boot,conf_interval=conf_interval)
    x_sig = xp_[ind_sig]
    y_sig = .2
    bott = np.zeros_like(x_sig)+ylim[1]
    ax.fill_between(x_sig,bott,bott-y_sig*time_sig,
        facecolor=c,alpha=.4)
    box_keys={'lw':1, 'c':'k'}
    ax.plot(x_sig,bott,**box_keys)
    ax.plot(x_sig,bott-y_sig,**box_keys)
    ax.plot([0,0],[bott[0],bott[0]-y_sig],**box_keys)
    ax.plot([10,10],[bott[0],bott[0]-y_sig],**box_keys)

    #plot the uv indicator
    loc = -delay
    ax.fill_between([loc,loc+pulse1/60],[-1,-1],[10,10],facecolor='thistle',alpha=.5,zorder=-20)
    loc += delay

    ax.fill_between([loc,loc+pulse2/60],[-1,-1],[10,10],facecolor='thistle',alpha=.5,zorder=-20)
    ax.set_xlabel('time (min)')
    ax.set_ylabel('Activity')
    plt.xlim(-delay-2,7)
    plt.ylim(*ylim)
    return fig, ax
#################################################################################

def compareDelays_rnai(conditions=['WT'],pulse1=5, pulse2=5, delay=[.5,1,3,5,30],
                 pop_measure=[totalResponse_pop,],n_boot=1e2,conf_interval=99,
                 plot_comparison=False,measure_compare='diff',experiment='uv-uv',
                 control_rnai=False):#responseDuration,peakResponse

    fig,ax = plt.subplots(nrows=len(pop_measure))
    width=.2 #width of violin plots
    if len(pop_measure)==1:
        ax = [ax]
    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey',yy=12):
        ax.scatter([loc,],yy,marker='*',color=c)
    #load data
    name = f'data/LDS_response_pairedPulse_{experiment}.pickle'
    with open(name,'rb') as f:
            result = pickle.load(f)
    ref_name = 'data/LDS_response_rnai.pickle'
    with open(ref_name,'rb') as f:
        result_ref = pickle.load(f)

    colors=['cornflowerblue','firebrick','goldenrod']
    ref_dist = [] #WT at each delay
    for ii,cond in enumerate(conditions):

        #solve for the reference (no pre-pulse)
        if pulse2==5:
            ref = data_of_interest(result_ref.keys(),[cond],['+','_','amp','Eye','1F'])#'WT'#'022421pc2'#
            if control_rnai and cond=='WT':
                ref = ref + data_of_interest(result_ref.keys(),['cntrl'],['+','_','amp','Eye','1F'])
        else:
            ref = data_of_interest(result_ref.keys(),[f'{cond}_{pulse2}s'],['+','1F'])
            if control_rnai and cond=='WT':
                ref = ref + data_of_interest(result_ref.keys(),[f'cntrl_{pulse2}s'],['+','_','amp','Eye','1F'])


        print(ref)
        xp_ref = result_ref['tau']
        yp_ref = np.concatenate([result_ref[ref_i] for ref_i in ref])
        #calculate reference population statistics
        for n_m, M in enumerate(pop_measure):
            loc=np.argmin(xp_ref**2)
            bott = 0
            if measure_compare in DIFFERENCE:
                y,rng,significant,dist = bootstrap_diff(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
            elif measure_compare in RELATIVE:
                bott = 1
                y,rng,significant,dist = bootstrap_relative(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
            else:
                y,rng,dist = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
                                       conf_interval=conf_interval,return_samples=True)
            x_loc = len(delay)+.2*ii
            if not plot_comparison:
                y,rng,dist = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
                                       conf_interval=conf_interval, return_samples=True)
            #v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],showextrema=False)
            ax[n_m].scatter([x_loc],[y-bott],color=colors[ii])
            v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],showextrema=False)
            for pc in v['bodies']:
                pc.set_facecolor('grey')
            v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],showextrema=False)
            for pc in v['bodies']:
                pc.set_facecolor(colors[ii])
                pc.set_alpha(.08)
            ax[n_m].plot([-.5,x_loc+.5],[y,y],c='grey',ls=':')

        c=colors[ii]

        #loop through conditions
        measure_distributions = [[] for _ in pop_measure]
        for i,d in enumerate(delay):

            test = [f'{cond}_{pulse1}s{pulse2}s_{d}mDelay']#f'WT_{pulse1}s{pulse2}s_{d}mDelay'
            if d<1:
                test = [f'{cond}_{pulse1}s{pulse2}s_{int(d*60)}sDelay']
            if control_rnai and cond=='WT':
                if d>=1:
                    test = test + [f'cntrl_{pulse1}s{pulse2}s_{d}mDelay']
                else:
                    test = test + [f'cntrl_{pulse1}s{pulse2}s_{int(d*60)}sDelay']

            test = data_of_interest(result.keys(),test,['_24h']+['081222cntrl_5s5s_1mDelay','081222cntrl_5s5s_2mDelay','081222cntrl_5s5s_3mDelay',])
            print(test)
            xp = result['tau']
            yp = np.concatenate([result[test_i]['secondary'] for test_i in test])
            print(f'n={yp.shape[0]}')
            #calculate population based measures
            for n_m, M in enumerate(pop_measure):
                loc_ref=np.argmin(xp_ref**2)
                loc=np.argmin(xp**2)
                bott = 0
                # measure_distributions[n_m].append(totalResponse(yp[:,loc:loc+15*120]))
                if True:
                    baseline =np.median( yp_ref[:,loc-(5*120):loc],axis=0).mean()
                    print(baseline)
                    residual = np.median(yp_ref[:,int(loc+d*120):int(loc+(d+10)*120)],axis=0).mean() - baseline
                    print(d,residual,baseline)
                    if int(loc+(d+10)*120)>yp_ref.shape[1]:
                        residual=0
                else:
                    residual = 0
                



                if measure_compare in DIFFERENCE:
                    y,rng,significant,dist = bootstrap_diff(yp[:,loc:loc+15*120]-residual,yp_ref[:,loc_ref:loc_ref+15*120]
                                               ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
                elif measure_compare in RELATIVE:
                    bott=1
                    y,rng,significant,dist = bootstrap_relative(yp[:,loc:loc+15*120]-residual,yp_ref[:,loc_ref:loc_ref+15*120]
                                               ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
                measure_distributions[n_m].append(dist)
                # else:
                #     y,rng,dist = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
                #                            conf_interval=conf_interval,return_samples=True)
                #     significant=False
                #test for difference in sensitization vs. WT
                if ii==0:
                    # _,__,ref_dist_i = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
                    #                        conf_interval=conf_interval,return_samples=True)
                    # ref_dist.append(ref_dist_i)
                    ref_dist.append(dist)
                    sig_sense=False
                else:
                   sig_sense = False
                   delta_sense = dist-ref_dist[i]
                   if np.percentile(delta_sense,(100-conf_interval)/2)>0 or np.percentile(delta_sense,100-(100-conf_interval)/2)<0:
                       sig_sense=True

                x_loc= i + ii*.2
                # if not plot_comparison:
                #     y,rng,dist = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
                #                            conf_interval=conf_interval,return_samples=True)
                v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],
                                            showmeans=False,showextrema=False)
                if i==0:
                    ax[n_m].scatter([x_loc],[y],color=c,label=cond)
                else:
                    ax[n_m].scatter([x_loc],[y],color=c,)
                for pc in v['bodies']:
                    pc.set_facecolor(c)
                if significant:
                    mark_sig(ax[n_m],x_loc,c=c,yy=rng[1]*1.1)
                if sig_sense:
                    mark_sig(ax[n_m],x_loc,c='k',yy=rng[1]*1.1+.1)
    ax[-1].set_xticks(np.arange(len(delay)+1))
    ax[-1].set_xticklabels(delay+[120])
    ax[-1].set_xlabel('Delay (min)')
    for a,M in zip(ax,pop_measure):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylabel(M.__name__)
        a.set_xlim(-.5,len(delay)+.5)
    fig.suptitle(f'{pulse1}s pulse 1, {pulse2}s pulse 2, variable delay')
    plt.legend()
    return fig,ax, measure_distributions

#################################################################################


# def compareDelays(pulse1=5, pulse2=5, delay=[.5,1,3,5,30],
#                   pop_measure=[totalResponse_pop,],n_boot=1e2,conf_interval=99,
#                  plot_comparison=False,measure_compare='diff'):#responseDuration,peakResponse
#
#     fig,ax = plt.subplots(nrows=len(pop_measure))
#     width=.2 #width of violin plots
#     if len(pop_measure)==1:
#         ax = [ax]
#     #keys for measure_compare
#     DIFFERENCE = ['diff','difference','subtract']
#     RELATIVE = ['relative','rel','divide']
#     #significance marker
#     def mark_sig(ax,loc,c='grey',yy=12):
#         ax.scatter([loc,],yy,marker='*',color=c)
#     #load data
#     name = 'data/LDS_response_pairedPulse_uv-uv.pickle'
#     with open(name,'rb') as f:
#             result = pickle.load(f)
#     ref_name = 'data/LDS_response_uvRange.pickle'
#     ref_name = 'data/LDS_response_rnai.pickle'
#     with open(ref_name,'rb') as f:
#         result_ref = pickle.load(f)
#
#     #solve for the reference
#     if pulse2==5:
#         ref = 'WT'#'022421pc2'#
#     else:
#         ref = f'WT_{pulse2}s'
#     xp_ref = result_ref['tau']
#     yp_ref = result_ref[ref]
#     #calculate reference population statistics
#     for n_m, M in enumerate(pop_measure):
#         loc=np.argmin(xp_ref**2)
#         bott = 0
#         if measure_compare in DIFFERENCE:
#             y,rng,significant,dist = bootstrap_diff(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
#                                        ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
#         elif measure_compare in RELATIVE:
#             bott = 1
#             y,rng,significant,dist = bootstrap_relative(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
#                                        ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
#         else:
#             y,rng,dist = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
#                                    conf_interval=conf_interval,return_samples=True)
#         x_loc = len(delay)
#         if not plot_comparison:
#             y,rng,dist = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
#                                    conf_interval=conf_interval, return_samples=True)
#         v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],showextrema=False)
#         ax[n_m].scatter([x_loc],[y-bott],color='grey')
#         for pc in v['bodies']:
#             pc.set_facecolor('grey')
#         ax[n_m].plot([-.5,x_loc+.5],[y,y],c='grey',ls=':')
# #         else:
# #             ax[n_m].bar([x_loc],[y-bott],color='grey',width=width,bottom=[bott],alpha=.2)
# #             ax[n_m].plot([x_loc,x_loc],rng,color='grey')
#
#
#
#     c='cornflowerblue'
#     #loop through conditions
#     for i,d in enumerate(delay):
#
#         test = f'WT_{pulse1}s{pulse2}s_{d}mDelay'#f'WT_{pulse1}s{pulse2}s_{d}mDelay'
#         if d<1:
#             test = f'WT_{pulse1}s{pulse2}s_{int(d*60)}sDelay'
#         xp = result['tau']
#         yp = result[test]['secondary']
#
#         #calculate population based measures
#         for n_m, M in enumerate(pop_measure):
#             loc_ref=np.argmin(xp_ref**2)
#             loc=np.argmin(xp**2)
#             bott = 0
#             if measure_compare in DIFFERENCE:
#                 y,rng,significant,dist = bootstrap_diff(yp[:,loc:loc+15*120],yp_ref[:,loc_ref:loc_ref+15*120]
#                                            ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
#             elif measure_compare in RELATIVE:
#                 bott=1
#                 y,rng,significant,dist = bootstrap_relative(yp[:,loc:loc+15*120],yp_ref[:,loc_ref:loc_ref+15*120]
#                                            ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
#             else:
#                 y,rng,dist = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
#                                        conf_interval=conf_interval,return_samples=True)
#                 significant=False
#             x_loc= i
#             if not plot_comparison:
#                 y,rng,dist = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
#                                        conf_interval=conf_interval,return_samples=True)
#             v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],
#                                         showmeans=False,showextrema=False)
#             ax[n_m].scatter([x_loc],[y],color=c)
#             for pc in v['bodies']:
#                 pc.set_facecolor(c)
#
# #             else:
# #                 ax[n_m].bar([x_loc],[y-bott],color=c,width=width,bottom=[bott],alpha=.2)
# #                 ax[n_m].plot([x_loc,x_loc],rng,color=c)
#             if significant:
#                 mark_sig(ax[n_m],x_loc,c=c,yy=rng[1]*1.1)
#     ax[-1].set_xticks(np.arange(len(delay)+1))
#     ax[-1].set_xticklabels(delay+[120])
#     ax[-1].set_xlabel('Delay (min)')
#     for a,M in zip(ax,pop_measure):
#         a.spines['right'].set_visible(False)
#         a.spines['top'].set_visible(False)
#         a.set_ylabel(M.__name__)
#         a.set_xlim(-.5,len(delay)+.5)
#     fig.suptitle(f'{pulse1}s pulse 1, {pulse2}s pulse 2, variable delay')
#     return fig,ax
#################################################################################
def compareFirstPulse(pulse1=[1,5,10], pulse2=5, delay=3,
                  pop_measure=[totalResponse_pop,],n_boot=1e2,conf_interval=99,
                 plot_comparison=False,measure_compare='diff'):#responseDuration,peakResponse

    fig,ax = plt.subplots(nrows=len(pop_measure))
    width=.2 #width of violin plots
    if len(pop_measure)==1:
        ax = [ax]
    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey',yy=12):
        ax.scatter([loc,],yy,marker='*',color=c)
    #load data
    name = 'data/LDS_response_pairedPulse_uv-uv.pickle'
    with open(name,'rb') as f:
            result = pickle.load(f)
    ref_name = 'data/LDS_response_uvRange.pickle'
    # ref_name = 'data/LDS_response_rnai.pickle'
    with open(ref_name,'rb') as f:
        result_ref = pickle.load(f)

    #solve for the reference
    if pulse2==5:
        ref = 'WT'
    else:
        ref = f'WT_{pulse2}s'
    xp_ref = result_ref['tau']
    yp_ref = result_ref[ref]
    #calculate reference population statistics
    for n_m, M in enumerate(pop_measure):
        loc=np.argmin(xp_ref**2)
        bott = 0
        if measure_compare in DIFFERENCE:
            y,rng,significant,dist = bootstrap_diff(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
                                       ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
        elif measure_compare in RELATIVE:
            bott = 1
            y,rng,significant,dist = bootstrap_relative(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
                                       ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
        else:
            y,rng,dist = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot,statistic=M,conf_interval=conf_interval,return_samples=True)
        print(pulse1)
        x_loc = 0
        if not plot_comparison:
            y,rng,dist = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
                                   conf_interval=conf_interval, return_samples=True)
        v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],showextrema=False)
        ax[n_m].scatter([x_loc],[y-bott],color='grey')
        for pc in v['bodies']:
            pc.set_facecolor('grey')
        ax[n_m].plot([-.5,len(pulse1)+.5],[y,y],c='grey',ls=':')
        # else:
        #     ax[n_m].bar([x_loc],[y-bott],color='grey',width=width,bottom=[bott],alpha=.2)
        #     ax[n_m].plot([x_loc,x_loc],rng,color='grey')


    c='cornflowerblue'
    #loop through conditions
    for i,p1 in enumerate(pulse1):
        # solve for pulse 1 reference
        if p1==5:
            pulse1_ref = 'WT'
        else:
            pulse1_ref = f'WT_{p1}s'
        yp_ref_1 = result_ref[pulse1_ref]


        test = f'WT_{p1}s{pulse2}s_{delay}mDelay'
        if delay<1:
            test = f'WT_{p1}s{pulse2}s_{int(delay*60)}sDelay'
        xp = result['tau']
        yp = result[test]['secondary']
        print(f'n={yp.shape[0]}')
        #calculate population based measures
        for n_m, M in enumerate(pop_measure):
            loc_ref=np.argmin(xp_ref**2)
            loc=np.argmin(xp**2)
            bott = 0
            if True:
                print(delay,loc_ref,yp_ref.shape)
                baseline = np.median(yp_ref_1[:,loc_ref-(120*5):loc_ref],axis=0).mean()
                residual = np.median(yp_ref_1[:,loc_ref+int(delay*120):loc_ref+int((delay+10)*120)],axis=0).mean()- baseline
                print(baseline, residual)
            else:
                residual = 0
            print('residual', residual)

            if measure_compare in DIFFERENCE:
                y,rng,significant,dist = bootstrap_diff(yp[:,loc:loc+15*120]-residual,yp_ref[:,loc_ref:loc_ref+15*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
            elif measure_compare in RELATIVE:
                bott=1
                y,rng,significant,dist = bootstrap_relative(yp[:,loc:loc+15*120]-residual,yp_ref[:,loc_ref:loc_ref+15*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
            else:
                y,rng,dist = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,conf_interval=conf_interval,return_samples=True)
                significant=False
            x_loc= i+1
            if not plot_comparison:
                y,rng,dist = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,
                                       conf_interval=conf_interval,return_samples=True)
            v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],
                                        showmeans=False,showextrema=False)
            ax[n_m].scatter([x_loc],[y],color=c)
            for pc in v['bodies']:
                pc.set_facecolor(c)

            # else:
            #     ax[n_m].bar([x_loc],[y-bott],color=c,width=width,bottom=[bott],alpha=.2)
            #     ax[n_m].plot([x_loc,x_loc],rng,color=c)
            if significant:
                mark_sig(ax[n_m],x_loc,c=c,yy=rng[1]*1.1)
    ax[-1].set_xticks(np.arange(len(pulse1)+1))
    ax[-1].set_xticklabels([0]+pulse1)
    ax[-1].set_xlabel('First pulse (s)')
    for a,M in zip(ax,pop_measure):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylabel(M.__name__)
        a.set_xlim(-.5,len(pulse1)+.5)
    fig.suptitle(f'variable pulse 1, {pulse2}s pulse 2, {delay}m delay')
    return fig,ax

#################################################################################
def regeneration_traces(condition, ref, n_boot=1e3, conf_interval=99,
                        pool=12,days=7,step=None,statistic=np.median,ylim=(0,1.5),
                        pulse_dur=[5,5],delay=3,pulse = 'primary',):
    name = 'data/LDS_response_pairedPulse_uv-uv_regen.pickle'
    ref_name = 'data/LDS_response_pairedPulse_uv-uv.pickle'
    if step is None:
        step = pool
    integrate=10
    fig,ax = plt.subplots(nrows=days,sharex=True,sharey=True,figsize=(8,12))
    #plot the reference
    with open(ref_name,'rb') as f:
        result_ref = pickle.load(f)
    xp = result_ref['tau']
    yp_ref = result_ref[ref][pulse]
    y,rng = bootstrap_traces(yp_ref,n_boot=n_boot,statistic=statistic)
    for a in ax:
        a.plot(xp,y,c='grey')
        a.fill_between(xp,*rng,alpha=.2,color='grey',lw=0,
                        edgecolor='None', zorder=-1)
    #plot each day of the regenerating
    with open(name,'rb') as f:
        result = pickle.load(f)
    xp = result['tau']
    data = result[condition][pulse]
    loc = np.argmin(np.abs(xp))
    for t in range(len(data)):
        if t%step>0:continue
        count = t//step
        if count>=ax.size: continue
        c = plt.cm.cool(count/len(ax))
        yp = np.concatenate(data[t:t+pool])
        print(f'n={yp.shape[0]}')
        y,rng = bootstrap_traces(yp,n_boot=n_boot,statistic=statistic)
        ax[count].plot(xp,y,c=c)
        ax[count].fill_between(xp,*rng,alpha=.2,color=c,lw=0,
                        edgecolor='None', zorder=-1)

        #Do a time-dependent test on the reference and current condition
        xp_ref = result_ref['tau']
        xp_ = result['tau']
        loc=np.argmin(xp_**2)
        ind_sig = np.arange(-10*120,20*120)+loc
        loc_ref=np.argmin(xp_ref**2)
        ind_sig_ref = np.arange(-10*120,20*120)+loc_ref

        time_sig = timeDependentDifference(yp[:,ind_sig],yp_ref[:,ind_sig_ref],
                                           n_boot=n_boot,conf_interval=conf_interval)
        x_sig = xp_[ind_sig]
        y_sig = .2
        bott = np.zeros_like(x_sig)+ylim[1]
        ax[count].fill_between(x_sig,bott,bott-y_sig*time_sig,
            facecolor=c,alpha=.4)
        box_keys={'lw':1, 'c':'k'}
        ax[count].plot(x_sig,bott,**box_keys)
        ax[count].plot(x_sig,bott-y_sig,**box_keys)
        # ax[count].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
        # ax[count].plot([10,10],[bott[0],bott[0]-y_sig],**box_keys)
    #draw on pulse region
    loc=0
    if pulse=='secondary':
        loc = -delay
    print(loc)
    for a in ax:
        a.fill_between([loc,loc+pulse_dur[0]/60],[-1,-1],[10,10],facecolor='thistle',alpha=.3,zorder=-20)
    loc += delay
    for a in ax:
        a.fill_between([loc,loc+pulse_dur[1]/60],[-1,-1],[10,10],facecolor='thistle',alpha=.3,zorder=-20)
    ax[-1].set_xlabel('time (min)')
    ax[days//2].set_ylabel('Activity')
    if pulse=='primary':
        plt.xlim(-2,15)
    else:
        plt.xlim(-5,12)
    plt.ylim(*ylim)
    return fig,ax
#################################################################################

def regeneration_qunatify(condition, ref, n_boot=1e3, conf_interval=99,
                        pool=12,step=None,statistic=np.mean,
                        pop_measure=[totalResponse_pop,],measure_compare='diff',ref_dpa=[]):

    width=.2 #width of violin plots
    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey',yy=12):
        ax.scatter([loc,],yy,marker='*',color=c)
    name = 'data/LDS_response_pairedPulse_uv-uv_regen.pickle'
    ref_name = 'data/LDS_response_regen.pickle'
    with open(name,'rb') as f:
            result = pickle.load(f)
    with open(ref_name,'rb') as f:
        result_ref = pickle.load(f)

    fig,ax = plt.subplots(nrows=len(pop_measure),figsize=(8,8))
    if len(pop_measure)==1:
        ax=[ax]
    #make the list of datasets to compare
    if type(ref) is str:
        data_ref = result_ref[ref]
    else:
        data_ref = []
        for name,day in zip(ref,ref_dpa):
            for ii,yy in enumerate(result_ref[name]):
                loc = ii+day
                if loc>=len(data_ref):
                    data_ref.append([])
                data_ref[loc].append(yy)
        data_ref = [np.concatenate(yy) for yy in data_ref]
    data = []
    step=12#hard coded to one day
    pool=12#hard coded to one day
    for t in range(len(result[condition]['secondary'])):
        if t%step>0:continue
        count = t//step
        data.append(np.concatenate(result[condition]['secondary'][t:t+pool]))

    # while len(data)<len(data_ref):
    #     data.append(data[-1])
    #loop through the datasets, get distributions and significance on each
    loc = np.argmin(result['tau']**2)+10
    loc_ref = np.argmin(result_ref['tau']**2)+10
    for i,(yp,yp_ref) in enumerate(zip(data,data_ref)):
        #calculate population based measures
        for n_m, M in enumerate(pop_measure):
            bott = 0
            if measure_compare in DIFFERENCE:
                y_diff,rng,significant,dist_diff = bootstrap_diff(yp[:,loc:loc+10*120],yp_ref[:,loc_ref:loc_ref+10*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
            elif measure_compare in RELATIVE:
                bott=1
                y,rng,significant,dist = bootstrap_relative(yp[:,loc:loc+10*120],yp_ref[:,loc_ref:loc_ref+10*120]
                                           ,n_boot=n_boot,measurement=M,conf_interval=conf_interval,return_samples=True)
            else:
                print('ERROR: invalid comparison term')
                return


            y,rng,dist = bootstrap(yp[:,loc:loc+10*120],n_boot=n_boot,statistic=M,
                                   conf_interval=conf_interval,return_samples=True)
            y_ref,rng_ref,dist_ref = bootstrap(yp_ref[:,loc_ref:loc_ref+10*120],n_boot=n_boot,statistic=M,
                                  conf_interval=conf_interval,return_samples=True)

            #plot ref
            x_loc= i #+ ii*.2
            v = ax[n_m].violinplot([dist_ref],positions=[x_loc],vert=True,widths=[width],
                                        showmeans=False,showextrema=False)
            for pc in v['bodies']:
                pc.set_facecolor('grey')
            if i==0:
                ax[n_m].scatter([x_loc],[y_ref],color='grey',label='single pulse')
            else:
                ax[n_m].scatter([x_loc],[y_ref],color='grey',)


            x_loc= i + .2
            c='forestgreen'
            v = ax[n_m].violinplot([dist],positions=[x_loc],vert=True,widths=[width],
                                        showmeans=False,showextrema=False,)
            for pc in v['bodies']:
                pc.set_facecolor(c)
            if i==0:
                ax[n_m].scatter([x_loc],[y],color=c,label='paired pulse')
            else:
                ax[n_m].scatter([x_loc],[y],color=c,)

            c='magenta'
            v = ax[n_m].violinplot([dist_diff],positions=[x_loc],vert=True,widths=[width],
                                        showmeans=False,showextrema=False,)
            for pc in v['bodies']:
                pc.set_facecolor(c)
            if i==0:
                ax[n_m].scatter([x_loc],[y_diff],color=c,label='Sensitization')
            else:
                ax[n_m].scatter([x_loc],[y_diff],color=c,)

            # if i==0:
            #     ax[n_m].scatter([x_loc],[y],color=c,label=cond)
            # else:
            #     ax[n_m].scatter([x_loc],[y],color=c,)
            # for pc in v['bodies']:
            #     pc.set_facecolor(c)
            if significant:
                mark_sig(ax[n_m],x_loc,c='k',yy=rng[1]*1.1)


    for a,M in zip(ax,pop_measure):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylabel(M.__name__)
    plt.legend()
    plt.xlabel('dpa')
    return fig, ax
