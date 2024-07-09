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
                    print(dat)
                    keep=False
                if keep: to_plot.append(dat)
    return to_plot

def rnai_response_layered(interest_list,exclude,n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[responseDuration,totalResponse_pop],
                          conf_interval=95, stat_testing=True,powers =[64,]): 
    
    '''compiles 5s and 30s data for given genes of interest and layers on plot'''
    #name = 'data/LDS_response_LONG.pickle'
    name = 'data/LDS_response_step_90m.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey'):
        ax.scatter([loc,],12,marker='*',color=c)
    #give extra n_boot to measurements
    n_boot_meas = max(n_boot, 3e2)

    fig, ax_all=plt.subplots(nrows=len(powers), ncols=2, sharex='col', sharey='col',figsize=(15,10))
    if len(powers)==1: ax_all = ax_all[None,:]
    ax = ax_all[:,0]
    ax2 = ax_all[:,1]
    tic_loc = []
    tic_name=[]
    #reference
    Y_REF = []
    
    for num,power in enumerate(powers):
        xp=result['tau']
        exclude_this=[]+exclude 
        interest_i = f'WT_30m2h{power}bp'    
        to_plot = data_of_interest(result.keys(),[interest_i],exclude_this)
        print(interest_i,to_plot)
        if len(to_plot)==0: continue
        yp_ref=[]
        for dat in to_plot:
            yp_ref.extend(result[dat])
        yp_ref = np.array(yp_ref)

        Y_REF.append(yp_ref)
        ind_t = np.where((xp>=-4)&(xp<=40))[0]
        y,rng = bootstrap_traces(yp_ref[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
        ax[num].plot(xp[ind_t],y,lw=1,color='grey',zorder=20,alpha=.6,label=f'WT {yp_ref.shape[0]}')
        ax[num].fill_between(xp[ind_t],*rng,alpha=.2,color='grey',lw=0,edgecolor='None', zorder=-1)
        ax[num].set_ylim(0,2)
        ax[num].set_yticks([0,1,2])
        # ax[num].plot([0,0],[0,5],c='k',ls=':')
        ax[num].fill_between([0,30,],[-1,-1],[10,10],facecolor='thistle',alpha=.3,zorder=-20)
#         #calculate reference population statistics
#         for n_m, M in enumerate(pop_measure):
#             loc=np.argmin(xp**2)
# #             y,rng = bootstrap(yp_ref[:,loc:loc+10*120],n_boot=n_boot,statistic=M)
#             if measure_compare in DIFFERENCE:
#                 y,rng,significant = bootstrap_diff(yp_ref[:,loc:loc+10*120],yp_ref[:,loc:loc+10*120]
#                                            ,n_boot=n_boot_meas,measurement=M,conf_interval=conf_interval)
#             elif measure_compare in RELATIVE:
#                 y,rng,significant = bootstrap_relative(yp_ref[:,loc:loc+10*120],yp_ref[:,loc:loc+10*120]
#                                            ,n_boot=n_boot_meas,measurement=M,conf_interval=conf_interval)
#             else:
#                 y,rng = bootstrap(yp_ref[:,loc:loc+10*120],n_boot=n_boot_meas,statistic=M,conf_interval=conf_interval)
#             x_loc = n_m#n_m*(len(interest_list)+1)
#             ax2[num].bar([x_loc],[y-1],color='grey',width=.1,bottom=[1],alpha=.2)
#             ax2[num].plot([x_loc,x_loc],rng,color='grey')
#             tic_loc.append(x_loc)
#             tic_name.append(M.__name__)

#         #calculate reference individual statistics
#         for n_m2, M in enumerate(ind_measure):
#             loc=np.argmin(xp**2)
#             value = M(yp_ref[:,loc:loc+10*120])
#             if measure_compare in DIFFERENCE:
#                 y,rng,significant = bootstrap_diff(value,value,n_boot=n_boot_meas,measurement=None,conf_interval=conf_interval)
#             elif measure_compare in RELATIVE:
#                 y,rng,significant = bootstrap_relative(value,value,n_boot=n_boot_meas,measurement=None,conf_interval=conf_interval)
#             else:
#                 y,rng = bootstrap(value,n_boot=n_boot_meas,conf_interval=conf_interval)

#             x_loc2 = n_m2+x_loc+1#n_m2*(len(interest_list)+1) + x_loc +1
#             ax2[num].scatter([x_loc2],[y],color='grey')
#             ax2[num].plot([x_loc2,x_loc2],rng,color='grey')
#             tic_loc.append(x_loc2)
#             tic_name.append(M.__name__)

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
            y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
            ax[num].plot(xp[ind_t],y,label=f'{interest} ({yp.shape[0]})',lw=1,color=c,zorder=-1)
            ax[num].fill_between(xp[ind_t],*rng,alpha=.1,color=c,lw=0,edgecolor='None',zorder=-2)
            #Do a time-dependent test on the reference and current knockdown
            loc=np.argmin(xp**2)
            ind_sig = np.arange(-5*120,45*120)+loc
            if stat_testing:
                time_sig = timeDependentDifference(yp[:,ind_sig],Y_REF[num][:,ind_sig],n_boot=n_boot,conf_interval=conf_interval)
                x_sig = xp[ind_sig]#np.arange(0,10,1/120)
                y_sig = .1
                if len(interest_list)>1:
                    y_sig=2*y_sig/len(interest_list)
                bott = np.zeros_like(x_sig)+2 - i*y_sig
                ax[num].fill_between(x_sig,bott,bott-y_sig*time_sig,
                    facecolor=c,alpha=.4)
                box_keys={'lw':1, 'c':'k'}
                ax[num].plot(x_sig,bott,**box_keys)
                ax[num].plot(x_sig,bott-y_sig,**box_keys)
                ax[num].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
                ax[num].plot([ind_sig[1],ind_sig[1]],[bott[0],bott[0]-y_sig],**box_keys)
                ax[num].set_title(f'{power} bit power')
#                 ax[num].set_ylim(bott[0]-y_sig,2)
                
#                 #calculate population based measures
#                 for n_m, M in enumerate(pop_measure):
#                     loc=np.argmin(xp**2)
#                     if measure_compare in DIFFERENCE:
#                         y,rng,significant = bootstrap_diff(yp[:,loc:loc+10*120],Y_REF[num][:,loc:loc+10*120]
#                                                    ,n_boot=n_boot_meas,measurement=M,conf_interval=conf_interval)
#                     elif measure_compare in RELATIVE:
#                         y,rng,significant = bootstrap_relative(yp[:,loc:loc+10*120],Y_REF[num][:,loc:loc+10*120]
#                                                    ,n_boot=n_boot_meas,measurement=M,conf_interval=conf_interval)
#                     else:
#                         y,rng = bootstrap(yp[:,loc:loc+10*120],n_boot=n_boot_meas,statistic=M,conf_interval=conf_interval)
#                         significant=False
#                     x_loc= n_m + (i+1)*.07   #n_m*(len(interest_list)+1)+i+1
#                     ax2[num].bar([x_loc],[y-1],color=c,width=.1,bottom=[1],alpha=.2)
#                     ax2[num].plot([x_loc,x_loc],rng,color=c)
#                     if significant:
#                         mark_sig(ax2[num],x_loc,c=c)
#                 #calculate individual based
#                 for n_m2, M in enumerate(ind_measure):
#                     loc=np.argmin(xp**2)
#                     value = M(yp[:,loc:loc+10*120])
#                     value_ref = M(Y_REF[num][:,loc:loc+10*120])
#                     if measure_compare in DIFFERENCE:
#                         y,rng,significant = bootstrap_diff(value,value_ref,n_boot=n_boot_meas,measurement=None,conf_interval=conf_interval)
#                     elif measure_compare in RELATIVE:
#                         y,rng,significant = bootstrap_relative(value,value_ref,n_boot=n_boot_meas,measurement=None,conf_interval=conf_interval)
#                     else:
#                         y,rng = bootstrap(value,n_boot=n_boot_meas,conf_interval=conf_interval)
#                         significant=False
#                     x_loc2 = len(pop_measure) + n_m2 + (i+1)*.07  #n_m2*(len(interest_list)+1)+i+1 + x_loc +1
#                     ax2[num].bar([x_loc2],[y-1],color=c,width=.1,bottom=[1],alpha=.2)
#                     ax2[num].plot([x_loc2,x_loc2],rng,color=c)
#                     if significant:
#                         mark_sig(ax2[num],x_loc,c=c)
    ax[num].legend()
    
#     for a in ax2:
#         a.plot([-.1,x_loc+.1],[1,1],c='k',lw=.5)
#     ax2[-1].set_xticks(tic_loc)
#     ax2[-1].set_xticklabels(tic_name)
    ax[-1].set_xlabel('time (min)')
    ax[len(ax)//2].set_ylabel('Z')
    ax[0].set_xlim(-4,40)
#     if measure_compare in RELATIVE:
#         ax2[-1].set_ylabel('<M(RNAi)/M(WT)>')
#         ax2[0].set_yscale('log')
#     elif measure_compare in DIFFERENCE:
#         ax2[-1].set_ylabel('<M(RNAi) - M(WT)>')
#     else:
#         ax2[-1].set_ylabel('M(gene)')
#     ax2[-1].set_xlabel('M')
    return fig, ax


def response_scaling(interest='WT',exclude=[],n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[responseDuration,totalResponse_pop],
                          conf_interval=95, stat_testing=True,powers =[4,8,16,32,64,128]): 
    
    '''compiles 5s and 30s data for given genes of interest and layers on plot'''
    name = 'data/LDS_response_LONG.pickle'
    with open(name,'rb') as f:
        result = pickle.load(f)
    #keys for measure_compare
    DIFFERENCE = ['diff','difference','subtract']
    RELATIVE = ['relative','rel','divide']
    #significance marker
    def mark_sig(ax,loc,c='grey'):
        ax.scatter([loc,],12,marker='*',color=c)
    #give extra n_boot to measurements
    n_boot_meas = max(n_boot, 3e2)

    fig, ax_all=plt.subplots(nrows=1, ncols=1, sharex='col', sharey='col',figsize=(12,5))
#     if len(powers)==1: ax_all = ax_all[None,:]
    ax = [ax_all]#[:,0]
#     ax2 = ax_all[:,1]
    tic_loc = []
    tic_name=[]
    #reference
    Y_REF = []
    
    for num,power in enumerate(powers):
        c = plt.cm.magma(num/len(powers))
        xp=result['tau']
        exclude_this=[]+exclude 
        interest_i = f'{interest}_30m2h{power}bp'    
        to_plot = data_of_interest(result.keys(),[interest_i],exclude_this)
        print(interest_i,to_plot)
        if len(to_plot)==0: continue
        yp_ref=[]
        for dat in to_plot:
            yp_ref.extend(result[dat])
        yp_ref = np.array(yp_ref)

        Y_REF.append(yp_ref)
        ind_t = np.where((xp>=-4)&(xp<=40))[0]
        y,rng = bootstrap_traces(yp_ref[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
        ax[0].plot(xp[ind_t],y,lw=1,color=c,zorder=-num,alpha=.6,label=f'{interest} {power}bp ({yp_ref.shape[0]})')
        ax[0].fill_between(xp[ind_t],*rng,alpha=.2,color=c,lw=0,edgecolor='None', zorder=-10)
        ax[0].set_ylim(0,2)
        ax[0].set_yticks([0,1,2])
        # ax[num].plot([0,0],[0,5],c='k',ls=':')
        ax[0].fill_between([0,30,],[-1,-1],[10,10],facecolor='thistle',alpha=.2,zorder=-20)
    ax[0].legend()
    ax[0].set_xlim(-3,35)
    ax[0].set_ylim(0,1.3)
    
    return fig, ax