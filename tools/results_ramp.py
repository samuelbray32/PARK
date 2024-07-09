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


def response_ramp_byGene(interest_list,exclude,powers =['1bp255bp','255bp1bp'],durations=[1,3,5,10],
                          n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[],
                          conf_interval=95, stat_testing=True,): 
    
    '''compiles 5s and 30s data for given genes of interest and layers on plot'''
    name = 'data/LDS_response_ramp.pickle'
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

    fig, ax_all=plt.subplots(nrows=len(durations), ncols=len(interest_list), sharex=True, sharey=True,figsize=(7*len(interest_list),10))
    if len(durations)==1: ax_all = ax_all[None,:]
    if len(interest_list)==1: ax_all=ax_all[:,None]   
    
    ax = ax_all#[:,0]
#     ax2 = ax_all[:,1]
    tic_loc = []
    tic_name=[]
    #reference
    Y_REF = [[] for _ in interest_list]
    
    xp = result['tau']
    ind_t = np.where((xp>=-10)&(xp<=30))[0]
    
#     for num,power in enumerate(powers):
#         xp=result['tau']
#         exclude_this=[]+exclude 
#         interest_i = f'WT_30m2h{power}bp'    
#         to_plot = data_of_interest(result.keys(),[interest_i],exclude_this)
#         print(interest_i,to_plot)
#         if len(to_plot)==0: continue
#         yp_ref=[]
#         for dat in to_plot:
#             yp_ref.extend(result[dat])
#         yp_ref = np.array(yp_ref)

#         Y_REF.append(yp_ref)
#         ind_t = np.where((xp>=-4)&(xp<=40))[0]
#         y,rng = bootstrap_traces(yp_ref[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
#         ax[num].plot(xp[ind_t],y,lw=1,color='grey',zorder=20,alpha=.6,label=f'WT {yp_ref.shape[0]}')
#         ax[num].fill_between(xp[ind_t],*rng,alpha=.2,color='grey',lw=0,edgecolor='None', zorder=-1)
#         ax[num].set_ylim(0,2)
#         ax[num].set_yticks([0,1,2])
#         # ax[num].plot([0,0],[0,5],c='k',ls=':')
#         ax[num].fill_between([0,30,],[-1,-1],[10,10],facecolor='thistle',alpha=.3,zorder=-20)

        
    for i,interest in enumerate(interest_list):
        print(interest)
        
        for j,power in enumerate(powers):
            c=plt.cm.Set1(j/9)
            
            for k,dur in enumerate(durations):
                if j==0:
                    ax[k,i].fill_between([0,dur],[-1,-1],[3,3],facecolor='thistle',alpha=.2)
                exclude_this = exclude.copy()
                if not '+' in interest:
                    exclude_this.append('+')
                search_name = f'{interest}_{power}_{dur}m' 
                to_plot = data_of_interest(result.keys(),[search_name],exclude_this)
                print(search_name,to_plot)
                if len(to_plot)==0: continue
                yp=[]
                for dat in to_plot:
                    yp.extend(result[dat]['data'])
                yp = np.array(yp)
                y,rng = bootstrap_traces(yp[:,ind_t],n_boot=n_boot,statistic=statistic,conf_interval=conf_interval)
                ax[k,i].plot(xp[ind_t],y,lw=1,color=c,zorder=-1,label=f'{interest}, {power} ({yp.shape[0]})')
                ax[k,i].fill_between(xp[ind_t],*rng,alpha=.25,color=c,lw=0,edgecolor='None',zorder=-2)
                
                if j==0:
                    Y_REF[i].append(yp)
                else:   
                    #Do a time-dependent test on the reference and current knockdown
                    loc=np.argmin(xp**2)
                    ind_sig = np.arange(-5*120,20*120)+loc
                    if stat_testing:
                        time_sig = timeDependentDifference(yp[:,ind_sig],Y_REF[i][k][:,ind_sig],n_boot=n_boot,conf_interval=conf_interval)
                        x_sig = xp[ind_sig]#np.arange(0,10,1/120)
                        y_sig = .1
#                         if len(powers)-1>1:
#                             y_sig=2*y_sig/len(interest_list)
                        bott = np.zeros_like(x_sig)+1.5 - (j-1)*y_sig
                        ax[k,i].fill_between(x_sig,bott,bott-y_sig*time_sig,
                            facecolor=c,alpha=.4)
                        box_keys={'lw':1, 'c':'k'}
                        ax[k,i].plot(x_sig,bott,**box_keys)
                        ax[k,i].plot(x_sig,bott-y_sig,**box_keys)
                        ax[k,i].plot([ind_sig[0],ind_sig[0]],[bott[0],bott[0]-y_sig],**box_keys)
                        ax[k,i].plot([ind_sig[1],ind_sig[1]],[bott[0],bott[0]-y_sig],**box_keys)
                        ax[k,i].set_title(f'{dur} min')
    ax[0,0].set_xlim(-4,15)
    ax[0,0].set_ylim(0,1.5)
    for a in ax[0,:]:
        a.legend()
    ax[-1,len(interest_list)//2].set_xlabel('time(min)')
    ax[len(durations)//2,0].set_ylabel('Activity')
                
#     ax[num].legend()
    

#     ax[-1].set_xlabel('time (min)')
#     ax[len(ax)//2].set_ylabel('Z')
#     ax[0].set_xlim(-4,40)
    return fig, ax
