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

def behavior_heatmap(interest_list,exclude,n_boot=1e3,statistic=np.median,
                          measure_compare=None,ind_measure=[],
                          pop_measure=[responseDuration,totalResponse_pop],
                          pulseTimes=[5,30],conf_interval=99, stat_testing=True,
                         plot_comparison=True, ylim=(0,2),control_rnai=False,
                         cmap_clip=3):
     fig=plt.figure(figsize=(12,12))
     ax=fig.gca()
     name = 'data/LDS_response_rnai.pickle'
     with open(name,'rb') as f:
         result = pickle.load(f)
     #keys for measure_compare
     DIFFERENCE = ['diff','difference','subtract']
     RELATIVE = ['relative','rel','divide']

     phenotypes = [[] for i in range(len(pop_measure)*len(pulseTimes))]
     phenotypes_names = []
     '''reference'''
     Y_REF = []
     phenotypes_names.append('WT')
     for num,pulse in enumerate(pulseTimes):
         xp=result['tau']-pulse/60
         exclude_this=['35mm','p']

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
         #calculate reference population statistics
         for n_m, M in enumerate(pop_measure):
             loc=np.argmin(xp**2)
 #             y,rng = bootstrap(yp_ref[:,loc:loc+10*120],n_boot=n_boot,statistic=M)
             bott = 0
             if measure_compare in DIFFERENCE:
                 y,rng,significant = bootstrap_diff(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
                                            ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
             elif measure_compare in RELATIVE:
                 bott = 1
                 y,rng,significant = bootstrap_relative(yp_ref[:,loc:loc+15*120],yp_ref[:,loc:loc+15*120]
                                            ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
             else:
                 y,rng = bootstrap(yp_ref[:,loc:loc+15*120],n_boot=n_boot,statistic=M,conf_interval=conf_interval)
             phenotypes[num*len(pop_measure)+n_m].append(y)
     ''' Loop through genes'''
     for i,interest in enumerate(interest_list):
        phenotypes_names.append(interest)
        for num,pulse in enumerate(pulseTimes):
            exclude_this = exclude.copy()
            '''
            if interest=='syt1':
                exclude_this.append('syt1a')

            '''
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
            if not '+' in interest:
                exclude_this.append('+')
            if not 'PreRegen' in interest:
                exclude_this.append('PreRegen')
            to_plot = data_of_interest(result.keys(),[interest_i],exclude_this)
            if len(to_plot)==0:
                for n_m, M in enumerate(pop_measure):
                    phenotypes[num*len(pop_measure)+n_m].append(np.nan)
                continue
            print(to_plot)
            yp=[]
            for dat in to_plot:
                yp.extend(result[dat])
            yp = np.array(yp)

            #calculate population based measures
            for n_m, M in enumerate(pop_measure):
                loc=np.argmin(xp**2)
                if measure_compare in DIFFERENCE:
                    y,rng,significant = bootstrap_diff(yp[:,loc:loc+15*120],Y_REF[num][:,loc:loc+15*120]
                                               ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
                elif measure_compare in RELATIVE:
                    bott=1
                    y,rng,significant = bootstrap_relative(yp[:,loc:loc+15*120],Y_REF[num][:,loc:loc+15*120]
                                               ,n_boot=n_boot,measurement=M,conf_interval=conf_interval)
                else:
                    y,rng = bootstrap(yp[:,loc:loc+15*120],n_boot=n_boot,statistic=M,conf_interval=conf_interval)
                    significant=False
                phenotypes[num*len(pop_measure)+n_m].append(y)
     phenotypes=np.array(phenotypes)
     if measure_compare in RELATIVE:
         phenotypes=np.log2(phenotypes+1e-6)
     c_max = min(np.nanmax(np.abs(phenotypes)),cmap_clip)
     print(c_max)
     plt.imshow(phenotypes.T,cmap='RdBu',clim=(-c_max,c_max))
     plt.colorbar()
     plt.yticks(np.arange(len(phenotypes_names)),labels=phenotypes_names)

     x_names = []
     for p in pulseTimes:
         for M in pop_measure:
             x_names.append(f'{p}s {M.__name__ }')
     plt.xticks(np.arange(len(x_names)),labels=x_names,rotation=90)
     return fig, ax, phenotypes
