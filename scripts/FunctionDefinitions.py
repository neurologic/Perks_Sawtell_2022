'''Import Packages'''
import sys
from brian2 import *
import sympy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy import signal
from scipy import optimize
from scipy import stats
from scipy.stats import gaussian_kde
import pickle
import random

import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42



'''Figure Styles'''
def figsave(figure_folder,figname):
    plt.savefig(figure_folder / (figname + '.pdf'),format='pdf',dpi=300, transparent = True)
    
def create_fig():
    hfig = plt.figure(figsize=[1.5,1.5])
    ax = hfig.add_axes([0.2,0.2,0.7,0.7],frameon=False)
    return hfig,ax

def create_fig_tuning():
    figsize=[1.2,1.5]
    hfig = plt.figure(figsize = figsize) 
    ax = hfig.add_axes([0.3,0.2,0.6,0.7])
    return hfig,ax

def set_fig_style():
    rc = matplotlib.rcParams
    fontsize = 7
    rc['font.size'] = fontsize
    rc['axes.labelsize'] = fontsize
    rc['axes.titlesize'] = fontsize
    rc['axes.linewidth'] = 0.5
    rc['axes.labelpad'] = 0
    rc['font.family'] = 'sans-serif'
    rc['font.sans-serif'] = ['Helvetica','Arial','sans-serif']
    rc['xtick.major.width'] = 0.5
    rc['ytick.major.width'] = 0.5
    rc['xtick.minor.width'] = 0.5
    rc['ytick.minor.width'] = 0.5
    rc['xtick.major.size'] = 2
    rc['ytick.major.size'] = 2
    rc['xtick.minor.size'] = 2
    rc['ytick.minor.size'] = 2
    rc['xtick.major.pad'] = 0
    rc['ytick.major.pad'] = 0
    rc['xtick.minor.pad'] = 0
    rc['ytick.minor.pad'] = 0
    rc['ytick.labelsize'] = fontsize
    rc['xtick.labelsize'] = fontsize
    rc['pdf.fonttype'] = 42
    rc['ps.fonttype'] = 42
#     matplotlib.rcParams.update(rc)
    return rc

def plot_corr_matrix_multigauss(rv):
    hfig = plt.figure(figsize = (15,5))
    ax1 = hfig.add_axes([0.1,0.2,0.4,0.7])
    ax1.axis('scaled')
    ax2 = hfig.add_axes([0.5,0.2,0.4,0.7])
    ax2.axis('scaled')

    covmat = rv.covariance
    sns.heatmap(covmat,
            ax = ax1,
            annot=True,
            cbar = False,
            fmt="0.3f",
            cmap="YlGnBu",
            xticklabels=['stretch','tau','offset'],#range(np.shape(dataset)[1]),
            yticklabels=['stretch','tau','offset'])
    ax1.set_title("Covariance matrix")

    corrmat = correlation_from_covariance(covmat)
    sns.heatmap(corrmat,
            ax = ax2,
            annot=True,
            cbar = False,
            fmt="0.3f",
            cmap="YlGnBu",
            xticklabels=['stretch','tau','offset'],#range(np.shape(dataset)[1]),
            yticklabels=['stretch','tau','offset'])
    ax2.set_title("Correlation matrix")


'''Analysis'''
def exp_fit(x, a, k, b):
    return a*np.exp(x*k) + b

def get_example_cmd(exptname,boutstr,sweepdur):
    ''' Format of inputs:
    exptname = '20191218_009'
    boutstr= '[expt.get_bout_win('R','Keyboard')[0],
            expt.get_bout_win('R','Keyboard')[1],
            expt.get_bout_win('N','Keyboard')[0],
            expt.get_bout_win('N','Keyboard')[1]]''
    sweepdur = 0.05'''
    expt = AmpShift_Stable()
    expt.load_expt(exptname, data_folder)
    expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')


    bout = eval(boutstr)
    marker_df = expt.get_marker_table()
    dt = expt.get_dt('lowgain')
    bout_df = expt.filter_marker_df_time(marker_df,bout)

    b_df = expt.filter_marker_df_code(bout_df,['B'])
    u_df = expt.filter_marker_df_code(bout_df,['U'])

    #from uncoupled trials get times for command responses
    cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_df.time.values])

    xtime,sweeps = expt.get_sweepsmat('lowgain',cmd_t,sweepdur)
    cmd_ = np.mean(sweeps,1)-np.mean(sweeps,1)[0]
    
    return xtime, cmd_

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def exclude_HighThreshAff(meta_df):
    x = np.unique(meta_df.ampshift) #array([-40,-30,-20,-10,-5,0,5,10,20,30,40])
    spk_thresh = 12

    n_excluded = 0
    n_included = 0
    expt_excluded = []

    meta_params_df = pd.DataFrame()
    colors = ['blue','orange','brown','green']
    for i,animal in enumerate(np.unique(meta_df['animalID'])):
        animal_df = meta_df[meta_df['animalID'] == animal]
        print(str(animal) + '(' + colors[i] + ')' + ' has a total of ' + str(len(np.unique(animal_df['exptname']))) + ' afferents recorded')

        params_all = []
        sse_all = []
        df_all = []

        for name in np.unique(animal_df['exptname']):
            expt_df = animal_df[animal_df['exptname'] == name]
            try:
                x_data = expt_df.dropna(0).ampshift.values
                y_data =  expt_df.dropna(0).fsl.values
                if len(np.unique(x_data)) > 5 :
                    n_included += 1
                    spk_thresh = np.max(y_data)
                    params, params_covariance = optimize.curve_fit(exp_fit, x_data, y_data, 
                                                                   p0=[1,-0.05, 3],
                                                                   bounds = ([0,-1,1],[np.inf,0,10]),
                                                                   maxfev=1000)
                    params_all.append(params)
                    c = []
                    for x_,y_ in zip(x_data,y_data):
                        yh = exp_fit(x_,params[0],params[1],params[2])
                        s = np.std(y_data[x_data==x_]) #calc std of y measure at each x to adjust chi2
                        c.append(((y_ - yh)**2))

                    sse_all.append(np.sum(c))    
                    df_all.append(len(y_data)-3)
                if len(np.unique(x_data)) <= 5:
                    n_excluded += 1
                    expt_excluded.append(name)
                    params = array([np.NaN,np.NaN,np.NaN])
                    params_all.append(params)
                    sse_all.append(np.NaN)    
                    df_all.append(len(y_data)-3)
            except:
                n_excluded += 1
                expt_excluded.append(name)
                params = array([np.NaN,np.NaN,np.NaN])
                params_all.append(params)
                sse_all.append(np.NaN)    
                df_all.append(len(y_data)-3)
                pass # doing nothing on exception

        exptname = animal_df.groupby('exptname').exptname.describe().top.values
        max_fsl = animal_df.groupby('exptname').fsl.max().values
        a = [p[0] for p in params_all]
        k = [p[1] for p in params_all]
        b = [p[2] for p in params_all]

        meta_params = {
            'exptname' : exptname,
            'animal' : animal,
            'stretch' : a,
            'tau' : k,
            'offset' : b,
            'max_fsl' : max_fsl,
            'sse' : sse_all,
            'df' : df_all
        }
        animal_df = pd.DataFrame(meta_params)
        meta_params_df = meta_params_df.append(animal_df,sort = False,ignore_index = False)


    print('number afferents excluded because first fsl threshold too high or could not be fit: ' + str(n_excluded))
    print('number afferents included (spike thresh at least 0%): ' + str(n_included))
    
    return meta_params_df,expt_excluded

def assess_fits(meta_params_df):
    un_fit = meta_params_df[meta_params_df['sse'].gt(meta_params_df['sse'].quantile(.9))]

    well_fit = meta_params_df[meta_params_df['sse'].lt(meta_params_df['sse'].quantile(.9))]
    plt.figure(figsize = (2,4))
    well_fit.boxplot('sse')
    
    print(un_fit)

    return well_fit,un_fit

def calc_peaks(xtime,sweeps, order, min_peakt,threshold_h,dt):
    R = sweeps
    
    nsamp=int(order/dt) #the window of comparison in nsamp for order; 2msec seemed good
    epsp_ = signal.argrelextrema(R,np.greater_equal,order = nsamp)[0]
    epsp_ = epsp_[np.where((epsp_*dt)>min_peakt)[0]]
    epsp = []
    measure = epsp_
    for i in measure:
        lb = int(min_peakt/dt)
        rb = len(R)-1
        min_height = np.min([abs(R[i]-R[lb]),abs(R[i]-R[rb])])
        if min_height>threshold_h:
            epsp.append(i)
    if len(epsp)>0:
        epsp = np.min(epsp)
    elif len(epsp)==0:
        epsp = np.NaN

    R_filt = R 
    y = signal.medfilt(np.concatenate([[0],np.diff(R_filt)]),[25]) 
    accel = signal.medfilt(np.concatenate([[0],np.diff(y)]),[11])   

    dvdt_start = int(0.002/dt)
    if ~np.isnan([epsp]).any():
        epsp_t = xtime[epsp]
        max_dvdt = np.max(y[dvdt_start:epsp])
        dvdt_threshold = np.max([0.01,0.15*max_dvdt])

        onset_options = np.where((np.sign(y-dvdt_threshold)>0) & (np.sign(accel)>=0))[0]
        valid_onsets = onset_options[(onset_options>dvdt_start)&(onset_options<epsp)]
        if len(valid_onsets) > 0:
            if (epsp_t-(np.min(valid_onsets)*dt)) > 0: #ensure that onset is before peak
                epsp_onset_ind = np.min(valid_onsets) #min after stim artifact
                epsp_amp = R[epsp]-R[epsp_onset_ind]
                epsp_onset = xtime[epsp_onset_ind]
            elif (epsp_t-(np.min(valid_onsets)*dt)) <= 0:
                epsp_t = np.NaN
                epsp_onset = np.NaN
                epsp_amp = 0
        elif len(valid_onsets)==0:
            epsp_t = np.NaN
            epsp_onset = np.NaN
            epsp_amp = 0
    elif np.isnan([epsp]).any():
        epsp_t = np.NaN
        epsp_onset = np.NaN
        epsp_amp = 0    

    return epsp_t, epsp_amp, epsp_onset


def exponential(x, a, k, b):
    return a*np.exp(x*k) + b


def get_afferents_subsampled(meta_df,n_inputs):
    # get spike times for afferent population
    subset = random.choices(np.unique(meta_df['exptname']),k=n_inputs) #divide by 4 if multispike
    multispike = []
    fsl = []
    # get spike times across stimamps
    for exptname in subset:
        expt_df = meta_df[meta_df['exptname']==exptname]
        multispike.append(expt_df.groupby('ampshift').mean()['s0t'].values)
        fsl.append(expt_df.groupby('ampshift').mean()['s0t'].values)
        multispike.append(expt_df.groupby('ampshift').mean()['s1t'].values)
        multispike.append(expt_df.groupby('ampshift').mean()['s2t'].values)
        multispike.append(expt_df.groupby('ampshift').mean()['s3t'].values)
    multispike = np.asarray(multispike).T
    fsl = np.asarray(fsl).T
    
    return multispike, fsl

def generate_y_fit_multispike_gauss(x,n_inputs,dataset,rv):
    #generate y_fit array on each iteration
    stretch_min = np.min(dataset[:,0])
    stretch_max = np.max(dataset[:,0])
    tau_min = np.min(dataset[:,1])
    tau_max = np.max(dataset[:,1])
    offset_min = np.min(dataset[:,2])
    offset_max = np.max(dataset[:,2])
#     max_fsl = np.max(dataset[:,3])
    max_fsl = 11
    
    offset_2 = 1.74 #(these are averages from empirical data in ms from first spikes)
    offset_3 = 3.66
    offset_4 = 6.37
    
    i = 0
    y_fit = []
    fsl = []
    while i <n_inputs:
        [s,t,o] = rv.resample(1)
        while (s<stretch_min)\
        | (s>stretch_max)\
        | (t<tau_min)\
        | (t>tau_max)\
        | (o<offset_min)\
        | (o>offset_max):
            [s,t,o] = rv.resample(1)
        i+=1
        y = exp_fit(x,s,t,o)
        spk_thresh = max_fsl
        y[y>spk_thresh] = np.NaN
        y_fit.append(y)
        fsl.append(y)
        
        y2 = y + offset_2
        y2[y2>spk_thresh] = np.NaN
        y_fit.append(y2)

        y3 = y + offset_3
        y3[y3>spk_thresh] = np.NaN
        y_fit.append(y3)

        y4 = y + offset_4
        y4[y4>spk_thresh] = np.NaN
        y_fit.append(y4)
    
    y_fit = np.asarray(y_fit).T
    fsl = np.asarray(fsl).T
    return y_fit, fsl
    #     y_fit = np.asarray(y_fit).T

def generate_y_fit_multispike_homogisi(meta_df,x,n_inputs,dataset,rv):
    basis, _ = generate_y_fit_multispike_gauss(x,1,dataset,rv)
    # select a subset of real afferent p(spike) to mimic
    subset = random.choices(np.unique(meta_df['exptname']),k=n_inputs) #divide by 4 if multispike
    multispike = []
    # get spike times across stimamps
    for expt_ in subset:
        spkt = meta_df[meta_df['exptname'].isin([expt_])].groupby(['ampshift']).mean()[['s0t','s1t','s2t','s3t']].values
        spk_mask = np.isnan(spkt)
        y = basis.copy()
        y[spk_mask] = np.nan
        multispike.extend(y.T)
    multispike = np.asarray(multispike).T
    
    return multispike
    
def generate_y_fit_multispike_homog(meta_df,x,n_inputs,dataset,rv):
    #generate y_fit array on each iteration
    stretch_static = np.median(dataset[:,0])
    tau_static = np.median(dataset[:,1])
    offset_static = np.median(dataset[:,2])
    max_fsl = 11
    
#     offset_min = np.min(dataset[:,2])
#     offset_max = np.max(dataset[:,2])

    
    offset_2 = 1.74 #(these are averages from empirical data in ms from first spikes)
    offset_3 = 3.66
    offset_4 = 6.37
    
    i = 0
    y_fit = []
    fsl = []
    
    for i in range(n_inputs):
        y = exp_fit(x,stretch_static,tau_static,offset_static)
        spk_thresh = max_fsl
        y[y>spk_thresh] = np.NaN
        y_fit.append(y)
        fsl.append(y)

        y2 = y + offset_2
        y2[y2>spk_thresh] = np.NaN
        y_fit.append(y2)

        y3 = y + offset_3
        y3[y3>spk_thresh] = np.NaN
        y_fit.append(y3)

        y4 = y + offset_4
        y4[y4>spk_thresh] = np.NaN
        y_fit.append(y4)
#     while i <n_inputs:
#         [s,t,o] = rv.resample(1)
#         while (o<offset_min)\
#         | (o>offset_max):
#             [s,t,o] = rv.resample(1)
#         i+=1
#         y = exp_fit(x,stretch_static,tau_static,o)
#         spk_thresh = max_fsl
#         y[y>spk_thresh] = np.NaN
#         y_fit.append(y)
#         fsl.append(y)
        
#         y2 = y + offset_2
#         y2[y2>spk_thresh] = np.NaN
#         y_fit.append(y2)

#         y3 = y + offset_3
#         y3[y3>spk_thresh] = np.NaN
#         y_fit.append(y3)

#         y4 = y + offset_4
#         y4[y4>spk_thresh] = np.NaN
#         y_fit.append(y4)
    
    y_fit = np.asarray(y_fit).T
    fsl = np.asarray(fsl).T
    return y_fit, fsl

def reset_namespace_defaults():
    namespace_sim = {
        'N_inputs' : 4*4, # 4 inputs with 4 possible spikes each
        'N_runs' : 15,
        'sim_dt' : 0.1*ms,
        'duration' : 0.05*second,
        'Cm' : 6*pF,
        'E_l' : -70*mV,
        'g_l' : 1*nS, # a 1MOhm cell has gl = 1*nS
        'E_e' : 0*mV,
        'E_e_lmi' : -80*mV,
        'V_th' : 0*mV,
        'V_r' : -70*mV,
        'tau_r' : 1*ms,
        'w_e' : 0.1*nS,
        'w_e_lmi' : 20*nS, #0,#either on and off... weight by logistic 0*nS,
        'tau_e1' : 4*ms,
        'tau_e2' : 1*ms,
        'tau_e_lmi' : 10*ms,
        'onset_offset' : 0, # 5msec is for figure making because data plotted with 5msec pre-stimonset #4.5, #ms, time of normal stimulus onset relative to cmd
        'e_lmi_delay' : 4, #ms
        'invpeak' : 1
    }
    return namespace_sim

def initialize_model(namespace_sim,meta_params):    
    start_scope()
    ################################################################################
    # Model definition
    ################################################################################

    ### Neurons
    neuron_eq = '''
    dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_e_lmi*(E_e_lmi-v))/Cm    : volt 
    dg_e/dt = (invpeak * s - g_e)/tau_e1 : siemens
    ds/dt = -s/tau_e2 : siemens
    dg_e_lmi/dt = -g_e_lmi/tau_e_lmi  : siemens  # post-synaptic electrosensory inhibitory conductance
    '''
    neurons = NeuronGroup(1, model=neuron_eq, method='euler',
                         name = 'grc',namespace=namespace_sim)
    neurons.v = -70*mV

    # set up monitor of post-synaptic neuron parameters
    state_mon = StateMonitor(neurons, ['v', 'g_e', 'g_e_lmi'], 
                             record=0, name = 'state_mon')

    n_aff = meta_params['N_inputs']
    aff_ind = arange(n_aff)
    aff_t = sorted(np.zeros(n_aff))*ms

    G_e = SpikeGeneratorGroup(n_aff, aff_ind, aff_t, name='aff_input')

    lmi_t = np.zeros(1)*ms

    G_e_lmi = SpikeGeneratorGroup(1, array([0]),lmi_t,name = 'lmi_input')

    syn_e = Synapses(G_e, neurons,on_pre='s += w_e',namespace=namespace_sim,name='aff_syn')
    syn_e_lmi = Synapses(G_e_lmi, neurons, on_pre='g_e_lmi_post += w_e_lmi',
                         delay=meta_params['e_lmi_delay'],namespace=namespace_sim,
                      name='aff_lmi_syn')

    #Now connect the synapses...
    syn_e.connect(i=aff_ind, j = 0)
    syn_e_lmi.connect(i=0, j = 0)

    net = Network(collect())
    
    return net

def run_4_model_conditions(x, n_runs, n_inputs):
    # get a canonical multispike tuning function from fsl generative model and offsets with basic spike threshold at 12ms
#     x = np.asarray([-40,-30,-20,-10,-5,0,5,10,20,30,40])
    
    net = initialize_model(namespace_sim,meta_params) 
    net.store('intialized',filename=sim_filepath) 
    
    dvdt = []
    R_amp = [] # initialize array to store the result of each run
    for _ in arange(n_runs):

        multispike = generate_y_fit_multispike_homogisi(x,n_inputs,dataset,rv) 
        
        R_amp_ = [] # initialize array to store responses across stimulus set
        dvdt_ = []
        for y in multispike:
            net.restore('intialized',filename=sim_filepath) # this resets simulation clock to 0
                # the state of the system at time of restore should affect the network operation accordingly

            y = y[~np.isnan(y)] # remove nan values from array of spike times
            spk_t_aff = np.asarray(y)*ms # create spike time array for SpikeGeneratorGroup, sorted
            ind_aff = np.empty(0) # create default afferent index array in case all were nan

            spk_t_lmi = np.empty(0)*ms # create default lmi spike time in case where no afferent input (all nan)
            ind_lmi = np.empty(0) # create default lmi index array in case all were nan

            if len(y)!=0: # if not all were nan, create index array for afferent spikes, lmi neuron, and lmi spike
                ind_aff = np.arange(len(y))
                ind_lmi = np.arange(1)
                spk_t_lmi = [np.min(y)]*ms

            # update SpikeGeneratorGroup neuron indices and spikes
            net['aff_input'].set_spikes(ind_aff,spk_t_aff)
            net['lmi_input'].set_spikes(ind_lmi,spk_t_lmi)

            net.run(duration = meta_params['duration'])

            r =net['state_mon'].v/mV # set r to voltage trace from simulation trial
            r = r.reshape(-1)
            R_amp_.append(np.max(r)-r[0]) # append response for this stimamp to response mat across all stimamp
            xtime = net['state_mon'].t/ms

            if len(y)!=0:
                dv = r[np.argmax(r)]-r[xtime>(np.min(y))][0]
                dt = xtime[np.argmax(r)]-xtime[xtime>(np.min(y))][0]
            if len(y)==0:
                dv = np.nan
                dt = np.nan
            dvdt_.append(dv/dt)

        R_amp.append((np.asarray(R_amp_).T).flatten()) #append the result from this run across stimamp
        dvdt.append(np.asarray(dvdt_).T) #append the result from this run across stimamp

    R_amp = np.asarray(R_amp).T
    dvdt = np.asarray(dvdt).T
    homog_isi = {'amp':R_amp,'dvdt':dvdt}# = R_amp #np.nanmean(R_amp,1) # append the average across
#     homog_isi = R_amp #np.nanmean(R_amp,1)

    ### subsample from data
    dvdt = []
    R_amp = [] # initialize array to store the result of each run
    for _ in arange(n_runs):

        # get spike times for afferent population
        multispike,fsl = get_afferents_subsampled(n_inputs)

        R_amp_ = [] # initialize array to store responses across stimulus set
        dvdt_ = []
        for y in multispike:
            net.restore('intialized',filename=sim_filepath) # this resets simulation clock to 0
                # the state of the system at time of restore should affect the network operation accordingly

            y = y[~np.isnan(y)] # remove nan values from array of spike times
            spk_t_aff = np.asarray(y)*ms # create spike time array for SpikeGeneratorGroup, sorted
            ind_aff = np.empty(0) # create default afferent index array in case all were nan

            spk_t_lmi = np.empty(0)*ms # create default lmi spike time in case where no afferent input (all nan)
            ind_lmi = np.empty(0) # create default lmi index array in case all were nan

            if len(y)!=0: # if not all were nan, create index array for afferent spikes, lmi neuron, and lmi spike
                ind_aff = np.arange(len(y))
                ind_lmi = np.arange(1)
                spk_t_lmi = [np.min(y)]*ms

            # update SpikeGeneratorGroup neuron indices and spikes
            net['aff_input'].set_spikes(ind_aff,spk_t_aff)
            net['lmi_input'].set_spikes(ind_lmi,spk_t_lmi)

            net.run(duration = meta_params['duration'])

            r =net['state_mon'].v/mV # set r to voltage trace from simulation trial
            r = r.reshape(-1)
            R_amp_.append(np.max(r)-r[0]) # append response for this stimamp to response mat across all stimamp
            xtime = net['state_mon'].t/ms

            if len(y)!=0:
                dv = r[np.argmax(r)]-r[xtime>(np.min(y))][0]
                dt = xtime[np.argmax(r)]-xtime[xtime>(np.min(y))][0]
            if len(y)==0:
                dv = np.nan
                dt = np.nan
            dvdt_.append(dv/dt)

        R_amp.append((np.asarray(R_amp_).T).flatten()) #append the result from this run across stimamp
        dvdt.append(np.asarray(dvdt_).T) #append the result from this run across stimamp

    R_amp = np.asarray(R_amp).T
    dvdt = np.asarray(dvdt).T
    subsamp = {'amp':R_amp,'dvdt':dvdt}# = R_amp #np.nanmean(R_amp,1) # append the average across

    ### heterogeneous tuning
    dvdt = []
    R_amp = [] # initialize array to store the result of each run
    for _ in arange(n_runs):

        # get spike times for afferent population
        multispike,fsl = generate_y_fit_multispike_gauss(x,n_inputs,dataset,rv)

        R_amp_ = [] # initialize array to store responses across stimulus set
        dvdt_ = []
        for y in multispike:
            net.restore('intialized',filename=sim_filepath) # this resets simulation clock to 0
                # the state of the system at time of restore should affect the network operation accordingly

            y = y[~np.isnan(y)] # remove nan values from array of spike times
            spk_t_aff = np.asarray(y)*ms # create spike time array for SpikeGeneratorGroup, sorted
            ind_aff = np.empty(0) # create default afferent index array in case all were nan

            spk_t_lmi = np.empty(0)*ms # create default lmi spike time in case where no afferent input (all nan)
            ind_lmi = np.empty(0) # create default lmi index array in case all were nan

            if len(y)!=0: # if not all were nan, create index array for afferent spikes, lmi neuron, and lmi spike
                ind_aff = np.arange(len(y))
                ind_lmi = np.arange(1)
                spk_t_lmi = [np.min(y)]*ms

            # update SpikeGeneratorGroup neuron indices and spikes
            net['aff_input'].set_spikes(ind_aff,spk_t_aff)
            net['lmi_input'].set_spikes(ind_lmi,spk_t_lmi)

            net.run(duration = meta_params['duration'])

            r =net['state_mon'].v/mV # set r to voltage trace from simulation trial
            r = r.reshape(-1)
            R_amp_.append(np.max(r)-r[0]) # append response for this stimamp to response mat across all stimamp
            xtime = net['state_mon'].t/ms

            if len(y)!=0:
                dv = r[np.argmax(r)]-r[xtime>(np.min(y))][0]
                dt = xtime[np.argmax(r)]-xtime[xtime>(np.min(y))][0]
            if len(y)==0:
                dv = np.nan
                dt = np.nan
            dvdt_.append(dv/dt)

        R_amp.append((np.asarray(R_amp_).T).flatten()) #append the result from this run across stimamp
        dvdt.append(np.asarray(dvdt_).T) #append the result from this run across stimamp

    R_amp = np.asarray(R_amp).T
    dvdt = np.asarray(dvdt).T
    homog_pspike = {'amp':R_amp,'dvdt':dvdt}# = R_amp #np.nanmean(R_amp,1) # append the average across

    R_amp = [] # initialize array to store the result of each run
    dvdt = []
    for _ in arange(n_runs):

        # get spike times for afferent population
        multispike,fsl = generate_y_fit_multispike_homog(x,n_inputs,dataset,rv)

        R_amp_ = [] # initialize array to store responses across stimulus set
        dvdt_ = []
        for y in multispike:
            net.restore('intialized',filename=sim_filepath) # this resets simulation clock to 0
                # the state of the system at time of restore should affect the network operation accordingly

            y = y[~np.isnan(y)] # remove nan values from array of spike times
            spk_t_aff = np.asarray(y)*ms # create spike time array for SpikeGeneratorGroup, sorted
            ind_aff = np.empty(0) # create default afferent index array in case all were nan

            spk_t_lmi = np.empty(0)*ms # create default lmi spike time in case where no afferent input (all nan)
            ind_lmi = np.empty(0) # create default lmi index array in case all were nan

            if len(y)!=0: # if not all were nan, create index array for afferent spikes, lmi neuron, and lmi spike
                ind_aff = np.arange(len(y))
                ind_lmi = np.arange(1)
                spk_t_lmi = [np.min(y)]*ms

            # update SpikeGeneratorGroup neuron indices and spikes
            net['aff_input'].set_spikes(ind_aff,spk_t_aff)
            net['lmi_input'].set_spikes(ind_lmi,spk_t_lmi)

            net.run(duration = meta_params['duration'])

            r =net['state_mon'].v/mV # set r to voltage trace from simulation trial
            r = r.reshape(-1)
            R_amp_.append(np.max(r)-r[0]) # append response for this stimamp to response mat across all stimamp
            xtime = net['state_mon'].t/ms

            if len(y)!=0:
                dv = r[np.argmax(r)]-r[xtime>(np.min(y))][0]
                dt = xtime[np.argmax(r)]-xtime[xtime>(np.min(y))][0]
            if len(y)==0:
                dv = np.nan
                dt = np.nan
            dvdt_.append(dv/dt)

        R_amp.append((np.asarray(R_amp_).T).flatten()) #append the result from this run across stimamp
        dvdt.append(np.asarray(dvdt_).T) #append the result from this run across stimamp

    R_amp = np.asarray(R_amp).T
    dvdt = np.asarray(dvdt).T
    homog_all = {'amp':R_amp,'dvdt':dvdt}# = R_amp #np.nanmean(R_amp,1) # append the average across
#     homog_all = R_amp 
    
    return subsamp, homog_all, homog_isi, homog_pspike