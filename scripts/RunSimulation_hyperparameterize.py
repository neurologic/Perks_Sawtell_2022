############
'''import packages'''

from tqdm import tqdm
import h5py
import tables as tb
import platform
import sys
os_name = platform.system()
if os_name == 'Darwin':
    sys.path.append('/Users/kperks/mnt/engram_share/home/kep2142/scripts/Python/Analysis/')
if os_name == 'Linux':
    sys.path.append('/mnt/engram/scripts/Python/Analysis/')
from ClassDef_AmplitudeShift_Stable import AmpShift_Stable
from Plotting import figsave, create_fig, create_fig_tuning, set_fig_style

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

from sklearn.linear_model import LinearRegression
from sklearn import datasets,linear_model
# from sklearn.cross_validation import train_test_split

import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42


############
'''set up parameters and filepaths'''

rootpath = Path('/Users/kperks/mnt/')
savepath = rootpath / 'engram_share/locker/GranularCellPaperResources/Draft_For_Submission/Code/Results'
exptpath = rootpath / 'engram_share/home/kep2142/spikedata'
data_folder = exptpath / 'data_raw'
df_folder = exptpath / 'data_processed' / 'Figures_GRC_properties' / 'Unsubtracted_CvsU' / 'df_cmdintact'
meta_data_folder = exptpath / 'data_processed' / 'GRC_properties_Meta'
figure_folder = Path('/Users/kperks/mnt/engram_share') / 'locker' / 'GranularCellPaperResources' / 'Figure_RawEPScomponents'

#for storing simulation states
sim_filename = 'grc_model_init.pickle'
sim_filepath = savepath / sim_filename 

n_runs = 20
n_inputs_list = [4,5,6,7]
x = np.asarray([-40,-30,-20,-10,-5,0,5,10,20,30,40])
sim_dt = 0.1
# xtime = np.linspace(0,50,int(0.05/sim_dt*1000))

meta_params = {
    'N_inputs' : 4*4, # 7 inputs with 4 possible spikes each
    'N_runs' : n_runs,
    'duration' : 0.05*second,
    'onset_offset' : 0, # 5msec is for figure making because data plotted with 5msec pre-stimonset #4.5,
    'tau_e1' : 4*ms,
    'tau_e2' : 1*ms,#ms, time of normal stimulus onset relative to cmd
    'e_lmi_delay' : 4*ms #ms
}

invpeak = (meta_params['tau_e2'] / meta_params['tau_e1']) ** \
        (meta_params['tau_e1'] / (meta_params['tau_e2'] - meta_params['tau_e1']))

namespace_sim = {
    'sim_dt' : 0.1*ms,
    'Cm' : 12*pF,
    'E_l' : -70*mV,
    'g_l' : 1*nS, # a 1MOhm cell has gl = 1*nS
    'E_e' : 0*mV,
    'E_e_lmi' : -90*mV,
    'V_th' : 0*mV,
    'V_r' : -70*mV,
    'w_e' : 0.1*nS,
    'w_e_lmi' : 0*nS, #0*nS,##0,#either on and off... weight by logistic 0*nS,
    'tau_e1' : 4*ms,
    'tau_e2' : 1*ms,
    'tau_e_lmi' : 5*ms,
    'invpeak' : invpeak
}


############
'''Define Functions'''

def exp_fit(x, a, k, b):
    return a*np.exp(x*k) + b

def exclude_HighThreshAff(meta_df):
    x = np.unique(meta_df.ampshift) #array([-40,-30,-20,-10,-5,0,5,10,20,30,40])
    spk_thresh = 11

    n_excluded = 0
    n_included = 0
    expt_excluded = []

    meta_params_df = pd.DataFrame()
    colors = ['blue','orange','brown','green']
    for i,animal in enumerate(np.unique(meta_df['animalID'])):
        animal_df = meta_df[meta_df['animalID'] == animal]
        # print(str(animal) + '(' + colors[i] + ')' + ' has a total of ' + str(len(np.unique(animal_df['exptname']))) + ' afferents recorded')

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


    # print('number afferents excluded because first fsl threshold too high or could not be fit: ' + str(n_excluded))
    # print('number afferents included (spike thresh at least 0%): ' + str(n_included))
    # plt.ylim(1.8,15)
    # plt.legend(bbox_to_anchor=(1.1, 1.05))
    
    return meta_params_df,expt_excluded

def assess_fits(meta_params_df):
    un_fit = meta_params_df[meta_params_df['sse'].gt(meta_params_df['sse'].quantile(.9))]

    well_fit = meta_params_df[meta_params_df['sse'].lt(meta_params_df['sse'].quantile(.9))]
    # plt.figure(figsize = (2,4))
    # well_fit.boxplot('sse')
    print('unfit afferents')
    print(un_fit)

    return well_fit,un_fit
        
def get_afferents_subsampled(n_inputs):
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

def generate_y_fit_multispike_homogisi(x,n_inputs,dataset,rv):
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
    
def generate_y_fit_multispike_homog(x,n_inputs,dataset,rv):
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
        
    y_fit = np.asarray(y_fit).T
    fsl = np.asarray(fsl).T
    return y_fit, fsl


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
    neurons = NeuronGroup(1, model=neuron_eq, method='euler', name = 'grc', namespace=namespace_sim)
    neurons.v = -70*mV

    # set up monitor of post-synaptic neuron parameters
    state_mon = StateMonitor(neurons, ['v'],record=0, name = 'state_mon')

    n_aff = meta_params['N_inputs']
    aff_ind = arange(n_aff)
    aff_t = sorted(np.zeros(n_aff))*ms

    G_e = SpikeGeneratorGroup(n_aff, aff_ind, aff_t, name='aff_input')

    lmi_t = np.zeros(1)*ms

    G_e_lmi = SpikeGeneratorGroup(1, array([0]),lmi_t,name = 'lmi_input')

    syn_e = Synapses(G_e, neurons,on_pre='s += w_e',namespace=namespace_sim,name='aff_syn')
    syn_e_lmi = Synapses(G_e_lmi, neurons, on_pre='g_e_lmi_post += w_e_lmi',delay=meta_params['e_lmi_delay'],namespace=namespace_sim,name='aff_lmi_syn')

    #Now connect the synapses...
    syn_e.connect(i=aff_ind, j = 0)
    syn_e_lmi.connect(i=0, j = 0)

    net = Network(collect())
    
    return net


############
'''MAIN'''


exptdate = 'all'
# meta_df.to_csv('DF_AfferentPopulation_' + exptdate + '.csv')
meta_df = pd.read_csv('/Users/kperks/mnt/engram_share/home/kep2142/spikedata/data_processed/DF_AfferentPopulation_all.csv')
# meta_df = pd.read_csv('DF_AfferentPopulation_' + exptdate + '.csv')
meta_params_df,expt_excluded = exclude_HighThreshAff(meta_df)
well_fit,un_fit = assess_fits(meta_params_df)
data_df = well_fit
dataset = list(zip(
    data_df['stretch'],
    data_df['tau'],
    data_df['offset'],
    data_df['max_fsl']
    ))
dataset = np.asarray(dataset)
rv = gaussian_kde(dataset[:,0:3].T)
max_fsl_global = np.max(meta_params_df['max_fsl'])
filtered_df = meta_df[meta_df.exptname.str.match('|'.join(well_fit.exptname.values))]

meta_df = pd.read_csv('/Users/kperks/mnt/engram_share/home/kep2142/spikedata/data_processed/DF_Afferent_MultiSpikePop_ALL.csv')

for a in n_inputs_list:
    filename = 'simulation_results_12pF_'+ str(a) + 'inputs.h5'
    model_h5_resultspath = savepath / filename

    meta_params['N_inputs'] = a*4

    with h5py.File(model_h5_resultspath, 'a') as h5file:
        group = h5file.require_group('results')
        group = h5file.require_group('metadata')
        group.create_dataset('stimamp',data=x)
        # group.create_dataset('xtime',data=xtime) # get this from subsamp simulation
        group.create_dataset('n_inputs',data=a)
        group.create_dataset('n_runs',data=n_runs)

    ############# def run_4_model_conditions(x, n_runs, n_inputs):
    # get a canonical multispike tuning function from fsl generative model and offsets with basic spike threshold at 12ms

    net = initialize_model(namespace_sim,meta_params) 
    net.store('intialized',filename=sim_filepath) 

    ### subsample from data
    R_wavs = []
    onset = []
    print('doing subsampled simulation')
    for _ in tqdm(range(n_runs)):
        
        # get spike times for afferent population
        multispike,fsl = get_afferents_subsampled(a)

        R_wavs_ = [] # initialize array to store responses across stimulus set
        onset_ = []
        for y in multispike:
            miny = np.NaN
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
                miny = np.min(y)

            # update SpikeGeneratorGroup neuron indices and spikes
            net['aff_input'].set_spikes(ind_aff,spk_t_aff)
            net['lmi_input'].set_spikes(ind_lmi,spk_t_lmi)

            net.run(duration = meta_params['duration'])

            r =net['state_mon'].v/mV # set r to voltage trace from simulation trial
            xtime = net['state_mon'].t/ms
            R_wavs_.append(r.reshape(-1)) # append response for this stimamp to response mat across all stimamp
            onset_.append(miny)

        R_wavs.append(np.asarray(R_wavs_).T) #append the result from this run across stimamp
        onset.append(np.asarray(onset_).T)

    R_wavs = np.asarray(R_wavs).T
    onset = np.asarray(onset).T

        
    with h5py.File(model_h5_resultspath, 'a') as h5file:
        group = h5file.get('metadata')
        group.create_dataset('xtime',data=xtime)
        group = h5file.get('results')
        group.create_dataset('subsamp',data = R_wavs)
        group.create_dataset('subsamp_onset',data = onset)

    # homogeneous isi
    #     multispike = generate_y_fit_multispike_homogisi(x,n_inputs,dataset,rv) 
    R_wavs = []
    onset = []
    print('doing homog_isi simulation')
    for _ in tqdm(range(n_runs)):
        
        # get spike times for afferent population
        multispike = generate_y_fit_multispike_homogisi(x,a,dataset,rv) 

        R_wavs_ = [] # initialize array to store responses across stimulus set
        onset_ = []
        for y in multispike:
            miny = np.NaN
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
                miny = np.min(y)

            # update SpikeGeneratorGroup neuron indices and spikes
            net['aff_input'].set_spikes(ind_aff,spk_t_aff)
            net['lmi_input'].set_spikes(ind_lmi,spk_t_lmi)

            net.run(duration = meta_params['duration'])

            r =net['state_mon'].v/mV # set r to voltage trace from simulation trial
            R_wavs_.append(r.reshape(-1)) # append response for this stimamp to response mat across all stimamp
            onset_.append(miny)

        R_wavs.append(np.asarray(R_wavs_).T) #append the result from this run across stimamp
        onset.append(np.asarray(onset_).T)

    R_wavs = np.asarray(R_wavs).T
    onset = np.asarray(onset).T

    with h5py.File(model_h5_resultspath, 'a') as h5file:
        group = h5file.get('results')
        group.create_dataset('homog_isi',data = R_wavs)
        group.create_dataset('homog_isi_onset',data = onset)


    # homog pspike
#         multispike,fsl = generate_y_fit_multispike_gauss(x,n_inputs,dataset,rv)
    R_wavs = []
    onset = []
    print('doing homog_pspike simulation')
    for _ in tqdm(range(n_runs)):
        
        # get spike times for afferent population
        multispike,fsl = generate_y_fit_multispike_gauss(x,a,dataset,rv)

        R_wavs_ = [] # initialize array to store responses across stimulus set
        onset_ = []
        for y in multispike:
            miny = np.NaN
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
                miny = np.min(y)

            # update SpikeGeneratorGroup neuron indices and spikes
            net['aff_input'].set_spikes(ind_aff,spk_t_aff)
            net['lmi_input'].set_spikes(ind_lmi,spk_t_lmi)

            net.run(duration = meta_params['duration'])

            r =net['state_mon'].v/mV # set r to voltage trace from simulation trial
            R_wavs_.append(r.reshape(-1)) # append response for this stimamp to response mat across all stimamp
            onset_.append(miny)

        R_wavs.append(np.asarray(R_wavs_).T) #append the result from this run across stimamp
        onset.append(np.asarray(onset_).T)

    R_wavs = np.asarray(R_wavs).T
    onset = np.asarray(onset).T

    with h5py.File(model_h5_resultspath, 'a') as h5file:
        group = h5file.get('results')
        group.create_dataset('homog_pspike',data = R_wavs)
        group.create_dataset('homog_pspike_onset',data = onset)


    # homog all
#          multispike,fsl = generate_y_fit_multispike_homog(x,n_inputs,dataset,rv)
    R_wavs = []
    onset = []
    print('doing homog_all simulation')
    for _ in tqdm(range(n_runs)):
        
        # get spike times for afferent population
        multispike,fsl = generate_y_fit_multispike_homog(x,a,dataset,rv)

        R_wavs_ = [] # initialize array to store responses across stimulus set
        onset_ = []
        for y in multispike:
            miny = np.NaN
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
                miny = np.min(y)

            # update SpikeGeneratorGroup neuron indices and spikes
            net['aff_input'].set_spikes(ind_aff,spk_t_aff)
            net['lmi_input'].set_spikes(ind_lmi,spk_t_lmi)

            net.run(duration = meta_params['duration'])

            r =net['state_mon'].v/mV # set r to voltage trace from simulation trial
            R_wavs_.append(r.reshape(-1)) # append response for this stimamp to response mat across all stimamp
            onset_.append(miny)

        R_wavs.append(np.asarray(R_wavs_).T) #append the result from this run across stimamp
        onset.append(np.asarray(onset_).T)

    R_wavs = np.asarray(R_wavs).T
    onset = np.asarray(onset).T

    with h5py.File(model_h5_resultspath, 'a') as h5file:
        group = h5file.get('results')
        group.create_dataset('homog_all',data = R_wavs)
        group.create_dataset('homog_all_onset',data = onset)

