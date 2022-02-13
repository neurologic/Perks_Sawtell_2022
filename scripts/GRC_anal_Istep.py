#!/usr/bin/env python
# coding: utf-8

#%%
print('initializing packages')
import platform
import sys
os_name = platform.system()
if os_name == 'Darwin':
    sys.path.append('/Users/kperks/mnt/engram/scripts/Python/Analysis/')
if os_name == 'Linux':
    sys.path.append('/mnt/engram/scripts/Python/Analysis/')
from ClassDef_AmplitudeShift_Stable import AmpShift_Stable

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
import random

import matplotlib
matplotlib.rcParams.update({'font.size': 22})

#%%
# print('changing to data_processed folder and defining folders used in script')
# chdir('/Users/kperks/mnt/engram/spikedata/data_processed/')

exptpath = Path.cwd().resolve().parents[0] #assumes running notebook from /data_processed
data_folder = exptpath / 'data_raw'
figure_folder = exptpath / 'data_processed' / 'Figures_GRC_properties'
df_folder = exptpath / 'data_processed' / 'df_GRC_properties'

cell_list = {
    '20170206_003': [0,1,2],
    '20170502_002': [0],
    '20170912_005': [0],
    '20171010_002': [0],
    '20171010_005': [0],
    '20171011_001': [0],
    '20171027_000': [0],
    '20171031_004': [0],
    '20171107_002': [0],
    '20180103_001': [0],
    '20180108_004': [0],
    '20180122_001': [0],
    '20180122_002': [0],
    '20180130_000': [0],
    '20181213_002': [0],
    '20190107_003': [0],
    '20190227_001': [0],
    '20191218_005': [0],
    '20191218_007': [0],
    '20191218_009': [0],
    '20200113_003': [0],
    '20200113_004': [0],
    '20200225_000': [0],
    '20200226_002': [0,1],
    '20200312_002': [0]
    }
win_list = {
    '20170206_003': [50,250],
    '20170502_002': [50,250],
    '20170912_005': [50,250],
    '20171010_002': [50,250],
    '20171010_005': [50,250],
    '20171011_001': [50,250],
    '20171027_000': [50,250],
    '20171031_004': [50,250],
    '20171107_002': [50,250],
    '20180103_001': [5,55],
    '20180108_004': [5,55],
    '20180122_001': [5,55],
    '20180122_002': [5,55],
    '20180130_000': [5,55],
    '20181213_002': [30,80],
    '20190107_003': [30,80],
    '20190227_001': [30,42],
    '20191218_005': [30,130],
    '20191218_007': [30,130],
    '20191218_009': [30,130],
    '20200113_003': [30,130],
    '20200113_004': [30,130],
    '20200225_000': [30,130],
    '20200226_002': [30,130],
    '20200312_002': [5,55]
    }
scale_factor = {
    '20170206_003': [10000],
    '20170502_002': [1],
    '20170912_005': [1],
    '20171010_002': [1],
    '20171010_005': [1],
    '20171011_001': [1],
    '20171027_000': [1],
    '20171031_004': [1],
    '20171107_002': [1],
    '20180103_001': [1],
    '20180108_004': [1],
    '20180122_001': [1],
    '20180122_002': [1],
    '20180130_000': [1],
    '20181213_002': [1],
    '20190107_003': [1],
    '20190227_001': [1],
    '20191218_005': [10000],
    '20191218_007': [10000],
    '20191218_009': [10000],
    '20200113_003': [10000],
    '20200113_004': [10000],
    '20200225_000': [10000],
    '20200226_002': [1],
    '20200312_002': [1]
    }

# fig1 = plt.figure(figsize=(10,5));
# ax1_0 = fig1.add_axes([0.2,0.2,0.3,0.7])
# ax1_0.margins(0.05, tight=True)
# ax1_1 = fig1.add_axes([0.625,0.3,0.3,0.5])
# ax1_1.axis('equal')

# fig2 = plt.figure(figsize=(5,8));
# ax2_0 = fig2.add_axes([0.3,0.45,0.6,0.3])
# ax2_1 = fig2.add_axes([0.3,0.1,0.6,0.3])

sweepdur = 0.3
subwin = 25

meta_df = pd.read_csv('DF_Istep.csv')#pd.DataFrame()
for exptname,val,win,scale in zip(list(cell_list.keys())[0:14],list(cell_list.values())[0:14],list(win_list.values())[0:14],list(scale_factor.values())[0:14]):
#     exptname = list(cell_list.keys())[exptind]
#     val = list(cell_list.values())[exptind]
    # ax1_0.cla()
    # ax1_1.cla()
    # ax2_0.cla()
    # ax2_1.cla()

    scale = scale[0]

    # set up expt object instance
    expt = AmpShift_Stable()
    expt.load_expt(exptname, data_folder)
    expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')
    marker_df = expt.get_marker_table()
    dt = expt.get_dt('lowgain')

    # get time windows of Isteps (bout = [[start,stop]]) from dictionary index
    s = 'bout = [expt.get_bout_win("I","Keyboard")[' + str(val[0]) + ']'
    if len(val) > 1:
        for i in val[1:]:
            s += ',expt.get_bout_win("I","Keyboard")[' + str(i) + ']'
    s += ']'
    exec(s)

    # use time windows to filter 'trigevt' events that start Istep trials
    event_chan = 'trigevt'
    #first get events
    events = expt.get_events(event_chan)
    trialT = expt.filter_events(events,bout)

    xtime_R,R = expt.get_sweepsmat('lowgain',trialT,sweepdur)
    xtime_I,I = expt.get_sweepsmat('current',trialT,sweepdur)
    
    dt_I = expt.get_dt('current')
    dt_R = expt.get_dt('lowgain')
    current_u = []
    onset_peak = []
    offset_peak = []
    for i,r in zip(I.T,R.T):
        curr_inj = scale * (np.mean(i[int(win[0]/dt_I/1000):int(win[1]/dt_I/1000)])-np.mean(i[0:int(win[0]/dt_I/1000)]))
        current_u.append(curr_inj)
        if curr_inj<0:
            onset_peak.append(np.min(r[int(win[0]/dt_R/1000):int(win[0]/dt_R/1000)+int(subwin/dt_R/1000)])-
                             np.median(r[0:int(win[0]/dt_R/1000)]))
            offset_peak.append(np.min(r[int(win[1]/dt_R/1000)-int(subwin/dt_R/1000):int(win[1]/dt_R/1000)])-
                             np.median(r[0:int(win[0]/dt_R/1000)]))
        if curr_inj>=0:
            onset_peak.append(np.max(r[int(win[0]/dt_R/1000):int(win[0]/dt_R/1000)+int(subwin/dt_R/1000)])-
                             np.median(r[0:int(win[0]/dt_R/1000)]))
            offset_peak.append(np.max(r[int(win[1]/dt_R/1000)-int(subwin/dt_R/1000):int(win[1]/dt_R/1000)])-
                             np.median(r[0:int(win[0]/dt_R/1000)]))

    trial_df = pd.DataFrame({
        'exptname' : exptname,
        'current_inj' : current_u,
        'onset_peak' : onset_peak,
        'offset_peak' : offset_peak
        })
    meta_df = meta_df.append(trial_df,sort = False,ignore_index = False)

meta_df.to_csv('DF_Istep.csv')
    # ax1_0.scatter(np.asarray(current_u),onset_peak)
    # ax1_0.set_ylabel('onset_peak')
    # ax1_0.set_xlabel('current injected')

    # ax1_1.scatter(onset_peak,offset_peak)
    # lim = [np.min([np.min(onset_peak),np.min(offset_peak)]),np.max([np.max(onset_peak),np.max(offset_peak)])]
    # ax1_1.plot(lim, lim)
    # ax1_1.set_ylabel('offset_peak')
    # ax1_1.set_xlabel('onset_peak')

    # plt.figure(fig1.number)
    # plt.savefig(('Figures_GRC_properties/Istep/' + exptname + '_quantified.png'),format='png')
    # plt.savefig(('Figures_GRC_properties/Istep/' + exptname + '_quantified.eps'),format='eps',dpi = 1200)

    
    # ax2_0.plot(xtime_R,R);
    # ax2_1.plot(xtime_I,I*scale);

    # plt.figure(fig2.number)
    # plt.savefig(('Figures_GRC_properties/Istep/' + exptname + '.png'),format='png')
    # plt.savefig(('Figures_GRC_properties/Istep/' + exptname + '.eps'),format='eps',dpi = 1200)