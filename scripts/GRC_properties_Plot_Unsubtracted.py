#!/usr/bin/env python
# coding: utf-8

#%%
print('initializing packages')
from os import chdir

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import seaborn as sns
from scipy import signal

import matplotlib
matplotlib.rcParams.update({'font.size': 22})

sys.path.append('/Users/kperks/mnt/engram/scripts/Python/Analysis/')
from ClassDef_AmplitudeShift_Stable import AmpShift_Stable

#%%
print('changing to data_processed folder and defining folders used in script')
chdir('/Users/kperks/mnt/engram/spikedata/data_processed/')

exptpath = Path.cwd().resolve().parents[0] #assumes running notebook from /data_processed
data_folder = exptpath / 'data_raw' 
figure_folder = exptpath / 'data_processed' / 'Figures_GRC_properties' / 'Unsubtracted_CvsU'
df_folder = exptpath / 'data_processed' / 'Figures_GRC_properties' / 'Unsubtracted_CvsU' / 'df_cmdintact'

#%%
print('setting sweep duration to 50 msec')
sweepdur = 0.045

#%%
print('defining functions')
def calc_peaks (xtime,sweeps, order, min_peakt, t0_offset,threshold_h,dt):
	min_peakt = min_peakt+t0_offset
	R = np.mean(sweeps,1)-np.mean(sweeps,1)[0]

	nsamp=int(order/dt) #the window of comparison in nsamp for order; 2msec seemed good
	ipsp_ = signal.argrelextrema(R,np.less_equal,order = nsamp)[0]
	epsp_ = signal.argrelextrema(R,np.greater_equal,order = nsamp)[0]

	epsp_ = epsp_[np.where((epsp_*dt)>min_peakt)[0]]
	ipsp_ = ipsp_[np.where((ipsp_*dt)>min_peakt)[0]]

	epsp = []
	measure = epsp_
	compare = ipsp_
	for i in measure:
		if len(compare[compare<i])>0:
			lb = np.max(compare[compare<i])
		elif len(compare[compare<i])==0:
			lb = int(min_peakt/dt)
		if len(compare[compare>i])>0:
			rb = np.min(compare[compare>i])
		elif len(compare[compare>i])==0:
			rb = len(R)-1
		min_height = np.min([abs(R[i]-R[lb]),abs(R[i]-R[rb])])
		if min_height>threshold_h:
			epsp.append(i)
	if len(epsp)>0:
		epsp = np.min(epsp)
	elif len(epsp)==0:
		epsp = np.NaN

	ipsp = []
	measure = ipsp_
	compare = epsp_
	for i in measure:
		if len(compare[compare<i])>0:
			lb = np.max(compare[compare<i])
		elif len(compare[compare<i])==0:
			lb = int(min_peakt/dt)
		if len(compare[compare>i])>0:
			rb = np.min(compare[compare>i])
		elif len(compare[compare>i])==0:
			rb = len(R)-1
		min_height = np.min([abs(R[i]-R[lb]),abs(R[i]-R[rb])])
		if min_height>threshold_h:
			ipsp.append(i)
	if len(ipsp)>0:
		ipsp = np.min(ipsp)
	elif len(ipsp)==0:
		ipsp = np.NaN

	R_filt = signal.medfilt(R,[11])
	y = signal.medfilt(np.concatenate([[0],np.diff(R_filt)]),[25]) #-threshold_dvdt
	accel = signal.medfilt(np.concatenate([[0],np.diff(y)]),[11])   

	dvdt_start = int((0.002+t0_offset)/dt)
	if ~np.isnan([epsp]).any():
		epsp_t = xtime[epsp]
		max_dvdt = np.max(y[dvdt_start:epsp])
		dvdt_threshold = np.max([0.01,0.15*max_dvdt])

		onset_options = np.where((np.sign(y-dvdt_threshold)>0) & (np.sign(accel)>=0))[0]
		valid_onsets = onset_options[(onset_options>dvdt_start)&(onset_options<epsp)]
		if len(valid_onsets) > 0:
			if (epsp_t-(np.min(valid_onsets)*dt)*1000) > 0: #ensure that onset is before peak
				epsp_onset_ind = np.min(valid_onsets) #min after stim artifact
				epsp_amp = R[epsp]-R[0] #R[epsp]-R[epsp_onset_ind]
				epsp_onset = xtime[epsp_onset_ind]
				epsp_dvdt = np.max(y[epsp_onset_ind:epsp])
				epsp_dvdt = epsp_dvdt/dt/1000 # convert to mV/msec from mV/sample
			elif (epsp_t-(np.min(valid_onsets)*dt)*1000) <= 0:
				epsp_t = np.NaN
				epsp_onset = np.NaN
				epsp_amp = np.NaN
				epsp_dvdt = np.NaN
		elif len(valid_onsets)==0:
			epsp_t = np.NaN
			epsp_onset = np.NaN
			epsp_amp = np.NaN
			epsp_dvdt = np.NaN
	elif np.isnan([epsp]).any():
		epsp_t = np.NaN
		epsp_onset = np.NaN
		epsp_amp = np.NaN     
		epsp_dvdt = np.NaN

	if ~np.isnan([ipsp]).any():
		ipsp_t = xtime[ipsp]
		max_dvdt = np.min(y[dvdt_start:ipsp])
		dvdt_threshold = np.min([-0.01,0.15*max_dvdt])

		onset_options = np.where((np.sign(y-dvdt_threshold)<0) & (np.sign(accel)<=0))[0]
		valid_onsets = onset_options[(onset_options>dvdt_start)&(onset_options<ipsp)]
		if len(valid_onsets) > 0:
			if (ipsp_t-(np.min(valid_onsets)*dt)*1000) > 0: #ensure that onset is before peak
				ipsp_onset_ind = np.min(valid_onsets) #min after stim artifact
				ipsp_amp = R[ipsp]-R[0] #R[ipsp_onset_ind]
				ipsp_onset = xtime[ipsp_onset_ind]
				ipsp_dvdt = np.min(y[ipsp_onset_ind:ipsp])
				ipsp_dvdt = ipsp_dvdt/dt/1000 # convert to mV/msec from mV/sample
			elif (ipsp_t-(np.min(valid_onsets)*dt)*1000) <= 0:
				ipsp_t = np.NaN
				ipsp_onset = np.NaN
				ipsp_amp = np.NaN
				ipsp_dvdt = np.NaN
		elif len(valid_onsets)==0:
			ipsp_t = np.NaN
			ipsp_onset = np.NaN
			ipsp_amp = np.NaN
			ipsp_dvdt = np.NaN
	elif np.isnan([ipsp]).any():
		ipsp_t = np.NaN
		ipsp_onset = np.NaN
		ipsp_amp = np.NaN
		ipsp_dvdt = np.NaN

	#calculate std of response starting at first psp onset with duration 20ms
	onset_vals = np.array([epsp_onset,ipsp_onset])
	if len(onset_vals[~np.isnan(onset_vals)])==2:
		if epsp<ipsp:
			onset_ind = int(epsp_onset/1000/dt)
			offset_ind = onset_ind + int(20/1000/dt)
			response_std = np.mean(np.std(sweeps[onset_ind:offset_ind,:],1))
			peak_std = np.std(sweeps[epsp]) 
		if ipsp<epsp:
			onset_ind = int(ipsp_onset/1000/dt)
			offset_ind = onset_ind + int(20/1000/dt)
			response_std = np.mean(np.std(sweeps[onset_ind:offset_ind,:],1))
			peak_std = np.std(sweeps[ipsp])       
	elif len(onset_vals[~np.isnan(onset_vals)])==1:
		if (~np.isnan(epsp_onset) & np.isnan(ipsp_onset)):
			onset_ind = int(epsp_onset/1000/dt)
			offset_ind = onset_ind + int(20/1000/dt)
			response_std = np.mean(np.std(sweeps[onset_ind:offset_ind,:],1))
			peak_std = np.std(sweeps[epsp]) 
		if (~np.isnan(ipsp_onset) & np.isnan(epsp_onset)):
			onset_ind = int(ipsp_onset/1000/dt)
			offset_ind = onset_ind + int(20/1000/dt)
			response_std = np.mean(np.std(sweeps[onset_ind:offset_ind,:],1))
			peak_std = np.std(sweeps[ipsp])       
	elif len(onset_vals[~np.isnan(onset_vals)])==0:
		response_std = np.NaN
		peak_std = np.NaN
	#calculate halfwidth of response
	if (~np.isnan(epsp_t) & ~np.isnan(epsp_amp)):
		if R[int(epsp_t/1000/dt)] >= epsp_amp:
			R_shifted = R
		elif R[int(epsp_t/1000/dt)] < epsp_amp: #if epsp peak is negative need to offset response to calc halfwidth
			R_shifted = R - R[int(epsp_t/1000/dt)] + epsp_amp
		if ~np.isnan(epsp_amp):
			rise_options = np.where((np.sign((R_shifted-(epsp_amp/2)))>0) & (np.sign(y)>0))[0]
			valid_rise = rise_options[rise_options>int(2/1000/dt)]
			rise_t = np.min(valid_rise) * dt
			fall_options = np.where((np.sign((R_shifted-(epsp_amp/2)))>0) & (np.sign(y)<0))[0]
			valid_fall = fall_options[fall_options>int(2/1000/dt)]
			fall_t = np.max(valid_fall) * dt
			epsp_hw = (fall_t - rise_t)*1000 #convert to ms
	elif (np.isnan(epsp_t) | np.isnan(epsp_amp)):
		epsp_hw = np.NaN

	return epsp_t, epsp_amp, epsp_onset, ipsp_t, ipsp_amp, ipsp_onset, response_std, peak_std, epsp_hw, epsp_dvdt, ipsp_dvdt

def get_results(expt,cmd_t,u_t,c_t,c_latency,do_plot,ax):
	#get command response
	xtime,sweeps = expt.get_sweepsmat('lowgain',cmd_t,sweepdur)
	cmd_ = np.mean(sweeps,1)-np.mean(sweeps,1)[0]
	# use these cmd trials to get Vm when measuring Estim responses
	Vm_baseline = np.mean(sweeps,1)[0]

	cell_data = {
		'exptname' : exptname,
		'Vm_baseline' : Vm_baseline,
		'c_latency' : c_latency}

	# calculate peak data for cmdR
	min_peakt = 0.003  #(s)
	threshold_h = 0.25 #(mV)
	order = 0.0025 #(s)
	#get command response
	xtime,sweeps = expt.get_sweepsmat('lowgain',cmd_t,sweepdur)
	result = calc_peaks (xtime,sweeps, order, min_peakt,0,threshold_h,dt)
	if do_plot==1:
		plot_peaks_result(ax,xtime,sweeps,result,'purple')
	this_dict = result_to_dict(result,'cmd')
	cell_data.update(this_dict)

	# calculate peak data for uncoupled response
	#get uncoupled response
	xtime,sweeps = expt.get_sweepsmat('lowgain',u_t-c_latency,sweepdur)
	result = calc_peaks (xtime,sweeps, order, min_peakt,c_latency,threshold_h,dt)
	if do_plot==1:
		plot_peaks_result(ax,xtime,sweeps,result,'green')
	this_dict = result_to_dict(result,'u')
	cell_data.update(this_dict)

	# calculate peak data for coupled response
	#need to subtract cmd Response
	#first get command response offset by c_latency
	# xtime,sweeps = expt.get_sweepsmat('lowgain',cmd_t+c_latency,sweepdur)
	# cmd_ = np.mean(sweeps,1)-np.mean(sweeps,1)[0]
	#get coupled response
	xtime,sweeps = expt.get_sweepsmat('lowgain',c_t-c_latency,sweepdur)
	# sweeps = np.asarray([sweep - cmd_ for sweep in sweeps.T]).T
	result = calc_peaks (xtime,sweeps, order, min_peakt,c_latency,threshold_h,dt)
	if do_plot==1:
		plot_peaks_result(ax,xtime,sweeps,result,'orange')
	this_dict = result_to_dict(result,'c')  
	cell_data.update(this_dict)
	mean_c = np.mean(sweeps,1)-np.mean(sweeps,1)[0]

	# calculate peak data for predicted response
	#need to subtract cmd Response
	#first get command response offset by c_latency
	xtime,sweeps = expt.get_sweepsmat('lowgain',cmd_t,sweepdur)
	cmd_ = np.mean(sweeps,1)-np.mean(sweeps,1)[0]
	#get uncoupled response
	xtime,sweeps = expt.get_sweepsmat('lowgain',u_t-c_latency,sweepdur)
	sweeps = np.asarray([sweep + cmd_ for sweep in sweeps.T]).T
	result = calc_peaks (xtime,sweeps, order, min_peakt,c_latency,threshold_h,dt)
	if do_plot==1:
		plot_peaks_result(ax,xtime,sweeps,result,'gray')
	this_dict = result_to_dict(result,'p')  
	cell_data.update(this_dict)
	mean_p = np.mean(sweeps,1)-np.mean(sweeps,1)[0]

	this_dict = {'max_diff_predicted' : np.max(mean_c-mean_p)}
	cell_data.update(this_dict)

	return cell_data

def plot_response(ax,xtime,sweeps,color_r):
	R = np.mean(sweeps,1)-np.mean(sweeps,1)[0]
	ax.plot(xtime,R,color = color_r)

def plot_peaks_result(ax,xtime,sweeps,result,color_r):
	R = np.mean(sweeps,1)-np.mean(sweeps,1)[0]
	epsp_t = result[0],
	epsp_amp = result[1],
	epsp_onset = result[2],
	ipsp_t = result[3],
	ipsp_amp = result[4],
	ipsp_onset = result[5],

	ax.plot(xtime,R,color = color_r)
	ax.vlines(epsp_onset,-2,2,color = 'red',linestyles='dashed')
	ax.vlines(ipsp_onset,-2,2,color = 'blue',linestyles='dashed')
	ax.plot(epsp_t, epsp_amp,"*",color = color_r)
	ax.plot(ipsp_t, ipsp_amp,"^",color = color_r)

def result_to_dict(result, rtype):
	this_dict = {
		rtype + '_epsp_t' : result[0],
		rtype + '_epsp_amp' : result[1],
		rtype + '_epsp_onset' : result[2],
		rtype + '_ipsp_t' : result[3],
		rtype + '_ipsp_amp' : result[4],
		rtype + '_ipsp_onset' : result[5],
		rtype + '_response_std' : result[6],
		rtype + '_peak_std' : result[7],
		rtype + '_epsp_hw' : result[8],
		rtype + '_epsp_dvdt' : result[9],
		rtype + '_ipsp_dvdt' : result[10]
		}
	return this_dict
	
#################################################
#################################################
#%%
print('initializing a figure')
save_plot = 1
fig = plt.figure(num=1)
ax = fig.add_axes([0.1,0.1,0.8,0.8])
# ax.set_visible(True)



# #################################################
# #################################################
# # %%
# ax.cla()
# exptname = '20200606_005'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# #from command only trials get times for command responses
# bout = [expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('R','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# u_t = u_df.time.values

# bout = [expt.get_bout_win('N','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['C'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)
    
# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# # %%
# ax.cla()
# exptname = '20200606_001'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[1],
# 		expt.get_bout_win('N','Keyboard')[1]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# # %%
# ax.cla()
# exptname = '20200607_005'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 		expt.get_bout_win('R','Keyboard')[1],
# 		expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# # %%
# ax.cla()
# exptname = '20200607_004'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 		expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# # %%
# ax.cla()
# exptname = '20200607_002'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 		expt.get_bout_win('R','Keyboard')[1],
# 		expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# # %%
# ax.cla()
# exptname = '20200607_000'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[1],
# 		expt.get_bout_win('R','Keyboard')[2],
# 		expt.get_bout_win('N','Keyboard')[1],
# 		expt.get_bout_win('N','Keyboard')[2]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# # %%
# ax.cla()
# exptname = '20200525_001'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 		expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# # %%
# ax.cla()
# exptname = '20200525_006'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 		expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# # %%
# ax.cla()
# exptname = '20200524_002'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[1],
# 		expt.get_bout_win('N','Keyboard')[1]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# # %%
# ax.cla()
# exptname = '20200312_002'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 		expt.get_bout_win('N','Keyboard')[0],
# 		expt.get_bout_win('N','Keyboard')[1]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# ################################################
# ################################################
# #%%
# ax.cla()
# exptname = '20200227_000'
# print(exptname)

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 	   expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20200226_002'
# print(exptname)

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 	   expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20200225_000' #* had to manually change dvdt_start param in calc_peaks to 0.001
# print(exptname)

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('N','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 	   expt.get_bout_win('N','Keyboard')[0]]
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20200115_002'
# print(exptname)

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 	   expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# ################################################
# ################################################
# #%%
# ax.cla()
# exptname = '20200113_003'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 	   expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20200113_004'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 		expt.get_bout_win('R','Keyboard')[1],
# 		expt.get_bout_win('N','Keyboard')[0],
# 		expt.get_bout_win('N','Keyboard')[1]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20191218_005'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 		expt.get_bout_win('R','Keyboard')[1],
# 		expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20191218_009'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 		expt.get_bout_win('R','Keyboard')[1],
# 		expt.get_bout_win('N','Keyboard')[0],
# 		expt.get_bout_win('N','Keyboard')[1]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20200122_001'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20200115_004'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('B','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('C','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20200312_000'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('N','Keyboard')[1]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20200309_000'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 	   expt.get_bout_win('N','Keyboard')[0],
# 	   expt.get_bout_win('N','Keyboard')[1]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20200122_002'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 	   expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20200121_006'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 	   expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# ################################################
# ################################################
# #%%
# ax.cla()
# exptname = '20200109_004'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')
# # # hyperpol but drifting (also did a bout without bias current)
# # bout = [expt.get_bout_win('R','Keyboard')[2],
# #        expt.get_bout_win('N','Keyboard')[1]]
# # at rest near beginning of expt so see spikes
# bout = [expt.get_bout_win('B','Keyboard')[0],
# 		expt.get_bout_win('B','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('C','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20191218_007'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# bout = [expt.get_bout_win('R','Keyboard')[0],
# 	   expt.get_bout_win('N','Keyboard')[0]]

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')
# bout_df = expt.filter_marker_df_time(marker_df,bout)

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# c_df = expt.filter_marker_df_code(bout_df,['C'])

# u_t = u_df.time.values
# c_t = c_df.time.values
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20180122_001'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# #note with exptname that this a hyperpolarized version of this cell
# cell_data['exptname'] = exptname + '_hyperpolarized'
# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# ############## then do at rest for this cell
# ax.cla()

# bout = [expt.get_bout_win('B','Keyboard')[0],expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# #note with exptname that this a hyperpolarized version of this cell
# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20190325_002'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values

# bout = [expt.get_bout_win('U','Keyboard')[0],expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)
# ################################################
# ################################################
# #%%
# ax.cla()
# exptname = '20171031_004'
# # need to adjust stimulus times by -0.003 because of how event marker detected artifact position

# #first look at hyperpolarized trials
# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[1],expt.get_bout_win('B','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# #note with exptname that this a hyperpolarized version of this cell
# cell_data['exptname'] = exptname + '_hyperpolarized'
# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# ############## then do at rest for this cell
# ###spikes are almost 10mV here!!!! good example to match in vitro
# ax.cla()

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[0],expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# #note with exptname that this a hyperpolarized version of this cell
# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20171010_006'
# # need to adjust stimulus times by -0.003 because of how event marker detected artifact position

# #first look at hyperpolarized trials
# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[0],expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20180130_000'
# # need to adjust stimulus times by -0.003 because of how event marker detected artifact position

# #first look at hyperpolarized trials
# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2],
# 		expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0],
# 		expt.get_bout_win('C','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20180108_004'
# # need to adjust stimulus times by -0.003 because of how event marker detected artifact position

# #first look at hyperpolarized trials
# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[1],expt.get_bout_win('B','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# #note with exptname that this a hyperpolarized version of this cell
# cell_data['exptname'] = exptname + '_hyperpolarized'
# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# ############## then do at rest for this cell
# ###spikes are almost 10mV here!!!! good example to match in vitro
# ax.cla()

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[0],expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = exptname + '.png'
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# #note with exptname that this a hyperpolarized version of this cell
# df = pd.DataFrame(cell_data,index=[0])
# savename = exptname + '.csv'
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20170912_003'
# # need to adjust stimulus times by -0.003 because of how event marker detected artifact position

# #first look at hyperpolarized trials
# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('B','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20170912_005'
# # need to adjust stimulus times by -0.003 because of how event marker detected artifact position

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #######also at rest but later in trace. spikes worse but cmd better
# ax.cla()
# bout = [expt.get_bout_win('U','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[2],
# 		expt.get_bout_win('B','Keyboard')[1],
# 		expt.get_bout_win('B','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# cell_data['exptname'] = exptname + '_bout2'
# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20190107_000'
# # need to adjust stimulus times by -0.003 because of how event marker detected artifact position

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2],
# 		expt.get_bout_win('U','Keyboard')[3],
# 		expt.get_bout_win('U','Keyboard')[4],
# 		expt.get_bout_win('U','Keyboard')[5],
# 		expt.get_bout_win('U','Keyboard')[6]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values

# bout = [expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0],
# 		expt.get_bout_win('C','Keyboard')[1],
# 		expt.get_bout_win('C','Keyboard')[2],
# 		expt.get_bout_win('C','Keyboard')[3],
# 		expt.get_bout_win('C','Keyboard')[4],
# 		expt.get_bout_win('C','Keyboard')[5]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20171027_000'
# # need to adjust stimulus times by -0.003 because of how event marker detected artifact position

# #first look at hyperpolarized trials
# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('B','Keyboard')[0],
# 		expt.get_bout_win('B','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20180103_001'
# # need to adjust stimulus times by -0.003 because of how event marker detected artifact position

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #######also at rest but later in trace. 
# ax.cla()
# bout = [expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# cell_data['exptname'] = exptname + '_bout2'
# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #######then finally hyperpolarized trials. 
# ax.cla()
# bout = [expt.get_bout_win('U','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[2],
# 		expt.get_bout_win('B','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# cell_data['exptname'] = exptname + '_bout2'
# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20190102_000'
# # need to adjust stimulus times by -0.003 because of how event marker detected artifact position

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2],
# 		expt.get_bout_win('U','Keyboard')[3]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values

# bout = [expt.get_bout_win('B','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2],
# 		expt.get_bout_win('U','Keyboard')[3]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0],
# 		expt.get_bout_win('C','Keyboard')[1],
# 		expt.get_bout_win('C','Keyboard')[2],
# 		expt.get_bout_win('C','Keyboard')[3]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# ################################################
# ################################################
# #%%
# ax.cla()
# exptname = '20180103_003'
# # need to adjust stimulus times by -0.0005 because of how event marker detected artifact position

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20190227_001'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[4]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values

# bout = [expt.get_bout_win('U','Keyboard')[4],
# 		expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[3]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20190104_000'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values

# bout = [expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[1],
# 		expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20190128_001'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[2],
# 		expt.get_bout_win('U','Keyboard')[3],
# 		expt.get_bout_win('U','Keyboard')[4]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values

# bout = [expt.get_bout_win('U','Keyboard')[2],
# 		expt.get_bout_win('U','Keyboard')[3],
# 		expt.get_bout_win('U','Keyboard')[4],
# 		expt.get_bout_win('B','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[1],
# 		expt.get_bout_win('C','Keyboard')[2],
# 		expt.get_bout_win('C','Keyboard')[3]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20190312_005'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20171010_002'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20180122_002'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20171010_005'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20181213_002'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0],
# 		expt.get_bout_win('C','Keyboard')[1],
# 		expt.get_bout_win('C','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values 
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20190107_003'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2],
# 		expt.get_bout_win('U','Keyboard')[3]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values 

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2],
# 		expt.get_bout_win('U','Keyboard')[3],
# 		expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0],
# 		expt.get_bout_win('C','Keyboard')[1],
# 		expt.get_bout_win('C','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values 
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20190110_000'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values 

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('U','Keyboard')[2],
# 		expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0],
# 		expt.get_bout_win('C','Keyboard')[1],
# 		expt.get_bout_win('C','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values 
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20200309_001'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('N','Keyboard')[1],
# 		expt.get_bout_win('R','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# u_t = u_df.time.values

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# c_df = expt.filter_marker_df_code(bout_df,['C'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values 
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20200303_000'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('B','Keyboard')[1],
# 		expt.get_bout_win('R','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['U'])
# u_t = u_df.time.values

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# bout = [expt.get_bout_win('R','Keyboard')[0]]
# c_df = expt.filter_marker_df_code(bout_df,['C'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values 
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20180105_000'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values -0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20180112_003'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values  - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%

# ax.cla()
# exptname = '20180108_001'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1],
# 		expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# b_df = expt.filter_marker_df_code(bout_df,['B'])
# #from uncoupled trials get times for command responses
# cmd_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in u_t])

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values  - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%

# ax.cla()
# exptname = '20170206_003'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t_all = u_df.time.values - 0.0003
# #from uncoupled trials get times for command responses
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_t = []
# for u in u_t_all:
# 	cmd_pre=np.NaN
# 	cmd_post=np.NaN
# 	if (len(b_df.time.values[b_df.time.values<u])>0):
# 		cmd_pre = u-np.max(b_df.time.values[b_df.time.values<u])
# 	if (len(b_df.time.values[b_df.time.values>u])>0):
# 		cmd_post = np.min(b_df.time.values[b_df.time.values>u])-u
# 	if ((cmd_pre>0.05) & (cmd_post>0.05)):
# 		u_t.append(u)
# u_t = np.asarray(u_t)

# bout = [expt.get_bout_win('B','Keyboard')[0],
# 		expt.get_bout_win('B','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# #################################################
# #################################################
# #%%

# ax.cla()
# exptname = '20171107_002'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values  - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20171011_001'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('B','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values  - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%

# ax.cla()
# exptname = '20171010_000'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('B','Keyboard')[0],
# 		expt.get_bout_win('B','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values  - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20171010_003'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0],
# 		expt.get_bout_win('U','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('B','Keyboard')[0],
# 		expt.get_bout_win('B','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values  - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# ######## then super hyperpolarized to prevent spiking
# ax.cla()

# bout = [expt.get_bout_win('U','Keyboard')[3]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('U','Keyboard')[3]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[2]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values  - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '_hyperpolarized' + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '_hyperpolarized' + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)

# #################################################
# #################################################
# #%%
# ax.cla()
# exptname = '20170502_002'

# expt = AmpShift_Stable()
# expt.load_expt(exptname, data_folder)
# expt.set_channels('CmdTrig','lowgain','spikes','SIU','DigMark')

# marker_df = expt.get_marker_table()
# dt = expt.get_dt('lowgain')

# bout = [expt.get_bout_win('U','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# u_df = expt.filter_marker_df_code(bout_df,['E'])
# u_t = u_df.time.values - 0.0003

# bout = [expt.get_bout_win('B','Keyboard')[0],
# 		expt.get_bout_win('B','Keyboard')[1]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# cmd_t = b_df.time.values

# bout = [expt.get_bout_win('C','Keyboard')[0]]
# bout_df = expt.filter_marker_df_time(marker_df,bout)
# c_df = expt.filter_marker_df_code(bout_df,['E'])
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# c_t = c_df.time.values  - 0.0003
# #calculate time will need to offset command response by to subtract from coupled estim response
# #use coupled trials
# cmd_coupled_t = np.asarray([np.max(b_df.time.values[b_df.time.values<t]) for t in c_t])
# c_latency = np.median(c_t-cmd_coupled_t)

# cell_data = get_results(expt,cmd_t,u_t,c_t,c_latency,1,ax)

# if save_plot:
# 	savename = cell_data['exptname'] + '.png' #altered name to save figure as
# 	fig.savefig(figure_folder / savename,format = 'png',dpi = 75)

# df = pd.DataFrame(cell_data,index=[0])
# savename = cell_data['exptname'] + '.csv' #altered name to save df as
# df.to_csv(df_folder / savename)
# #################################################
# #################################################
# #%%




# #############
# # %%
# ####### if freerun stim and need to elminate some
# ####### stim trials because too close to cmd
# ###############
# u_t_all = u_df.time.values - 0.0003
# #from uncoupled trials get times for command responses
# b_df = expt.filter_marker_df_code(bout_df,['B'])
# u_t = []
# for u in u_t_all:
# 	cmd_pre=np.NaN
# 	cmd_post=np.NaN
# 	if (len(b_df.time.values[b_df.time.values<u])>0):
# 		cmd_pre = u-np.max(b_df.time.values[b_df.time.values<u])
# 	if (len(b_df.time.values[b_df.time.values>u])>0):
# 		cmd_post = np.min(b_df.time.values[b_df.time.values>u])-u
# 	if ((cmd_pre>0.05) & (cmd_post>0.05)):
# 		u_t.append(u)
# u_t = np.asarray(u_t)