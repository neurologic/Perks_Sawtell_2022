from neo import Spike2IO
import numpy as np
from scipy import stats
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans


class AmpShift_Stable():
    def _init_(self):
        print('object created')

    def load_expt(self,exptname,data_folder):
        # exptpath = Path.cwd().resolve().parents[0]
        # data_folder = exptpath / 'data_raw'
        # file_to_open = data_folder / exptname[0:exptname.find('_')]/ Path(exptname + '.smr')
        file_to_open = data_folder / Path(exptname + '.smr')
        print(file_to_open)
        bl = Spike2IO(file_to_open.as_posix(),try_signal_grouping=False).read_block()
        seg = bl.segments[0]
        self.seg = seg
        self.exptname = exptname
 
    def set_amps(self,n_amp,amp_array):
        self.n_amp = n_amp
        self.amp_array = amp_array

    def set_channels(self,trigger_chan,vm_chan,spk_chan,siu_chan,marker_chan):
        self.trigger_chan = trigger_chan
        self.vm_chan = vm_chan
        self.spk_chan = spk_chan
        self.siu_chan = siu_chan
        self.marker_chan = marker_chan

    def get_marker_table(self):
        markerT = np.asarray(self.seg.events[[s.name for s in self.seg.events].index(self.marker_chan)].magnitude)
        # markerC = np.asarray([int(s.decode("utf-8")) for s in self.seg.events[[s.name for s in self.seg.events].index(self.marker_chan)].labels]);
        markerC = np.asarray([int(s) for s in self.seg.events[[s.name for s in self.seg.events].index(self.marker_chan)].labels]);
        markerC = np.asarray([chr(int(hex(n),16)) for n in markerC])
        marker_df = pd.DataFrame({
            'time' : markerT,
            'code' : markerC,
        })
        
        return marker_df

    def get_events(self,channame):
        events = np.asarray(self.seg.events[[s.name for s in self.seg.events].index(channame)].magnitude)
        return events

    def get_signal(self,channame):
        signal = self.seg.analogsignals[[s.name for s in self.seg.analogsignals].index(channame)]
        return signal

    def get_bout_win(self,boutstr,marker_chan):
        codeS = np.asarray([int(s) for s in self.seg.events[[s.name for s in self.seg.events].index(marker_chan)].get_labels()])
        codeT = np.asarray(self.seg.events[[s.name for s in self.seg.events].index(marker_chan)].magnitude)
        stopstr = 'S'
        startcode = ord(boutstr)
        stopcode = ord(stopstr)
        alloff = codeT[codeS==stopcode]
        bouton = codeT[codeS == startcode]
        boutoff = [np.min(alloff[alloff>t]) for t in bouton]
        bout_win = [[o,f] for o,f in zip(bouton,boutoff)]
        
        return bout_win

    def filter_marker_df_time(self,marker_df,time_win):
        sub_df = marker_df[(marker_df['time']>time_win[0][0]) & (marker_df['time']<time_win[0][1])]
        if np.shape(time_win)[0]>1:
            for t in time_win[1:]:
                sub_df = pd.concat([sub_df,marker_df[(marker_df['time']>t[0]) & (marker_df['time']<t[1])]])

        return sub_df
    
    def filter_marker_df_code(self,marker_df,codestr):
        sub_df = marker_df[marker_df['code'].str.match(codestr[0])]
        if np.shape(codestr)[0]>1:
            for c in codestr[1:]:
                sub_df = pd.concat([sub_df,marker_df[sub_df['code'].str.match(c)]])

        return sub_df

    def filter_events(self,events,time_win):
        these_events = []
        for sublist in np.asarray([events[np.where((events>win[0])&(events<win[1]))] for win in time_win]):
            for item in sublist:
                these_events.append(item)
        these_events = np.asarray(these_events)
        
        return these_events

    def get_sweepsmat(self,signal_chan,times,sweepdur):
        thischan = self.seg.analogsignals[[s.name for s in self.seg.analogsignals].index(signal_chan)]
        sweepsmat = []
        v = thischan.magnitude
        dt = float(thischan.sampling_period)
        nsamp = int(sweepdur/dt)
        for t in times:
            inds = [int(t/dt),int(t/dt)+nsamp]
            sweepsmat.append(v[inds[0]:inds[1]].flatten())
        sweepsmat = np.asarray(sweepsmat).T
        
        xtime = np.linspace(0,sweepdur,int(sweepdur/dt))*1000
        
        return xtime,sweepsmat
    
    def get_dt(self,signal_chan):
        thischan = self.seg.analogsignals[[s.name for s in self.seg.analogsignals].index(signal_chan)]
        dt = float(thischan.sampling_period)
        
        return dt

    def cluster_event_Amp(self,event_Amp,event_0_Amp):
        ampshift = np.asarray([((A/event_0_Amp)*100)-100 for A in event_Amp]).reshape(-1, 1)
        km = KMeans(n_clusters=self.n_amp)
        p = km.fit_predict(ampshift)
        c = [_c[0] for _c in km.cluster_centers_]
        for i,v in zip(np.argsort(c),self.amp_array):
            c[i] = v
        ampshift_round = [c[v] for v in p]

        return ampshift, ampshift_round