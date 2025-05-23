from scipy.interpolate import interp1d
import numpy as np
from scipy.signal import butter, lfilter, freqz
import scipy.io as sio

class LabData:
    def __init__(self, path = None, snp = 10918, c = 10):
        """Provide the path of the .mat file. 
        c is the criterium to define transition duration in %"""
        
        self.c = c
        self.D = sio.loadmat(path)
        self.Name = path.split('/')[-2]
        cam = self.D['cam'].flatten()
        self.on = len(cam[cam > 4])
        self.fs = 5000
        tf = self.D['tf'].flatten()
        tf_on = tf[:self.on]
        tf_cam = np.zeros(snp)
        self.tf_cam = tf_on[-1] - 1/self.fs*np.arange((snp),0,-1)
        self.index = np.where(tf_on >= tf_cam[0])
        self.tf_rec = tf_on[self.index]
    
    def time(self):
        """"Return physical time when camera was ON"""
        return self.tf_cam
    
    def alpha(self):
        """"Return staging value when camera was ON"""
        alpha_value = (self.D['pilotef']/(self.D['pilotef'] + self.D['TOf'])).flatten() * 100
        alpha_on = alpha_value[:self.on]
        alpha_rec = alpha_on[self.index]
        alpha_interpolation = interp1d(self.tf_rec, alpha_rec, kind='linear', fill_value='extrapolate')
        alpha = alpha_interpolation(self.tf_cam)
        return alpha
    
    def Microphone(self, micro = 'M1'):
        """"Return Microphone signal when camera was ON.
        If micro isn't specify return M1"""
        
        M_value = self.D[micro].flatten()
        M_on = M_value[:self.on]
        M_rec = M_on[self.index]
        M_interpolation = interp1d(self.tf_rec, M_rec, kind='linear', fill_value='extrapolate')
        M = M_interpolation(self.tf_cam)
        return M
    
    

    def Transition(self):
        """"Return Transition point index and the filtered squared microphone signal"""

        def butter_lowpass(cutoff, fs, order=6):
            return butter(order, cutoff, fs=fs, btype='low', analog=False)

        def butter_lowpass_filter(data, cutoff, fs, order=5):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = lfilter(b, a, data)
            return y
        
        fs = 5000       # sample rate, Hz
        cutoff = 162.5
        M1 = self.Microphone('M1')
        normalized_data =  M1/ np.max(abs(M1))
        # filt = uniform_filter(nor_data, size= 500)
        y = butter_lowpass_filter(normalized_data, cutoff, fs, order=6)
        filt = y**2
        nix1 = np.where(filt == np.max(filt))[0]
    
        return [nix1, filt]

    def Transition_duration(self):
        """"Return Transition duration in ms"""

        nix1, filt = self.Transition()
        percentage_maximun  = self.c / 100
        normalized_filter = filt / np.max(abs(filt))
        longitud1 =  np.where(normalized_filter[:nix1[0]] <= percentage_maximun * np.max(normalized_filter))
        longitud2 =  np.where(normalized_filter[nix1[0]:] <= percentage_maximun * np.max(normalized_filter))
        interv1 = longitud1[0][-1]
        interv2 = longitud2[0][0] + nix1[0]
        dt = np.round((self.tf_cam[interv2] -  self.tf_cam[interv1]) * 1000,5)
        
        return [dt, (interv1, interv2)]
    
    def Transition_alpha(self):
        """"Return Staging value at transition point"""
        return np.round(self.alpha()[self.Transition()[0]],4)

    def Transition_type(self):
        """"Return the Transition type V = V - Flame, T = Tulip like - Flame"""
        _, (i1,i2) = self.Transition_duration()
        fr = np.fft.rfft(self.Microphone('M1')[i2:i2+2500])
        Tpe = ['V' if fr[0] > 60000 else 'T'][0]
        if Tpe == 'V':
            self.type = 'V Shape'
        else:
            self.type ='Tulip like Shape' 
        
        print(self.type)        
        return Tpe

    def Transition_labels(self, n = 3):
        """" n is the numbers of labels, 2,3 or 4. 
        if n is 4 will label Tulip like flame as 2 and V flame as 3"""

        n_labels = n
        Label = np.zeros(len(self.tf_cam))
        _, (i1,i2) = self.Transition_duration()
        nix1 = self.Transition()[0][0]

        if n_labels == 2:
            Label[:nix1] = 0
            Label[nix1:] = 1

        
        if n_labels == 3:
            Label[:i1] = 1
            Label[i1:i2] = 0
            Label[i2:] = 2
        
        if n_labels == 4:
            Label[:i1] = 1
            Label[i1:i2] = 0
            if self.Transition_type() == 'T':
                Label[i2:] = 2
            else:
                Label[i2:] = 3

        return Label