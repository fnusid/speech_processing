import sounddevice as sd
import scipy.io.wavfile
import numpy as np
from scipy import signal
import pdb
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io.wavfile import write, read
import pdb
def calculate_autocorr(waveform, model_order):
    R=[]
    R.append(np.mean(waveform*waveform))
    for idx in range(1, model_order+1):
        # waveform_duplicate = np.zeros((waveform.shape[0]))
        # waveform_duplicate[idx:] = waveform[:waveform.shape[0]-idx]
        # r= np.dot(waveform, waveform_duplicate)
        # R.append(r)
        r = np.mean(waveform[idx:] * waveform[:-idx])
        R.append(r)
    # breakpoint()
    return R

def record_voice(duration, channels, fs, choose_file,save='True'):
    s = sd.rec(int(duration * fs), samplerate=fs,  
           channels=channels); sd.wait()
    s_int = np.int16(s * 32767)
    if save=='True':
        # Save as WAV file
        write(f"/Users/sidharth/Desktop/speech_processing/audio/{choose_file.split('.wav')[0]}.wav", fs, s_int)
    return s

def plot_mag_spec(original_signal, lpc_signal, fs, choose_file):
    X_f_original = np.fft.fft(original_signal)
    X_f_lpc = np.fft.fft(lpc_signal)
    # f_freqs = np.linspace(0, fs/2, int(n_samples/2))  # Frequency range for positive frequencies
    # w = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
    f_freq_orig = np.fft.fftfreq(len(original_signal), d=1)*fs
    f_freq_lpc = np.fft.fftfreq(len(lpc_signal), d=1)*fs

    # Magnitude spectrum

    # Plotting frequency domain
    plt.figure(figsize=(10, 5))
    plt.plot(f_freq_orig, 20*np.log10(np.abs(X_f_original)), label=f'original fft_waveform of {choose_file} ')
    plt.plot(f_freq_lpc, 20*np.log10(np.abs(X_f_lpc)), color='orange', label=f'lpc waveform of {choose_file} ')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude Spectrum' )
    # plt.xlim(0, fs/2)  # Set x-axis limits to show relevant frequencies
    plt.grid(True)
    plt.legend()
    plt.savefig(f"/Users/sidharth/Desktop/speech_processing/figures/{choose_file.split('.wav')[0]}.jpg")
    plt.show()
 

def lpc_coeff_calculation(waveform, model_order):
    R_vec = []
    toeplitz = np.zeros((model_order, model_order))

    R = calculate_autocorr(waveform, model_order) #0-p

    np.fill_diagonal(toeplitz, R[0]) #substituting the diagonal of the topeplitz matrix with R(0)
    for x in range(model_order):
        for y in range(x+1, model_order-x):
            toeplitz[x,y] = toeplitz[y,x] = R[y]
    R_vec = R[1:].copy() #R(1)--R(p)
    a_vec = scipy.linalg.inv(toeplitz)@np.array(R_vec)
    return a_vec


if __name__ == "__main__":
    fs = 16000  # Sampling frequency
    t = np.linspace(0, 1, fs, endpoint=False)  # Time axis
    f = 5  # Frequency of the sawtooth wave
    periodic_signal = signal.sawtooth(2 * np.pi * f * t)
    write(f"/Users/sidharth/Desktop/speech_processing/audio/prob4-per.wav", fs, periodic_signal)
    # Plot the sawtooth waveform
    plt.plot(t, periodic_signal, label='periodic sigal', color='orange')
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("Periodic signal")
    plt.savefig("/Users/sidharth/Desktop/speech_processing/figures/prob4-per.jpg")
    plt.show()
    

    noise_signal = np.random.normal(0, 1, int(4*16000))
    write(f"/Users/sidharth/Desktop/speech_processing/audio/prob4-noi.wav", fs, noise_signal)
    plt.plot(np.linspace(0, 4, int(4*16000)), noise_signal, label='noise sigal', color='orange')
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("Noise signal")
    plt.savefig("/Users/sidharth/Desktop/speech_processing/figures/prob4-noi.jpg")
    plt.show()

    for choose_file in ['prob4-a.wav','prob4-E.wav','prob4-i.wav','prob4-u.wav', 'prob4-:U.wav']:
        if os.path.exists(f"/Users/sidharth/Desktop/speech_processing/audio/{choose_file}")==False:
            print(f"Recording {choose_file}")
            record_voice(duration=5, channels=1, fs=16000, choose_file=choose_file)
            print("recording done")
        fs, data = read(f"/Users/sidharth/Desktop/speech_processing/audio/{choose_file}")
        window_length = 16000
        hann = np.hanning(window_length)
        start_idx = 40000-int(window_length/2)
        end_index = 40000+int(window_length/2)
        # breakpoint()
        waveform = data[start_idx:end_index]*hann
        lpc_coeff = lpc_coeff_calculation(waveform=waveform, model_order=8)
        print(f"lpc coefficients of {choose_file} are", lpc_coeff)
        b=[1]
        for idx in range(len(lpc_coeff)):
            b.append(-lpc_coeff[idx])
        b=np.asarray(b)
        a = [1]
        z_wav = scipy.signal.lfilter(b, a, data)
        sd.play(z_wav)
        z_wav_int = np.int16(z_wav*32767)
        # if os.path.exists(f"/Users/sidharth/Desktop/speech_processing/audio/{choose_file.split('.wav')[0]}-g.wav")==True:
        write(f"/Users/sidharth/Desktop/speech_processing/audio/{choose_file.split('.wav')[0]}-g.wav",16000 ,z_wav_int )
        for sig_name, sig in {'periodic_signal':periodic_signal, 'noise_signal' :noise_signal }.items():
            z_sig = scipy.signal.lfilter(b, a, sig)
            z_int_sig = np.int16(z_sig*32767)
            if sig_name=='periodic_signal':
                addendum = 'p'
            else: addendum='n'
            # if os.path.exists(f"/Users/sidharth/Desktop/speech_processing/audio/{choose_file.split('.wav')[0]}-{addendum}.wav")==True:
            write(f"/Users/sidharth/Desktop/speech_processing/audio/{choose_file.split('.wav')[0]}-{addendum}.wav",16000 ,z_int_sig )
            
        plot_mag_spec(waveform, z_wav, fs=fs, choose_file=choose_file)




    








