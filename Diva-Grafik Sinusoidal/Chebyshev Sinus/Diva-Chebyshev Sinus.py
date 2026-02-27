import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. SETUP PARAMETER SINYAL
fs = 100000  # Frekuensi sampling
t = np.linspace(0, 0.005, int(fs * 0.005), endpoint=False) 
f_sinyal = 1000  # Frekuensi Sinus 1000 Hz
amplitudo = 1

# Membuat Sinyal Sinusoidal Murni
x_clean = amplitudo * np.sin(2 * np.pi * f_sinyal * t)

# Menambahkan Noise (SNR -3dB)
# SNR -3dB berarti daya noise lebih besar daripada daya sinyal
snr_db = -3
p_signal = np.mean(x_clean**2)
p_noise = p_signal / (10**(snr_db / 10))
noise = np.random.normal(0, np.sqrt(p_noise), len(t))
x_noisy = x_clean + noise

# 2. PERANCANGAN FILTER CHEBYSHEV TYPE I (LOW PASS)
cutoff = 1500  # Frekuensi cutoff 1500 Hz
rp = 1         # Passband ripple sebesar 1 dB
order = 4      # Orde filter 4
sos = signal.cheby1(order, rp, cutoff, btype='low', fs=fs, output='sos')
y_filtered = signal.sosfilt(sos, x_noisy)

# 3. VISUALISASI HASIL
plt.figure(figsize=(12, 10))

# Subplot 1: Domain Waktu (Melihat kehalusan sinyal)
plt.subplot(2, 1, 1)
plt.plot(t, x_noisy, color='pink', label='Noisy Sinusoidal (-3dB)', alpha=0.7)
plt.plot(t, y_filtered, color='purple', label='Chebyshev Filtered', linewidth=2)
plt.plot(t, x_clean, 'k--', label='Original Sinusoidal', alpha=0.5)
plt.title("Chebyshev Type I Filter: Time Domain Analysis (Sinusoidal)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Subplot 2: Domain Frekuensi (FFT untuk melihat redaman noise)
plt.subplot(2, 1, 2)
n_fft = 1024
freq = np.fft.fftfreq(n_fft, 1/fs)[:n_fft//2]
fft_noisy = np.abs(np.fft.fft(x_noisy, n_fft)[:n_fft//2])
fft_filtered = np.abs(np.fft.fft(y_filtered, n_fft)[:n_fft//2])

plt.semilogy(freq, fft_noisy, color='pink', label='FFT Noisy')
plt.semilogy(freq, fft_filtered, color='purple', label='FFT Chebyshev')
plt.title("Analisis Spektrum (FFT 1024 Point)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (log)")
plt.xlim(0, 5000)
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.savefig("Diva-Chebyshev/Sinusoidal_Chebyshev.png") # Opsional: simpan otomatis
plt.show()

'''
PENJELASAN GRAFIK (ANALISA):
---------------------------
1. GRAFIK DOMAIN WAKTU (ATAS):
   - Garis Pink (Noisy): Sinyal sinusoidal yang sangat kotor akibat noise dengan daya tinggi (SNR -3dB).
   - Garis Ungu (Filtered): Filter Chebyshev berhasil memulihkan bentuk gelombang sinus dengan efektif. 
     Dibandingkan Butterworth, Chebyshev memiliki transisi yang lebih tajam dalam menyaring noise.
   - Delay/Phase Lag: Terdapat sedikit pergeseran waktu (fase) antara sinyal asli (putus-putus) 
     dan hasil filter, yang merupakan karakteristik umum filter IIR.

2. GRAFIK DOMAIN FREKUENSI / FFT (BAWAH):
   - Puncak di 1000 Hz: Frekuensi utama sinyal sinusoidal berhasil dilewatkan dengan baik.
   - Karakteristik Low Pass: Terlihat penurunan magnitude yang sangat curam setelah frekuensi 
     cutoff 1500 Hz. Chebyshev memberikan "roll-off" yang lebih curam daripada Butterworth.
   - Passband Ripple: Sesuai teori, Chebyshev Type I memiliki riak (ripple) pada area passband (0-1500 Hz), 
     namun hal ini memberikan keuntungan berupa pemotongan noise yang lebih tajam di area stopband.
'''