import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. SETUP PARAMETER SINYAL
fs = 100000  # Frekuensi sampling
t = np.linspace(0, 0.005, int(fs * 0.005), endpoint=False) # Durasi 5ms
f_sinyal = 1000  # 1000 Hz
amplitudo = 1

# Membuat Sinyal Sinusoidal Murni
x_clean = amplitudo * np.sin(2 * np.pi * f_sinyal * t)

# Menambahkan Noise (SNR -3dB)
snr_db = -3
p_signal = np.mean(x_clean**2)
p_noise = p_signal / (10**(snr_db / 10))
noise = np.random.normal(0, np.sqrt(p_noise), len(t))
x_noisy = x_clean + noise

# 2. PERANCANGAN FILTER BUTTERWORTH (LOW PASS)
cutoff = 1500  # Frekuensi cutoff
order = 4      # Orde filter
sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
y_filtered = signal.sosfilt(sos, x_noisy)

# 3. VISUALISASI HASIL (TIME DOMAIN)
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(t, x_noisy, color='pink', label='Noisy Sinusoidal (-3dB)', alpha=0.7)
plt.plot(t, y_filtered, color='blue', label='Butterworth Filtered', linewidth=2)
plt.plot(t, x_clean, 'k--', label='Original Sinusoidal', alpha=0.5)
plt.title("Butterworth Filter: Time Domain Analysis (Sinusoidal)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# 4. ANALISIS SPEKTRUM (FFT)
plt.subplot(2, 1, 2)
n_fft = 1024
freq = np.fft.fftfreq(n_fft, 1/fs)[:n_fft//2]
fft_noisy = np.abs(np.fft.fft(x_noisy, n_fft)[:n_fft//2])
fft_filtered = np.abs(np.fft.fft(y_filtered, n_fft)[:n_fft//2])

plt.semilogy(freq, fft_noisy, color='pink', label='FFT Noisy')
plt.semilogy(freq, fft_filtered, color='blue', label='FFT Butterworth')
plt.title("Analisis Spektrum (FFT 1024 Point)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (log)")
plt.xlim(0, 5000)
plt.legend()
plt.grid(True)

plt.tight_layout()
# Simpan hasil untuk GitHub
# plt.savefig("Diva-Butterworth/Sinusoidal_Butterworth.png") 
plt.show()

'''
PENJELASAN GRAFIK (ANALISA):
---------------------------
1. GRAFIK DOMAIN WAKTU (ATAS):
   - Garis Pink (Noisy): Menunjukkan sinyal sinus yang tertutup noise ekstrem (SNR -3dB).
   - Garis Biru (Filtered): Filter Butterworth berhasil memulihkan bentuk sinus murni dengan sangat mulus.
   - Berbeda dengan sinyal kotak, pada sinyal sinus ini tidak terjadi distorsi bentuk yang signifikan karena sinus 
     adalah komponen frekuensi tunggal yang berada di bawah frekuensi cutoff (1500Hz).
   - Terdapat sedikit pergeseran fase (sinyal biru sedikit bergeser ke kanan dibanding garis hitam putus-putus) 
     yang merupakan karakteristik alami dari filter IIR seperti Butterworth.

2. GRAFIK DOMAIN FREKUENSI / FFT (BAWAH):
   - Puncak di 1000 Hz: Menunjukkan frekuensi utama sinyal sinusoidal yang tetap dipertahankan.
   - Karakteristik Low Pass: Magnitude pada frekuensi di atas 1500 Hz (cutoff) ditekan secara drastis (garis biru 
     turun jauh di bawah garis pink).
   - Keunggulan Butterworth: Area passband (0-1500 Hz) terlihat sangat rata (maximally flat), memastikan 
     amplitudo sinyal sinusoidal tetap terjaga tanpa adanya riak (ripple).
'''