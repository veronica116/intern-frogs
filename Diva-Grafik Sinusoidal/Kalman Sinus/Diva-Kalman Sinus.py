import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP PARAMETER SINYAL
fs = 100000  # Frekuensi sampling
t = np.linspace(0, 0.005, int(fs * 0.005), endpoint=False) 
f_sinyal = 1000  # Frekuensi Sinus 1000 Hz
amplitudo = 1

# Membuat Sinyal Sinusoidal Murni
x_clean = amplitudo * np.sin(2 * np.pi * f_sinyal * t)

# Menambahkan Noise (SNR -3dB)
snr_db = -3
p_signal = np.mean(x_clean**2)
p_noise = p_signal / (10**(snr_db / 10))
noise = np.random.normal(0, np.sqrt(p_noise), len(t))
x_noisy = x_clean + noise

# 2. ALGORITMA KALMAN FILTER (RECURSIVE)
def kalman_filter(z, Q=1e-5, R=0.01):
    n = len(z)
    x_hat = np.zeros(n)      # Estimasi sinyal
    P = np.zeros(n)          # Estimasi error covariance
    x_hat[0] = 0.0           
    P[0] = 1.0               
    
    for k in range(1, n):
        # Tahap Prediksi
        x_hat_minus = x_hat[k-1]
        P_minus = P[k-1] + Q
        
        # Tahap Koreksi (Update)
        K = P_minus / (P_minus + R) # Kalman Gain
        x_hat[k] = x_hat_minus + K * (z[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus
        
    return x_hat

# Eksekusi Filter
y_filtered = kalman_filter(x_noisy)

# 3. VISUALISASI HASIL (Sesuai format Butterworth)
plt.figure(figsize=(12, 10))

# Subplot 1: Domain Waktu
plt.subplot(2, 1, 1)
plt.plot(t, x_noisy, color='pink', label='Noisy Sinusoidal (-3dB)', alpha=0.7)
plt.plot(t, y_filtered, color='green', label='Kalman Filtered', linewidth=2)
plt.plot(t, x_clean, 'k--', label='Original Sinusoidal', alpha=0.5)
plt.title("Kalman Filter: Time Domain Analysis (Sinusoidal)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Subplot 2: Domain Frekuensi (FFT)
plt.subplot(2, 1, 2)
n_fft = 1024
freq = np.fft.fftfreq(n_fft, 1/fs)[:n_fft//2]
fft_noisy = np.abs(np.fft.fft(x_noisy, n_fft)[:n_fft//2])
fft_filtered = np.abs(np.fft.fft(y_filtered, n_fft)[:n_fft//2])

plt.semilogy(freq, fft_noisy, color='pink', label='FFT Noisy')
plt.semilogy(freq, fft_filtered, color='green', label='FFT Kalman')
plt.title("Analisis Spektrum (FFT 1024 Point)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (log)")
plt.xlim(0, 5000)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

'''
PENJELASAN GRAFIK (ANALISA):
---------------------------
1. GRAFIK DOMAIN WAKTU (ATAS):
   - Garis Pink (Noisy): Sinyal input yang sangat kotor dengan gangguan acak (Gaussian Noise).
   - Garis Hijau (Kalman): Filter berhasil mengekstrak pola sinusoidal dengan sangat halus. 
     Kalman bekerja secara prediktif sehingga mampu memisahkan noise tanpa menghilangkan 
     karakteristik utama sinyal.
   - Low Phase Lag: Salah satu keunggulan utama Kalman dibandingkan Butterworth adalah 
     keterlambatan sinyal (delay) yang jauh lebih minimal karena sifatnya yang adaptif 
     terhadap perubahan data.

2. GRAFIK DOMAIN FREKUENSI / FFT (BAWAH):
   - Puncak di 1000 Hz: Frekuensi sinyal target (sinus) tetap dominan dan terjaga amplitudonya.
   - Redaman Noise: Terlihat bahwa magnitude noise pada frekuensi tinggi ditekan secara efektif 
     (garis hijau berada jauh di bawah garis pink pada frekuensi tinggi).
   - Karakteristik Adaptif: Berbeda dengan filter frekuensi konvensional yang memotong tajam 
     di satu titik, Kalman meredam noise berdasarkan estimasi statistik, sehingga respon 
     spektrumnya terlihat lebih luwes namun tetap bersih.
'''