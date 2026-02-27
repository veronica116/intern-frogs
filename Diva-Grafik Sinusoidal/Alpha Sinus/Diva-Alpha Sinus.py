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

# 2. ALGORITMA ALPHA FILTER (LOW PASS DIGITAL)
def alpha_filter(z, alpha=0.05):
    """
    Alpha (0 < alpha < 1) menentukan tingkat penghalusan.
    Semakin kecil alpha, semakin mulus tapi delay semakin besar.
    """
    y = np.zeros(len(z))
    y[0] = z[0]  # Inisialisasi awal
    for k in range(1, len(z)):
        y[k] = alpha * z[k] + (1 - alpha) * y[k-1]
    return y

# Eksekusi Filter dengan alpha 0.05
y_filtered = alpha_filter(x_noisy, alpha=0.05)

# 3. VISUALISASI HASIL (Sesuai format Butterworth)
plt.figure(figsize=(12, 10))

# Subplot 1: Domain Waktu
plt.subplot(2, 1, 1)
plt.plot(t, x_noisy, color='pink', label='Noisy Sinusoidal (-3dB)', alpha=0.7)
plt.plot(t, y_filtered, color='cyan', label='Alpha Filtered (a=0.05)', linewidth=2)
plt.plot(t, x_clean, 'k--', label='Original Sinusoidal', alpha=0.5)
plt.title("Alpha Filter: Time Domain Analysis (Sinusoidal)")
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
plt.semilogy(freq, fft_filtered, color='cyan', label='FFT Alpha')
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
   - Garis Pink (Noisy): Sinyal sinusoidal dengan gangguan noise frekuensi tinggi yang dominan.
   - Garis Cyan (Alpha): Menunjukkan efek penghalusan (smoothing) yang signifikan. Alpha filter bekerja 
     dengan merata-ratakan nilai saat ini dengan nilai sebelumnya secara eksponensial.
   - Phase Lag & Amplitudo: Terlihat pergeseran fase (delay) yang paling besar dibandingkan Butterworth 
     atau Kalman. Selain itu, puncak amplitudo sedikit teredam karena sifat akumulatif filter.

2. GRAFIK DOMAIN FREKUENSI / FFT (BAWAH):
   - Karakteristik Low Pass: Alpha filter bertindak sebagai filter lolos rendah sederhana. Magnitude 
     noise pada frekuensi tinggi berhasil ditekan (garis cyan berada di bawah garis pink).
   - Respon Frekuensi: Tidak seperti Butterworth yang memiliki cutoff tajam, Alpha filter memiliki 
     kemiringan redaman yang lebih landai, sehingga beberapa noise di frekuensi menengah 
     mungkin masih sedikit lolos.
   - Efisiensi: Sangat ringan secara komputasi karena hanya memerlukan satu operasi perkalian dan 
     penjumlahan per sampel data, menjadikannya pilihan utama untuk sistem embedded sederhana.
'''