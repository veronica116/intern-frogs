import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. SETUP PARAMETER SINYAL
fs = 100000
t = np.linspace(0, 0.005, int(fs * 0.005), endpoint=False)
f_sinyal = 1000
amplitudo = 1

# Sinyal Bersih & Noisy (-3dB)
x_clean = amplitudo * signal.square(2 * np.pi * f_sinyal * t)
snr_db = -3
p_signal = np.mean(x_clean**2)
p_noise = p_signal / (10**(snr_db / 10))
noise = np.random.normal(0, np.sqrt(p_noise), len(x_clean))
x_noisy = x_clean + noise

# 2. ALGORITMA KALMAN FILTER
def kalman_filter(z, Q, R):
    n = len(z)
    x_hat = np.zeros(n)      # Hasil estimasi
    P = np.zeros(n)          # Error covariance
    x_hat_minus = np.zeros(n) # Prediksi
    P_minus = np.zeros(n)     # Prediksi error
    K = np.zeros(n)          # Kalman Gain

    # Inisialisasi awal
    x_hat[0] = 0
    P[0] = 1.0

    for k in range(1, n):
        # Tahap Prediksi
        x_hat_minus[k] = x_hat[k-1]
        P_minus[k] = P[k-1] + Q

        # Tahap Koreksi (Update)
        K[k] = P_minus[k] / (P_minus[k] + R)
        x_hat[k] = x_hat_minus[k] + K[k] * (z[k] - x_hat_minus[k])
        P[k] = (1 - K[k]) * P_minus[k]
        
    return x_hat

# Parameter Kalman
Q = 1e-4  
R = 0.1   
x_kalman = kalman_filter(x_noisy, Q, R)

# 3. ANALISIS FFT 1024 TITIK
n_fft = 1024
freq = np.fft.fftfreq(n_fft, 1/fs)[:n_fft//2]
mag_noisy = np.abs(np.fft.fft(x_noisy, n_fft)[:n_fft//2])
mag_kalman = np.abs(np.fft.fft(x_kalman, n_fft)[:n_fft//2])

# 4. PLOT HASIL (EDITED)
plt.figure(figsize=(12, 10))

# Time Domain
plt.subplot(2, 1, 1)
plt.plot(t, x_noisy, color='red', alpha=0.3, label='Noisy (-3dB)')
plt.plot(t, x_kalman, color='green', label='Kalman Filtered', linewidth=2)
# --- TAMBAHAN DI SINI ---
plt.plot(t, x_clean, 'k--', alpha=0.7, label='Original Square Wave') 
# ------------------------
plt.title("Kalman Filter: Time Domain Analysis")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)

# Frequency Domain
plt.subplot(2, 1, 2)
plt.semilogy(freq, mag_noisy, color='red', alpha=0.3, label='FFT Noisy')
plt.semilogy(freq, mag_kalman, color='green', label='FFT Kalman')
plt.title("Analisis Spektrum FFT 1024 Point")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (log)")
plt.xlim(0, 5000)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

"""
Penjelasan Grafik
Grafik ini menunjukkan bagaimana filter frekuensi klasik bekerja pada sinyal kotak yang "tenggelam" dalam noise parah (SNR -3dB).
1. Grafik Atas: Time Domain Analysis (Domain Waktu)
- Bentuk sinyal terhadap waktu (0 sampai 0.005 detik).
- Garis Merah (Noisy -3dB): Amplitudo noise seringkali lebih tinggi daripada sinyal aslinya (mencapai angka 4 sementara sinyal asli cuma di angka 1). Bentuk kotak aslinya hampir tidak terlihat sama sekali.
- Garis Hitam Putus-putus (Original Square Wave): Sinyal kotak yang berpindah secara instan antara 1 dan -1 setiap 0.0005 detik (setengah periode dari 1000Hz).
- Kalman Filtered (Hijau Tebal):
    Karakteristik Adaptif: Berbeda dengan Butterworth atau Chebyshev yang membuat sinyal menjadi sangat bulat (seperti sinus), Kalman Filter berusaha mempertahankan bentuk kotak. Perhatikan transisinya yang cenderung lebih cepat naik/turun.
    Keseimbangan (Trade-off): Garis hijau terlihat jauh lebih tenang dibandingkan noise merah, namun tetap memiliki sedikit "gerigi" halus. Ini menandakan Kalman Filter sedang menyeimbangkan antara mempercayai data sensor yang kotor dengan model prediksi internalnya.
    Keakuratan Fase: Sinyal hijau ini sangat sinkron dengan garis putus-putus hitam (minim delay). Ini adalah keunggulan utama Kalman dibandingkan filter IIR yang biasanya menyebabkan pergeseran fase (sinyal bergeser ke kanan).
2. Grafik Bawah: Analisis Spektrum (FFT 1024 Point)
- Puncak Utama (1000 Hz): Terdapat lonjakan magnitude yang sangat jelas pada frekuensi 1000 Hz. Ini membuktikan bahwa Kalman Filter berhasil mengidentifikasi dan meloloskan frekuensi utama dari sinyal kotak.
- Penekanan Noise (Denoising): Garis hijau (FFT Kalman) secara konsisten berada di bawah garis merah (FFT Noisy) di hampir seluruh rentang frekuensi, terutama di atas 1500 Hz. Hal ini menunjukkan bahwa algoritma Kalman secara statistik berhasil membedakan mana lonjakan yang merupakan noise dan mana yang merupakan sinyal, lalu menekan noise tersebut hingga amplitudonya turun signifikan.
- Harmonisa Sinyal: Karena sinyal outputnya masih memiliki bentuk agak "kotak", kamu bisa melihat puncak-puncak kecil tambahan pada harmonisa ganjil (seperti di area dekat 3000 Hz). Filter Kalman berusaha menjaga komponen frekuensi ini agar bentuk kotaknya tidak berubah menjadi sinus sempurna seperti pada Butterworth.
"""