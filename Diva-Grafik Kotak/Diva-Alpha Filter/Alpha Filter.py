import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. SETUP PARAMETER (Sama biar konsisten)
fs = 100000
t = np.linspace(0, 0.005, int(fs * 0.005), endpoint=False)
f_sinyal = 1000
amplitudo = 1

x_clean = amplitudo * signal.square(2 * np.pi * f_sinyal * t)
snr_db = -3
p_signal = np.mean(x_clean**2)
p_noise = p_signal / (10**(snr_db / 10))
x_noisy = x_clean + np.random.normal(0, np.sqrt(p_noise), len(x_clean))

# 2. ALGORITMA ALPHA FILTER (Simple Exponential Smoothing)
def alpha_filter(z, alpha):
    """
    z: sinyal kotor
    alpha: faktor smoothing (0 sampai 1)
           Semakin kecil alpha, semakin bersih tapi delay makin parah.
    """
    n = len(z)
    x_hat = np.zeros(n)
    x_hat[0] = z[0] # Inisialisasi awal
    
    for k in range(1, n):
        # Rumus: Estimasi = (1 - alpha) * Estimasi_Lama + alpha * Data_Baru
        x_hat[k] = (1 - alpha) * x_hat[k-1] + alpha * z[k]
        
    return x_hat

# Eksekusi (Coba alpha kecil untuk noise berat -3dB)
alpha_val = 0.05 
x_alpha = alpha_filter(x_noisy, alpha_val)

# 3. FFT 1024
n_fft = 1024
freq = np.fft.fftfreq(n_fft, 1/fs)[:n_fft//2]
mag_noisy = np.abs(np.fft.fft(x_noisy, n_fft)[:n_fft//2])
mag_alpha = np.abs(np.fft.fft(x_alpha, n_fft)[:n_fft//2])

# 4. PLOTTING
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(t, x_noisy, color='red', alpha=0.3, label='Noisy (-3dB)')
plt.plot(t, x_alpha, color='cyan', label=f'Alpha Filter (a={alpha_val})', linewidth=2)
plt.plot(t, x_clean, 'k--', alpha=0.7, label='Original Square Wave')
plt.title("Alpha Filter (Single Exponential Smoothing): Time Domain")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogy(freq, mag_noisy, color='red', alpha=0.3, label='FFT Noisy')
plt.semilogy(freq, mag_alpha, color='cyan', label='FFT Alpha')
plt.title("FFT Analysis 1024 Point")
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
- Alpha Filter (Sian/Biru Muda):
    Efek Smoothing: Filter ini berhasil meredam noise menjadi garis yang jauh lebih halus.
    Kelemahan Fase (Lag): Terlihat jelas adanya keterlambatan posisi sinyal sian dibandingkan garis putus-putus hitam. Saat sinyal asli sudah naik ke 1, sinyal Alpha masih merangkak naik perlahan.
    Distorsi Bentuk: Karena hanya menggunakan satu parameter (alpha = 0.05), filter ini tidak mampu mengikuti perubahan instan sinyal kotak. Hasilnya, bentuk kotak berubah menjadi deretan kurva pengisian/pengosongan kapasitor (eksponensial) yang tumpul.
2. Grafik Bawah: Analisis Spektrum (FFT 1024 Point)
- Puncak Utama (1000 Hz): Terdapat puncak magnitudo di frekuensi 1000 Hz yang menandakan frekuensi dasar sinyal kotak tetap terjaga, meskipun kekuatannya sedikit teredam.
- Penekanan Noise: Garis sian (FFT Alpha) secara konsisten berada di bawah garis merah (FFT Noisy). Ini membuktikan bahwa filter ini efektif sebagai Low Pass Filter sederhana yang membuang komponen frekuensi tinggi (noise).
- Karakteristik Penurunan: Berbeda dengan Butterworth atau Chebyshev yang memiliki potongan tajam, penurunan magnitude pada Alpha filter cenderung lebih landai di frekuensi tinggi.
"""