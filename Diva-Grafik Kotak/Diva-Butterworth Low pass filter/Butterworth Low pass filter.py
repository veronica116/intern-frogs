import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. SETUP PARAMETER SINYAL
fs = 100000              
t = np.linspace(0, 0.005, int(fs * 0.005), endpoint=False) 
f_sinyal = 1000          
amplitudo = 1

# Sinyal Kotak
x_clean = amplitudo * signal.square(2 * np.pi * f_sinyal * t)

# Tambah Noise SNR -3dB
snr_db = -3
p_signal = np.mean(x_clean**2)
p_noise = p_signal / (10**(snr_db / 10))
noise = np.random.normal(0, np.sqrt(p_noise), len(x_clean))
x_noisy = x_clean + noise

# 2. FILTER BUTTERWORTH
cutoff = 1500 
order = 4 

# Membuat filter (SOS format lebih stabil untuk order tinggi)
sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
x_filtered = signal.sosfilt(sos, x_noisy)

# 3. ANALISIS FFT 1024 TITIK (zooming)
def get_fft(data, fs, n=1024):
    yf = np.fft.fft(data, n)
    xf = np.fft.fftfreq(n, 1/fs)
    return xf[:n//2], np.abs(yf[:n//2])

freq, mag_noisy = get_fft(x_noisy, fs)
_, mag_filtered = get_fft(x_filtered, fs)

# 4. PLOT HASIL (EDITED)
plt.figure(figsize=(12, 10))

# Grafik Waktu (Time Domain)
plt.subplot(2, 1, 1)
plt.plot(t, x_noisy, color='red', alpha=0.3, label='Noisy (-3dB)')
plt.plot(t, x_filtered, color='blue', label='Butterworth Filtered', linewidth=2)
# --- TAMBAHAN DI SINI ---
plt.plot(t, x_clean, 'k--', alpha=0.7, label='Original Square Wave') 
# ------------------------
plt.title("Butterworth Low Pass Filter: Time Domain Analysis")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)

# Grafik Frekuensi (Frequency Domain)
plt.subplot(2, 1, 2)
plt.semilogy(freq, mag_noisy, color='red', alpha=0.3, label='FFT Noisy')
plt.semilogy(freq, mag_filtered, color='blue', label='FFT Filtered')
plt.title("Analisis Spektrum (FFT 1024 Point)")
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
- Garis Biru (Butterworth Filtered): Noise frekuensi tinggi ("gerigi" tajam) berhasil dihilangkan sepenuhnya.
- Distorsi Bentuk: Karena kita membuang frekuensi tinggi, sinyal kotak berubah menjadi mirip sinus. Sudut tajam pada kotak membutuhkan frekuensi tinggi (harmonisa) untuk terbentuk; karena dipotong di 1500Hz, sudut itu jadi tumpul/bulat.
- Phase Lag (Keterlambatan): Puncak garis biru agak bergeser ke kanan dibanding puncak garis putus-putus hitam. Inilah efek "delay" dari filter analog/IIR.

2. Grafik Bawah: Analisis Spektrum (FFT 1024 Point)
- Puncak Utama (1000 Hz): Baik pada garis merah maupun biru, terdapat puncak tertinggi tepat di angka 1000 (1k). Ini adalah frekuensi dasar, Filter Butterworth berhasil menjaga puncak ini tetap utuh.
- Area di atas 1500 Hz (Stopband): Garisnya meluncur turun (drop) dengan sangat konsisten.
- Dibandingkan garis merah yang tetap tinggi (noise), garis biru berada jauh di bawahnya. Ini membuktikan bahwa filter berhasil menekan (attenuate) noise frekuensi tinggi secara efektif.
- Karakteristik "Flat": Garis biru di sekitar puncak 1000Hz terlihat cukup datar sebelum mulai turun. Inilah ciri khas Butterworth yang disebut maximally flat passband—dia tidak menambah riak/gelombang aneh pada frekuensi yang kita inginkan.
"""