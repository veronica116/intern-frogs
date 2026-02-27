import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. SETUP PARAMETER SINYAL
fs = 100000              
t = np.linspace(0, 0.005, int(fs * 0.005), endpoint=False) 
f_sinyal = 1000          
amplitudo = 1

# Sinyal Kotak Murni
x_clean = amplitudo * signal.square(2 * np.pi * f_sinyal * t)

# Tambah Noise SNR -3dB
snr_db = -3
p_signal = np.mean(x_clean**2)
p_noise = p_signal / (10**(snr_db / 10))
noise = np.random.normal(0, np.sqrt(p_noise), len(x_clean))
x_noisy = x_clean + noise

# 2. FILTER CHEBYSHEV (Type I)
cutoff = 1500 
order = 4 
rp = 1  # Maximum ripple (dB) yang diizinkan di passband. 
        # Semakin besar rp, potongan semakin tajam tapi riak semakin banyak.

# Membuat filter dalam format SOS (Second-Order Sections)
sos = signal.cheby1(order, rp, cutoff, btype='low', fs=fs, output='sos')
x_filtered = signal.sosfilt(sos, x_noisy)

# 3. ANALISIS FFT 1024 TITIK
def get_fft(data, fs, n=1024):
    yf = np.fft.fft(data, n)
    xf = np.fft.fftfreq(n, 1/fs)
    return xf[:n//2], np.abs(yf[:n//2])

freq, mag_noisy = get_fft(x_noisy, fs)
_, mag_filtered = get_fft(x_filtered, fs)

# 4. PLOT HASIL
plt.figure(figsize=(12, 10))

# Grafik Waktu (Time Domain)
plt.subplot(2, 1, 1)
plt.plot(t, x_noisy, color='red', alpha=0.3, label='Noisy (-3dB)')
plt.plot(t, x_filtered, color='purple', label='Chebyshev Type I Filtered', linewidth=2)
plt.plot(t, x_clean, 'k--', alpha=0.7, label='Original Square Wave') 

plt.title("Chebyshev Type I Filter: Time Domain Analysis")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)

# Grafik Frekuensi (Frequency Domain)
plt.subplot(2, 1, 2)
plt.semilogy(freq, mag_noisy, color='red', alpha=0.3, label='FFT Noisy')
plt.semilogy(freq, mag_filtered, color='purple', label='FFT Chebyshev')
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
Grafik ini menunjukkan bagaimana filter frekuensi klasik bekerja pada sinyal kotak yang "tenggelam" dalam noise (SNR -3dB).
1. Grafik Atas: Time Domain Analysis (Domain Waktu)
- Bentuk sinyal terhadap waktu (0 sampai 0.005 detik).
- Garis Merah (Noisy -3dB): Amplitudo noise seringkali lebih tinggi daripada sinyal aslinya (mencapai angka 4 sementara sinyal asli cuma di angka 1). Bentuk kotak aslinya hampir tidak terlihat sama sekali.
- Garis Hitam Putus-putus (Original Square Wave): Sinyal kotak yang berpindah secara instan antara 1 dan -1 setiap 0.0005 detik (setengah periode dari 1000Hz).
- Garis Ungu (Chebyshev Type I Filtered):
    Kemampuan Denoising: Filter ini berhasil membuang noise yang tajam dan menghasilkan garis yang jauh lebih halus dibandingkan garis merah.
    Distorsi Bentuk: Sama seperti Butterworth, sinyal kotak berubah menjadi gelombang yang lebih melingkar (sinusoidal). Hal ini terjadi karena sudut tajam sinyal kotak dibentuk oleh frekuensi tinggi (harmonisa), yang mana frekuensi tersebut dipotong oleh filter Low Pass ini.
    Ciri Khas Chebyshev: Jika kamu perhatikan sangat teliti pada puncak gelombangnya, Chebyshev cenderung memiliki transisi yang sedikit lebih tegak dibandingkan Butterworth sebelum mulai melengkung, namun mengakibatkan adanya sedikit ripple (riak) atau osilasi kecil sebelum mencapai titik stabil.
2. Grafik Bawah: Analisis Spektrum (FFT 1024 Point)
- Puncak di 1000 Hz: Kamu bisa melihat puncak tertinggi yang tajam tepat di frekuensi 1000 Hz. Ini membuktikan bahwa filter tetap meloloskan komponen utama sinyal.
- Karakteristik Penekanan Noise (Stopband):Garis ungu menunjukkan penurunan magnitude yang sangat curam setelah melewati frekuensi cutoff.
- Dibandingkan dengan garis merah (noise murni), garis ungu berada jauh di bawahnya pada rentang frekuensi tinggi (2000Hz - 5000Hz). Ini menandakan filter Chebyshev sangat efektif menekan noise frekuensi tinggi hingga ke level magnitude yang sangat rendah (sekitar 10^0 atau 1).
- Passband Ripple: Ciri khas utama Chebyshev Type I adalah adanya riak di area frekuensi yang diloloskan (passband). Pada grafik, ini terlihat dari garis ungu yang tidak "se-datar" Butterworth di area sebelum 1000Hz, melainkan ada sedikit gelombang naik-turun kecil pada amplitudonya.
"""