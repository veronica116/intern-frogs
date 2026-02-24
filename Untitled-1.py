import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, cheby1, cheby2, ellip, filtfilt


# Parameter
f = 1000          # Frekuensi (Hz)
A = 1             # Amplitudo
fs = 100000       # Sampling frequency (Hz)
t = np.arange(0, 0.01, 1/fs)  # Waktu 10 ms

# Membuat sinyal kotak
square_wave = A * signal.square(2 * np.pi * f * t)

# ===== AWGN SNR = -3 dB =====
SNR_dB = -3
signal_power = np.mean(square_wave**2) # Daya sinyal 
SNR_linear = 10**(SNR_dB/10) # Konversi SNR dari dB ke linear
noise_power = signal_power / SNR_linear # Daya noise yang diperlukan untuk mencapai SNR yang diinginkan
noise = np.random.normal(0, np.sqrt(noise_power), len(square_wave)) # Generate noise Gaussian
noisy_signal = square_wave + noise # menambahkan noise 

# ===== FFT 1024 POINT =====
NFFT = 1024 # Jumlah titik FFT
fft_signal = np.fft.fft(noisy_signal, NFFT) # Melakukan FFT pada sinyal noisy
fft_magnitude = np.abs(fft_signal) / NFFT # Normalisasi magnitude FFT
freq_axis = np.fft.fftfreq(NFFT, 1/fs) # Membuat sumbu frekuensi untuk FFT

# Ambil frekuensi positif saja
half = NFFT // 2 # Karena FFT simetris, kita hanya perlu setengahnya
freq_axis = freq_axis[:half] # Ambil magnitude untuk frekuensi positif
fft_magnitude = fft_magnitude[:half] # Ambil frekuensi positif dan magnitude yang sesuai

# ===== Parameter Filter =====
cutoff_freq = 3000 # Frekuensi cutoff (Hz)
order = 4 # Orde filter
nyquist = fs / 2 # Frekuensi Nyquist
normal_cutoff = cutoff_freq / nyquist # Normalisasi frekuensi cutoff

# ===== Filter Butterworth 4th Order =====
b, a = butter(order, normal_cutoff, btype='low') # Hitung koefisien filter Butterworth
filtered_butter = filtfilt(b, a, noisy_signal) # Terapkan filter pada sinyal noisy

# ===== Filter Chebyshev Type I 4th Order =====
rp = 1 # Ripple dalam dB
b_cheby, a_cheby = cheby1(order, rp, normal_cutoff, btype='low') # Hitung koefisien filter Chebyshev
filtered_cheby = filtfilt(b_cheby, a_cheby, noisy_signal) # Terapkan filter pada sinyal noisy

# ===== Filter Chebyshev Type II 4th Order =====
rs = 40 # Attenuation dalam dB
b_cheby2, a_cheby2 = cheby2(order, rs, normal_cutoff, btype='low') # Hitung koefisien filter Chebyshev Type II
filtered_cheby2 = filtfilt(b_cheby2, a_cheby2, noisy_signal) # Terapkan filter pada sinyal noisy

# ===== Filter Elliptic 4th Order =====
rp_ellip = 1 # Ripple dalam dB
rs_ellip = 40 # Stopband Attenuation dalam dB
b_ellip, a_ellip = ellip(order, rp_ellip, rs_ellip, normal_cutoff, btype='low') # Hitung koefisien filter Elliptic
filtered_ellip = filtfilt(b_ellip, a_ellip, noisy_signal) # Terapkan filter pada sinyal noisy

# ===== Kalman Filter 1D =====
Q = 0.001 # Variansi proses (process noise) [semakin kecil semakin halus hasil filter, tetapi bisa kehilangan detail]
R = noise_power # Variansi pengukuran (measurement noise) [sesuaikan dengan daya noise yang digunakan]
x_est = np.zeros_like(noisy_signal) # Inisialisasi array untuk estimasi sinyal
P = 1 # Inisialisasi error covariance
for i in range(len(noisy_signal)):
    # Prediction step
    if i == 0:
        x_pred = noisy_signal[0] # Untuk iterasi pertama, gunakan nilai pengukuran sebagai prediksi awal
    else:
        x_pred = x_est[i-1] # Prediksi nilai sinyal (gunakan nilai sebelumnya)
    
    P_pred = P + Q # Prediksi error covariance

    # Kalman gain
    K = P_pred / (P_pred + R) 
    # Update step
    x_est[i] = x_pred + K * (noisy_signal[i] - x_pred) # Update estimasi dengan pengukuran
    P = (1 - K) * P_pred # Update error covariance

# Plot 
#plt.figure()
#plt.plot(t, square_wave, label="Original") # untuk menampilkan sinyal asli

# plot sinyal noisy tanpa fft
# plt.plot(t, noisy_signal) #tanpa fft
# plt.title("Sinyal Kotak dengan AWGN (SNR = -3 dB)")
# plt.xlabel("Waktu (s)")
# plt.ylabel("Amplitudo")
# plt.grid(True)
# plt.show()

#plot sinyal dengan fft 1024 point
max_freq = 10000 # Batas frekuensi untuk plot
mask = freq_axis <= max_freq # Buat mask untuk frekuensi yang diinginkan
freq_plot = freq_axis[mask] # Ambil frekuensi yang diinginkan
magnitude_plot = fft_magnitude[mask] # Ambil magnitude yang sesuai dengan frekuensi yang diinginkan
plt.figure()
plt.plot(freq_plot, magnitude_plot)
plt.title("FFT 1024-Point (0-10000Hz)")
plt.xlabel("Frekuensi (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)


#plot sinyal hasil filter butterworth 4th order
plt.figure()
plt.plot(t, filtered_butter)
plt.title("Sinyal Hasil Filter Butterworth 4th Order")
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo")
plt.grid(True)


#plot sinyal hasil filter chebyshev Type I 4th order
plt.figure()
plt.plot(t, filtered_cheby)
plt.title("Sinyal Hasil Filter Chebyshev Type I 4th Order")
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo")
plt.grid(True)

#plot sinyal hasil filter chebyshev Type II 4th order
plt.figure()
plt.plot(t, filtered_cheby2)
plt.title("Sinyal Hasil Filter Chebyshev Type II 4th Order")
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo")
plt.grid(True)

#plot sinyal hasil filter elliptic 4th order
plt.figure()
plt.plot(t, filtered_ellip)
plt.title("Sinyal Hasil Filter Elliptic 4th Order")
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo")
plt.grid(True)

#plot sinyal hasil filter kalman 1D
plt.figure()
plt.plot(t, x_est)
plt.title("Sinyal Hasil Filter Kalman 1D")
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo")
plt.grid(True)



plt.show()