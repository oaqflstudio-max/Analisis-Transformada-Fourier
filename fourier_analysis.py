import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Configuración global de tiempo y frecuencia
# ==========================================
fs = 1000  # Frecuencia de muestreo (Hz)
t_start, t_end = -2.0, 2.0
t = np.linspace(t_start, t_end, int(fs * (t_end - t_start)), endpoint=False)
N = len(t)
freqs = np.fft.fftfreq(N, 1/fs)

def compute_fft(signal):
    """Calcula la FFT centrada y sus frecuencias correspondientes."""
    fft_vals = np.fft.fft(signal)
    fft_shifted = np.fft.fftshift(fft_vals)
    freqs_shifted = np.fft.fftshift(freqs)
    
    # Magnitud normalizada y Fase
    magnitude = np.abs(fft_shifted) / N
    phase = np.angle(fft_shifted)
    # Filtrar ruido numérico de fase para magnitudes despreciables
    phase[magnitude < 1e-4] = 0.0
    return freqs_shifted, magnitude, phase

def plot_signal_and_fft(t, signal, freqs, magnitude, phase, title, freq_lim=(-10, 10)):
    """Genera gráficas en tiempo, magnitud y fase del espectro."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    #Dominio del tiempo
    axs[0].plot(t, signal, color='blue')
    axs[0].set_title(f'{title} - Dominio del Tiempo')
    axs[0].set_xlabel('Tiempo [s]')
    axs[0].set_ylabel('Amplitud')
    axs[0].grid(True)
    
    # Dominio de la frecuencia (Magnitud)
    axs[1].stem(freqs, magnitude, linefmt='b-', markerfmt='bo', basefmt='r-')
    axs[1].set_title(f'{title} - Espectro de Magnitud')
    axs[1].set_xlabel('Frecuencia [Hz]')
    axs[1].set_ylabel('Magnitud')
    axs[1].set_xlim(freq_lim)
    axs[1].grid(True)
    
    # Dominio de la frecuencia (Fase)
    axs[2].plot(freqs, phase, color='green')
    axs[2].set_title(f'{title} - Espectro de Fase')
    axs[2].set_xlabel('Frecuencia [Hz]')
    axs[2].set_ylabel('Fase [rad]')
    axs[2].set_xlim(freq_lim)
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 1. Definición y Análisis de Señales Elementales
# ==========================================

# A. Pulso Rectangular (Ancho T_rect = 1s)
rect = np.where(np.abs(t) <= 0.5, 1.0, 0.0)
f_rect, mag_rect, phase_rect = compute_fft(rect)
plot_signal_and_fft(t, rect, f_rect, mag_rect, phase_rect, "Pulso Rectangular")

# B. Función Escalón Unitario (Heaviside suavizado en rango)
step = np.where(t >= 0, 1.0, 0.0)
f_step, mag_step, phase_step = compute_fft(step)
plot_signal_and_fft(t, step, f_step, mag_step, phase_step, "Función Escalón Unitario")

# C. Función Senoidal (f0 = 5 Hz)
f0 = 5.0
sinusoid = np.sin(2 * np.pi * f0 * t)
f_sin, mag_sin, phase_sin = compute_fft(sinusoid)
plot_signal_and_fft(t, sinusoid, f_sin, mag_sin, phase_sin, "Senoidal (5 Hz)", freq_lim=(-15, 15))

# ==========================================
# 2. Verificación de Propiedades de la Transformada de Fourier
# ==========================================

# Propiedad 1: Linealidad -> a*x1(t) + b*x2(t)
x1 = np.sin(2 * np.pi * 3 * t)
x2 = np.sin(2 * np.pi * 7 * t)
x_linear = 2 * x1 + 0.5 * x2
f_lin, mag_lin, phase_lin = compute_fft(x_linear)
plot_signal_and_fft(t, x_linear, f_lin, mag_lin, phase_lin, "Linealidad: 2*Sin(3Hz) + 0.5*Sin(7Hz)", freq_lim=(-15, 15))

# Propiedad 2: Desplazamiento en el Tiempo -> x(t - t0)
t0 = 0.3
rect_shifted = np.where(np.abs(t - t0) <= 0.5, 1.0, 0.0)
f_shift, mag_shift, phase_shift = compute_fft(rect_shifted)
plot_signal_and_fft(t, rect_shifted, f_shift, mag_shift, phase_shift, f"Desplazamiento en Tiempo (t0={t0}s)")

# Propiedad 3: Escalamiento en Frecuencia/Tiempo -> x(a*t)
# Si a > 1 (compresión en tiempo), el espectro se dilata en frecuencia
a = 2.0
rect_scaled = np.where(np.abs(a * t) <= 0.5, 1.0, 0.0)
f_scale, mag_scale, phase_scale = compute_fft(rect_scaled)
plot_signal_and_fft(t, rect_scaled, f_scale, mag_scale, phase_scale, f"Escalamiento en Tiempo (a={a})")
