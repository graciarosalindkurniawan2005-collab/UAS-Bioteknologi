# UAS-Bioteknologi
Kinetic model simulation
# %matplotlib inline
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =====================================================================
# 1. DEFINISI PARAMETER (Sesuai Tabel Parameter pada Soal) [cite: 81]
# =====================================================================
V1_max = 5.0      # Kecepatan maksimum reaksi v1 [cite: 81]
Km1    = 2.0      # Konstanta Michaelis untuk v1 [cite: 81]
Ki     = 3.0      # Konstanta inhibisi non-kompetitif [cite: 81]
X      = 10.0     # Konsentrasi substrat eksternal (konstan) [cite: 81]
k2     = 1.0      # Konstanta laju orde pertama untuk A -> B [cite: 81]
k3     = 0.8      # Konstanta laju orde pertama untuk B -> P [cite: 81]
k4     = 0.3      # Konstanta laju orde pertama untuk A -> byproduct [cite: 81]

# =====================================================================
# 2. FORMULASI PERSAMAAN KINETIK (ODEs) [cite: 71, 72]
# =====================================================================
def metabolic_system(t, y):
    A, B, P = y  # Mengambil nilai konsentrasi metabolit internal [cite: 47, 73]
    
    # Persamaan laju reaksi v1 dengan Allosteric Inhibition [cite: 48, 79]
    v1 = (V1_max * X) / ((Km1 + X) * (1 + (P / Ki))) # [cite: 79]
    
    # Persamaan laju reaksi v2, v3, v4 (First-order kinetics) [cite: 81]
    v2 = k2 * A
    v3 = k3 * B
    v4 = k4 * A
    
    # Berdasarkan Mass Balance dari Q2B [cite: 71, 72]
    dAdt = v1 - v2 - v4
    dBdt = v2 - v3
    dPdt = v3
    
    return [dAdt, dBdt, dPdt]

# =====================================================================
# 3. KONDISI AWAL DAN EKSEKUSI SIMULASI
# =====================================================================
y0 = [0.0, 0.0, 0.0]  # Konsentrasi awal A, B, P di t=0
t_span = (0, 40)      # Rentang waktu pengamatan
t_eval = np.linspace(0, 40, 400)

sol = solve_ivp(metabolic_system, t_span, y0, t_eval=t_eval, method='RK45')

# =====================================================================
# 4. VISUALISASI GRAFIK UNTUK GITHUB [cite: 91]
# =====================================================================
# Menggunakan resolusi tinggi (dpi=150) agar teks dan garis tajam di layar GitHub
plt.figure(figsize=(10, 6), dpi=150)

plt.plot(sol.t, sol.y[0], label='Metabolit A', color='#1f77b4', linewidth=2.5)
plt.plot(sol.t, sol.y[1], label='Metabolit B', color='#ff7f0e', linewidth=2.5)
plt.plot(sol.t, sol.y[2], label='Product P', color='#2ca02c', linewidth=2.5)

# Pengaturan anotasi grafik yang bersih
plt.title('Simulasi Kinetika Jalur Metabolik (Dengan Allosteric Feedback)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Waktu (t)', fontsize=11)
plt.ylabel('Konsentrasi Metabolit', fontsize=11)
plt.xlim(0, 40)
plt.ylim(0, max(sol.y[2]) * 1.1)
plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

# Menyimpan cadangan gambar statis otomatis
plt.savefig('simulation_result.png', bbox_inches='tight', dpi=300)
plt.show() 
