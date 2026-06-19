# UAS-Bioteknologi
Kinetic model simulation
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =====================================================================
# 1. DEFINISI PARAMETER (Sesuai Tabel Parameter pada Soal)
# =====================================================================
V1_max = 5.0      # Kecepatan maksimum reaksi v1
Km1    = 2.0      # Konstanta Michaelis untuk v1
Ki     = 3.0      # Konstanta inhibisi non-kompetitif
X      = 10.0     # Konsentrasi substrat eksternal (konstan)
k2     = 1.0      # Konstanta laju orde pertama untuk A -> B
k3     = 0.8      # Konstanta laju orde pertama untuk B -> P
k4     = 0.3      # Konstanta laju orde pertama untuk A -> Byproduct

# =====================================================================
# 2. FORMULASI PERSAMAAN KINETIK (ODEs)
# =====================================================================
def metabolic_system(t, y):
    A, B, P = y  # Mengambil nilai konsentrasi metabolit internal saat ini
    
    # Menghitung laju fluks reaksi berdasarkan persamaan kinetika
    # v1 menggunakan model Allosteric Non-Competitive Inhibition
    v1 = (V1_max * X) / ((Km1 + X) * (1 + (P / Ki)))
    
    # v2, v3, v4 menggunakan model kinetika orde pertama (first-order)
    v2 = k2 * A
    v3 = k3 * B
    v4 = k4 * A
    
    # Persamaan diferensial berdasar neraca massa (mass balance dari Q2B)
    dAdt = v1 - v2 - v4
    dBdt = v2 - v3
    dPdt = v3
    
    return [dAdt, dBdt, dPdt]

# =====================================================================
# 3. KONDISI AWAL DAN RENTANG WAKTU SIMULASI
# =====================================================================
# Mengasumsikan pada t=0, konsentrasi internal awal [A, B, P] adalah 0
y0 = [0.0, 0.0, 0.0]

# Rentang waktu simulasi (dari jam 0 hingga 40 untuk melihat kondisi stabil)
t_span = (0, 40)
t_eval = np.linspace(0, 40, 400)

# ODE Solver
sol = solve_ivp(metabolic_system, t_span, y0, t_eval=t_eval, method='RK45')

# =====================================================================
# 4. Simulation Figure
# =====================================================================
plt.figure(figsize=(9, 5.5))
plt.plot(sol.t, sol.y[0], label='Metabolit A', color='#1f77b4', linewidth=2.5)
plt.plot(sol.t, sol.y[1], label='Metabolit B', color='#ff7f0e', linewidth=2.5)
plt.plot(sol.t, sol.y[2], label='Product P', color='#2ca02c', linewidth=2.5)

plt.title('Simulasi Dinamika Sistem Jalur Metabolik dengan Regulasi Inhibisi', fontsize=12, fontweight='bold')
plt.xlabel('Waktu (t)', fontsize=11)
plt.ylabel('Konsentrasi Metabolit', fontsize=11)
plt.xlim(0, 40)
plt.ylim(0, max(sol.y[2]) * 1.1)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

# Showing graph
plt.tight_layout()
plt.show()
