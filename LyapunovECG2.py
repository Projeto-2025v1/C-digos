import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -------------------------
# Parâmetros do sistema
# -------------------------
# Parâmetros
mu0 = 0.5
mu1 = 0.08
mu2 = 0.05
mu3 = 0.45

alphaSA = 2.3
alphaAV = 3
alphaHP = 4

vSA1 = 0.2
vSA2 = -1.9
vAV1 = 0.1
vAV2 = -0.1
vHP1 = 1
vHP2 = -1

dSA1 = 3
eSA1 = 5.5
dAV1 = 3
eAV1 = 3
dHP1 = 3
eHP1 = 7

dSA2 = 1
eSA2 = 1
dAV2 = 1
eAV2 = 1
dHP2 = 1
eHP2 = 1

kAVSA = 0
kAVSAtau = 0
kHPSA = 0
kHPSAtau = 0
kSAAV = 5
kSAAVtau = 5
kHPAV = 0
kHPAVtau = 0
kSAPH = 0
kSAPHtau = 0
kAVHP = 20
kAVHPtau = 20

tauSAAV = 0.1
tauAVHP = 0.8

n = 10

# -------------------------
# Sistema de EDOs
# -------------------------
def sistema_edo(t, y):
    # Estados principais
    x1_0 = y[0]
    x2 = y[1]
    x3 = y[2]
    x4 = y[3]
    x5 = y[4]
    x6 = y[5]

    # Definir índices das cadeias
    num_chain_states = n + 1
    idx_x1_chain_start = 6
    idx_x1_chain_end = idx_x1_chain_start + num_chain_states
    idx_x3_chain_start = idx_x1_chain_end

    # Extrair estados das cadeias
    x1_estados = y[idx_x1_chain_start : idx_x1_chain_end]
    x3_estados = y[idx_x3_chain_start:]

    dydt = np.zeros_like(y)

    # --- Dinâmicas principais ---

    # Nó SA (x1, x2)
    dydt[0] = x2
    dydt[1] = -alphaSA * x2 * (x1_0 - vSA1) * (x1_0 - vSA2) - (x1_0 * (x1_0 + dSA1) * (x1_0 + eSA1)) / (dSA2 * eSA2)

    # Nó AV (x3, x4)
    x1_delay_saav = x1_estados[-1]
    dydt[2] = x4
    dydt[3] = -alphaAV * x4 * (x3 - vAV1) * (x3 - vAV2) - (x3 * (x3 + dAV1) * (x3 + eAV1)) / (dAV2 * eAV2) - kSAAV * x3 + kSAAVtau * x1_delay_saav

    # Nó HP (x5, x6)
    x3_delay_avhp = x3_estados[-1]
    dydt[4] = x6
    dydt[5] = -alphaHP * x6 * (x5 - vHP1) * (x5 - vHP2) - (x5 * (x5 + dHP1) * (x5 + eHP1)) / (dHP2 * eHP2) - kAVHP * x5 + kAVHPtau * x3_delay_avhp

    # ---Cadeias de Atraso---

    # Cadeia 1 (x1_0 -> SAAV)
    if tauSAAV > 1e-9:
        taxa_saav = n / tauSAAV

        dydt[idx_x1_chain_start] = taxa_saav * (x1_0 - x1_estados[0])

        for i in range(1, n + 1): # Loop de 1 até n
            idx = idx_x1_chain_start + i
            dydt[idx] = taxa_saav * (x1_estados[i - 1] - x1_estados[i])
    else:
        # Sem atrasos
        dydt[idx_x1_chain_start : idx_x1_chain_end] = x2


    # Cadeia 2 (x3 -> AVHP)
    if tauAVHP > 1e-9:
        taxa_avhp = n / tauAVHP

        # 1. Primeiro estado (índice 0)
        dydt[idx_x3_chain_start] = taxa_avhp * (x3 - x3_estados[0])

        # 2. Estados intermediários (índices 1 a n)
        for i in range(1, n + 1):
            idx = idx_x3_chain_start + i
            dydt[idx] = taxa_avhp * (x3_estados[i - 1] - x3_estados[i])
    else:
        # Se não há atraso, todos os estados da cadeia têm a derivada de x3
        dydt[idx_x3_chain_start:] = x4

    return dydt

# -------------------------
# Condições iniciais e integração
# -------------------------
num_chain_states = n + 1
total_states = 6 + num_chain_states + num_chain_states
y0 = np.zeros(total_states)

# Estados principais
y0[0] = -0.1    # x1_0
y0[1] = 0.025   # x2
y0[2] = -0.6    # x3
y0[3] = 0.1     # x4
y0[4] = -3.3    # x5
y0[5] = 10 / 15 # x6

# Índices da cadeia 1 (delay x1)
idx_x1_chain_start = 6
idx_x1_chain_end = idx_x1_chain_start + num_chain_states
y0[idx_x1_chain_start : idx_x1_chain_end] = y0[0]

# Índices da cadeia 2 (delay x3)
idx_x3_chain_start = idx_x1_chain_end
y0[idx_x3_chain_start:] = y0[2]

print(f"Resolvendo sistema de EDOs com {n} estados de discretização por cadeia...")
print(f"Total de estados: {total_states}")
t_span = (0.0, 200.0)
t_eval = np.linspace(t_span[0], t_span[1], 20000)

dt_lyap = t_eval[1] - t_eval[0]
print(f"dt real da simulação: {dt_lyap:.6f}")

sol = solve_ivp(sistema_edo, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)

if not sol.success:
    print("Integração falhou com RK45:", sol.message)
    sol = solve_ivp(sistema_edo, t_span, y0, t_eval=t_eval, method='BDF', rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError("Integração falhou com ambos os métodos.")

t = sol.t
x1sol = sol.y[0, :]
x2sol = sol.y[1, :]
x3sol = sol.y[2, :]
x4sol = sol.y[3, :]
x5sol = sol.y[4, :]
x6sol = sol.y[5, :]

# -------------------------
# Construção do ECG
# -------------------------

ecg_signal = 0.5 + (mu1 * x1sol) + (mu2 * x3sol) + (mu3 * x5sol)

mask = t > 50.0
ecg_clean = ecg_signal[mask]
t_clean = t[mask]
print(f"Sinal ECG limpo: {len(ecg_clean)} pontos")

# -------------------------
# Parâmetros embedding
# -------------------------
tau_opt = 23
m_opt = 8

print(f"Usando delay: {tau_opt}")
print(f"Usando dimensão: {m_opt}")

def reconstruct_phase_space(data, m, tau):
    n_data = len(data)
    N = n_data - (m - 1) * tau
    if N <= 0:
        raise ValueError("Parâmetros de embedding inválidos: N <= 0")
    embedded = np.zeros((N, m))
    for i in range(N):
        for j in range(m):
            embedded[i, j] = data[i + j * tau]
    return embedded

print("Reconstruindo espaço de fases...")
try:
    phase_space = reconstruct_phase_space(ecg_clean, m_opt, tau_opt)
    print("Espaço de fases reconstruído:", phase_space.shape)
except ValueError as e:
    print("Erro na reconstrução:", e)
    if len(ecg_clean) < tau_opt * (m_opt - 1):
        tau_opt = max(1, len(ecg_clean) // (2 * m_opt))
        print("Ajustando tau para:", tau_opt)
        phase_space = reconstruct_phase_space(ecg_clean, m_opt, tau_opt)

# -------------------------
# Algoritmo de Wolf
# -------------------------

def find_nearest_neighbor_wolf(data, current_idx, theiler_window, max_separation=None):

    N, m = data.shape
    current_point = data[current_idx]
    best_dist = np.inf
    best_idx = -1

    for i in range(N):
        # Aplicar janela de Theiler
        if abs(i - current_idx) <= theiler_window:
            continue

        if max_separation is not None and i + max_separation >= N:
            continue

        dist = np.linalg.norm(data[i] - current_point)

        if dist < best_dist and dist > 1e-10: # Evitar pontos idênticos
            best_dist = dist
            best_idx = i

    return best_idx, best_dist

def wolf_lyapunov_corrected(phase_space, dt, evolve_steps, theiler_window, max_replacements=1000):

    N, m = phase_space.shape
    sum_log = 0.0
    num_replacements = 0

    # Iniciar em um ponto com espaço suficiente para evoluir
    current_idx = 0

    # Encontrar o primeiro vizinho
    neighbor_idx, dist_initial = find_nearest_neighbor_wolf(
        phase_space, current_idx, theiler_window, max_separation=evolve_steps
    )

    if neighbor_idx == -1:
        print("[Wolf] ERRO: Não foi possível encontrar um vizinho inicial.")
        return 0.0, 0.0, 0

    print(f"[Wolf] Iniciando... Vizinho inicial {neighbor_idx} (dist={dist_initial:.4e})")

    # Loop principal: seguir a trajetória fiducial
    while (current_idx + evolve_steps < N and
           neighbor_idx + evolve_steps < N and
           num_replacements < max_replacements):

        # Evoluir ambos os pontos
        evolved_idx = current_idx + evolve_steps
        evolved_neighbor_idx = neighbor_idx + evolve_steps

        # Calcular a nova distância
        dist_final = np.linalg.norm(phase_space[evolved_idx] - phase_space[evolved_neighbor_idx])

        # VALIDAÇÃO: Verificar se a divergência é válida
        if dist_initial > 1e-10 and dist_final > 1e-10:
            # Calcular razão de expansão
            expansion_ratio = dist_final / dist_initial

            if 1e-6 < expansion_ratio < 1e6:
                sum_log += np.log(expansion_ratio)
                num_replacements += 1
            else:
                print(f"[Wolf] Aviso: Expansão anormal ignorada ({expansion_ratio:.2e})")

        # Etapa de Substituição: avançar o ponto fiducial
        current_idx = evolved_idx

        # Encontrar novo vizinho
        neighbor_idx, dist_initial = find_nearest_neighbor_wolf(
            phase_space, current_idx, theiler_window, max_separation=evolve_steps
        )

        # Se não encontrar vizinho, parar
        if neighbor_idx == -1:
            print(f"[Wolf] Parando no índice {current_idx}, vizinho não encontrado.")
            break

        # Progresso a cada 100 substituições
        if num_replacements % 100 == 0 and num_replacements > 0:
            print(f"[Wolf] Substituições: {num_replacements}/{max_replacements}")

    min_substitutions = 50
    if num_replacements < min_substitutions:
        print(f"[Wolf] AVISO: Apenas {num_replacements} substituições (mínimo recomendado: {min_substitutions})")
        print("[Wolf] Resultado pode não ser confiável.")

    # Cálculo final
    if num_replacements > 0:
        total_time = num_replacements * evolve_steps * dt
        lyap_exp = sum_log / total_time
    else:
        lyap_exp = 0.0

    return lyap_exp, sum_log, num_replacements

# -------------------------
# Algoritmo de Rosenstein
# -------------------------
def lyarosenstein(phase_space, meanperiod, maxiter, dt):
    """
    Algoritmo de Rosenstein para cálculo do maior expoente de Lyapunov
    """
    M = phase_space.shape[0]

    print(f"[Rosenstein] Iniciando busca de vizinhos mais próximos...")
    neardis = np.zeros(M)
    nearpos = np.zeros(M, dtype=int)

    # Encontrar os vizinhos mais próximos
    for i in range(M):
        if i % 1000 == 0:
            print(f"[Rosenstein] Processando ponto {i}/{M}")

        x0 = np.tile(phase_space[i, :], (M, 1))
        distance = np.sqrt(np.sum((phase_space - x0)**2, axis=1))

        # Excluir pontos temporalmente próximos
        for j in range(M):
            if abs(j - i) <= meanperiod:
                distance[j] = 1e10

        neardis[i] = np.min(distance)
        nearpos[i] = np.argmin(distance)

    d = np.zeros(maxiter)

    print(f"[Rosenstein] Calculando divergência ao longo do tempo...")
    # Calcular a divergência ao longo do tempo
    for k in range(maxiter):
        if k % 20 == 0:
            print(f"[Rosenstein] Iteração {k}/{maxiter}")

        maxind = M - k - 1
        evolve = 0
        pnt = 0

        for j in range(M):
            if j <= maxind and nearpos[j] <= maxind:
                dist_k = np.sqrt(np.sum((phase_space[j + k, :] - phase_space[nearpos[j] + k, :])**2))

                if dist_k > 1e-10:
                    evolve += np.log(dist_k)
                    pnt += 1

        if pnt > 0:
            d[k] = evolve / pnt
        else:
            d[k] = 0

    # Ajuste linear para calcular o LLE
    tlinear = np.arange(15, min(79, maxiter))

    if len(tlinear) > 2 and len(d[tlinear]) > 2:
        F = np.polyfit(tlinear, d[tlinear], 1)
        lle = F[0] / dt
    else:
        lle = 0.0
        F = [0, 0]

    return d, lle, F

# -------------------------
# Estimar período médio
# -------------------------
signal_1d = phase_space[:, 0]
signal_normalized = (signal_1d - np.mean(signal_1d)) / (np.std(signal_1d) if np.std(signal_1d) > 0 else 1.0)
peaks, _ = find_peaks(signal_normalized, height=0.5, distance=10)
if len(peaks) > 2:
    mean_period = int(np.mean(np.diff(peaks)))
else:
    mean_period = max(10, int(phase_space.shape[0] / 50))

print(f"\nPeríodo médio estimado: {mean_period} pontos ({mean_period * dt_lyap:.4f} s)")

# -------------------------
# WOLF
# -------------------------
print("\n" + "="*60)
print("CALCULANDO EXPOENTE DE LYAPUNOV - ALGORITMO DE WOLF")
print("="*60)

theiler_window_wolf = mean_period
evolve_steps_wolf = max(3, mean_period // 4) # Menor que antes, mais seguro
max_replacements_wolf = 500 # Limite de substituições

print(f"[Wolf] Parâmetros:")
print(f"  evolve_steps = {evolve_steps_wolf} ({evolve_steps_wolf * dt_lyap:.4f} s)")
print(f"  theiler_window = {theiler_window_wolf} ({theiler_window_wolf * dt_lyap:.4f} s)")
print(f"  max_replacements = {max_replacements_wolf}")

lyap_wolf, sum_log_wolf, nsubs_wolf = wolf_lyapunov_corrected(
    phase_space,
    dt_lyap,
    evolve_steps_wolf,
    theiler_window_wolf,
    max_replacements_wolf
)
print(f"[Wolf] Expoente de Lyapunov: {lyap_wolf:.6f} (substituições: {nsubs_wolf})")

# -------------------------
# ROSENSTEIN
# -------------------------
meanperiod_ros = mean_period
maxiter_ros = min(100, phase_space.shape[0] // 10)

print("\n" + "="*60)
print("CALCULANDO EXPOENTE DE LYAPUNOV - ALGORITMO DE ROSENSTEIN")
print("="*60)
d_rosenstein, lle_rosenstein, fit_params = lyarosenstein(
    phase_space,
    meanperiod_ros,
    maxiter_ros,
    dt_lyap
)
print(f"[Rosenstein] Expoente de Lyapunov: {lle_rosenstein:.6f}")
print(f"[Rosenstein] Coeficiente angular do ajuste: {fit_params[0]:.6f}")

# -------------------------
# Plots comparativos
# -------------------------
fig = plt.figure(figsize=(18, 12))

# Espaço de fases 2D
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(phase_space[:, 0], phase_space[:, 1], '-', alpha=0.7, linewidth=0.5)
ax1.set_xlabel('x(t)')
ax1.set_ylabel('x(t+τ)')
ax1.set_title('Espaço de Fases Reconstruído (2D)')
ax1.grid(True, alpha=0.3)

# Espaço de fases 3D
if m_opt >= 3:
    from mpl_toolkits.mplot3d import Axes3D
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    lim = min(2000, phase_space.shape[0])
    ax2.plot(phase_space[:lim, 0], phase_space[:lim, 1], phase_space[:lim, 2], '-', alpha=0.7, linewidth=0.5)
    ax2.set_xlabel('x(t)')
    ax2.set_ylabel('x(t+τ)')
    ax2.set_zlabel('x(t+2τ)')
    ax2.set_title('Espaço de Fases 3D')

# Sinal ECG
ax4 = fig.add_subplot(2, 3, 4)
lim_t = min(2000, len(t_clean))
ax4.plot(t_clean[:lim_t], ecg_clean[:lim_t], '-', linewidth=1)
ax4.set_xlabel('Tempo (s)')
ax4.set_ylabel('Amplitude')
ax4.set_title('Sinal ECG')
ax4.grid(True, alpha=0.3)

# -------------------------
# Resultados finais
# -------------------------
print("\n" + "="*60)
print("RESULTADOS FINAIS")
print("="*60)
print(f"Algoritmo de Wolf")
print(f"  λ_max = {lyap_wolf:.6f}")
print(f"Algoritmo de Rosenstein:")
print(f"  λ_max = {lle_rosenstein:.6f}")

plt.tight_layout()
plt.show()