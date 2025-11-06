import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import qr
import matplotlib.pyplot as plt

# Parâmetros do sistema
mu0 = 1
mu1 = 0.06
mu2 = 0.1
mu3 = 0.3
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
kSAAV = 5
kSAAVtau = 5
kAVHP = 20
kAVHPtau = 20
tauSAAV = 0.1
tauAVHP = 0.8

# Discretização do atraso
n = 5  # Discretização para tauSAAV
m = 5  # Discretização para tauAVHP
dim_sistema = 6 + (n + 1) + (m + 1)  # Dimensão total do sistema

def sistema_edo(t, y):
    """Sistema de EDOs com atraso discretizado"""
    # Estados principais
    x1_0 = y[0]
    x2 = y[1]
    x3 = y[2]
    x4 = y[3]
    x5 = y[4]
    x6 = y[5]

    # Estados da cadeia de atraso 1 (x1)
    x1_estados = y[6 : 6+n+1]

    # Estados da cadeia de atraso 2 (x3)
    idx_start_x3 = 6 + n + 1
    x3_estados = y[idx_start_x3 : idx_start_x3 + m + 1]

    dydt = np.zeros_like(y)

    # Equações principais
    dydt[0] = x2
    dydt[1] = -alphaSA * x2 * (x1_0 - vSA1) * (x1_0 - vSA2) - \
              x1_0 * (x1_0 + dSA1) * (x1_0 + eSA1) / (dSA2 * eSA2)

    dydt[2] = x4
    dydt[3] = -alphaAV * x4 * (x3 - vAV1) * (x3 - vAV2) - \
              x3 * (x3 + dAV1) * (x3 + eAV1) / (dAV2 * eAV2) - \
              kSAAV * x3 + kSAAVtau * x1_estados[-1]

    dydt[4] = x6

    dydt[5] = -alphaHP * x6 * (x5 - vHP1) * (x5 - vHP2) - \
              x5 * (x5 + dHP1) * (x5 + eHP1) / (dHP2 * eHP2) - \
              kAVHP * x5 + kAVHPtau * x3_estados[-1]

    # Discretização do atraso 1 (x1)
    dydt[6] = x2
    for i in range(1, n):
        dydt[6+i] = n / (2 * tauSAAV) * (x1_estados[i-1] - x1_estados[i+1])
    dydt[6+n] = n / tauSAAV * (x1_estados[n-1] - x1_estados[n])

    # Discretização do atraso 2 (x3)
    dydt[idx_start_x3] = x4
    for i in range(1, m):
        dydt[idx_start_x3 + i] = m / (2 * tauAVHP) * (x3_estados[i-1] - x3_estados[i+1])
    dydt[idx_start_x3 + m] = m / tauAVHP * (x3_estados[m-1] - x3_estados[m])

    return dydt

def jacobiano(t, y):
    """Calcula a matriz Jacobiana do sistema"""
    J = np.zeros((dim_sistema, dim_sistema))

    x1_0, x2, x3, x4, x5, x6 = y[0:6]

    x1_estados = y[6 : 6+n+1]
    idx_start_x3 = 6 + n + 1
    x3_estados = y[idx_start_x3 : idx_start_x3 + m + 1]

    # Derivadas para x1_0 e x2
    J[0, 1] = 1  # dx1_0/dx2
    J[1, 0] = (-alphaSA * x2 * ((x1_0 - vSA2) + (x1_0 - vSA1)) -
               ((x1_0 + dSA1)*(x1_0 + eSA1) + x1_0*(x1_0 + eSA1) + x1_0*(x1_0 + dSA1)) / (dSA2 * eSA2))
    J[1, 1] = -alphaSA * (x1_0 - vSA1) * (x1_0 - vSA2)

    # Derivadas para x3 e x4
    J[2, 3] = 1  # dx3/dx4
    J[3, 2] = (-alphaAV * x4 * ((x3 - vAV2) + (x3 - vAV1)) -
               ((x3 + dAV1)*(x3 + eAV1) + x3*(x3 + eAV1) + x3*(x3 + dAV1)) / (dAV2 * eAV2) -
               kSAAV)
    J[3, 3] = -alphaAV * (x3 - vAV1) * (x3 - vAV2)
    J[3, 6+n] = kSAAVtau

    # Derivadas para x5 e x6
    J[4, 5] = 1  # dx5/dx6
    J[5, 4] = (-alphaHP * x6 * ((x5 - vHP2) + (x5 - vHP1)) -
               ((x5 + dHP1)*(x5 + eHP1) + x5*(x5 + eHP1) + x5*(x5 + dHP1)) / (dHP2 * eHP2) -
               kAVHP)
    J[5, 5] = -alphaHP * (x5 - vHP1) * (x5 - vHP2)

    idx_ultimo_x3 = idx_start_x3 + m
    J[5, idx_ultimo_x3] = kAVHPtau

    # Derivadas para o sistema de atraso 1 (x1)
    J[6, 1] = 1 # Conexão x2 -> input cadeia 1
    for i in range(1, n):
        J[6+i, 6+i-1] = n / (2 * tauSAAV)
        J[6+i, 6+i+1] = -n / (2 * tauSAAV)
    J[6+n, 6+n-1] = n / tauSAAV
    J[6+n, 6+n] = -n / tauSAAV

    # Derivadas para o sistema de atraso 2 (x3)
    J[idx_start_x3, 3] = 1
    for i in range(1, m):
        J[idx_start_x3 + i, idx_start_x3 + i - 1] = m / (2 * tauAVHP)
        J[idx_start_x3 + i, idx_start_x3 + i + 1] = -m / (2 * tauAVHP)
    J[idx_start_x3 + m, idx_start_x3 + m - 1] = m / tauAVHP
    J[idx_start_x3 + m, idx_start_x3 + m] = -m / tauAVHP

    return J

def sistema_extendido(t, y_ext):
    """
    Sistema estendido para cálculo dos expoentes de Lyapunov
    y_ext[:dim] = estado do sistema
    y_ext[dim:dim+dim*dim] = matriz de perturbações (achatada)
    """
    dim = dim_sistema
    y = y_ext[:dim]
    W = y_ext[dim:dim+dim*dim].reshape((dim, dim))

    # Calcular o sistema principal
    dydt = sistema_edo(t, y)

    # Calcular o Jacobiano
    J = jacobiano(t, y)

    # Evoluir as perturbações: dW/dt = J * W
    dWdt = J @ W

    return np.concatenate([dydt, dWdt.flatten()])

def calcular_expoentes_lyapunov_com_evolucao(t_final=1000, dt=0.01, transiente=200):
    """
    Calcula os expoentes de Lyapunov e retorna a evolução ao longo das iterações
    """
    print("Calculando expoentes de Lyapunov...")

    # Condições iniciais
    y0 = np.zeros(dim_sistema)
    y0[0] = -0.1
    y0[1] = 0.025
    y0[2] = -0.6
    y0[3] = 0.1
    y0[4] = -3.3
    y0[5] = 10/15

    # Inicializa a cadeia de atraso 1 (x1) com o valor de x1_0
    y0[6:6+n+1] = -0.1

    # Inicializa a cadeia de atraso 2 (x3) com o valor de x3
    idx_start_x3 = 6 + n + 1
    y0[idx_start_x3 : idx_start_x3 + m + 1] = -0.6

    # Inicializar matriz de perturbações como identidade
    W0 = np.eye(dim_sistema)

    # Estado estendido inicial
    y_ext0 = np.concatenate([y0, W0.flatten()])

    # Integrar sistema transiente
    print("Integrando transiente...")
    sol_trans = solve_ivp(sistema_edo, (0, transiente), y0,
                         method='RK45', rtol=1e-8, atol=1e-10)
    y0_trans = sol_trans.y[:, -1]

    # Inicializar para cálculo dos expoentes
    y_ext = np.concatenate([y0_trans, W0.flatten()])

    # Arrays para armazenar evolução
    evolucao_expoentes = []
    lyapunov_sums = np.zeros(dim_sistema)
    num_iterations = 0

    # Parâmetros do algoritmo
    reorthonormalization_time = 1.0  # Tempo entre reortonormalizações
    t_current = transiente

    print("Executando algoritmo de Lyapunov...")

    while t_current < t_final:
        t_next = min(t_current + reorthonormalization_time, t_final)

        # Integrar até o próximo ponto de reortonormalização
        sol_step = solve_ivp(sistema_extendido, [t_current, t_next], y_ext,
                           method='RK45', rtol=1e-8, atol=1e-10)

        if not sol_step.success:
            print("Aviso: Problema na integração, tentando BDF...")
            sol_step = solve_ivp(sistema_extendido, [t_current, t_next], y_ext,
                               method='BDF', rtol=1e-6, atol=1e-8)

        y_ext = sol_step.y[:, -1]

        # Extrair matriz de perturbações
        W = y_ext[dim_sistema:].reshape((dim_sistema, dim_sistema))

        # Fatoração QR
        Q, R = qr(W)

        # Acumular logaritmo dos valores diagonais de R
        # Usar np.maximum para evitar log(0) se R for singular
        diag_R = np.abs(np.diag(R))
        lyapunov_sums += np.log(np.maximum(diag_R, 1e-100)) # Adicionado robustez

        num_iterations += 1

        # Calcular expoentes atuais
        expoentes_atuais = lyapunov_sums / (num_iterations * reorthonormalization_time)
        evolucao_expoentes.append(expoentes_atuais.copy())

        # Reinicializar com matriz ortonormal
        y_ext[dim_sistema:] = Q.flatten()

        t_current = t_next

        # Progresso
        if num_iterations % 10 == 0:
            progress = ((t_current - transiente) / (t_final - transiente)) * 100
            print(f"Progresso: {progress:.1f}%")

    # Calcular expoentes finais
    lyapunov_exponents = lyapunov_sums / (num_iterations * reorthonormalization_time)

    return lyapunov_exponents, np.array(evolucao_expoentes)

# Executar cálculo e plotar evolução
if __name__ == "__main__":
    # Calcular expoentes de Lyapunov com evolução
    exponents, evolucao = calcular_expoentes_lyapunov_com_evolucao(t_final=500, dt=0.01, transiente=100)

    print("\n=== Resultados dos Expoentes de Lyapunov ===")
    print(f"Dimensão do sistema: {dim_sistema}") # <-- ALTERAÇÃO (Valor maior)
    print(f"Número de expoentes calculados: {len(exponents)}")
    print(f"\nExpoentes de Lyapunov:")
    for i, exp in enumerate(exponents):
        print(f"λ_{i+1} = {exp:.6f}")

    print(f"\nSoma dos expoentes: {np.sum(exponents):.6f}")
    print(f"Número de expoentes positivos: {np.sum(exponents > 0)}")
    print(f"Maior expoente: {np.max(exponents):.6f}")

    # Plotar a evolução dos expoentes principais
    plt.figure(figsize=(15, 10))

    # Plot dos 6 primeiros expoentes (mais relevantes)
    num_expoentes_plot = min(6, dim_sistema)
    iteracoes = range(1, len(evolucao) + 1)

    plt.subplot(2, 1, 1)
    for i in range(num_expoentes_plot):
        plt.plot(iteracoes, evolucao[:, i], label=f'λ_{i+1}', linewidth=2)

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Número de Iterações')
    plt.ylabel('Expoente de Lyapunov')
    plt.title('Evolução dos Expoentes de Lyapunov Principais')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot de todos os expoentes
    plt.subplot(2, 1, 2)
    for i in range(dim_sistema):
        # Não plotar todos os labels para não poluir
        if i < 10:
             plt.plot(iteracoes, evolucao[:, i], label=f'λ_{i+1}', alpha=0.7)
        else:
             plt.plot(iteracoes, evolucao[:, i], alpha=0.5)

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Número de Iterações')
    plt.ylabel('Expoente de Lyapunov')
    plt.title(f'Evolução de Todos os {dim_sistema} Expoentes de Lyapunov')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()