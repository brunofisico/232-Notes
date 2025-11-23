"""Demonstração compacta de inversão da rotação interna usando splittings mistos.

O código constrói um perfil de rotação suave, gera kernels de modos mistos
realistas e recupera o perfil por regularização de Tikhonov. Também compara
frequências observadas e modeladas sem uso de argumentos de linha de comando.
"""

import smplotlib  # importação solicitada; módulo não é usado diretamente
import matplotlib.pyplot as plt
import numpy as np


def gerar_perfil_realista(n_pontos: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Cria um perfil de rotação suave decrescente do núcleo para a superfície."""
    raio = np.linspace(0.0, 1.0, n_pontos)
    nucleo_lento = 520.0 - 180.0 / (1.0 + np.exp(-(raio - 0.18) / 0.05))
    envelope = 220.0 + 65.0 * np.exp(-((raio - 0.82) / 0.18) ** 2)
    omega_real = nucleo_lento + envelope * np.exp(-(raio / 0.9) ** 3)
    return raio, omega_real


def gerar_kernels(n_modos: int, raio: np.ndarray) -> np.ndarray:
    """Gera kernels de modos mistos concentrados no núcleo e envelope."""
    kernels = []
    m_central = 0.55 * n_modos
    for j in range(n_modos):
        peso_nucleo = 0.35 + 0.55 * np.exp(-((j - m_central) / (0.32 * n_modos)) ** 2)
        peso_envelope = 1.0 - peso_nucleo
        componente_g = np.exp(-((raio - 0.22) / 0.07) ** 2)
        componente_p = np.exp(-((raio - 0.82) / 0.23) ** 2)
        kernel = peso_nucleo * componente_g + peso_envelope * componente_p
        kernel /= np.trapz(kernel, raio)
        kernels.append(kernel)
    return np.vstack(kernels)


def gerar_splittings(kernels: np.ndarray, omega_real: np.ndarray, raio: np.ndarray) -> np.ndarray:
    """Calcula splittings a partir dos kernels e do perfil verdadeiro."""
    return np.array([np.trapz(kernel * omega_real, raio) for kernel in kernels])


def gerar_frequencias_base(n_modos: int) -> np.ndarray:
    """Cria uma sequência de frequências de modos dipolares em µHz."""
    return 95.0 + 11.5 * np.arange(n_modos)


def inverter_rotacao(kernels: np.ndarray, splittings_obs: np.ndarray, ruido: np.ndarray, raio: np.ndarray,
                     lambda_reg: float = 1e-3) -> np.ndarray:
    """Aplica inversão com regularização de Tikhonov de primeira derivada."""
    n_r = raio.size
    primeira_derivada = -2.0 * np.eye(n_r)
    primeira_derivada += np.eye(n_r, k=1) + np.eye(n_r, k=-1)
    primeira_derivada[0, 0] = primeira_derivada[-1, -1] = -1.0
    W = np.diag(1.0 / (ruido ** 2))
    A = kernels
    lhs = A.T @ W @ A + lambda_reg * (primeira_derivada.T @ primeira_derivada)
    rhs = A.T @ W @ splittings_obs
    solucao, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return solucao


def sintetizar_observaveis(n_modos: int = 18, ruido_splitting: float = 12.0) -> dict:
    """Combina perfil, kernels e ruídos para gerar observáveis realistas."""
    raio, omega_real = gerar_perfil_realista()
    kernels = gerar_kernels(n_modos, raio)
    splittings_limpos = gerar_splittings(kernels, omega_real, raio)
    ruido = np.full_like(splittings_limpos, ruido_splitting)
    splittings_obs = splittings_limpos + np.random.normal(0.0, ruido_splitting, size=n_modos)
    frequencias_modelo = gerar_frequencias_base(n_modos)
    frequencias_obs = frequencias_modelo + 0.001 * splittings_obs
    return {
        "raio": raio,
        "omega_real": omega_real,
        "kernels": kernels,
        "splittings_obs": splittings_obs,
        "ruido": ruido,
        "frequencias_modelo": frequencias_modelo,
        "frequencias_obs": frequencias_obs,
    }


def resumo_tabelado(frequencias_obs: np.ndarray, frequencias_modelo: np.ndarray, splittings_obs: np.ndarray,
                    splittings_rec: np.ndarray) -> str:
    linhas = ["Modo | nu_obs [µHz] | nu_mod [µHz] | split_obs [nHz] | split_rec [nHz]"]
    for idx, (fo, fm, so, sr) in enumerate(zip(frequencias_obs, frequencias_modelo, splittings_obs, splittings_rec)):
        linhas.append(f"{idx:>4d} | {fo:>11.3f} | {fm:>11.3f} | {so:>13.2f} | {sr:>13.2f}")
    return "\n".join(linhas)


def plotar_perfis_rotacao(raio: np.ndarray, omega_real: np.ndarray, omega_rec: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(raio, omega_real, label="Perfil verdadeiro", linewidth=2.0)
    ax.plot(raio, omega_rec, label="Perfil recuperado", linestyle="--", linewidth=2.0)
    ax.set_xlabel("raio fracionário r/R")
    ax.set_ylabel(r"$\Omega$ [nHz]")
    ax.set_title("Perfil de rotação interno")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()


def plotar_kernels(kernels: np.ndarray, raio: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    indices = np.linspace(0, kernels.shape[0] - 1, 5, dtype=int)
    for idx in indices:
        ax.plot(raio, kernels[idx], label=f"Kernel {idx}")
    ax.set_xlabel("raio fracionário r/R")
    ax.set_ylabel("Sensibilidade normalizada")
    ax.set_title("Kernels de modos mistos")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()


def plotar_splittings(frequencias_obs: np.ndarray, splittings_obs: np.ndarray, splittings_rec: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.errorbar(frequencias_obs, splittings_obs, yerr=splittings_obs * 0.0 + 12.0, fmt="o", label="Observado")
    ax.plot(frequencias_obs, splittings_rec, "x", label="Reconstruído")
    ax.set_xlabel("Frequência observada [µHz]")
    ax.set_ylabel("Splitting [nHz]")
    ax.set_title("Splittings observados vs. recuperados")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()


if __name__ == "__main__":
    np.random.seed(7)
    observaveis = sintetizar_observaveis()
    raio = observaveis["raio"]
    omega_real = observaveis["omega_real"]
    kernels = observaveis["kernels"]
    splittings_obs = observaveis["splittings_obs"]
    ruido = observaveis["ruido"]
    frequencias_modelo = observaveis["frequencias_modelo"]
    frequencias_obs = observaveis["frequencias_obs"]

    omega_rec = inverter_rotacao(kernels, splittings_obs, ruido, raio)
    splittings_rec = gerar_splittings(kernels, omega_rec, raio)

    print("Perfil verdadeiro e recuperado (nHz) em pontos radiais selecionados:\n")
    pontos = np.linspace(0, 1, 6)
    for p in pontos:
        idx = np.argmin(np.abs(raio - p))
        print(f"r/R={raio[idx]:.2f}  Omega_real={omega_real[idx]:7.2f}  Omega_rec={omega_rec[idx]:7.2f}")

    print("\nFrequências observadas versus modeladas e splittings:\n")
    print(resumo_tabelado(frequencias_obs, frequencias_modelo, splittings_obs, splittings_rec))

    plotar_perfis_rotacao(raio, omega_real, omega_rec)
    plotar_kernels(kernels, raio)
    plotar_splittings(frequencias_obs, splittings_obs, splittings_rec)
    plt.show()
