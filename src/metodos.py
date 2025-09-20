import numpy as np

def metodo_autovector(P):
    """
    Calcula la distribución estacionaria usando el autovector del autovalor 1.

    Args:
        P: Matriz de transición (n×n)

    Returns:
        pi: Distribución estacionaria
    """
    # Calcular autovalores y autovectores de P^T
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Encontrar el índice del autovalor ≈ 1
    idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)

    # Extraer y normalizar el autovector correspondiente
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)
    pi = pi / pi.sum()

    return pi


def metodo_tiempos_retorno(P):
    """
    Calcula la distribución estacionaria usando tiempos medios de retorno.
    Implementación según la formulación matricial del documento teórico.

    Args:
        P: Matriz de transición (n×n)

    Returns:
        pi: Distribución estacionaria
    """
    n = P.shape[0]
    tiempos_retorno = np.zeros(n)

    # Pre-calcular matriz identidad reducida (reusar memoria)
    I_reducida = np.eye(n - 1)
    b = np.ones(n - 1)

    for j in range(n):
        # Para calcular el tiempo medio de retorno al estado j:
        # 1. Removemos fila j y columna j de P para obtener P_{-j}
        # 2. Resolvemos (I - P_{-j}) * m = 1 para obtener tiempos desde estados i≠j hacia j
        # 3. Calculamos μ_jj = 1 + Σ p_jk * m_kj para k≠j

        # Índices sin el estado j
        indices_sin_j = np.concatenate([np.arange(j), np.arange(j+1, n)])

        # Extraer submatriz P_{-j} (sin fila j y columna j)
        P_reducida = P[indices_sin_j][:, indices_sin_j]

        # Resolver sistema (I - P_{-j}) * m = 1
        # donde m contiene los tiempos medios desde cada estado i≠j hacia j
        m = np.linalg.solve(I_reducida - P_reducida, b)

        # Calcular tiempo medio de retorno desde j
        # μ_jj = 1 + Σ p_jk * m_kj para k≠j
        p_j_sin_j = P[j, indices_sin_j]  # Probabilidades desde j hacia otros estados
        tiempos_retorno[j] = 1 + p_j_sin_j @ m

    # Calcular distribución estacionaria: π_j = 1/μ_jj
    pi = 1.0 / tiempos_retorno
    return pi / pi.sum()  # Normalizar


def generar_caminata_aleatoria(n, p):
    """
    Genera matriz de transición para caminata aleatoria cíclica.

    Args:
        n: Número de estados
        p: Probabilidad de ir al siguiente estado (q = 1-p al anterior)

    Returns:
        P: Matriz de transición
    """
    q = 1 - p
    P = np.zeros((n, n))

    for i in range(n):
        P[i, (i + 1) % n] = p  # Probabilidad de ir al siguiente (mod n para ciclo)
        P[i, (i - 1) % n] = q  # Probabilidad de ir al anterior (mod n para ciclo)

    return P