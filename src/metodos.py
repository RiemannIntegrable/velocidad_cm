import numpy as np

def metodo_autovector(P):

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