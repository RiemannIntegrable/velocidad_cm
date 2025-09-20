import numpy as np

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