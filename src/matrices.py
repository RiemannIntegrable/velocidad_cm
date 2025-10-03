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


def generar_caminata_doble(n, p, r):
    """
    Genera matriz de transición para caminata aleatoria en forma de "8".
    
    Args:
        n: Número total de estados (debe ser par)
        p: Probabilidad de ir al siguiente estado dentro de la misma mitad
        r: Probabilidad de ir de una mitad a la otra
    
    Returns:
        P: Matriz de transición
    """
    if n % 2 != 0:
        raise ValueError("n debe ser par para dividir en dos mitades")
    
    P = np.zeros((n, n))
    mitad = n // 2
    
    # Primera mitad: estados 0 a mitad-1
    for i in range(mitad):
        if i < mitad - 1:
            # Estados intermedios de la primera mitad
            P[i, i + 1] = p  # Siguiente en la misma mitad
            P[i, i - 1 if i > 0 else mitad - 1] = 1 - p  # Anterior (cíclico)
        else:
            # Último estado de la primera mitad (i = mitad-1)
            P[i, 0] = p  # Va al primero de la misma mitad
            P[i, i - 1] = 1 - p - r  # Anterior en la misma mitad
            P[i, mitad] = r  # Conexión a la segunda mitad
    
    # Segunda mitad: estados mitad a n-1
    for i in range(mitad, n):
        if i < n - 1:
            # Estados intermedios de la segunda mitad
            P[i, i + 1] = p  # Siguiente en la misma mitad
            P[i, i - 1 if i > mitad else n - 1] = 1 - p  # Anterior (cíclico)
        else:
            # Último estado de la segunda mitad (i = n-1)
            P[i, mitad] = p  # Va al primero de la segunda mitad
            P[i, i - 1] = 1 - p - r  # Anterior en la misma mitad
            P[i, mitad - 1] = r  # Conexión a la primera mitad
    
    return P


def generar_matriz_perturbada(n):
    """
    Genera matriz de transición basada en identidad con perturbación.
    
    Args:
        n: Número de estados
    
    Returns:
        P: Matriz de transición estocástica
    """
    epsilon = 1e-8
    P = np.eye(n)
    
    # Restar epsilon de la diagonal principal
    P -= epsilon * np.eye(n)
    
    # Añadir epsilon/2 a las diagonales superior e inferior
    for i in range(n - 1):
        P[i, i + 1] += epsilon / 2  # Diagonal superior
        P[i + 1, i] += epsilon / 2  # Diagonal inferior
    
    # Normalizar filas para asegurar que sumen 1
    row_sums = P.sum(axis=1)
    P = P / row_sums[:, np.newaxis]
    
    return P