"""
Generador de matrices de transición para cadenas de Markov.

Autor: José Miguel Acuña Hernández
Universidad Nacional de Colombia
"""

import numpy as np
from typing import Optional, Tuple


class TransitionMatrixGenerator:
    """Clase para generar y validar matrices de transición."""

    @staticmethod
    def validate_stochastic_matrix(P: np.ndarray, tol: float = 1e-10) -> Tuple[bool, str]:
        """
        Valida que una matriz sea estocástica.

        Args:
            P: Matriz a validar
            tol: Tolerancia numérica

        Returns:
            (es_valida, mensaje_error)
        """
        if P.ndim != 2:
            return False, "La matriz debe ser bidimensional"

        if P.shape[0] != P.shape[1]:
            return False, "La matriz debe ser cuadrada"

        if np.any(P < -tol):
            return False, "La matriz contiene elementos negativos"

        row_sums = np.sum(P, axis=1)
        if not np.allclose(row_sums, 1.0, atol=tol):
            return False, f"Las filas no suman 1. Sumas: {row_sums}"

        return True, "Matriz estocástica válida"

    @staticmethod
    def is_irreducible(P: np.ndarray, max_power: int = 100) -> bool:
        """
        Verifica si una matriz de transición es irreducible.

        Una cadena es irreducible si desde cualquier estado se puede
        llegar a cualquier otro estado en un número finito de pasos.

        Args:
            P: Matriz de transición
            max_power: Máximo número de pasos a verificar

        Returns:
            True si la matriz es irreducible
        """
        n = P.shape[0]
        P_sum = np.zeros_like(P)
        P_power = np.eye(n)

        for _ in range(max_power):
            P_power = P_power @ P
            P_sum += P_power

            if np.all(P_sum > 0):
                return True

        return False

    @staticmethod
    def is_aperiodic(P: np.ndarray, max_steps: int = 100) -> bool:
        """
        Verifica si una matriz de transición es aperiódica.

        Una cadena es aperiódica si el MCD de los tiempos de retorno
        posibles a cualquier estado es 1.

        Args:
            P: Matriz de transición
            max_steps: Máximo número de pasos a verificar

        Returns:
            True si la matriz es aperiódica
        """
        n = P.shape[0]

        for i in range(n):
            return_times = []
            P_power = P.copy()

            for step in range(1, max_steps + 1):
                if P_power[i, i] > 0:
                    return_times.append(step)
                P_power = P_power @ P

            if return_times:
                from math import gcd
                from functools import reduce
                period = reduce(gcd, return_times)
                if period == 1:
                    return True

        return False

    @staticmethod
    def generate_random_stochastic(n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Genera una matriz estocástica aleatoria.

        Args:
            n: Tamaño de la matriz
            seed: Semilla para reproducibilidad

        Returns:
            Matriz estocástica n×n
        """
        if seed is not None:
            np.random.seed(seed)

        P = np.random.random((n, n))
        P = P / P.sum(axis=1, keepdims=True)

        return P

    @staticmethod
    def generate_sparse_stochastic(n: int, density: float = 0.1,
                                  seed: Optional[int] = None) -> np.ndarray:
        """
        Genera una matriz estocástica dispersa.

        Args:
            n: Tamaño de la matriz
            density: Proporción de elementos no-cero
            seed: Semilla para reproducibilidad

        Returns:
            Matriz estocástica dispersa n×n
        """
        if seed is not None:
            np.random.seed(seed)

        P = np.zeros((n, n))

        for i in range(n):
            num_nonzero = max(2, int(n * density))
            indices = np.random.choice(n, size=num_nonzero, replace=False)
            values = np.random.random(num_nonzero)
            P[i, indices] = values / values.sum()

        return P

    @staticmethod
    def compute_stationary_distribution_power(P: np.ndarray, tol: float = 1e-10,
                                             max_iter: int = 10000) -> np.ndarray:
        """
        Calcula la distribución estacionaria usando el método de la potencia.

        Args:
            P: Matriz de transición
            tol: Tolerancia para convergencia
            max_iter: Máximo número de iteraciones

        Returns:
            Distribución estacionaria
        """
        n = P.shape[0]
        pi = np.ones(n) / n

        for _ in range(max_iter):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi) < tol:
                return pi_new / pi_new.sum()
            pi = pi_new

        return pi / pi.sum()

    @staticmethod
    def compute_mixing_time(P: np.ndarray, epsilon: float = 0.25) -> int:
        """
        Estima el tiempo de mezcla de la cadena.

        El tiempo de mezcla es el número de pasos necesarios para que
        la distribución esté epsilon-cerca de la estacionaria.

        Args:
            P: Matriz de transición
            epsilon: Distancia de variación total objetivo

        Returns:
            Tiempo de mezcla estimado
        """
        n = P.shape[0]
        pi_star = TransitionMatrixGenerator.compute_stationary_distribution_power(P)

        P_t = np.eye(n)
        for t in range(1, 10000):
            P_t = P_t @ P

            max_tvd = 0
            for i in range(n):
                tvd = 0.5 * np.sum(np.abs(P_t[i, :] - pi_star))
                max_tvd = max(max_tvd, tvd)

            if max_tvd < epsilon:
                return t

        return -1

    @staticmethod
    def compute_spectral_gap(P: np.ndarray) -> float:
        """
        Calcula el gap espectral de la matriz.

        El gap espectral es 1 - |λ₂|, donde λ₂ es el segundo
        autovalor más grande en valor absoluto.

        Args:
            P: Matriz de transición

        Returns:
            Gap espectral
        """
        eigenvalues = np.linalg.eigvals(P)
        abs_eigenvalues = np.abs(eigenvalues)
        abs_eigenvalues.sort()
        abs_eigenvalues = abs_eigenvalues[::-1]

        if len(abs_eigenvalues) > 1:
            return 1 - abs_eigenvalues[1]
        else:
            return 1.0