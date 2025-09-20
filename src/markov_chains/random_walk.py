"""
Implementación de caminata aleatoria simple en grafos cíclicos.

Autor: José Miguel Acuña Hernández
Universidad Nacional de Colombia
"""

import numpy as np
from typing import Optional, Dict, Any
from .transition_matrix import TransitionMatrixGenerator


class RandomWalk:
    """Caminata aleatoria simple en un grafo cíclico."""

    def __init__(self, n_states: int, p: float, validate: bool = True):
        """
        Inicializa una caminata aleatoria.

        Args:
            n_states: Número de estados en el ciclo
            p: Probabilidad de ir al siguiente estado (q = 1-p para ir al anterior)
            validate: Si validar los parámetros

        Raises:
            ValueError: Si los parámetros no son válidos
        """
        if validate:
            if n_states < 2:
                raise ValueError("Se requieren al menos 2 estados")
            if not 0 < p < 1:
                raise ValueError("p debe estar en (0, 1)")

        self.n_states = n_states
        self.p = p
        self.q = 1 - p
        self._P = None
        self._stationary = None
        self._spectral_gap = None

    @property
    def transition_matrix(self) -> np.ndarray:
        """Obtiene la matriz de transición (lazy loading)."""
        if self._P is None:
            self._P = self._build_transition_matrix()
        return self._P

    def _build_transition_matrix(self) -> np.ndarray:
        """
        Construye la matriz de transición para la caminata aleatoria.

        Returns:
            Matriz de transición n×n
        """
        n = self.n_states
        P = np.zeros((n, n))

        for i in range(n):
            P[i, (i + 1) % n] = self.p
            P[i, (i - 1) % n] = self.q

        return P

    def get_stationary_distribution_theoretical(self) -> np.ndarray:
        """
        Calcula la distribución estacionaria teórica.

        Para caminata aleatoria simétrica (p = q = 0.5), la distribución
        es uniforme. Para p ≠ q, usa la fórmula analítica si existe.

        Returns:
            Distribución estacionaria
        """
        n = self.n_states

        if np.isclose(self.p, 0.5):
            return np.ones(n) / n
        else:
            if self.p != self.q:
                ratio = self.p / self.q
                if np.isclose(ratio, 1.0):
                    return np.ones(n) / n
                else:
                    pi = np.zeros(n)
                    for i in range(n):
                        pi[i] = ratio ** i
                    pi = pi / pi.sum()
                    return pi
            else:
                return np.ones(n) / n

    def simulate_path(self, initial_state: int, n_steps: int,
                     seed: Optional[int] = None) -> np.ndarray:
        """
        Simula una trayectoria de la caminata aleatoria.

        Args:
            initial_state: Estado inicial
            n_steps: Número de pasos a simular
            seed: Semilla para reproducibilidad

        Returns:
            Array con los estados visitados
        """
        if seed is not None:
            np.random.seed(seed)

        path = np.zeros(n_steps + 1, dtype=int)
        path[0] = initial_state

        for t in range(n_steps):
            current = path[t]
            if np.random.random() < self.p:
                path[t + 1] = (current + 1) % self.n_states
            else:
                path[t + 1] = (current - 1) % self.n_states

        return path

    def compute_empirical_distribution(self, n_steps: int = 10000,
                                      initial_state: int = 0,
                                      seed: Optional[int] = None) -> np.ndarray:
        """
        Calcula la distribución empírica mediante simulación.

        Args:
            n_steps: Número de pasos a simular
            initial_state: Estado inicial
            seed: Semilla para reproducibilidad

        Returns:
            Distribución empírica
        """
        path = self.simulate_path(initial_state, n_steps, seed)

        counts = np.bincount(path, minlength=self.n_states)
        return counts / len(path)

    def compute_hitting_time(self, start: int, target: int,
                            max_steps: int = 100000,
                            n_simulations: int = 1000) -> Dict[str, float]:
        """
        Estima el tiempo de primera llegada de start a target.

        Args:
            start: Estado inicial
            target: Estado objetivo
            max_steps: Máximo número de pasos por simulación
            n_simulations: Número de simulaciones

        Returns:
            Diccionario con estadísticas del tiempo de llegada
        """
        hitting_times = []

        for _ in range(n_simulations):
            current = start
            for step in range(1, max_steps + 1):
                if np.random.random() < self.p:
                    current = (current + 1) % self.n_states
                else:
                    current = (current - 1) % self.n_states

                if current == target:
                    hitting_times.append(step)
                    break

        if hitting_times:
            hitting_times = np.array(hitting_times)
            return {
                'mean': np.mean(hitting_times),
                'std': np.std(hitting_times),
                'min': np.min(hitting_times),
                'max': np.max(hitting_times),
                'median': np.median(hitting_times),
                'success_rate': len(hitting_times) / n_simulations
            }
        else:
            return {
                'mean': np.inf,
                'std': np.inf,
                'min': np.inf,
                'max': np.inf,
                'median': np.inf,
                'success_rate': 0.0
            }

    def compute_return_time(self, state: int, n_simulations: int = 1000,
                           max_steps: int = 100000) -> Dict[str, float]:
        """
        Estima el tiempo medio de retorno a un estado.

        Args:
            state: Estado de interés
            n_simulations: Número de simulaciones
            max_steps: Máximo número de pasos por simulación

        Returns:
            Diccionario con estadísticas del tiempo de retorno
        """
        return_times = []

        for _ in range(n_simulations):
            current = state
            for step in range(1, max_steps + 1):
                if np.random.random() < self.p:
                    current = (current + 1) % self.n_states
                else:
                    current = (current - 1) % self.n_states

                if current == state:
                    return_times.append(step)
                    break

        if return_times:
            return_times = np.array(return_times)
            return {
                'mean': np.mean(return_times),
                'std': np.std(return_times),
                'min': np.min(return_times),
                'max': np.max(return_times),
                'median': np.median(return_times),
                'theoretical': self.n_states if np.isclose(self.p, 0.5) else None
            }
        else:
            return {
                'mean': np.inf,
                'std': np.inf,
                'min': np.inf,
                'max': np.inf,
                'median': np.inf,
                'theoretical': self.n_states if np.isclose(self.p, 0.5) else None
            }

    def get_spectral_gap(self) -> float:
        """
        Calcula el gap espectral de la caminata aleatoria.

        Returns:
            Gap espectral (1 - |λ₂|)
        """
        if self._spectral_gap is None:
            eigenvalues = np.linalg.eigvals(self.transition_matrix)
            abs_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
            if len(abs_eigenvalues) > 1:
                self._spectral_gap = 1 - abs_eigenvalues[1]
            else:
                self._spectral_gap = 1.0
        return self._spectral_gap

    def get_properties(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de las propiedades de la caminata.

        Returns:
            Diccionario con propiedades
        """
        gen = TransitionMatrixGenerator()
        P = self.transition_matrix

        return {
            'n_states': self.n_states,
            'p': self.p,
            'q': self.q,
            'is_symmetric': np.isclose(self.p, 0.5),
            'is_irreducible': gen.is_irreducible(P),
            'is_aperiodic': gen.is_aperiodic(P),
            'spectral_gap': self.get_spectral_gap(),
            'mixing_time_025': gen.compute_mixing_time(P, epsilon=0.25),
            'stationary_uniform': np.isclose(self.p, 0.5)
        }

    def theoretical_hitting_time(self, start: int, target: int) -> Optional[float]:
        """
        Calcula el tiempo teórico de llegada para casos especiales.

        Args:
            start: Estado inicial
            target: Estado objetivo

        Returns:
            Tiempo esperado de llegada (None si no hay fórmula cerrada)
        """
        n = self.n_states
        dist = min((target - start) % n, (start - target) % n)

        if np.isclose(self.p, 0.5):
            return n * dist - dist * dist
        elif self.p > 0.5:
            forward_dist = (target - start) % n
            if forward_dist < n / 2:
                return forward_dist / (2 * self.p - 1) if self.p != 0.5 else None
        return None