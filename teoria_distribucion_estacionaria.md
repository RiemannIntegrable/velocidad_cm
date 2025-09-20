# Teoría de Distribuciones Estacionarias en Cadenas de Markov
## Análisis Comparativo de Métodos Computacionales

**Autor:** José Miguel Acuña Hernández
**Universidad Nacional de Colombia - Maestría en Actuaría y Finanzas**
**Fecha:** Septiembre 2025

---

## 1. Fundamentos Matemáticos de Cadenas de Markov

### 1.1 Definiciones Fundamentales

Una **cadena de Markov** es un proceso estocástico {Xₙ}ₙ₌₀^∞ con espacio de estados S (finito o numerable) que satisface la propiedad de Markov:

```
P(Xₙ₊₁ = j | X₀ = i₀, X₁ = i₁, ..., Xₙ = i) = P(Xₙ₊₁ = j | Xₙ = i) = pᵢⱼ
```

#### Propiedades Clave para Nuestro Estudio:

1. **Homogénea**: Las probabilidades de transición pᵢⱼ no dependen del tiempo n.

2. **Irreducible**: Para todo par de estados i, j ∈ S, existe n ≥ 0 tal que P(Xₙ = j | X₀ = i) > 0.
   - Intuitivamente: desde cualquier estado puedo llegar a cualquier otro.

3. **Aperiódica**: Para todo estado i, el máximo común divisor del conjunto {n ≥ 1 : pᵢᵢ⁽ⁿ⁾ > 0} es 1.
   - Intuitivamente: no hay ciclos determinísticos.

### 1.2 Matriz de Transición

La **matriz de transición** P = [pᵢⱼ] es una matriz estocástica donde:
- pᵢⱼ ≥ 0 para todo i, j
- ∑ⱼ pᵢⱼ = 1 para todo i (cada fila suma 1)

Para una cadena con n estados, P es una matriz n×n.

### 1.3 Distribución Estacionaria

Una distribución de probabilidad π = (π₁, π₂, ..., πₙ) es **estacionaria** si:

```
πP = π
```

Es decir, π es un autovector izquierdo de P con autovalor 1.

#### Teorema Fundamental (Teorema Ergódico):
Si una cadena de Markov es finita, irreducible y aperiódica, entonces:
1. Existe una única distribución estacionaria π
2. lim_{n→∞} P^n = 1π^T (convergencia a matriz con filas idénticas iguales a π)
3. πᵢ = 1/μᵢᵢ donde μᵢᵢ es el tiempo esperado de retorno al estado i

---

## 2. Método del Autovector (Eigenvalue Method)

### 2.1 Fundamento Teórico

La ecuación πP = π puede reescribirse como:
```
π(P - I) = 0
```

Equivalentemente, buscamos el autovector izquierdo de P correspondiente al autovalor λ = 1.

**Observación Clave**: Por el teorema de Perron-Frobenius, para una matriz estocástica irreducible y aperiódica:
- λ = 1 es el autovalor dominante
- |λᵢ| < 1 para todos los demás autovalores
- El autovector correspondiente a λ = 1 tiene componentes no negativas

### 2.2 Implementación Algorítmica

#### Opción A: Descomposición Espectral Directa
```python
def metodo_autovector_directo(P):
    # Calculamos autovalores y autovectores de P^T
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Encontramos el índice del autovalor ≈ 1
    idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)

    # Extraemos el autovector correspondiente
    pi = np.real(eigenvectors[:, idx])

    # Normalizamos para que sume 1
    pi = pi / np.sum(pi)

    return pi
```

#### Opción B: Método de la Potencia (Power Method)
```python
def metodo_potencia(P, tol=1e-10, max_iter=1000):
    n = P.shape[0]
    pi = np.ones(n) / n  # Distribución uniforme inicial

    for _ in range(max_iter):
        pi_nuevo = pi @ P
        if np.linalg.norm(pi_nuevo - pi) < tol:
            break
        pi = pi_nuevo

    return pi
```

### 2.3 Análisis de Complejidad

#### Descomposición Espectral:
- **Complejidad temporal**: O(n³)
  - La descomposición en valores propios requiere O(n³) operaciones
- **Complejidad espacial**: O(n²)
  - Almacenar la matriz de autovectores

#### Método de la Potencia:
- **Complejidad temporal**: O(kn²)
  - k = número de iteraciones (depende de |λ₂|, el segundo autovalor más grande)
  - Cada iteración: multiplicación vector-matriz O(n²)
- **Complejidad espacial**: O(n)
  - Solo necesitamos almacenar el vector actual

---

## 3. Método de Tiempos Medios de Primer Arribo

### 3.1 Fundamento Teórico

Este método se basa en el siguiente resultado fundamental:

**Teorema**: Para una cadena irreducible y aperiódica, la probabilidad estacionaria de estar en el estado i es:
```
πᵢ = 1/μᵢᵢ
```
donde μᵢᵢ es el **tiempo medio de retorno** al estado i partiendo desde i.

### 3.2 Cálculo de Tiempos Medios

Los tiempos medios de primer paso mᵢⱼ satisfacen el sistema de ecuaciones:

```
mᵢⱼ = 1 + ∑_{k≠j} pᵢₖ mₖⱼ    para i ≠ j
mⱼⱼ = 0
```

En forma matricial, para calcular los tiempos medios al estado j:
```
(I - P_{-j}) m_{·j} = 1
```
donde P_{-j} es P con la j-ésima fila y columna removidas.

### 3.3 Implementación Algorítmica

```python
def metodo_tiempos_medios(P):
    n = P.shape[0]
    tiempos_retorno = np.zeros(n)

    for j in range(n):
        # Construimos el sistema para el estado j
        # Necesitamos calcular m_jj (tiempo medio de retorno)

        # Paso 1: Resolver para tiempos de primera llegada desde j
        P_reducida = np.delete(np.delete(P, j, axis=0), j, axis=1)
        I_reducida = np.eye(n-1)
        b = np.ones(n-1)

        # Sistema: (I - P_reducida) * m = b
        m_reducido = np.linalg.solve(I_reducida - P_reducida, b)

        # Paso 2: Calcular tiempo medio de retorno
        p_j = P[j, :]
        p_j_sin_j = np.delete(p_j, j)
        tiempos_retorno[j] = 1 + np.dot(p_j_sin_j, m_reducido)

    # Distribución estacionaria
    pi = 1 / tiempos_retorno
    pi = pi / np.sum(pi)  # Normalizar

    return pi
```

### 3.4 Análisis de Complejidad

- **Complejidad temporal**: O(n³)
  - Resolver n sistemas lineales de tamaño (n-1)×(n-1)
  - Cada sistema: O((n-1)³) ≈ O(n³)
  - Total: O(n × n³) = O(n⁴)

  **Optimización**: Si solo necesitamos π, podemos usar un enfoque más eficiente resolviendo un solo sistema aumentado.

- **Complejidad espacial**: O(n²)
  - Almacenar las matrices reducidas

---

## 4. Complejidad Computacional: Conceptos Fundamentales

### 4.1 Notación Big-O

La notación O(·) describe el comportamiento asintótico de un algoritmo cuando el tamaño del problema (n) tiende a infinito.

**Definición matemática**: f(n) = O(g(n)) si existen constantes c > 0 y n₀ tal que:
```
f(n) ≤ c·g(n) para todo n ≥ n₀
```

### 4.2 Clases de Complejidad Comunes

| Complejidad | Nombre | Ejemplo | Tiempo para n=1000 |
|------------|--------|---------|-------------------|
| O(1) | Constante | Acceso a array[i] | ~1 μs |
| O(log n) | Logarítmica | Búsqueda binaria | ~10 μs |
| O(n) | Lineal | Suma de array | ~1 ms |
| O(n log n) | Linearítmica | Ordenamiento eficiente | ~10 ms |
| O(n²) | Cuadrática | Multiplicación matriz-vector | ~1 s |
| O(n³) | Cúbica | Multiplicación matricial | ~17 min |

### 4.3 Análisis Empírico vs Teórico

**Teórico**: Analiza el número de operaciones fundamentales.
**Empírico**: Mide tiempo real de ejecución y uso de memoria.

Factores que afectan el rendimiento real:
- Constantes ocultas en Big-O
- Localidad de caché
- Optimizaciones del compilador/intérprete
- Paralelización implícita (BLAS, etc.)

---

## 5. Caso de Estudio: Caminata Aleatoria Simple

### 5.1 Definición del Modelo

Consideramos una caminata aleatoria en un grafo cíclico con n estados:
- Estados: {0, 1, 2, ..., n-1}
- Transiciones:
  - P(i → i+1 mod n) = p
  - P(i → i-1 mod n) = q = 1-p
  - p, q > 0

### 5.2 Matriz de Transición

```
     [ 0   p   0   ...  0   q ]
     [ q   0   p   ...  0   0 ]
P =  [ 0   q   0   ...  0   0 ]
     [ .   .   .   ...  .   . ]
     [ p   0   0   ...  q   0 ]
```

### 5.3 Propiedades Especiales

1. **Simetría**: Si p = q = 0.5, la distribución estacionaria es uniforme: πᵢ = 1/n

2. **Reversibilidad**: La cadena es reversible si satisface:
   ```
   πᵢ pᵢⱼ = πⱼ pⱼᵢ
   ```

3. **Gap Espectral**: λ₂ = cos(2π/n) determina la velocidad de convergencia

---

## 6. Métricas de Comparación

### 6.1 Métricas de Eficiencia Temporal

1. **Tiempo de CPU**: Tiempo de procesamiento puro
   ```python
   import time
   start = time.process_time()
   # código
   tiempo_cpu = time.process_time() - start
   ```

2. **Tiempo de Wall-Clock**: Tiempo real transcurrido
   ```python
   start = time.perf_counter()
   # código
   tiempo_real = time.perf_counter() - start
   ```

### 6.2 Métricas de Eficiencia Espacial

1. **Memoria Peak**: Máxima memoria utilizada
   ```python
   from memory_profiler import memory_usage
   mem = memory_usage(funcion)
   memoria_max = max(mem) - min(mem)
   ```

2. **Memoria Promedio**: Uso promedio durante ejecución

### 6.3 Métricas de Precisión

1. **Error Absoluto**: ||π_calculado - π_verdadero||₁
2. **Error Relativo**: ||π_calculado - π_verdadero||₁ / ||π_verdadero||₁
3. **Distancia de Variación Total**: ½ ∑|πᵢ_calc - πᵢ_verd|

---

## 7. Análisis Comparativo Esperado

### 7.1 Ventajas del Método del Autovector

1. **Precisión numérica**: Usa bibliotecas optimizadas (LAPACK)
2. **Estabilidad**: Menos sensible a condicionamiento
3. **Información adicional**: Obtiene todos los autovalores (útil para análisis de convergencia)

### 7.2 Ventajas del Método de Tiempos Medios

1. **Interpretabilidad**: Tiempos medios tienen significado directo
2. **Flexibilidad**: Puede calcular tiempos entre estados específicos
3. **Robustez**: Menos sensible a casi-periodicidad

### 7.3 Factores que Afectan el Rendimiento

1. **Tamaño del espacio de estados (n)**:
   - Ambos métodos son O(n³) en el peor caso
   - Para n grande, memoria puede ser limitante

2. **Estructura de P**:
   - Matrices dispersas favorecen métodos iterativos
   - Matrices densas favorecen métodos directos

3. **Gap espectral (1 - |λ₂|)**:
   - Afecta convergencia del método de potencia
   - No afecta métodos directos

---

## 8. Herramientas de Medición Recomendadas

### 8.1 Para Tiempo de Ejecución

```python
import timeit
import time

# Opción 1: timeit para mediciones precisas
tiempo = timeit.timeit(lambda: funcion(), number=100) / 100

# Opción 2: time.perf_counter para mediciones únicas
start = time.perf_counter()
resultado = funcion()
elapsed = time.perf_counter() - start
```

### 8.2 Para Uso de Memoria

```python
from memory_profiler import profile, memory_usage
import tracemalloc

# Opción 1: memory_profiler
mem_usage = memory_usage((funcion, (args,)))

# Opción 2: tracemalloc (nativo de Python)
tracemalloc.start()
resultado = funcion()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
```

### 8.3 Para Perfilado Completo

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
resultado = funcion()
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 funciones
```

---

## 9. Preguntas de Investigación

### Pregunta 1: ¿Cuál método es más eficiente?

**Hipótesis**: Para matrices pequeñas (n < 100), el método del autovector será más eficiente debido a implementaciones optimizadas. Para matrices grandes y dispersas, métodos iterativos podrían ser superiores.

**Métricas a evaluar**:
- Tiempo de ejecución vs. n
- Memoria utilizada vs. n
- Escalabilidad (pendiente en escala log-log)

### Pregunta 2: ¿En qué condiciones uno es más eficiente?

**Factores a investigar**:
1. **Sesgo de la caminata** (p vs. q):
   - p ≈ q: cadena casi simétrica
   - p >> q o p << q: cadena con deriva fuerte

2. **Tamaño del espacio**:
   - Pequeño: n ∈ [10, 50]
   - Mediano: n ∈ [50, 500]
   - Grande: n ∈ [500, 5000]

3. **Precisión requerida**:
   - Alta precisión: ε < 10⁻¹⁰
   - Precisión moderada: ε ≈ 10⁻⁶

### Pregunta 3: ¿Qué tan más eficiente es uno vs. otro?

**Análisis cuantitativo**:
```
Factor de mejora = Tiempo_método_A / Tiempo_método_B
Eficiencia relativa = (1 - Tiempo_mejor/Tiempo_peor) × 100%
```

**Visualizaciones propuestas**:
1. Gráfica log-log de tiempo vs. n para ambos métodos
2. Heatmap de eficiencia relativa en función de (n, p)
3. Curvas de trade-off precisión vs. tiempo

---

## 10. Implementación Práctica: Consideraciones

### 10.1 Optimizaciones Numéricas

1. **Uso de BLAS**: NumPy automáticamente usa BLAS optimizado
2. **Vectorización**: Evitar loops explícitos en Python
3. **Precisión numérica**: Usar float64 para evitar acumulación de errores

### 10.2 Validación de Resultados

Verificaciones esenciales:
1. ∑πᵢ = 1 (normalización)
2. πᵢ ≥ 0 para todo i (no negatividad)
3. ||πP - π|| < ε (estacionariedad)

### 10.3 Casos Especiales

1. **Cadenas periódicas**: Detectar y reportar error
2. **Cadenas reducibles**: Identificar clases de comunicación
3. **Matrices mal condicionadas**: Usar regularización si necesario

---

## Referencias Sugeridas

1. **Teoría de Cadenas de Markov**:
   - Norris, J.R. (1997). *Markov Chains*. Cambridge University Press.
   - Brémaud, P. (1999). *Markov Chains: Gibbs Fields, Monte Carlo Simulation, and Queues*.

2. **Análisis Numérico**:
   - Golub, G.H. & Van Loan, C.F. (2013). *Matrix Computations*. Johns Hopkins.
   - Stewart, W.J. (1994). *Introduction to the Numerical Solution of Markov Chains*.

3. **Complejidad Computacional**:
   - Cormen, T.H. et al. (2009). *Introduction to Algorithms*. MIT Press.
   - Papadimitriou, C.H. (1994). *Computational Complexity*. Addison-Wesley.

---

## Apéndice: Código de Referencia Rápida

### A.1 Generación de Matriz de Caminata Aleatoria

```python
def generar_caminata_aleatoria(n, p):
    """
    Genera matriz de transición para caminata aleatoria cíclica.

    Args:
        n: número de estados
        p: probabilidad de ir al siguiente estado

    Returns:
        P: matriz de transición n×n
    """
    q = 1 - p
    P = np.zeros((n, n))

    for i in range(n):
        P[i, (i+1) % n] = p
        P[i, (i-1) % n] = q

    return P
```

### A.2 Verificación de Distribución Estacionaria

```python
def verificar_estacionaria(P, pi, tol=1e-10):
    """
    Verifica que pi sea distribución estacionaria de P.

    Returns:
        (es_valida, error)
    """
    pi_nuevo = pi @ P
    error = np.linalg.norm(pi_nuevo - pi)
    es_valida = error < tol

    return es_valida, error
```

### A.3 Cálculo de Gap Espectral

```python
def calcular_gap_espectral(P):
    """
    Calcula el gap espectral (1 - |λ₂|).
    """
    eigenvalues = np.linalg.eigvals(P)
    eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]

    if len(eigenvalues) > 1:
        gap = 1 - eigenvalues_sorted[1]
    else:
        gap = 1

    return gap
```

---

**Nota Final**: Este documento proporciona la base teórica completa para implementar y comparar ambos métodos. La implementación práctica debe considerar las optimizaciones mencionadas y validar cuidadosamente los resultados para garantizar precisión numérica.