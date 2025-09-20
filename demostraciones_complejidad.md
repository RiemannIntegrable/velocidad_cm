# Demostraciones de Complejidad Computacional
## Análisis Detallado de los Algoritmos de Distribución Estacionaria

**Autor:** José Miguel Acuña Hernández
**Universidad Nacional de Colombia**

---

## 1. Complejidad del Método del Autovector (Sección 2.3)

### 1.1 Descomposición Espectral Directa

#### **Afirmación**: O(n³) temporal, O(n²) espacial

**DEMOSTRACIÓN COMPLETA:**

El algoritmo `np.linalg.eig(P.T)` realiza los siguientes pasos:

**Paso 1: Reducción a forma de Hessenberg** (para matrices no simétricas)
- Se aplica la transformación de Householder iterativamente
- Para cada columna k = 1, ..., n-2:
  - Calcular vector de Householder: O(n-k) operaciones
  - Aplicar transformación H·P·H^T: O((n-k)²) operaciones
- **Costo total paso 1**: ∑_{k=1}^{n-2} (n-k)² = O(n³)

**Paso 2: Algoritmo QR con shifts**
- Cada iteración QR:
  - Descomposición QR de matriz Hessenberg: O(n²)
  - Multiplicación RQ: O(n²)
- Número de iteraciones típico: O(n) para convergencia
- **Costo total paso 2**: O(n) × O(n²) = O(n³)

**Paso 3: Cálculo de autovectores**
- Back-substitution para cada autovector: O(n²)
- Para n autovectores: n × O(n²) = O(n³)

**TOTAL TEMPORAL**: O(n³) + O(n³) + O(n³) = **O(n³)**

**ANÁLISIS ESPACIAL:**
- Matriz P: n² elementos
- Matriz de autovectores: n² elementos
- Vectores auxiliares: O(n)
- **TOTAL ESPACIAL**: O(n²) + O(n²) + O(n) = **O(n²)**

---

### 1.2 Método de la Potencia

#### **Afirmación**: O(kn²) temporal, O(n) espacial

**DEMOSTRACIÓN COMPLETA:**

```python
for _ in range(max_iter):      # k iteraciones
    pi_nuevo = pi @ P           # Operación clave
    if norma < tol: break
    pi = pi_nuevo
```

**ANÁLISIS DE UNA ITERACIÓN:**

**Multiplicación vector-matriz `pi @ P`:**
- Para calcular el elemento j-ésimo del resultado:
  ```
  (pi @ P)[j] = ∑_{i=0}^{n-1} pi[i] × P[i,j]
  ```
  - Requiere n multiplicaciones y n-1 sumas: 2n-1 ≈ O(n) operaciones

- Para calcular todos los n elementos: n × O(n) = O(n²)

**Cálculo de norma:**
- `||pi_nuevo - pi||` requiere:
  - n restas
  - n valores absolutos (para norma L1)
  - n-1 sumas
  - Total: O(n)

**COSTO POR ITERACIÓN**: O(n²) + O(n) = O(n²)

**NÚMERO DE ITERACIONES (k):**

La convergencia depende del gap espectral. Si λ₂ es el segundo autovalor más grande:
- Error después de k iteraciones: O(|λ₂|^k)
- Para error < ε: k = O(log(ε) / log(|λ₂|))

En el peor caso: k = O(n) para matrices mal condicionadas
En caso típico: k = O(log(n)) para matrices bien condicionadas

**TOTAL TEMPORAL**: k × O(n²) = **O(kn²)**

**ANÁLISIS ESPACIAL:**
- Vector pi actual: n elementos
- Vector pi_nuevo: n elementos
- **TOTAL ESPACIAL**: O(n) + O(n) = **O(n)**

---

## 2. Complejidad del Método de Tiempos Medios (Sección 3.4)

### 2.1 Análisis Original (No Optimizado)

#### **Afirmación**: O(n⁴) temporal en el peor caso

**DEMOSTRACIÓN COMPLETA:**

```python
for j in range(n):                              # n iteraciones
    P_reducida = np.delete(np.delete(P, j, axis=0), j, axis=1)
    m_reducido = np.linalg.solve(I_reducida - P_reducida, b)
    tiempos_retorno[j] = 1 + np.dot(p_j_sin_j, m_reducido)
```

**ANÁLISIS POR ITERACIÓN:**

**Paso 1: Construcción de P_reducida**
- `np.delete(P, j, axis=0)`:
  - Copia n×(n-1) elementos: O(n²)
- `np.delete(..., j, axis=1)`:
  - Copia (n-1)×(n-1) elementos: O((n-1)²) = O(n²)
- **Total construcción**: O(n²)

**Paso 2: Resolver sistema lineal**
- Sistema de tamaño (n-1)×(n-1)
- `np.linalg.solve` usa descomposición LU:

  **2.1: Factorización LU**
  ```
  Para k = 0 hasta n-2:
      Para i = k+1 hasta n-1:
          L[i,k] = A[i,k] / A[k,k]        # O(1)
          Para j = k+1 hasta n-1:
              A[i,j] -= L[i,k] * A[k,j]   # O(1)
  ```

  - Bucle externo: n-1 iteraciones
  - Bucle medio: (n-1-k) iteraciones
  - Bucle interno: (n-1-k) iteraciones
  - **Costo LU**: ∑_{k=0}^{n-2} (n-1-k)² ≈ n³/3 operaciones

  **2.2: Forward substitution (Ly = b)**
  ```
  Para i = 0 hasta n-2:
      y[i] = b[i] - ∑_{j=0}^{i-1} L[i,j] * y[j]
  ```
  - **Costo**: ∑_{i=0}^{n-2} i = (n-1)(n-2)/2 ≈ n²/2

  **2.3: Backward substitution (Ux = y)**
  - Similar a forward: O(n²/2)

- **Total resolver sistema**: O(n³/3) + O(n²/2) + O(n²/2) = O(n³)

**Paso 3: Producto punto**
- `np.dot(p_j_sin_j, m_reducido)`: n-1 multiplicaciones + n-2 sumas = O(n)

**COSTO POR ITERACIÓN**: O(n²) + O(n³) + O(n) = O(n³)

**COSTO TOTAL**: n iteraciones × O(n³) = **O(n⁴)**

---

### 2.2 Por qué la Afirmación de O(n³) es Incorrecta

La documentación original dice "O(n³)" pero el análisis muestra O(n⁴).

**POSIBLE CONFUSIÓN:**
- Si resolvemos UN SOLO sistema de tamaño n×n: O(n³)
- Pero resolvemos n sistemas de tamaño (n-1)×(n-1): n × O(n³) = O(n⁴)

---

## 3. Optimización Propuesta: Reducir a O(n³)

### 3.1 Método de la Matriz Fundamental

En lugar de resolver n sistemas separados, podemos usar:

```python
def metodo_optimizado(P):
    # Resolver (I - P^T + ee^T)z = e
    A = np.eye(n) - P.T + np.ones((n,n))
    z = np.linalg.solve(A, np.ones(n))  # UN solo sistema n×n
    return z / z.sum()
```

**DEMOSTRACIÓN DE CORRECTITUD:**

La matriz A = I - P^T + ee^T tiene propiedades especiales:
- rank(A) = n (es no singular)
- Resolviendo Az = e obtenemos z ∝ π

**ANÁLISIS DE COMPLEJIDAD:**
- Construir A: O(n²)
- Resolver sistema n×n: O(n³)
- Normalizar: O(n)
- **TOTAL**: O(n²) + O(n³) + O(n) = **O(n³)**

---

## 4. Comparación de Constantes Ocultas

### 4.1 Método del Autovector

**Operaciones exactas** para n×n:
- Reducción Hessenberg: ≈ 10n³/3 flops
- Iteración QR: ≈ 6n³ flops (promedio)
- Autovectores: ≈ 3n³ flops
- **TOTAL**: ≈ 10n³ flops

### 4.2 Método de Tiempos (no optimizado)

**Operaciones exactas**:
- Por cada estado j:
  - Factorización LU: ≈ 2(n-1)³/3 flops
  - Substitución: ≈ 2(n-1)² flops
- Para n estados: ≈ 2n⁴/3 flops
- **TOTAL**: ≈ 0.67n⁴ flops

### 4.3 Razón de Tiempos

Para n = 100:
- Autovector: 10 × 100³ = 10⁷ flops
- Tiempos: 0.67 × 100⁴ = 6.7 × 10⁷ flops
- **Razón**: ≈ 6.7x más lento

Para n = 1000:
- Autovector: 10 × 10⁹ flops
- Tiempos: 0.67 × 10¹² flops
- **Razón**: ≈ 67x más lento

---

## 5. Efecto de Optimizaciones de NumPy

### 5.1 BLAS/LAPACK

NumPy usa bibliotecas optimizadas:
- **BLAS Level 3**: Multiplicación matricial con blocking de caché
- **LAPACK**: Algoritmos estables y paralelizados

**Speedup típico**:
- Operaciones matriciales: 10-100x vs Python puro
- Uso de múltiples cores: 2-8x adicional

### 5.2 Vectorización

```python
# Lento (Python puro)
for i in range(n):
    for j in range(n):
        C[i,j] = A[i,j] + B[i,j]  # n² llamadas a intérprete

# Rápido (NumPy)
C = A + B  # Una sola llamada a código C optimizado
```

---

## 6. Medición Empírica vs Teórica

### 6.1 Por qué los Exponentes Empíricos Pueden Diferir

**Factores que afectan las mediciones:**

1. **Tamaños pequeños**: Para n < 100, términos de orden inferior dominan
2. **Caché**: Matrices que caben en L2/L3 cache son más rápidas
3. **Paralelización automática**: BLAS puede usar múltiples threads
4. **Overhead del intérprete**: Constante aditiva significativa para n pequeño

### 6.2 Ejemplo de Análisis Empírico

Si medimos tiempos T(n) y ajustamos T(n) = c·n^α:

```
log(T) = log(c) + α·log(n)
```

Regresión lineal en escala log-log nos da:
- Pendiente = α (exponente empírico)
- Intercepto = log(c) (constante multiplicativa)

**Resultado típico**:
- Teórico: O(n³)
- Empírico: n^2.8 a n^3.2 (depende del rango de n)

---

## 7. Conclusiones

### Resumen de Complejidades

| Método | Teórica | Empírica Típica | Memoria |
|--------|---------|-----------------|---------|
| Autovector (eigen) | O(n³) | O(n^2.9) | O(n²) |
| Tiempos (no optimizado) | O(n⁴) | O(n^3.5) | O(n²) |
| Tiempos (optimizado) | O(n³) | O(n^3.1) | O(n²) |
| Potencia (k iters) | O(kn²) | O(n^2.5) para k=O(√n) | O(n) |

### Recomendaciones

1. **Para implementación práctica**: Usar método del autovector (NumPy optimizado)
2. **Para análisis teórico**: Considerar constantes ocultas
3. **Para matrices grandes**: Evaluar métodos iterativos dispersos
4. **Para precisión**: Método del autovector es más estable numéricamente

---

**Nota**: Las constantes exactas pueden variar según:
- Arquitectura del procesador
- Versión de NumPy/BLAS
- Tamaño de caché
- Número de cores disponibles