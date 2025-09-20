# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic research project comparing computational complexity of stationary distribution methods for Markov chains. The project is part of a Master's degree in Actuarial Science and Finance at Universidad Nacional de Colombia.

## Core Architecture

### Module Structure

- `src/metodos.py`: Contains two main algorithms for computing stationary distributions:
  - `metodo_autovector(P)`: Computes stationary distribution using eigenvalue decomposition
  - `metodo_tiempos_retorno(P)`: Computes stationary distribution using mean return times matrix formulation

- `src/matrices.py`: Matrix generation utilities
  - `generar_caminata_aleatoria(n, p)`: Generates cyclic random walk transition matrices

- `notebooks/analisis_comparativo.ipynb`: Main analysis notebook comparing both methods' performance

### Mathematical Foundation

The project implements two approaches from `teoria_distribucion_estacionaria.md`:
1. **Eigenvalue method**: Finding eigenvector for eigenvalue 1 of transition matrix P^T
2. **Mean return times**: Using πᵢ = 1/μᵢᵢ where μᵢᵢ is mean return time to state i

## Development Commands

```bash
# Activate conda environment
conda activate velocidad_cm_python

# Run tests on methods
python test_metodos.py

# Execute notebook analysis
jupyter notebook notebooks/analisis_comparativo.ipynb

# Run performance comparison
python run_notebook.py
```

## Key Implementation Details

### Corrected Implementation Notes

1. **metodo_tiempos_retorno**: Must iterate over destination state `j`, not source state `i`. The algorithm:
   - Removes row j and column j from P to get P_{-j}
   - Solves (I - P_{-j})m = 1 for mean hitting times
   - Computes return time: μⱼⱼ = 1 + Σ pⱼₖ·mₖⱼ

2. **generar_caminata_aleatoria**: Generates true cyclic random walk without diagonal entries. Each state has probability `p` to next state and `1-p` to previous state (modulo n).

### Complexity Analysis

Both methods show empirical complexity around O(n^2.1) to O(n^2.4), though theoretical complexity is O(n³). The eigenvalue method is consistently ~3x faster due to optimized LAPACK implementations in NumPy.

## Testing Approach

When testing modifications:
1. Verify both methods produce identical distributions (difference < 1e-14)
2. Check stationarity: ||πP - π|| < 1e-10
3. Ensure row sums of transition matrix equal 1

## Performance Measurement

Use `time.perf_counter()` for timing measurements, not `timeit` for single runs. Run multiple iterations (typically 20) to get stable averages.