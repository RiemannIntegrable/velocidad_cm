# CLAUDE.md - Tarea Cadenas de Markov Velocidad de convergencia

## 📋 Información del Proyecto
- **Nombre**: Tarea Cadenas de Markov Velocidad de convergencia
- **Descripción**: Tarea de Cadenas de Markov
- **Tipo**: analysis
- **Empresa**: UNAL
- **Autor**: José Miguel Acuña Hernández

## 🔧 Stack Técnico
- **Python**: 3.11


## 🎯 Contexto del Proyecto

### Objetivo Principal
[Completar: ¿Qué problema actuarial/analítico vas a resolver?]

### Datos del Proyecto
- **Fuente principal**: [Completar]
- **Tipo de datos**: [Siniestros, pólizas, mortalidad, etc.]
- **Volumen estimado**: [Completar]

### Metodología
- **Enfoque**: [Descriptivo, predictivo, prescriptivo]
- **Técnicas planeadas**: [GLM, supervivencia, clustering, etc.]

## 🏗️ Estructura del Proyecto
```
velocidad_cm/
├── data/              # Datos del proyecto
│   ├── raw/          # Datos originales
│   ├── processed/    # Datos listos para modelar
│   └── interim/      # Datos intermedios
├── notebooks/        # Análisis y exploración
├── src/              # Código reutilizable
├── reports/          # Reportes generados
└── models/           # Modelos entrenados
```

## 🔧 Comandos Útiles

- `/init` - Inicializar estructura CRISP-DM completa
- `/complete-context` - Completar contexto del proyecto interactivamente


## 📚 Contexto Especializado

**IMPORTANTE**: Para contexto detallado, lee estos archivos cuando sea relevante:

### 🏛️ Marco Regulatorio y Actuarial
**Lee:** `context/actuarial-context.md`
- Regulación colombiana (Superfinanciera)
- Conceptos actuariales clave (primas, reservas, solvencia)
- KPIs estándar del sector asegurador

### 📊 Esquemas y Datos
**Lee:** `context/data-schema.md` 
- Estructura típica de datos actuariales (pólizas, siniestros)
- Convenciones de nomenclatura
- Validaciones de calidad de datos

### 🔬 Metodologías Específicas
**Lee:** `context/methodology.md`
- CRISP-DM adaptado para actuaría
- Enfoques por línea de negocio (vida vs generales)
- Técnicas de validación cruzada temporal

### 📝 Estándares de Código
**Lee:** `context/standards.md`
- Convenciones Python/R para actuaría
- Templates de docstrings actuariales
- Estándares de visualización corporativa

### 💡 Ejemplos y Plantillas
**Lee cuando necesites implementar:**
- `context/examples/frequency_modeling.py` - Modelo GLM Poisson completo
- `context/templates/new_model_template.py` - Base para nuevos modelos


## 💡 Instrucciones para Claude

### Contexto Actuarial
- Usar terminología técnica apropiada para seguros
- Considerar regulación colombiana (Superfinanciera) cuando aplique
- Aplicar mejores prácticas de modelación actuarial
- Documentar supuestos claramente

- **Siempre consultar archivos en `context/` antes de implementar funcionalidades actuariales**

### Estilo de Código

**Python**:
- Seguir PEP 8
- Usar type hints
- Documentar con docstrings
- Tests con pytest




### Enfoque
- Priorizar precisión y robustez
- Explicar conceptos complejos claramente
- Incluir validaciones de datos
- Mantener trazabilidad de decisiones

### 📋 Guía de Uso del Contexto

**Antes de cualquier tarea actuarial, SIEMPRE:**
1. **Modelación de frecuencia/severidad** → Lee `context/examples/frequency_modeling.py`
2. **Nuevos modelos** → Lee `context/templates/new_model_template.py` 
3. **Validación de datos** → Lee `context/data-schema.md`
4. **Reportes regulatorios** → Lee `context/actuarial-context.md`
5. **Dudas de nomenclatura** → Lee `context/standards.md`
6. **Metodología CRISP-DM** → Lee `context/methodology.md`

**Instrucción clave:** Cada vez que te pidan implementar algo relacionado con actuaría, lee PRIMERO el archivo de contexto correspondiente, luego implementa.