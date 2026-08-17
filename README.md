# Predicción de fuga de clientes (customer churn) — clasificación binaria

Pipeline completo de Machine Learning supervisado sobre un problema de **churn en
telecomunicaciones**: desde el dato crudo hasta la elección del umbral de decisión y la
comparación de modelos.

Trabajo desarrollado en el módulo de *Minería de datos y Modelización Predictiva* del
Máster en Big Data, Data Science e Inteligencia Artificial (Universidad Complutense de
Madrid, curso 2025-26).

## El problema

Dado el comportamiento de uso, la información geográfica y las características del plan
contratado por cada cliente, predecir cuáles tienen alta probabilidad de darse de baja.
El dataset tiene **9.200 registros y 20 variables predictoras anonimizadas** (`V1`–`V20`)
más la variable objetivo binaria `Y`.

Al ser un problema con clases desbalanceadas y coste asimétrico (perder un cliente cuesta
mucho más que una campaña de retención innecesaria), el foco no está en la *accuracy* sino
en el compromiso precisión/recall y en dónde se coloca el umbral.

## Metodología

1. **Duplicados** — detectados y tratados manteniendo en paralelo la versión con y sin
   duplicados, para medir su efecto real sobre las métricas en lugar de asumirlo.
2. **Valores perdidos** — *heatmap* y correlación de nulos para distinguir un patrón
   MCAR de uno estructurado; imputación por mediana/moda con `SimpleImputer` ajustado
   **solo sobre train** para no filtrar información del test.
3. **Valores extremos** — doble criterio, rango intercuartílico (IQR) e `IsolationForest`,
   comparando qué marca cada uno antes de decidir.
4. **Reducción de dimensionalidad** — análisis de componentes principales (PCA) y
   reanálisis con 2 componentes; clustering para caracterizar segmentos de cliente.
5. **Modelado** — regresión logística, árboles de decisión y perceptrón multicapa (MLP),
   con búsqueda de hiperparámetros por *grid search* y validación cruzada estratificada.
6. **Selección del umbral** — barrido de *thresholds* sobre las probabilidades predichas
   en lugar del 0,5 por defecto, con matrices de confusión para cada punto de corte.
7. **Comparación final** — *boxplots* de la distribución de métricas entre configuraciones,
   importancia de variables e intervalos de confianza sobre las predicciones.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `notebooks/churn_clasificacion.ipynb` | Análisis completo, de la carga de datos a la comparación final de modelos. |
| `data/BBDD_ML_TAREA.csv` | Dataset de trabajo (9.200 × 21). |
| `docs/memoria_pca_clustering.pdf` | Memoria escrita: PCA, clustering y comparación de predicciones. |
| `scripts/funciones_auxiliares.py` | Funciones reutilizadas por el notebook. |
| `figuras/` | Matrices de confusión, selección de umbral e importancia de variables (SVG). |

## Ejecución

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/churn_clasificacion.ipynb
```

## Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter

---

**Juan Peñas Utrilla** — [LinkedIn](https://www.linkedin.com/in/juan-penas-utrilla/) · [GitHub](https://github.com/JuanUtrilla)
