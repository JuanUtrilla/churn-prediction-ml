# Predicción de fuga de clientes (churn)

¿Se puede saber con antelación qué clientes de una compañía telefónica van a darse
de baja? Con los datos de uso y de contrato de unos 3.500 clientes, el modelo final
reconoce a 8 de cada 10 clientes que acaban yéndose, en datos que no vio al entrenar, y
3 de cada 4 de sus avisos son correctos.

Trabajo del módulo de Minería de datos y Modelización Predictiva del Máster en Big
Data, Data Science e IA de la Universidad Complutense (curso 2025-26).

## Cómo está hecho

El CSV trae 9.200 filas, pero 5.662 son copias exactas. Al quitarlas quedan 3.538
clientes y solo uno de cada cinco se va; con las copias parecía que se iba la mitad.
Los valores perdidos se rellenan con estadísticos calculados solo sobre entrenamiento, y
los extremos se revisan con rango intercuartílico e `IsolationForest`.

Se comparan dos flujos que terminan en una red neuronal pequeña y que eligen sus cinco
variables de forma distinta (con una regresión logística o con un árbol de decisión).
Los dos se quedan con casi las mismas: minutos de día y de tarde, llamadas a atención al
cliente y tener plan internacional.

Como se va tan poca gente, cortar la probabilidad en 0,5 deja escapar a muchos. Barriendo
umbrales, el que mejor equilibra aciertos y falsas alarmas en el flujo con logística
está en 0,36.

![Barrido de umbrales en entrenamiento](img/seleccion_umbral.png)

El modelo final combina los dos flujos con un umbral de 0,35 y, en prueba, detecta el
81 % de las bajas con una precisión del 76 %.

## Qué hay en el repo

El análisis completo está en `notebooks/churn_clasificacion.ipynb`, los datos en
`data/`, la memoria (con componentes principales y segmentación de clientes) en `docs/`,
las funciones auxiliares en `scripts/` y las gráficas en `figuras/`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/churn_clasificacion.ipynb
```

Python, pandas, scikit-learn, Matplotlib y Seaborn.
