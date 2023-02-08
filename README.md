# Contenido

<!-- vim-markdown-toc Marked -->

* [Introducción al aprendizaje automático](#introducción-al-aprendizaje-automático)
    * [Aprendizaje supervisado contra no supervisado](#aprendizaje-supervisado-contra-no-supervisado)
* [Regresión lineal](#regresión-lineal)

<!-- vim-markdown-toc -->

# Introducción al aprendizaje automático

El aprendizaje automático es un subconjunto de la inteligencia artificial que se centra en el desarrollo de algoritmos y modelos estadísticos que permiten a los ordenadores aprender de los datos y hacer predicciones o tomar medidas basadas en la entrada de datos. Implica el uso de técnicas matemáticas y estadísticas para extraer conocimientos e ideas de los datos, sin programarlos explícitamente. Los modelos pueden mejorar continuamente a medida que reciben más datos, por lo que son capaces de adaptarse y aprender con el tiempo. El aprendizaje automático se utiliza ampliamente en diversos sectores, como las finanzas, la sanidad y el comercio electrónico, y está cambiando rápidamente la forma en que interactuamos con la tecnología.

## Aprendizaje supervisado contra no supervisado

El aprendizaje supervisado es un tipo de aprendizaje automático donde se proporciona al modelo un conjunto de datos etiquetados, es decir, con las respuestas correctas, con el objetivo de que el modelo aprenda a hacer predicciones precisas en nuevos datos.

Por otro lado, el aprendizaje no supervisado es un tipo de aprendizaje automático donde el modelo se entrena en un conjunto de datos sin etiquetas, es decir, sin respuestas correctas proporcionadas, con el objetivo de descubrir patrones o relaciones en los datos. Este tipo de aprendizaje se utiliza para tareas de clustering, reducción de dimensionalidad, entre otras.

En resumen, en el aprendizaje supervisado se proporciona información para guiar el aprendizaje, mientras que en el aprendizaje no supervisado el modelo debe encontrar patrones en los datos por sí solo.

# Regresión lineal

La regresión lineal es un tipo de modelo de aprendizaje supervisado que se utiliza para predecir una variable numérica en función de otras variables. Es llamado "lineal" porque el modelo supone una relación lineal entre las variables predictoras y la variable objetivo.

En la regresión lineal, se ajusta una recta (o hiperplano en caso de múltiples variables) a los datos de entrenamiento de manera que se minimice la suma de los errores cuadráticos entre las predicciones y los valores reales. La recta ajustada se puede usar para hacer predicciones en nuevos datos.

La regresión lineal es una herramienta muy útil en muchos campos, como la economía, la finanzas, la biología, entre otros, y se puede usar tanto para modelar relaciones simples como para resolver problemas más complejos.

Función implementada en Python:

https://github.com/FelipeSanchezSoberanis/aprendizaje-automatico/blob/5803dfb189a756101fec3d8948551f7e475ee9b3/linear_regression.py#L5-L18

