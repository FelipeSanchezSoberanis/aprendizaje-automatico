# Contenido

<!-- vim-markdown-toc Marked -->

* [Introducción al aprendizaje automático](#introducción-al-aprendizaje-automático)
    * [Aprendizaje supervisado contra no supervisado](#aprendizaje-supervisado-contra-no-supervisado)
* [Regresión lineal](#regresión-lineal)
* [Ecuación normal](#ecuación-normal)
* [Cálculos de error](#cálculos-de-error)
    * [Error absoluto medio](#error-absoluto-medio)
    * [Error cuadrático medio con raiz](#error-cuadrático-medio-con-raiz)
    * [Error cuadrático medio](#error-cuadrático-medio)
* [Gradiente descendente](#gradiente-descendente)
* [Regresión logística](#regresión-logística)
    * [Modelo](#modelo)
* [Support vector machine](#support-vector-machine)
    * [Clasificador de margen grande](#clasificador-de-margen-grande)
        * [Margen duro vs margen suave](#margen-duro-vs-margen-suave)

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

# Ecuación normal

La ecuación normal es una ecuación matemática utilizada en regresión lineal para encontrar los coeficientes que minimizan la suma de los errores cuadrados entre los valores pronosticados y reales de la variable dependiente. La ecuación normal se expresa de la siguiente manera:

$$ \Theta = (X^TX)^{-1}X^Ty $$

donde $\Theta$ es un vector de coeficientes, $X$ es una matriz de variables independientes (también conocida como matriz de diseño), $y$ es un vector de la variable dependiente, y $^T$ denota la transposición de una matriz.

En esencia, la ecuación normal resuelve los coeficientes que minimizan la suma de las diferencias cuadradas entre los valores pronosticados y reales de la variable dependiente. Lo hace encontrando el punto donde el gradiente de la suma de los errores cuadrados es cero, lo que es equivalente a resolver un sistema de ecuaciones lineales. La ecuación normal se puede utilizar tanto para regresión lineal simple como para regresión lineal múltiple.

Función implementada en Python:

https://github.com/FelipeSanchezSoberanis/aprendizaje-automatico/blob/91b63479e46955195d9789dc53339c38842a1cdb/normal_equation.py#L4-L6

# Cálculos de error

## Error absoluto medio

El Error Absoluto Medio (MAE, por sus siglas en inglés) es una métrica utilizada para medir la diferencia promedio entre los valores reales y los valores predichos en un conjunto de datos. Es una métrica comúnmente utilizada en el análisis de regresión.

La fórmula para el MAE es:

$$ \text{MAE} = \frac{1}{n} \sum_{i = 1}^{n} |\text{real}_i - \text{predicho}_i| $$

donde:

- n es el número total de observaciones en el conjunto de datos
- real_i es el valor real de la i-ésima observación
- predicho_i es el valor predicho de la i-ésima observación

El MAE se calcula tomando la diferencia absoluta entre cada valor real y predicho, sumando estas diferencias y luego dividiendo por el número total de observaciones. El resultado es un valor no negativo, donde un valor más pequeño indica un mejor rendimiento del modelo.

El MAE es útil en situaciones donde los grandes errores son indeseables y la magnitud del error es importante. Por ejemplo, en el contexto de predecir los precios de las casas, se preferiría un modelo con un MAE más pequeño porque esto resultaría en predicciones más precisas y posiblemente mejores resultados para los compradores y vendedores.

## Error cuadrático medio con raiz

El Error Cuadrático Medio (RMSE, por sus siglas en inglés) es una métrica utilizada para medir la diferencia promedio entre los valores reales y los valores predichos en un conjunto de datos. Es una métrica comúnmente utilizada en el análisis de regresión.

La fórmula para el RMSE es:

$$ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\text{real}_i - \text{predicho}_i)^2} $$

donde:

- n es el número total de observaciones en el conjunto de datos
- real_i es el valor real de la i-ésima observación
- predicho_i es el valor predicho de la i-ésima observación

El RMSE se calcula tomando el cuadrado de la diferencia entre cada valor real y predicho, sumando estos cuadrados, dividiendo por el número total de observaciones y luego tomando la raíz cuadrada del resultado. El resultado es un valor no negativo, donde un valor más pequeño indica un mejor rendimiento del modelo.

El RMSE es útil en situaciones donde los grandes errores son indeseables y la magnitud del error es importante, pero la presencia de valores atípicos (outliers) en los datos puede sesgar los resultados. A diferencia del MAE, el RMSE penaliza más severamente los errores grandes debido al cuadrado de la diferencia. Sin embargo, esto también puede hacerlo más sensible a los valores atípicos.

Por ejemplo, en el contexto de predecir el consumo de energía, se preferiría un modelo con un RMSE más pequeño porque esto resultaría en predicciones más precisas y posiblemente mejores resultados para las compañías de energía y los consumidores.

## Error cuadrático medio

El Error Cuadrático Medio (MSE, por sus siglas en inglés) es una métrica utilizada para medir la diferencia promedio al cuadrado entre los valores reales y los valores predichos en un conjunto de datos. Es una métrica comúnmente utilizada en el análisis de regresión.

La fórmula para el MSE es:

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\text{real}_i - \text{predicho}_i)^2 $$

donde:

- n es el número total de observaciones en el conjunto de datos
- real_i es el valor real de la i-ésima observación
- predicho_i es el valor predicho de la i-ésima observación

El MSE se calcula tomando la diferencia al cuadrado entre cada valor real y predicho, sumando estos cuadrados y luego dividiendo por el número total de observaciones. El resultado es un valor no negativo, donde un valor más pequeño indica un mejor rendimiento del modelo.

El MSE es útil en situaciones donde los grandes errores son indeseables y la magnitud del error es importante. Sin embargo, como el RMSE, el MSE puede ser sensible a los valores atípicos (outliers).

Por ejemplo, en el contexto de predecir los precios de las acciones, se preferiría un modelo con un MSE más pequeño porque esto resultaría en predicciones más precisas y posiblemente mejores resultados para los inversores.

# Gradiente descendente

Gradiente descendente (gradient descent en inglés) es un algoritmo iterativo de optimización utilizado en aprendizaje automático y otros modelos matemáticos para minimizar una función de coste ajustando los parámetros del modelo. Funciona calculando el gradiente (o la derivada) de la función de coste con respecto a los parámetros del modelo, y luego actualizando los parámetros en la dirección opuesta del gradiente para minimizar la función de coste.

El gradiente es un vector que apunta en la dirección del mayor aumento en la función de coste, por lo que moverse en la dirección opuesta del gradiente te llevará al punto del mayor descenso. Es por eso que el gradiente descendente actualiza iterativamente los parámetros del modelo en la dirección opuesta del gradiente hasta que alcanza un mínimo local, donde la función de coste ya no se puede reducir.

Existen diferentes variaciones del gradiente descendente, incluyendo el gradiente descendente por lotes (batch gradient descent), el gradiente descendente estocástico (stochastic gradient descent) y el gradiente descendente por mini-lotes (mini-batch gradient descent), cada uno con sus propias ventajas y desventajas. El gradiente descendente se utiliza ampliamente en el aprendizaje profundo (deep learning) y otros algoritmos de aprendizaje automático para optimizar modelos y mejorar su precisión.

# Regresión logística

La regresión logística es un método estadístico utilizado para modelizar la probabilidad de un resultado binario (es decir, una variable dicotómica que sólo puede tomar dos valores posibles) en función de una o más variables predictoras. Es un tipo de análisis de regresión utilizado habitualmente en campos como el aprendizaje automático, la estadística y las ciencias sociales.

El modelo de regresión logística utiliza una función logística (también llamada función sigmoidea) para estimar la probabilidad de una variable de respuesta binaria, dados los valores de las variables predictoras. La función logística es una curva en forma de S que asigna cualquier valor de entrada a un valor de probabilidad entre 0 y 1.

La regresión logística puede utilizarse tanto para resultados binarios como multinomiales (es decir, más de dos categorías). En la regresión logística binaria, la variable de respuesta sólo toma dos valores, como "éxito" o "fracaso", "sí" o "no", "0" o "1", etc. En la regresión logística multinomial, la variable de respuesta toma más de dos categorías, como ingresos "bajos", "medios" o "altos".

La regresión logística puede utilizarse para diversas aplicaciones, como predecir si un cliente comprará o no un producto, predecir si un estudiante aprobará o no un examen o predecir la probabilidad de que un paciente padezca una determinada enfermedad en función de su historial médico.

## Modelo

Se desea

$$
h_\theta(x) = g(\theta^T \cdot x)
$$

donde $g$ es la función sigmoide.

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

$$
h_\theta(x) = \frac{1}{1 + e^{-\theta^{T} \cdot x}}
$$

donde $h_\theta(x)$ es la probabilidad de que $y=1$ para la entrada $x$.

Después, se pasa por lo siguiente:

$$
\hat{y} =
\left\{
    \begin{matrix*}[l]
        1, & h_\theta(x) >= 0.5 \\
        0, & h_\theta(x) < 0.5 \\
    \end{matrix*}
\right.
$$

# Support vector machine

Support Vector Machine (SVM) es un poderoso algoritmo de aprendizaje automático que se utiliza tanto para la clasificación como para el análisis de regresión. SVM intenta encontrar un hiperplano en un espacio de alta dimensión que separa de manera máxima las diferentes clases en el conjunto de datos.

En términos simples, SVM es un tipo de algoritmo de aprendizaje supervisado que se utiliza para la clasificación o el análisis de regresión. El objetivo de SVM es encontrar un límite que separe las diferentes clases en el conjunto de datos lo más ampliamente posible, lo que se llama "margen máximo". Los puntos más cercanos al límite se llaman vectores de soporte, de ahí el nombre "Support Vector Machine".

SVM es particularmente útil en situaciones en las que el número de características es mucho mayor que el número de puntos de datos, y los datos no son linealmente separables. SVM se puede utilizar con diferentes funciones de núcleo, como lineal, polinómico, función de base radial (RBF) y sigmoidal.

En general, SVM es un algoritmo de aprendizaje automático versátil y potente que ha encontrado aplicaciones en diversos campos, como la clasificación de imágenes, la clasificación de texto, la bioinformática y las finanzas, entre otros.

## Clasificador de margen grande

Un clasificador de margen grande es un tipo de clasificador utilizado en algoritmos de aprendizaje automático, que tiene como objetivo encontrar un límite de decisión o hiperplano en un espacio de alta dimensión que maximiza el margen o la distancia entre el límite y los puntos de datos más cercanos.

El margen se define como la distancia entre el límite de decisión y los puntos de datos más cercanos de cada clase. Cuanto mayor sea el margen, mayor será la confianza en la decisión de clasificación, ya que indica que el límite de decisión es menos probable que se vea afectado por ruido o valores atípicos en los datos.

Un clasificador de margen grande, como el Support Vector Machine (SVM), es particularmente útil cuando se trabaja con datos que no son linealmente separables, ya que puede encontrar un límite de decisión no lineal que maximice el margen entre las clases. El concepto de un clasificador de margen grande se basa en la idea de minimización del riesgo estructural, que busca minimizar el error de generalización encontrando un límite de decisión que esté lo más lejos posible de los datos de entrenamiento.

En general, el clasificador de margen grande es un concepto poderoso en el aprendizaje automático que se ha utilizado en diversas aplicaciones, incluyendo la clasificación de imágenes, el procesamiento de lenguaje natural y la predicción financiera, entre otros.

### Margen duro vs margen suave

Los clasificadores de margen duro y margen suave son dos tipos de clasificadores lineales utilizados en el aprendizaje automático para problemas de clasificación binaria.

Un clasificador de margen duro intenta encontrar un hiperplano que separe las dos clases con un margen de al menos 1. Esto significa que el clasificador no permite que ningún punto de datos sea clasificado incorrectamente o caiga dentro del margen. Los clasificadores de margen duro funcionan bien cuando los datos son separables linealmente, lo que significa que las dos clases pueden ser completamente separadas por una línea recta o hiperplano.

Por otro lado, un clasificador de margen suave permite cierta cantidad de clasificación incorrecta o puntos de datos que caigan dentro del margen. Encuentra un hiperplano que maximiza el margen permitiendo cierta cantidad de error o clasificación incorrecta. El nivel de error se controla mediante un hiperparámetro llamado parámetro de regularización o C. Cuanto mayor sea el valor de C, más estricto será el clasificador, lo que significa que intentará minimizar la cantidad de clasificaciones incorrectas a expensas de un margen más pequeño. Cuanto menor sea el valor de C, más flexible será el clasificador, lo que significa que permitirá más clasificaciones incorrectas pero resultará en un margen más grande. Los clasificadores de margen suave funcionan bien cuando los datos no son perfectamente separables linealmente o cuando hay ruido en los datos.

En resumen, los clasificadores de margen duro funcionan bien cuando los datos son separables linealmente y no hay ruido, mientras que los clasificadores de margen suave funcionan bien cuando los datos no son perfectamente separables o hay algún ruido. La elección entre los dos depende de la naturaleza de los datos y el equilibrio entre el tamaño del margen y el error de clasificación.

