# Proyecto #1: Algoritmos de regresión "A"

## 1. En clase se estudió el algoritmo "Batch Gradient Descent". Quedó pendiente mostrar el procedimiento de "vectorización" de la fórmula para actualizar los pesos; dicho procedimiento debe incluirse en el documento a entregar.

Tomando en cuenta el gradiente de la función objetivo:

$$
\nabla \text{obj} = \sum_{i = 0}^{n} (\hat{y} - y) D(XW)
$$

donde D es la derivada de la matriz obtenida de la multiplicación de las matrices X, que son los datos observados, y W, que son los pesos.

Esto quiere decir que, para cada epoch (iteración de entrenamiento), los pesos se actualizarán de la siguiente manera:

$$
w_j = w_j - \text{learning rate} \times \nabla \text{obj}_j
$$

Así que, en cada epoch, los pesos son actualizados en sentido contrario a la gradiente, es decir, hacia el mínimo; en un paso del tamaño definido por el "learning rate" (razón de aprendizaje).

Para poder utilizar la forma de multiplicación matricial, se toman en cuenta, para ejemplificar, 2 matrices $X_{3x2}$ y $W_{2x1}$:

$$
X =
    \begin{bmatrix}
        x_1 & x_2 \\
        x_3 & x_4 \\
        x_5 & x_6 \\
    \end{bmatrix}
$$

$$
W =
    \begin{bmatrix}
        w_1  \\
        w_2  \\
    \end{bmatrix}
$$

Esto quiere decir que:

$$
XW =
    \begin{bmatrix}
        x_1 w_1 + x_4 w_2 \\
        x_2 w_1 + x_5 w_2 \\
        x_3 w_1 + x_6 w_2 \\
    \end{bmatrix}
$$

Obteniendo la derivada parcial para cada uno de los pesos:

$$
\frac{\partial XW}{\partial w_1} =
\begin{bmatrix}
    \frac{\partial(x_1 w_1 + x_4 w_2)}{\partial w_1} \\
    \frac{\partial(x_2 w_1 + x_5 w_2)}{\partial w_1} \\
    \frac{\partial(x_3 w_1 + x_6 w_2)}{\partial w_1} \\
\end{bmatrix} =
\begin{bmatrix}
    x_1 \\
    x_2 \\
    x_3 \\
\end{bmatrix}
$$

$$
\frac{\partial XW}{\partial w_2} =
\begin{bmatrix}
    \frac{\partial(x_1 w_1 + x_4 w_2)}{\partial w_2} \\
    \frac{\partial(x_2 w_1 + x_5 w_2)}{\partial w_2} \\
    \frac{\partial(x_3 w_1 + x_6 w_2)}{\partial w_2} \\
\end{bmatrix} =
\begin{bmatrix}
    x_4 \\
    x_5 \\
    x_6 \\
\end{bmatrix}
$$

Lo que se puede simplificar a:

$$
\frac{\partial XW}{\partial w_1} =
X
\begin{bmatrix}
    1 \\
    0 \\
\end{bmatrix}
$$

$$
\frac{\partial XW}{\partial w_2} =
X
\begin{bmatrix}
    0 \\
    1 \\
\end{bmatrix}
$$

Esto quiere decir, que la derivada de todos los pesos puede ser calculada por medio de la utilización de una matriz con una diagonal unitaria de dimensión $(p+1 \times p+1)$.

## 2. Investigar acerca de las siguientes variantes del algoritmo: (a) Stochastic Gradient Descent y (b) Mini-batch Gradient Descent. ¿Cuáles son las diferencias y las ventajas/desventajas entre estas variantes? ¿Cuáles son las condiciones bajo las cuales se prefieren aplicar estas variantes?

### Stochastic gradient descent

"Stochastic Gradient Descent" (SGD) es una variante del algoritmo de "Gradient Descent" utilizado para optimizar modelos de aprendizaje automático. En esta variante, sólo se utiliza un ejemplo de entrenamiento aleatorio para calcular el gradiente y actualizar los parámetros en cada iteración.

Ventajas:

- Velocidad: SGD es más rápido que otras variantes de Gradient Descent como Batch Gradient Descent y Mini-Batch Gradient Descent ya que sólo utiliza un ejemplo para actualizar los parámetros.

- Eficiencia de memoria: Dado que SGD actualiza los parámetros para cada ejemplo de entrenamiento de uno en uno, es eficiente en memoria y puede manejar grandes conjuntos de datos.

- Evita mínimos locales: Debido a las actualizaciones ruidosas en SGD, tiene la capacidad de escapar de los mínimos locales y converger a un mínimo global.

Desventajas:

- Actualizaciones ruidosas: Las actualizaciones en SGD son ruidosas y tienen una alta varianza, lo que puede hacer que el proceso de optimización sea menos estable y provocar oscilaciones alrededor del mínimo.

- Convergencia lenta: SGD puede requerir más iteraciones para converger al mínimo ya que actualiza los parámetros para cada ejemplo de entrenamiento de uno en uno.

- Sensibilidad a la tasa de aprendizaje: La elección de la tasa de aprendizaje puede ser crítica en SGD, ya que utilizar una tasa de aprendizaje alta puede hacer que el algoritmo sobrepase el mínimo, mientras que una tasa de aprendizaje baja puede hacer que el algoritmo converja lentamente.

- Menos preciso: Debido a las actualizaciones ruidosas, el SGD puede no converger al mínimo global exacto y dar lugar a una solución subóptima. Esto puede mitigarse utilizando técnicas como la programación de la tasa de aprendizaje y las actualizaciones basadas en el impulso.

### Mini-batch gradient descent

"Mini-batch Gradient Descent" (MGD) es una variación del algoritmo de "Gradient Descent" que divide el entrenamiento en pequeños lotes que son usados para calcular el modelo del error y actualizar los coeficientes. Esto quiere decir que, suponiendo que se tengan 1000 datos y un tamaño de lote definido de 50, se tomarán 50 datos al azar y se utilizarán para calcular el error y actualizar los coeficientes.

Ventajas:

- La frecuencia de actualización del modelo es mayor que en BGD: En MGD no estamos esperando los datos enteros, sólo pasamos 50 registros o 200 o 100 o 256, y luego pasamos a la optimización.

- La dosificación permite tanto la eficiencia de no tener todos los datos de entrenamiento en memoria como la implementación de algoritmos. También controlamos el consumo de memoria para almacenar las errores de los datos.

- Las actualizaciones por lotes proporcionan un proceso computacionalmente más eficiente que SGD.

Desventajas:

- No hay garantía de convergencia de un error de mejor manera.

- Dado que el tamaño de muestra que tomamos no está representando las propiedades (o varianza) de conjuntos de datos enteros, no seremos capaces de obtener una convergencia es decir, no obtendremos mínimos absolutos globales o locales.

- Al utilizar MGD, ya que estamos tomando los registros en lotes, por lo que, podría suceder que en algunos lotes, tenemos algún error y en otros lotes, tenemos algún otro error. Por lo tanto, tendremos que controlar la tasa de aprendizaje por nosotros mismos, siempre que utilicemos MGD. Si la tasa de aprendizaje es muy baja, la tasa de convergencia también disminuirá. Si la tasa de aprendizaje es demasiado alta, no obtendremos un mínimo absoluto global o local. Así que tenemos que controlar la tasa de aprendizaje.

## 3. Implementar en Python los algoritmos (a) Stochastic Gradient Descent y (b) Mini-batch Gradient Descent, en la solución de un problema de "nube de puntos artificial" siendo de particular interés la gráfica de la función costo vs. iteraciones. Comparar las soluciones con las obtenidas previamente en clase.

## 4. Investigar acerca del algoritmo "Polynomial Regression". ¿Cuándo se aplica?, ¿Qué problemas puede presentar una solución basada en este algoritmo?

## 5. Implementar en Python el algoritmo de "Polynomial Regression" para la solución de un conjunto (nube) de datos generados artificialmente (véase ejemplo de clase).

## 6. Conclusiones generales a nivel de equipo; comentarios individuales.

## 7. Elaborar una "Presentación" (rúbrica Reporte Digital 1 en presentación del primer día de clase). Debe documentarse la información de los incisos 1-6.

## 8. Elaborar una video-entrega (todos los integrantes del equipo deberán participar). Como parte de la evaluación se consideran el dominio, la comprensión y la profundidad plasmadas en las explicaciones. Debe presentarse la información de los incisos 1-6.

