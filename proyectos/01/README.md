# Proyecto #1: Algoritmos de regresión "A"

## 1. En clase se estudió el algoritmo "Batch Gradient Descent". Quedó pendiente mostrar el procedimiento de "vectorización" de la fórmula para actualizar los pesos; dicho procedimiento debe incluirse en el documento a entregar.

## 2. Investigar acerca de las siguientes variantes del algoritmo: (a) Stochastic Gradient Descent y (b) Mini-batch Gradient Descent. ¿Cuáles son las diferencias y las ventajas/desventajas entre estas variantes? ¿Cuáles son las condiciones bajo las cuales se prefieren aplicar estas variantes?

### Stochastic Gradient Descent

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

## 3. Implementar en Python los algoritmos (a) Stochastic Gradient Descent y (b) Mini-batch Gradient Descent, en la solución de un problema de "nube de puntos artificial" siendo de particular interés la gráfica de la función costo vs. iteraciones. Comparar las soluciones con las obtenidas previamente en clase.

## 4. Investigar acerca del algoritmo "Polynomial Regression". ¿Cuándo se aplica?, ¿Qué problemas puede presentar una solución basada en este algoritmo?

## 5. Implementar en Python el algoritmo de "Polynomial Regression" para la solución de un conjunto (nube) de datos generados artificialmente (véase ejemplo de clase).

## 6. Conclusiones generales a nivel de equipo; comentarios individuales.

## 7. Elaborar una "Presentación" (rúbrica Reporte Digital 1 en presentación del primer día de clase). Debe documentarse la información de los incisos 1-6.

## 8. Elaborar una video-entrega (todos los integrantes del equipo deberán participar). Como parte de la evaluación se consideran el dominio, la comprensión y la profundidad plasmadas en las explicaciones. Debe presentarse la información de los incisos 1-6.

