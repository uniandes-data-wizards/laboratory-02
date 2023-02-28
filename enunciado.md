# Laboratorio 2 - Regresión

[Objetivos](#objetivos)

[Herramientas](#herramientas)

[Enunciado](#enunciado)

[Construcción de pipelines](#pipelines)

[Entregables](#entregables)

[Rúbrica de clasificación](#rubrica)


## <a name="objetivos"></a> Objetivos

- Construir modelos analíticos para estimar una variable objetivo continua a partir de una serie de variables observadas.

- Comprender el proceso para la construcción de modelos analíticos que responden a una tarea de regresión.

- Automatizar el proceso de construcción de modelos analíticos con el uso de pipelines de tal forma que puedan ser usados en ambiente de producción.

- Extraer información útil para el negocio a partir de los resultados de los modelos de regresión.


## <a name="herramientas"></a> Herramientas

Durante este laboratorio trabajaremos con las siguientes herramientas:

- Librerías principales de Python para procesamiento y visualización de datos como: 
    - pandas
    - sklearn
    - seaborn
    - numpy
    - matplotlib
- Se recomienda usar la última distribución disponible de Anaconda Individual Edition, pueden encontrar el instalador en este [enlace](https://www.anaconda.com/products/individual). 
- Ambiente de desarrollo: JupyterLab en distribución de Anaconda o trabajar sobre Google Colab.  

## <a name="enunciado"></a> Enunciado

### Descripción de negocio

MotorAlpes es una empresa dedicada a la venta y compra de vehículos usados, la cual cuenta con una plataforma en la que sus usuarios (individuos o compañías) pueden publicar vehículos sobre los cuales otros usuarios pueden hacer ofertas. Uno de sus principales objetivos es asegurar que los precios sean justos y además coherentes con las características principales y las condiciones en las que se encuentre el vehículo en cuestión. De esta manera pueden evitar que se presenten estafas u otras situaciones que puedan darle una mala imagen a la empresa.

Actualmente, MotorAlpes trabaja con un grupo de expertos que sirven como intermediarios en el proceso de negociación del precio de un vehículo, estableciendo un rango sobre el cual ambas partes pueden llegar a un acuerdo. Sin embargo, debido a la cantidad de usuarios que interactúan constantemente con la plataforma, este proceso es cada vez más lento y se ha encontrado que en muchas ocasiones no es del todo preciso. 

La empresa busca entonces una alternativa mucho más práctica y para lograrlo ha puesto a su disposición un [conjunto de datos](data/MotorAlpes_data.csv) que reúne información general de todo tipo de vehículos usados junto a los precios que se consideran apropiados. El objetivo será entonces construir un modelo que apoye en la resolución de las siguientes tareas:

- Identificar las variables que más impactan el precio de un vehículo usado.
- Predecir el precio de un vehículo usado a partir de las variables de mayor interés.

El detalle de cada una de las variables se encuentra en el siguiente [diccionario](data/MotorAlpes_dictionary.pdf).

Adicionalmente, usted cuenta con unos [datos de prueba](data/MotorAlpes_test.csv), los cuales serán utilizados por la empresa para probar el resultado de su modelo y validar si cumple con las expectativas.


### Instrucciones 


MotorAlpes desea que usted los ayude a realizar el análisis de los datos, para lo cual, su equipo debe desarrollar un modelo que permita estimar el precio de un vehículo usado a partir de su información básica. Al igual que seguir los siguientes pasos para garantizar el uso de la metodología "ASUM-DM":

1. **Entendimiento de los datos:** en esta etapa recuerde que debe describir la característica de los datos e incluir el análisis de calidad de datos.

2. **Identificación de variables a utilizar**: su equipo debe identificar las variables más relevantes que puedan utilizarse en el proceso de estimación. 

3. **Preparación de datos:** en esta etapa su equipo debe identificar y solucionar cualquier problema de inconsistencia o ruido que se pueda tener en los datos. Además, deben tener en cuenta el preprocesamiento necesario para el uso de regresiones. No olvide ejecutar un esquema de manejo de variables faltantes, identificación/manejo de datos atípicos y de normalizar en caso de ser necesario.

4. **Modelamiento:** a partir de las variables identificadas anteriormente, se debe plantear una regresión que estime la variable objetivo y medir su desempeño.

5. **Evaluación cuantitativa:** a partir de las métricas seleccionadas para comparar y seleccionar el mejor modelo, explicar el resultado obtenido desde el punto de vista cuantitativo y contestar la pregunta:
    * ¿Su equipo recomienda instalar el modelo de estimación en producción o es mejor continuar usando expertos para la tarea?
	* En caso de no recomendar el uso de un modelo de regresión ¿Qué otras posibilidades tiene la empresa? ¿Hacia dónde debe seguir con esta tarea?

6. **Evaluación cualitativa:** responder a la pregunta del negocio 	¿Qué obtuvieron con el ejercicio de regresión? ¿Cuáles son las variables más influyentes y que tan confiables son los resultados?.

      **- Validación de supuestos:** realice los ajustes necesarios para que su modelo cumpla con las suposiciones necesarias para la inferencia estadística con regresiones.

      **- Interpretación de los coeficientes:** realice una interpretación de los coeficientes de la regresión, identificando los más relevantes para la tarea y cómo afectan la variable objetivo.

7. **Visualizar el resultado del modelo:** integrar el resultado obtenido con el modelo de regresión a un tablero de control para apoyar en el logro del objetivo de la empresa.
8. **Exportar el modelo:** su equipo debe exportar el modelo (creado utilizando pipelines) para poder ser usado sobre datos recientes en el ambiente de producción del cliente.

## <a name="pipelines"></a> Construcción de pipelines
Para realizar esta sección se recomienda utilizar JupyterLab para la construcción del Pipeline y la exportación del modelo. 

 El objetivo de crear un [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)  es automatizar todos los pasos realizados sobre los datos. Desde que salen de su fuente hasta que son ingresados al modelo de aprendizaje automático. Para un problema clásico, estos pasos incluyen: la selección de características o columnas, la imputación de valores no existentes, la codificación de variables categóricas utilizando diferentes técnicas como Label Encoding o One Hot Encoding y el escalamiento de variables numéricas en caso de ser necesario. Sin embargo, note que para problemas como el procesamiento de textos los pasos necesarios son diferentes. Además, como último paso, el pipeline contiene el modelo que recibe los datos después de la tranformación para realizar predicciones. Finalmente, estos pipelines pueden resultar muy útiles a la hora de calibrar y comparar modelos, pues se tiene la certeza de que los datos de entrada son los mismos para todos. Incluso, pueden ser utilizados para realizar validación cruzada utilizando GridSerchCV o RandomizedSerchCV. Así mismo, pueden ser exportados para llevar los modelos a producción por medio de la serialización de estos en archivos .pkl o .joblib. 

La librería Scikit Learn cuenta con API para la creación de pipelines en la que pueden ser utilizados diferentes pasos para la transformación de los datos que serán aplicados secuencialmente. Note que estos pasos implementan los métodos **fit** y **transform** para ser invocados desde el pipeline. Por otro lado, los modelos que serán la parte final del proceso de automatización solo cuentan con método fit. Una vez construido el modelo es posible serializar este haciendo uso de la función **dump** de la librería joblib, para posteriormente deserializar, cargar (mediante la función **load**) y utilizar el modelo en cualquier otra aplicación o ambiente. Tenga en cuenta que la serialización de un modelo solo incluye la estructura y configuraciones realizadas sobre el pipeline, más no las instancias de los objetos que lo componen. Pues estos son provistos por la librería, por medio de la importación, en cualquiera que sea su ambiente de ejecución. Esto significa que si usted construye transformaciones personalizadas, debe incluir por separado estas en el ambiente donde cargará y ejecutará el modelo una vez sea exportado, ya que estas no están incluidas en la serialización. 

Basándose en los pasos realizados para la calibración de su modelo de regresión del laboratorio 3. Construya un pipeline que incluya todos los pasos necesarios para transformar los datos desde el archivo fuente para que estos puedan ser utilizados para realizar predicciones.

A continuación puede encontrar algunos artículos que pueden ser de utilidad para la construcción de pipelines. 
<br>
<br>
[Scikit-learn Pipeline Tutorial with Parameter Tuning and Cross-Validation](https://towardsdatascience.com/scikit-learn-pipeline-tutorial-with-parameter-tuning-and-cross-validation-e5b8280c01fb)
<br>
[Data Science Quick Tip #003: Using Scikit-Learn Pipelines!](https://towardsdatascience.com/data-science-quick-tip-003-using-scikit-learn-pipelines-66f652f26954)
<br>
[Data Science Quick Tip #004: Using Custom Transformers in Scikit-Learn Pipelines!](https://towardsdatascience.com/data-science-quick-tip-004-using-custom-transformers-in-scikit-learn-pipelines-89c28c72f22a)
<br>
[Creating custom scikit-learn Transformers](https://www.andrewvillazon.com/custom-scikit-learn-transformers/)


## <a name="entregables"></a> Entregables

* Informe del laboratorio, que puede ser el mismo notebook, con el desarrollo y la evidencia de las etapas del 1 al 6.
* Modelo completo exportado en un archivo .joblib, usando la librería *joblib*. 
* Presentación con los resultados para MotorAlpes.
* Visualización de los resultados y código del tablero de control construido.
* El notebook a entregar debe estar ejecutado.
	
Se espera que el informe no supere las 8 páginas y que incluya **JUSTIFICACIONES** de las decisiones tomadas en la construcción e interpretación de los modelos.



**Nota:** 
El modelo entregado será utilizado para estimar la variable objetivo de los datos que separó el cliente y le compartió. 
El cliente cargará el modelo entregado se lo aplicará a los datos y comparará el resultado con el valor real de la variable  objetivo. 
Ustedes tienen acceso a una copia de estos datos (sin la variable objetivo). Recuerde que el cliente no planea hacer ningún tipo de alteración sobre estos datos antes de entregárselos al modelo, por lo que su equipo debe exportar un *pipeline* capaz de hacer la manipulaciones  necesarias para hacer la tarea de estimación. Este archivo de datos se llama: *MotorAlpes_test.csv*.

El código que usará el cliente es el siguiente:

```python
import numpy as np
import pandas as pd
import joblib

# Proceso de prueba del cliente
filename = 'modelo.joblib' # Ubicación del archivo entregado
df_recent = pd.read_csv('MotorAlpes_test.csv') # Lectura de los datos recientes

# Lee el archivo y carga el modelo
pipeline = load(filename)

y_true = pd.read_csv('MotorAlpes_validation.csv') # La columna que solo el cliente tiene
y_predicted =  pipeline.predict(df_recent)

# Calcula el desempeño del modelo
np.sqrt(mse(y_true, y_predicted))

```


## Instrucciones de Entrega
- El laboratorio se entrega en grupos de mínimo 2 y máximo 3 estudiantes
- Recuerde hacer la entrega por la sección unificada en Bloque Neón, antes del domingo 05 de marzo a las 22:00.   
  Este será el único medio por el cual se recibirán entregas.
- En la entrega indique la etapa o etapas realizada por cada uno de los miembros del grupo.


## <a name="rubrica"></a> Rúbrica de Calificación

A continuación se encuentra la rúbrica de calificación.


| Concepto | Porcentaje |
|:---:|:---:|
| 1. Descripción del entendimiento de datos  | 10% |
| 2. Descripción del proceso de identificación de variables | 10% |
| 3. Descripción de la limpieza y preparación de los datos  | 5% |
| 4. Implementación de la regresión lineal y extracción de sus métricas de calidad  | 10% |
| 5. Exploración de los supuestos a partir del modelo de regresión planteado | 5% |
| 6. Realizar las transformaciones necesarias  para cumplir los supuestos | 10% |
| 7. Incorporar las transformaciones y el estimador al modelo en un *pipeline* y exportarlo | 15% |
| 8. Presentación para MotorAlpes con resultados a nivel cuantitativo y cualitativo del modelo construido, recomendaciones dadas a la empresa y visualización | 15% |
| 9. Resultado del modelo al estimar el conjunto de datos que separó el cliente | 5% |
| 10. Visualización del resultado del modelo para ser utilizado por el cliente| 10% |
| 11. Notebook asociado y ejecutado junto con el código de la visualización del resultado del modelo| 5% |

La nota individual se calculará de acuerdo con las etapas realizadas por cada miembro del grupo. Se espera que el estudiante 1 del grupo se encargue del modelo analítico, el estudiante 2 del proceso del pipeline y el estudiante 3 de la visualización.

