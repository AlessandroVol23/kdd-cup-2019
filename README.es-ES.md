

kdd-cup-2019
==============================
Este es el repositorio para el curso práctico de Big Data Science en la LMU. El objetivo del curso era participar en el KDD Cup de este año, organizado por Baidu. Nuestro equipo fue __Los Bocadillos__ y logramos la posición 149 / 1.696 con una puntuación F1 del 69.43 % (compárese con el 1er puesto: 70.42 %).

# Descripción del proyecto

Una breve descripción del proyecto:
El KDD Cup de este año se realiza en colaboración con Baidu. Baidu es un motor de búsqueda chino (comparable a Google) que también ofrece un servicio de mapas.
El desafío tiene 2 tareas en total:
[1] Para consultas en los mapas de Baidu, debemos modelar la relevancia de los diferentes modos de transporte ofrecidos y predecir en cuál hará clic el usuario.
[2] Es un caso de uso abierto sobre cómo utilizar los datos proporcionados de manera significativa. 

Los datos proporcionados ya están divididos en conjuntos de entrenamiento y prueba.
Básicamente, para cada sesión [consulta de un usuario sobre cómo ir de A a B] tenemos una característica espacial [coordenadas del origen y destino],
las características específicas del modo de transporte [distancia, tiempo, precio] y las IDs personales específicas del usuario [63k categorías, incluidas NAs]

__Nota breve sobre los datos__:

Dado que hemos alojado todos los conjuntos de datos en un sistema interno DVC (control de versiones de datos), primero debe descargar todos los conjuntos de datos y luego ejecutar los scripts de preprocesamiento para obtenerlos. Coloque todos los datos brutos del usuario en `data/raw`.
Puede encontrar todas las fuentes de datos aquí:

__Fuentes de datos__:
* User data: https://dianshi.baidu.com/competition/29
* Weather: https://github.com/yaoxuefeng6/Paddle_baseline_KDD2019
* Holidays: https://publicholidays.cn/2018-dates/
* Subway stations: https://en.wikipedia.org/wiki/List_of_Beijing_Subway_stations

# Preprocesamiento 

## [1] Preprocesamiento de datos brutos
Para el preprocesamiento inicial de datos brutos, ejecute el archivo `src/data/raw_features`. 
Esto fusionará los 4 DFs brutos diferentes en uno solo y extraerá las primeras características básicas.

* Elija `row` o `col` dependiendo del modelo que desee entrenar [enfoque multiclase | enfoque de ranking]
* Elija `first` o `last` dependiendo de qué modo de transporte prefiera seleccionar cuando se ofrezca el mismo modo varias veces. El que se muestre primero en la lista de rutas o el que se muestre último.

Esto agrega las características 'brutas':
* Coordenadas
* Perfiles
* Objetivos
* Desapilamiento de rutas
 
``` bash
python ./src/data/raw_features.py <PATH_TO_KDD_CUP_2019/DATA> 'col' 'first'
```

## [2] Preprocesamiento de características externas
Esto agrega las características externas a los DFs fusionados (basado en [1]):

* Distancia a la estación de metro más cercana
* Características de tiempo
* Festivos
* Características meteorológicas

```shell
python ./src/features/external_features.py /path/to/kdd-cup-2019/data
```

## [3] División de SIDs para CV k-fold
--> Muestree SIDs del conjunto de entrenamiento, divídalos en k fragmentos de tamaño igual y devuelva una lista con k sublistas que contengan los SIDs para cada pliegue.
Si no desea tener 5 pliegues, abra el script y cambie el parámetro "amount_folds" a cualquier entero > 1.
Los resultados se guardarán en "data/processed/split_test_train/k-fold/SIDs.pickle"

```shell
python ./src/features/split_SIDs_for_CV.py
```

## [4] Ajuste de modelos
Todos los scripts para ajustar modelos se encuentran en las subcarpetas correspondientes dentro de "src/models".
Todos los scripts están documentados, por favor revise los archivos en sí para ajustar parámetros, seleccionar tipos de modelos, etc.
Básicamente, tienen la siguiente estructura:
    [1] Cargar paquetes
    [2] Definir funciones
    [3] main() --> ¡llamar a todas las funciones!

### [4-1] LGBM Multiclase
Ajustar un modelo LGBM Multiclase a los datos de "processed/multiclass".

### [4-2] LGBM Ranking
Ajustar un modelo LGBM Lamda Rank a los datos de "processed/ranking".

### [4-3] Modelos multivariantes
Ajustar cualquier modelo multiclase de la biblioteca sklearn a los datos de "processed/multiclass".

### [4-4] Modelos de Stacking
Apilar las predicciones de los submodelos para crear un metaaprendizaje.

### [4-5] TFR
Ajustar un modelo de Tensorflow Ranking a los datos de "processed/ranking".
Los parámetros se configuran en la sección "#%% Set the global params".

Organización del proyecto
------------

    ├── LICENSE
    ├── .dvc               <- Carpeta para el control de versiones de datos
    ├── .git               <- Carpeta para todas las configuraciones de GIT [incluyendo más subcarpetas]
    ├── README.md          <- El README de nivel superior para los desarrolladores que usan este proyecto.
    ├── data               <- carpeta que contiene todos los datos utilizados en este proyecto
    │   │   
    │   ├── raw            <- El volcado de datos original e inmutable.
    │   │   │  
    │   │   ├─── data_set_phase1        <- contiene todos los datos brutos originales publicados por el desafío KDD
    │   │   │                              [múltiples DFs que se fusionan en un DF grande ('./src/data/make_raw_preprocessing.py')]
    │   │   │    
    │   │   └─── data_set_phase1_joined <- todos los datos brutos unidos en un DF grande - para enfoques de ranking o multiclase en el modelado
    │   │                                  para más detalles ver: ('./src/data/make_raw_preprocessing.py')
    │   │
    │   ├── external       <- Datos de fuentes de terceros que no provienen del concurso de datos KDD en sí
    │   │   │    
    │   │   ├─── districts              <- dataframes para asignaciones de distritos
    │   │   │
    │   │   └─── external_features      <- dataframes con información externa
    │   │                                  [p.ej. coordenadas de estaciones de metro, festivos nacionales, ...]
    │   │                            
    │   ├── processed      <- Conjuntos de datos finales para modelado.
    │   │   ├─── split_test_train       <- SIDs utilizados para dividir los datos para CV
    │   │   │                              [tiene subcarpetas como: '5fold', '10fold', ... (correspondientes a la táctica de CV)]
    │   │   │
    │   │   ├─── features               <- contiene "multiclass_1.pickle",..., "multiclass_3.pickle" 
    │   │   │                              con los diferentes nombres de características que usamos para modelar [qué características]
    │   │   │
    │   │   ├─── ranking                <- todos los DFs de prueba y entrenamiento que pueden usarse para modelos de Ranking 
    │   │   │                              [TFRanking, LamdaRank,...]
    │   │   │      
    │   │   └─── multiclass             <- todos los DFs de prueba y entrenamiento que pueden usarse para modelos Multiclase
    │   │                                  [Diferencia: first_all / last_all --> 
    │   │                                  a veces se recomienda el mismo modo de transporte dos veces: mantenga el primero o el último]
    │   │  
    │   └── interim        <- Datos intermedios que han sido transformados.
    │
    ├── docs               <- Documentación
    │
    ├── environments       <- Carpeta para todos los entornos conda diferentes [todos guardados como archivos '.yml']
    │    
    ├── models             <- Modelos entrenados | predicciones del modelo | resúmenes del modelo
    │   │                     Estructura como sigue:
    │   │                       - 1. CARPETA: Categoría del modelo (multiclase, ranking, stacking,...)
    │   │                       - 2. Carpeta: Nombre del espacio de características utilizado (1, 2, 3, ...)
    │   │                       - 3. carpeta: Nombre del modelo concreto (KNN, MLP, XGboost)
    │   │                                    --> esta carpeta puede contener:   - archivos de resumen (F1, Matriz de confusión,...)
    │   │                                                                   - CV_predicitons (Probabilidades de clase predichas en CV)
    │   │                                                                                     [usado para stacking]
    │   │                                                                   - final_mod (modelo que fue entrenado con todos los datos de entrenamiento)
    │   │                                                                     que usamos para predecir en el conjunto de prueba.
    │   │                                                                     [¡el nombre corresponde al archivo de resumen- / CV_predicitons!]
    │   │  
    │   ├── multiclass     <- todos los modelos entrenados, predicciones de CV y resúmenes para los enfoques multiclase
    │   │   ├─── 1         <- todos los modelos que se crearon con nombres de características en data/processed/features/multiclass_1.pickle
    │   │   │                 [contiene subcarpetas con nombres de algoritmos multiclase]
    │   │
    │   ├── stacking       <- todos los modelos entrenados, predicciones de CV y resúmenes para el enfoque apilado
    │   │
    │   └── ranking        <- todos los modelos entrenados, predicciones de CV y resúmenes para los enfoques de ranking
    │       ├─── LGBM      <- todos los modelos LGBM [contiene subcarpetas con nombres de algoritmos multiclase]
    │       └─── TFR       <- todos los resúmenes TFR, predicciones, etc... [¡sin subcarpetas!]
    │
    ├── notebooks          <- Cuadernos Jupyter [solo exploratorios y experimentales]
    |                         Convención de nombres: número (para ordenar), las iniciales del creador y una breve descripción.
    |                         p.ej. '0.4-DS-data-exploration'.                          
    |
    ├── submissions        <- archivos de envío en formato .csv [SID, y_hat]
    │
    ├── references         <- Diccionarios de datos, manuales y todos los demás materiales explicativos.
    │
    ├── reports            <- Análisis generados en HTML, PDF, LaTeX, etc.
    │   └── figures           <- Gráficos y figuras generados para usar en los informes
    │
    ├── requirements.txt   <- El archivo de requisitos para reproducir el entorno de análisis, p.ej.
    │                         generado con `pip freeze > requirements.txt`
    │
    ├── setup.py           <- hace que el proyecto sea instalable con pip (pip install -e .) para que src pueda importarse
    │
    ├── src                <- Código fuente para usar en este proyecto.
    │   │                     Para descripciones más detalladas, consulte la sección 'Orden de scripts', subsección [4]                    
    │   │
    │   ├── __init__.py    <- Hace que src sea un módulo de Python
    │   │
    │   ├── data           <- Scripts para fusionar datos, agregar características (externas) y crear las divisiones para prueba y entrenamiento
    │   │
    │   ├── evaluator      <- Script para obtener la puntuación F1 de un archivo de envío
    │   │
    │   ├── features       <- Scripts con las funciones necesarias para la extracción de características 
    │   │   │                 [necesarios para la extracción de datos en "src/data"]
    │   │   ├── add_features.py
    │   │   └── build_features.py
    │   │
    │   ├── MLFlow         <- Script para registrar resultados de modelos con MLFlow
    │   │
    │   ├── models         <- Scripts para entrenar modelos y utilizar modelos entrenados para realizar
    │   │                     predicciones [cada tipo de modelo tiene su propia subcarpeta]
    │   │                     [para documentación, consulte los archivos en sí O la sección 'Orden de scripts', subsección [4]]
    │   │
    │   └── visualization  <- Scripts para crear visualizaciones exploratorias y orientadas a resultados
    │
    └── tox.ini            <- archivo tox con la configuración para ejecutar tox; consulte tox.testrun.org
--------
