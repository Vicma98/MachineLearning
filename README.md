# MachineLearning

cd /workspaces/RP_AD2026Q4
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade huggingface_hub
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyarrow openpyxl  # opcional: parquet y xlsx
python scripts/build_notebooks.py   # opcional: regenerar notebooks
jupyter lab

# autenticar Hugging Face
hf auth login
hf auth whoami

# Abrir el jupyter
jupyter notebook



## Documentacion de Hallazgos — Actividad Evaluativa 3
### VICTOR MANUEL HERNANDEZ FERNANDEZ 
### Monitoreo y Predicción de Deforestación con Machine Learning

**Institución Universitaria Digital de Antioquia (IU Digital)**
**Repositorio:** [https://github.com/Vicma98/MachineLearning/tree/developAE3]

### 1. Respuesta a la Pregunta de Negocio
**Pregunta:** 
> **¿Cuáles parcelas de bosque presentan un alto riesgo de sufrir deforestación severa a partir de sus condiciones climáticas y el uso del suelo, para que los guardabosques prioricen patrullajes e incentivos de conservación antes de que ocurra la tala?**

* **¿Quién necesita la respuesta?** La dirección de guardabosques, entidades de planificación territorial y autoridades ambientales encargadas de la protección forestal.
* **¿Qué decisión apoya?** Priorizar rutas de vigilancia preventiva en cuadrantes de alto riesgo y focalizar apoyos o incentivos de conservación donde los recursos operativos son limitados.

---

**Respuesta del modelo:** Con un **Random Forest Classifier** (`n_estimators=100`, `max_depth=8`, `random_state=42`) evaluado sobre **3,000 registros de prueba** (`12,000` de entrenamiento), el modelo logra detectar correctamente el **96.2%** de las parcelas con alto riesgo de deforestación (`Recall = 0.962`) y alcanza una exactitud global del **94.1%** (`Accuracy = 0.941`, `F1-Score = 0.9422`). Esto permite enfocar las brigadas de guardabosques y los recursos de conservación en las zonas más críticas sin desperdiciar recorridos en áreas de bajo riesgo.

### 2. Dataset Utilizado
* **Identificador en Hugging Face:** `electricsheepafrica/africa-synth-climate-deforestation-land-use-change-all`
* **Enlace:** [https://huggingface.co/datasets/electricsheepafrica/africa-synth-climate-deforestation-land-use-change-all]
* **Split utilizado:** `train` (15,000 registros y 18 columnas iniciales).
* **Por qué permite responder la pregunta:** Reúne variables climáticas, de uso del suelo e indicadores asociados a la deforestación, permitiendo entrenar un modelo supervisado para estimar si una parcela cae en un escenario de alto riesgo (`riesgo_alto_deforestacion`: `0 = Bajo riesgo`, `1 = Alto riesgo`) sin incluir la columna original de pérdida forestal entre las predictoras.

### 3. Modelo Seleccionado y Funcionamiento Paso a Paso
En la **Actividad Evaluativa 3** se utilizó un enfoque de **Aprendizaje Supervisado (Clasificación Binaria)** mediante el algoritmo **Random Forest Classifier** (`n_estimators=100`, `max_depth=8`, `random_state=42`).

#### ¿Cómo funciona el algoritmo paso a paso?
1. **Muestras *bootstrap* (bagging):** A partir del conjunto de entrenamiento (12,000 registros), se generan múltiples submuestras aleatorias con reemplazo para construir distintos árboles de decisión.
2. **Selección aleatoria de variables en cada nodo:** Al buscar la mejor división en cada nodo de un árbol, se evalúa un subconjunto aleatorio de características en lugar de todas a la vez, reduciendo la correlación entre árboles.
3. **División por impureza:** Cada árbol divide los datos según la variable y el umbral que mejor separan las clases (`0` y `1`), repitiendo el proceso hasta alcanzar criterios de parada como `max_depth=8`.
4. **Votación mayoritaria:** Para predecir sobre una nueva parcela (en los 3,000 registros de prueba), cada árbol emite su voto de clase y gana la categoría con mayor votación.

* **Justificación sobre escalado:** Los modelos basados en árboles de decisión realizan cortes por umbrales en cada variable y no dependen de distancias geométricas entre puntos, por lo que no requieren escalado de variables.


### 4. Hallazgos Principales y Evidencia
#### Respuesta a la Pregunta de Negocio
Evaluado sobre **3,000 registros de prueba** (tras entrenar con **12,000** y codificar variables categóricas con *One-Hot Encoding* a **6,039** columnas predictoras), el **Random Forest Classifier** alcanza una **Exactitud (`Accuracy`) de `0.941`**, una **Sensibilidad (`Recall`) de `0.962`**, una **Precisión de `0.9232`** y un **F1-Score de `0.9422`**, con matriz de confusión `[[1380, 120], [57, 1443]]`. En términos prácticos, el modelo detecta el **96.2%** de las parcelas críticas en prueba, permitiendo enfocar patrullajes y acciones preventivas en las zonas con mayor probabilidad de deforestación.

#### Hallazgos Principales
* **Alta cobertura de casos críticos (`Recall = 0.962`):** De 1,500 casos de alto riesgo en prueba, se identificaron **1,443** verdaderos positivos y solo **57** falsos negativos, reduciendo el riesgo de pasar por alto zonas vulnerables.
* **Control de falsas alarmas (`Precision = 0.9232`):** Se registraron **1,380** verdaderos negativos y **120** falsos positivos, manteniendo un balance sólido entre alertas útiles y carga operativa en campo.
* **Alta dimensionalidad tras codificación:** Pasar de 18 columnas iniciales a 6,039 variables predictoras tras *One-Hot Encoding* permite capturar patrones detallados, pero incrementa el costo computacional.

#### Confiabilidad y Limitaciones
* **Confiabilidad:** La separación estratificada 80/20 (`train_test_split`) y la exclusión de la variable base usada para construir la etiqueta evitan fuga de información hacia el conjunto de prueba.
* **Limitaciones:**
  * El conjunto de datos tiene carácter sintético, por lo que sus resultados no implican causalidad directa ni sustituyen la validación territorial con datos locales actualizados.
  * La alta cantidad de columnas generadas por codificación puede volver más lento el entrenamiento si el volumen de datos crece.
  * Persisten errores residuales (57 falsos negativos y 120 falsos positivos), por lo que el modelo debe apoyar —no reemplazar— la supervisión humana.

#### Recomendaciones
1. **Priorizar patrullajes preventivos** en las parcelas predichas como de alto riesgo (`1`) antes de periodos secos o de mayor presión sobre el suelo.
2. **Focalizar programas de conservación e incentivos** en zonas de riesgo alto identificadas por el modelo como lo son programas de apoyo agrícola sostenible o pagos por conservación a las zonas de riesgo alto antes de épocas críticas de clima seco o expansión del uso del suelo
3. **Evaluar reducción de dimensionalidad o selección de variables** y calibrar el umbral de alerta según la capacidad real de las brigadas.

---


