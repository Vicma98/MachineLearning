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

## Documento de Hallazgos — Actividad Evaluativa 3

### 1. Respuesta a la Pregunta de Negocio
**Pregunta:** ¿Cuáles parcelas de bosque tienen un alto riesgo de sufrir deforestación severa según su clima y el uso del suelo, para que los guardabosques prioricen patrullajes y ayudas de conservación antes de la tala?

**Respuesta del modelo:** Con un **Random Forest Classifier** (`n_estimators=100`, `max_depth=8`, `random_state=42`) evaluado sobre **3,000 registros de prueba** (`12,000` de entrenamiento), el modelo logra detectar correctamente el **96.2%** de las parcelas con alto riesgo de deforestación (`Recall = 0.962`) y alcanza una exactitud global del **94.1%** (`Accuracy = 0.941`, `F1-Score = 0.9422`). Esto permite enfocar las brigadas de guardabosques y los recursos de conservación en las zonas más críticas sin desperdiciar recorridos en áreas de bajo riesgo.

### 2. Hallazgos Principales y Evidencia
* **Alta capacidad de detección temprana (`Recall = 0.962`):** De **1,500 casos reales de alto riesgo** en la muestra de prueba, el modelo identificó **1,443** (Verdaderos Positivos) y omitió solo **57** (Falsos Negativos). En conservación ambiental, minimizar los falsos negativos es clave para evitar pérdidas irreversibles de cobertura forestal.
* **Control de falsas alarmas (`Precision = 0.9232`):** De las parcelas clasificadas como de bajo riesgo, **1,380** fueron clasificadas correctamente (Verdaderos Negativos) frente a **120** Falsos Positivos, manteniendo un costo operativo manejable al verificar zonas en campo.
* **Impacto de variables y codificación:** Tras limpiar duplicados e imputar nulos (mediana para numéricos y moda para categóricos), la codificación *One-Hot* generó **6,039 variables predictoras** a partir de las **18 columnas iniciales**, capturando patrones entre condiciones climáticas, uso del suelo y riesgo de deforestación.

### 3. Confiabilidad y Limitaciones
* **Confiabilidad:** Los datos se dividieron **80% entrenamiento / 20% prueba** de forma estratificada, evitando fuga de datos (*data leakage*) al no incluir la variable original usada para definir el umbral objetivo[cite: 5]. Además, al basarse en árboles de decisión, no requiere escalado de variables[cite: 4].
* **Limitaciones:**
  1. El dataset (`electricsheepafrica/africa-synth-climate-deforestation-land-use-change-all`) tiene componentes sintéticos, por lo que antes de operar en un territorio real debe calibrarse con datos locales recientes.
  2. La codificación *One-Hot* sobre columnas categóricas eleva la dimensionalidad a **6,039 columnas**, lo que aumenta el consumo de memoria y tiempo de cómputo si el dataset crece mucho.
  3. Persisten **57 falsos negativos** y **120 falsos positivos**, por lo que el modelo debe funcionar como apoyo a la decisión humana y no como reemplazo total de la inspección técnica.

### 4. Recomendaciones Accionables
1. **Priorizar patrullajes preventivos:** Asignar primero las rutas de vigilancia y monitoreo a las parcelas marcadas como `riesgo_alto_deforestacion = 1`, concentrando los esfuerzos donde el modelo detecta el **96.2%** de los casos críticos.
2. **Focalizar incentivos y asistencia técnica:** Dirigir programas de apoyo agrícola sostenible o pagos por conservación a las zonas de riesgo alto antes de épocas críticas de clima seco o expansión del uso del suelo.
3. **Ajustar el umbral de alerta y depurar variables:** Evaluar el umbral de probabilidad (`predict_proba`)[cite: 6] según el presupuesto disponible de patrullas y revisar si conviene reducir o seleccionar variables para disminuir las **6,039 columnas** generadas por *One-Hot Encoding*.


