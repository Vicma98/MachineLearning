
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class DeforestationClassifier:
    """
    esta clase encapsula el preprocesamiento, entrenamiento y métricas para responder a la pregunta de negocio de deforestación.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 8, random_state: int = 42):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state
        }
        self.model = RandomForestClassifier(**self.params)
        self.features = []

    def preparar_datos(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Limpia datos, convierte variables categóricas a numéricas
        y separa en train y test (80% / 20%).
        """
        datos = df.copy().drop_duplicates()

        # Si el target no es binario o viene continuo, discretizamos con el percentil 75
        if target_col not in datos.columns:
            # Seleccionamos la columna numérica que mida deforestación/cobertura
            num_cols = datos.select_dtypes(include=[np.number]).columns
            col_ref = [c for c in num_cols if "deforest" in c.lower() or "loss" in c.lower()]
            col_eval = col_ref[0] if col_ref else num_cols[0]
            umbral = datos[col_eval].median()
            datos[target_col] = (datos[col_eval] > umbral).astype(int)
            datos = datos.drop(columns=[col_eval])  # Eliminar original para evitar data leakage

        # Rellenar nulos
        for col in datos.columns:
            if datos[col].dtype in [np.float64, np.int64]:
                datos[col] = datos[col].fillna(datos[col].median())
            else:
                datos[col] = datos[col].fillna("Desconocido")

        # Separar X e y
        y = datos[target_col].values
        X_df = datos.drop(columns=[target_col])
        
        # Convertir variables de texto/categorías a números mediante get_dummies
        X_encoded = pd.get_dummies(X_df, drop_first=True)
        self.features = list(X_encoded.columns)

        # División 80% entrenamiento / 20% prueba
        return train_test_split(X_encoded.values, y, test_size=0.2, random_state=42, stratify=y)

    def entrenar_y_evaluar(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Ajusta el modelo con los datos de entrenamiento y extrae métricas con los de prueba.
        """
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        return {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "matriz_confusion": cm.tolist()
        }