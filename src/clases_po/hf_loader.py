"""
Módulo para cargar datasets desde Hugging Face y convertirlos a DataFrame.
Ubicación: src/clases_po/hf_loader.py
"""
import os
from pathlib import Path
from typing import Optional
from datasets import load_dataset
import pandas as pd


class HFDatasetLoader:
    """
    Descarga un dataset desde Hugging Face y lo devuelve como DataFrame de pandas.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        filename: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.filename = filename
        self.file_type = file_type

    def load(self) -> pd.DataFrame:
        """
        Descarga el dataset desde Hugging Face usando una ruta corta en Windows
        para evitar el límite de 260 caracteres. Si falla la red, usa el respaldo local.
        """
        try:
            print(f"[INFO] Conectando a Hugging Face: '{self.dataset_name}'...")
            # En Windows ('nt') usamos C:/hfc para que la ruta sea corta y no dé FileNotFoundError
            carpeta_cache = "C:/hfc" if os.name == "nt" else None

            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=carpeta_cache,
            )
            df = dataset.to_pandas()

            # Guardamos copia actualizada en datos/ si se indicó filename
            if self.filename:
                ruta_salida = Path(self.filename)
                ruta_salida.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(ruta_salida, index=False)

            print(f"[INFO] ¡Listo! Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")
            return df

        except Exception as error:
            # Plan B: si ya existe el CSV en datos/ de la actividad anterior, lo cargamos
            if self.filename and Path(self.filename).exists():
                print(f"[AVISO] Usando respaldo local ya descargado en: {self.filename}")
                return pd.read_csv(self.filename)
            raise RuntimeError(f"No se pudo cargar el dataset: {error}") from error