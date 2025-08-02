import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

# ------------------- RUTAS -------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Output/Correlation"
DATA_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH1 = BASE_DIR / "Output"
# ------------------- CARGA DE DATOS -------------------
df = pd.read_csv(DATA_PATH1 / "all_teams_all_metrics.csv")
df_numeric = df.select_dtypes(include=[np.number])

# ------------------- MATRIZ DE CORRELACIÓN -------------------
corr_matrix = df_numeric.corr()
corr_matrix.to_csv(DATA_PATH / "correlation_matrix.csv")

# ------------------- ELIMINAR VARIABLES ALTAMENTE CORRELACIONADAS -------------------
def eliminar_variables_correlacionadas(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()

    for column in upper.columns:
        # Asegurarse de que upper[column] existe y no es todo NaN
        col_corr = upper.get(column)
        if col_corr is not None:
            correladas = col_corr[col_corr > threshold].index.tolist()
            if correladas:
                to_drop.add(column)

    return df.drop(columns=to_drop), list(to_drop)

df_filtrado, eliminadas = eliminar_variables_correlacionadas(df_numeric)

# Guardar el nuevo dataset filtrado
filtered_csv_path = DATA_PATH / "filtered_features.csv"
df_filtrado.to_csv(filtered_csv_path, index=False)
print(f"{len(eliminadas)} variables eliminadas por alta correlación.")
print(f"Dataset filtrado guardado en: {filtered_csv_path}")
print("Forma del dataset filtrado:", df_filtrado.shape)
print("Forma del dataset original:", df_numeric.shape)
# ------------------- MAPA DE CALOR DEL FILTRADO -------------------
filtered_corr = df_filtrado.corr()
mask = np.triu(np.ones_like(filtered_corr, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(filtered_corr, mask=mask, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title('Mapa de calor de correlaciones (filtrado)')

# Guardar imagen
heatmap_path = DATA_PATH / "filtered_correlation_heatmap.png"
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"Mapa de calor filtrado guardado en: {heatmap_path}")

# Mostrar imagen
plt.show()