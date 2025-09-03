import pandas as pd
from pathlib import Path

# Ruta base = carpeta raíz del proyecto (padre de src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Archivos
INPUT_FILE = BASE_DIR / "output" / "Final" / "all_metrics_with_season_liga.csv"
OUTPUT_FILE = BASE_DIR / "output" / "Final" / "all_teams_with_season_liga_clean.csv"

def limpiar_datos():
    df = pd.read_csv(INPUT_FILE)
    print("Shape inicial:", df.shape)

    # Columnas a eliminar
    cols_to_drop = [
        # Defensive
        'Unnamed: 2_level_0 90s_for', 'Unnamed: 1_level_0 # Pl_for', 'Tackles Tkl_for',
        'Challenges Att_for', 'Unnamed: 16_level_0 Tkl+Int_for', 'Unnamed: 2_level_0 90s_against',
        'Unnamed: 1_level_0 # Pl_against', 'Tackles Tkl_against', 'Challenges Att_against',
        'Unnamed: 16_level_0 Tkl+Int_against',
        # Misc
        'Unnamed: 2_level_0 90s_for', 'Unnamed: 2_level_0 90s_against',
        'Unnamed: 1_level_0 # Pl_for', 'Unnamed: 1_level_0 # Pl_against',
        'Aerial Duels Won%_against', 'Aerial Duels Won%_for',
        # Passing
        'Expected A-xAG_for', 'Expected A-xAG_against',
        'Long Cmp%_against', 'Long Cmp%_for', 'Short Cmp%_against', 'Short Cmp%_for',
        'Medium Cmp%_against', 'Medium Cmp%_for', 'Total Cmp%_against', 'Total Cmp%_for',
        # PassTypes
        'Unnamed: 2_level_0 90s_for', 'Unnamed: 2_level_0 90s_against',
        'Unnamed: 1_level_0 # Pl_for', 'Unnamed: 1_level_0 # Pl_against'
    ]

    # Identificar columnas realmente eliminables
    cols_present = set(df.columns)
    cols_found = list(set(cols_to_drop) & cols_present)
    cols_missing = list(set(cols_to_drop) - cols_present)

    # Eliminar solo las que están presentes
    df_clean = df.drop(columns=cols_found, errors="ignore")

    print("✅ Eliminadas:", len(cols_found), "columnas")
    print("⚠️  No encontradas en el CSV:", len(cols_missing), "columnas")
    if cols_missing:
        print("   →", cols_missing[:10], "...")  # solo mostrar las primeras 10 por si son muchas

    print("Shape limpio:", df_clean.shape)

    # Asegurar directorio de salida
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Archivo limpio guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    limpiar_datos()
