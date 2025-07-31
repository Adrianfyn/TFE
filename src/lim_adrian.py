import pandas as pd
import numpy as np
from pathlib import Path

# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "Datos La Liga 17-25"

# Leer los CSV con rutas absolutas
defensive = pd.read_csv(DATA_PATH / "Defensive.csv")
defensive_clean = defensive.drop(columns=[
    'Unnamed: 2_level_0 90s_for', 'Unnamed: 1_level_0 # Pl_for', 'Tackles Tkl_for',
    'Challenges Att_for', 'Unnamed: 16_level_0 Tkl+Int_for', 'Unnamed: 2_level_0 90s_against',
    'Unnamed: 1_level_0 # Pl_against', 'Tackles Tkl_against', 'Challenges Att_against',
    'Unnamed: 16_level_0 Tkl+Int_against'
])
defensive_clean.to_csv(BASE_DIR / "output/defensive_clean.csv", index=False)

misc = pd.read_csv(DATA_PATH / "misc.csv")
misc_clean = misc.drop(columns=[
    'Unnamed: 2_level_0 90s_for', 'Unnamed: 2_level_0 90s_against',
    'Unnamed: 1_level_0 # Pl_for', 'Unnamed: 1_level_0 # Pl_against',
    'Aerial Duels Won%_against', 'Aerial Duels Won%_for'
])
misc_clean.to_csv(BASE_DIR / "output/misc_clean.csv", index=False)

passing = pd.read_csv(DATA_PATH / "Passing.csv")
passing_clean = passing.drop(columns=[
    'Unnamed: 2_level_0 90s_for', 'Unnamed: 2_level_0 90s_against',
    'Unnamed: 1_level_0 # Pl_for', 'Unnamed: 1_level_0 # Pl_against',
    'Expected A-xAG_for', 'Expected A-xAG_against', 'Long Cmp%_against',
    'Long Cmp%_for', 'Short Cmp%_against', 'Short Cmp%_for',
    'Medium Cmp%_against', 'Medium Cmp%_for', 'Total Cmp%_against', 'Total Cmp%_for'
])
passing_clean.to_csv(BASE_DIR / "output/passing_clean.csv", index=False)

pass_types = pd.read_csv(DATA_PATH / "PassTypes.csv")
pass_types_clean = pass_types.drop(columns=[
    'Unnamed: 2_level_0 90s_for', 'Unnamed: 2_level_0 90s_against',
    'Unnamed: 1_level_0 # Pl_for', 'Unnamed: 1_level_0 # Pl_against'
])
pass_types_clean.to_csv(BASE_DIR / "output/pass_types_clean.csv", index=False)

# Unir DataFrames
Clean_adrian = pd.merge(defensive_clean, misc_clean, on=["Squad", "Season"], how='left', suffixes=('_df', '_misc'))
Clean_adrian = pd.merge(Clean_adrian, passing_clean, on=["Squad", "Season"], how='left', suffixes=('', '_ps'))
Clean_adrian = pd.merge(Clean_adrian, pass_types_clean, on=["Squad", "Season"], how='left', suffixes=('', '_ps_t'))

# Mostrar el resultado
print(Clean_adrian.head())

# Guardar el DataFrame limpio
Clean_adrian.to_csv(BASE_DIR / "output/Clean_adrian.csv", index=False)
