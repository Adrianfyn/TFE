import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from pathlib import Path

def clean_for_clustering(
    df: pd.DataFrame,
    zero_var_thresh: float = 0.0,
    corr_thresh: float = 0.9999,
    miss_thresh: float = 0.5,
    dom_thresh: float = 0.95,
    drop_name_patterns: list[str] = None,
    drop_exact: list[str] = None
) -> pd.DataFrame:
    """
    Limpia un DataFrame para clustering:
      1) Elimina IDs/textos no numéricos.
      2) Columnas únicas (nunique==nfilas).
      3) Columnas con > miss_thresh % de NAs.
      4) Categorías dominantes > dom_thresh.
      5) Varianza ≤ zero_var_thresh.
      6) Correlación absoluta > corr_thresh.
      7) Patrones de nombre y borrado exacto.
    """
    df = df.copy()
    
    # 1) IDs/texto (salvo los que queramos mantener)
    drop_exact = drop_exact or []
    non_num = df.select_dtypes(include=['object','category']).columns
    to_drop = [c for c in non_num if c not in drop_exact]
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    # 2) Columnas únicas
    uniques = [c for c in df.columns if df[c].nunique() == len(df)]
    df.drop(columns=uniques, inplace=True, errors='ignore')

    # 3) Muchas NAs
    to_drop = [c for c in df.columns if df[c].isna().mean() > miss_thresh]
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    # 4) Categorías dominantes
    cat_cols = df.select_dtypes(include=['object','category']).columns
    to_drop = [
        c for c in cat_cols
        if df[c].value_counts(normalize=True, dropna=False).iloc[0] > dom_thresh
    ]
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    # 5) Varianza casi nula
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        vt = VarianceThreshold(threshold=zero_var_thresh)
        vt.fit(df[num_cols])
        low_var = list(num_cols[~vt.get_support()])
        df.drop(columns=low_var, inplace=True, errors='ignore')

    # 6) Correlación perfecta
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr = df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > corr_thresh)]
        df.drop(columns=to_drop, inplace=True, errors='ignore')

    # 7) Patrones de nombre y exactos
    drop_name_patterns = drop_name_patterns or []
    for pat in drop_name_patterns:
        to_drop = [c for c in df.columns if pat in c]
        df.drop(columns=to_drop, inplace=True, errors='ignore')
    df.drop(columns=drop_exact, inplace=True, errors='ignore')

    return df

def process_season_column(df, league_name):
    """
    Procesa la columna Season para crear una nueva columna Liga
    """
    df = df.copy()
    
    if 'Season' in df.columns:
        # Para La Liga (formato simple)
        if league_name == "La Liga":
            df['Liga'] = 'La Liga'
        
        # Para Liga Femenina (formato con sufijo)
        elif league_name == "Liga Femenina":
            df['Liga'] = 'Liga Femenina'
            df['Season'] = df['Season'].str.replace('-femenino', '')
        
        # Para Top 5 (múltiples ligas en una celda)
        elif league_name == "Top 5":
            # Dividir por el punto y coma y expandir
            expanded_df = df.copy()
            expanded_df['Season'] = expanded_df['Season'].str.split('; ')
            expanded_df = expanded_df.explode('Season')
            
            # Extraer la liga de la temporada
            expanded_df['Liga'] = expanded_df['Season'].str.split(' ').str[1:].str.join(' ')
            expanded_df['Season'] = expanded_df['Season'].str.split(' ').str[0]
            
            return expanded_df
    
    return df

def load_and_process_league_data(data_path, league_name):
    """
    Carga y procesa los datos para una liga específica
    """
    # Cargar y limpiar datos defensivos
    defensive = pd.read_csv(data_path / "Defensive.csv")
    defensive_clean = defensive.drop(columns=[
        'Unnamed: 2_level_0 90s_for', 'Unnamed: 1_level_0 # Pl_for', 'Tackles Tkl_for',
        'Challenges Att_for', 'Unnamed: 16_level_0 Tkl+Int_for', 'Unnamed: 2_level_0 90s_against',
        'Unnamed: 1_level_0 # Pl_against', 'Tackles Tkl_against', 'Challenges Att_against',
        'Unnamed: 16_level_0 Tkl+Int_against'
    ])
    
    # Cargar y limpiar datos misceláneos
    misc = pd.read_csv(data_path / "misc.csv")
    misc_clean = misc.drop(columns=[
        'Unnamed: 2_level_0 90s_for', 'Unnamed: 2_level_0 90s_against',
        'Unnamed: 1_level_0 # Pl_for', 'Unnamed: 1_level_0 # Pl_against',
        'Aerial Duels Won%_against', 'Aerial Duels Won%_for'
    ])
    
    # Cargar y limpiar datos de pases
    passing = pd.read_csv(data_path / "Passing.csv")
    passing_clean = passing.drop(columns=[
        'Unnamed: 2_level_0 90s_for', 'Unnamed: 2_level_0 90s_against',
        'Unnamed: 1_level_0 # Pl_for', 'Unnamed: 1_level_0 # Pl_against',
        'Expected A-xAG_for', 'Expected A-xAG_against', 'Long Cmp%_against',
        'Long Cmp%_for', 'Short Cmp%_against', 'Short Cmp%_for',
        'Medium Cmp%_against', 'Medium Cmp%_for', 'Total Cmp%_against', 'Total Cmp%_for'
    ])
    
    # Cargar y limpiar tipos de pases
    pass_types = pd.read_csv(data_path / "PassTypes.csv")
    pass_types_clean = pass_types.drop(columns=[
        'Unnamed: 2_level_0 90s_for', 'Unnamed: 2_level_0 90s_against',
        'Unnamed: 1_level_0 # Pl_for', 'Unnamed: 1_level_0 # Pl_against'
    ])
    
    # Procesar la columna Season para cada DataFrame
    defensive_clean = process_season_column(defensive_clean, league_name)
    misc_clean = process_season_column(misc_clean, league_name)
    passing_clean = process_season_column(passing_clean, league_name)
    pass_types_clean = process_season_column(pass_types_clean, league_name)
    
    # Unir todos los DataFrames de esta liga
    merged_df = pd.merge(defensive_clean, misc_clean, on=["Squad", "Season", "Liga"], how='left', suffixes=('_df', '_misc'))
    merged_df = pd.merge(merged_df, passing_clean, on=["Squad", "Season", "Liga"], how='left', suffixes=('', '_ps'))
    merged_df = pd.merge(merged_df, pass_types_clean, on=["Squad", "Season", "Liga"], how='left', suffixes=('', '_ps_t'))
    
    return merged_df

def main():
    # Configuración de paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "Data"
    OUTPUT_DIR = BASE_DIR / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Definir las carpetas de datos y sus nombres de liga
    leagues = [
        {"path": DATA_DIR / "Datos LaLiga 17-25", "name": "La Liga"},
        {"path": DATA_DIR / "Datos Liga F 24-25", "name": "Liga Femenina"},
        {"path": DATA_DIR / "Datos top 5 24-25", "name": "Top 5"}
    ]
    
    # Lista para almacenar los DataFrames de cada liga
    all_leagues_data = []
    
    # Procesar cada liga
    for league in leagues:
        print(f"Procesando {league['name']}...")
        try:
            league_data = load_and_process_league_data(league["path"], league["name"])
            all_leagues_data.append(league_data)
            print(f"✅ {league['name']} procesada correctamente")
        except Exception as e:
            print(f"❌ Error procesando {league['name']}: {str(e)}")
    
    # Combinar todos los datos
    if all_leagues_data:
        combined_df = pd.concat(all_leagues_data, ignore_index=True)
        
        # Limpieza final del DataFrame combinado
        final_clean = clean_for_clustering(
            combined_df,
            zero_var_thresh=0.0,
            corr_thresh=0.9999,
            miss_thresh=0.5,
            dom_thresh=0.95,
            drop_name_patterns=['90'],
            drop_exact=[]  # No eliminamos Season y Liga para poder filtrar
        )
        
        # Asegurarnos de que Squad, Season y Liga se mantienen
        for col in ['Squad', 'Season', 'Liga']:
            if col not in final_clean.columns and col in combined_df.columns:
                final_clean[col] = combined_df[col]
        
        # Guardar resultados
        final_clean.to_csv(OUTPUT_DIR / "Cleaned_Data_All_Leagues.csv", index=False)
        print(f"✅ Datos limpios de todas las ligas guardados en: {OUTPUT_DIR / 'Cleaned_Data_All_Leagues.csv'}")
        
        # Mostrar las ligas únicas para verificar
        if 'Liga' in final_clean.columns:
            print("\n🔍 Ligas presentes en el dataset:")
            for liga in final_clean['Liga'].unique():
                print(f"   - {liga}")
                
        # Mostrar información del dataset final
        print(f"\n📊 Dataset final: {final_clean.shape[0]} filas, {final_clean.shape[1]} columnas")
    else:
        print("❌ No se pudo procesar ninguna liga")

if __name__ == "__main__":
    main()