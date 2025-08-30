import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from pathlib import Path

def merge_and_save_all_clean_csvs(
    output_dir: Path | str,
    on: list[str] = ['Season', 'Squad'],
    how: str = 'inner',
    merged_filename: str = 'all_teams_all_metrics.csv'
) -> pd.DataFrame:
    output_path = Path(output_dir)
    csv_files = sorted(output_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {output_dir!r}")
    merged_df = pd.read_csv(csv_files[0])
    for csv_file in csv_files[1:]:
        df = pd.read_csv(csv_file)
        merged_df = merged_df.merge(df, on=on, how=how)
    merged_path = output_path / merged_filename
    merged_df.to_csv(merged_path, index=False)
    return merged_df

def clean_for_clustering(
    df: pd.DataFrame,
    zero_var_thresh: float = 0.0,
    corr_thresh: float = 0.9999,
    miss_thresh: float = 0.5,
    dom_thresh: float = 0.95,
    drop_name_patterns: list[str] = None,
    drop_exact: list[str] = None
) -> pd.DataFrame:
   
    df = df.copy()
    
    #IDs/texto 
    drop_exact = drop_exact or []
    non_num = df.select_dtypes(include=['object','category']).columns
    to_drop = [c for c in non_num if c not in drop_exact]
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    #Columnas únicas
    uniques = [c for c in df.columns if df[c].nunique() == len(df)]
    df.drop(columns=uniques, inplace=True, errors='ignore')

    #NAS
    to_drop = [c for c in df.columns if df[c].isna().mean() > miss_thresh]
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    #Categorías dominantes
    cat_cols = df.select_dtypes(include=['object','category']).columns
    to_drop = [
        c for c in cat_cols
        if df[c].value_counts(normalize=True, dropna=False).iloc[0] > dom_thresh
    ]
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    #Varianza casi nula
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        vt = VarianceThreshold(threshold=zero_var_thresh)
        vt.fit(df[num_cols])
        low_var = list(num_cols[~vt.get_support()])
        df.drop(columns=low_var, inplace=True, errors='ignore')

    #Correlación perfecta
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr = df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > corr_thresh)]
        df.drop(columns=to_drop, inplace=True, errors='ignore')

    #Patrones de nombre y exactos
    drop_name_patterns = drop_name_patterns or []
    for pat in drop_name_patterns:
        to_drop = [c for c in df.columns if pat in c]
        df.drop(columns=to_drop, inplace=True, errors='ignore')
    df.drop(columns=drop_exact, inplace=True, errors='ignore')
    df = df.loc[:, ~df.columns.str.contains('Pl', case=False)]
    df = df.loc[:, ~df.columns.str.contains('Age', case=False)]
    return df

if __name__ == "__main__":

    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir.parent / 'Data' / 'Datos La Liga 17-25' / 'squad-shooting.csv'
    df_original = pd.read_csv(csv_path)


    df_clean = clean_for_clustering(
        df_original,
        zero_var_thresh=0.0,
        corr_thresh=0.9999,
        miss_thresh=0.5,
        dom_thresh=0.95,
        drop_name_patterns=['90'],
        drop_exact=['Season','Squad']
    )
    df_clean[['Season','Squad']] = df_original[['Season','Squad']]
    cols = ['Season', 'Squad'] + [c for c in df_clean.columns if c not in ('Season','Squad')]
    df_clean = df_clean[cols]
    output_dir = script_dir.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_filename = f"{csv_path.stem}_clean.csv"
    df_clean.to_csv(output_dir / clean_filename, index=False)

    
    print(f"✅ Dataset limpio guardado en: {output_dir / clean_filename}")

    output_directory = Path(__file__).resolve().parent.parent / 'output' / 'total'
    df_all = merge_and_save_all_clean_csvs(output_directory)
    df_all.shape
    df_all.head(10)
    print("Shape del DataFrame fusionado:", df_all.shape)
