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

if __name__ == "__main__":
    # 1) Cargar el CSV original desde la carpeta Data
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir.parent / 'Data' / 'Datos La Liga 17-25' / 'posesion.csv'
    df_original = pd.read_csv(csv_path)

    # 2) Limpiar para clustering
    df_clean = clean_for_clustering(
        df_original,
        zero_var_thresh=0.0,
        corr_thresh=0.9999,
        miss_thresh=0.5,
        dom_thresh=0.95,
        drop_name_patterns=['90'],
        drop_exact=['Season','Squad']
    )

    # 3) Crear carpeta de salida y guardar con nombre "<nombre_csv>_clean.csv"
    output_dir = script_dir.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_filename = f"{csv_path.stem}_clean.csv"
    df_clean.to_csv(output_dir / clean_filename, index=False)

    print(f"✅ Dataset limpio guardado en: {output_dir / clean_filename}")
