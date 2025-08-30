import pandas as pd
from pathlib import Path

def merge_csvs_with_report(base_dir, output_dir):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Agrupar archivos por nombre
    files_by_name = {}
    for p in base_dir.rglob("*.csv"):
        files_by_name.setdefault(p.name, []).append(p)

    # Procesar cada grupo de CSV
    for name, paths in files_by_name.items():
        col_sets = []
        dfs = []
        for p in paths:
            df = pd.read_csv(p)
            col_sets.append(set(df.columns))
            dfs.append(df)

        # Columnas comunes a todos los archivos
        common_cols = set.intersection(*col_sets)
        # Columnas distintas
        all_cols = set.union(*col_sets)
        diff_cols = all_cols - common_cols

        if diff_cols:
            print(f"⚠️ {name}: las columnas NO coinciden en todos los archivos")
            print(f"   - Comunes: {sorted(common_cols)}")
            print(f"   - Diferentes: {sorted(diff_cols)}")
        else:
            print(f"✅ {name}: columnas coinciden en todos los archivos")

        # Fusionar (Pandas añade NaN donde falten columnas)
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_dir / name, index=False)

        print(f"   Guardado en {output_dir / name} con {len(merged)} filas (desde {len(paths)} archivos)\n")

def merge_selected_metrics(
    base_dir: str | Path = "output/merged_cleaned",
    out_path: str | Path = "output/Final/all_teams_all_metrics.csv",
    keys=("Season","Squad"),
    how="inner",   # usa "outer" si quieres conservar todas las filas
):
    base_dir = Path(base_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cargar lista de CSV y filtrar exclusiones
    csvs = sorted(base_dir.glob("*.csv"))
    def exclude(p: Path) -> bool:
        name = p.stem.lower()
        return ("overall" in name) or ("goalkeeping" in name)  # excluye ambos: goalkeeping y advanced goalkeeping
    csvs = [p for p in csvs if not exclude(p)]
    if not csvs:
        raise FileNotFoundError(f"No hay CSV válidos en {base_dir}")

    print("[INFO] Archivos a fusionar:")
    for p in csvs: print("  -", p.name)

    # Merge progresivo
    merged = pd.read_csv(csvs[0])
    for p in csvs[1:]:
        right = pd.read_csv(p)
        suffix = f"__{p.stem}"
        merged = merged.merge(right, on=list(keys), how=how, suffixes=("", suffix))

    # Guardar
    merged.to_csv(out_path, index=False)
    print(f"✅ Guardado: {out_path}  | shape: {merged.shape}")
    return merged

if __name__ == "__main__":
    #merge_csvs_with_report("Data", "output/merged_not_cleaned")
    merge_selected_metrics(
        base_dir=Path(__file__).resolve().parent.parent / "output" / "merged_cleaned",
        out_path=Path(__file__).resolve().parent.parent / "output" / "Final" / "all_metrics.csv",
        keys=("Season","Squad"),
        how="inner",   
    )