import re
import pandas as pd

IN_PATH  = "output/Final/all_metrics.csv"
OUT_PATH = "output/Final/all_metrics_with_season_liga.csv"


def normalize_liga(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name)).strip()
    if re.fullmatch(r"(?i)la liga", name):
        return "La Liga"
    if name.isupper():
        return name
    return name.title()

def split_season_liga(season_str: str):
    s = str(season_str).strip()
    if re.fullmatch(r"\d{4}-\d{4}", s):
        return s, "La Liga"
    m_space = re.match(r"(\d{4}-\d{4})\s+(.+)$", s)
    if m_space:
        return m_space.group(1), normalize_liga(m_space.group(2))
    m_dash = re.match(r"(\d{4}-\d{4})-(.+)$", s)
    if m_dash:
        return m_dash.group(1), normalize_liga(m_dash.group(2))
    return s, "Desconocida"

def main():
    df = pd.read_csv(IN_PATH)
    if "Season" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"Season": "season"})
    elif "season" not in df.columns:
        raise ValueError("No se encontró la columna 'Season' ni 'season' en el CSV.")
    split_values = df["season"].apply(split_season_liga)
    df["season"] = split_values.apply(lambda x: x[0])
    df["liga"]   = split_values.apply(lambda x: x[1])
    df.to_csv(OUT_PATH, index=False)
    print(f"Guardado en: {OUT_PATH}")
    print(df[["season", "liga"]].head(10))

if __name__ == "__main__":
    main()
