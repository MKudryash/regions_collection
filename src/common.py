"""Общие константы и функции проекта «Цифровое развитие регионов РФ, 2021».

Все ноутбуки импортируют этот модуль: пути, проекция, коды показателей,
загрузка исходных файлов, построение матрицы соседства и метрики разбиений.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MAPS = ROOT / "maps"
for _p in (RAW, PROCESSED, RESULTS, MAPS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 2021

# Равновеликая коническая проекция Альберса для России: стандартные параллели 52° и 64° с. ш.,
# осевой меридиан 100° в. д. Площади сохраняются, Чукотка не разрывается на 180° меридиане.
CRS_AEA = "+proj=aea +lat_1=52 +lat_2=64 +lat_0=0 +lon_0=100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

INDICATORS = {
    "17080100": ("pc", "Домохозяйства с ПК, %"),
    "17080200": ("internet", "Домохозяйства с доступом в интернет, %"),
    "17080300": ("broadband", "Домохозяйства с широкополосным доступом, %"),
    "17090200": ("daily_users", "Ежедневные пользователи интернета, % населения 15–74 лет"),
}
FEATURES = [v[0] for v in INDICATORS.values()]
LABELS = {v[0]: v[1] for v in INDICATORS.values()}
SHORT = {"pc": "ПК", "internet": "Интернет", "broadband": "ШПД", "daily_users": "Ежедневно онлайн"}
POPULATION_CODE = "01010000"  # численность населения (для ограничения Max-p)
YEAR = 2021
MISSING_CODE = -99999999.0

SOURCES = {
    "regions_collection": "https://github.com/tochno-st/regions_collection",
    "regions_collection_rar": "https://raw.githubusercontent.com/tochno-st/regions_collection/HEAD/"
                              "data/processed/data_regions_collection_102_v1.0.rar",
    "regions_etalon": "https://raw.githubusercontent.com/tochno-st/regions_collection/HEAD/src/data/regions_etalon.yaml",
    "boundaries": "https://raw.githubusercontent.com/timurkanaz/Russia_geojson_OSM/HEAD/"
                  "GeoJson%27s/Countries/Russia_regions.geojson",
    "boundaries_repo": "https://github.com/timurkanaz/Russia_geojson_OSM",
}


def download(url: str, dst: Path) -> Path:
    """Скачать файл, если его еще нет (для воспроизводимости без хранения тяжелых файлов в git)."""
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("download", url)
        urllib.request.urlretrieve(url, dst)
    return dst


# ---------------------------------------------------------------- матрица соседства
def connected_queen(gdf, k_link: int = 1):
    """Queen-смежность по общим границам + соединение несвязных компонент.

    Регионы без сухопутных соседей (Калининградская, Сахалинская области) и компоненты,
    отделенные морем (Крым + Севастополь), присоединяются к ближайшему по расстоянию
    между границами региону другой компоненты. Возвращает (W, список добавленных связей).
    Связный граф нужен методам регионализации (SKATER, Ward со связностью, Max-p).
    """
    from libpysal.weights import Queen, W
    from scipy.sparse.csgraph import connected_components

    wq = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    neigh = {i: list(v) for i, v in wq.neighbors.items()}
    added = []
    while True:
        w = W(neigh, silence_warnings=True)
        ncomp, lab = connected_components(w.sparse, directed=False)
        if ncomp == 1:
            break
        # самая маленькая компонента присоединяется к ближайшему региону вне ее
        sizes = np.bincount(lab)
        c = int(np.argmin(sizes))
        inside = np.where(lab == c)[0]
        outside = np.where(lab != c)[0]
        best = (np.inf, None, None)
        for i in inside:
            d = gdf.geometry.iloc[outside].distance(gdf.geometry.iloc[i]).to_numpy()
            j = int(np.argmin(d))
            if d[j] < best[0]:
                best = (d[j], int(i), int(outside[j]))
        _, i, j = best
        neigh[i].append(j)
        neigh[j].append(i)
        added.append((i, j, best[0] / 1000))
    w = W(neigh, silence_warnings=True)
    return w, added


# ---------------------------------------------------------------- метрики разбиений
def partition_metrics(X: np.ndarray, labels: np.ndarray, w, gdf=None) -> dict:
    """Метрики качества разбиения в пространстве признаков и в географическом пространстве."""
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from scipy.sparse.csgraph import connected_components

    labels = np.asarray(labels)
    k = len(np.unique(labels))
    sst = ((X - X.mean(0)) ** 2).sum()
    ssw = sum(((X[labels == c] - X[labels == c].mean(0)) ** 2).sum() for c in np.unique(labels))
    out = {
        "k": k,
        "R2": 1 - ssw / sst,
        "silhouette": silhouette_score(X, labels) if k > 1 else np.nan,
        "CH": calinski_harabasz_score(X, labels) if k > 1 else np.nan,
        "DB": davies_bouldin_score(X, labels) if k > 1 else np.nan,
        "min_size": int(np.bincount(pd.factorize(labels)[0]).min()),
    }
    # пространственная связность: число связных частей у каждого кластера в графе W
    A = w.sparse.tocsr()
    parts = 0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n, _ = connected_components(A[idx][:, idx], directed=False)
        parts += n
    out["fragments"] = parts           # = k, если все кластеры связны
    # доля связей соседства внутри одного кластера (пространственная однородность меток)
    rows, cols = A.nonzero()
    out["within_links"] = float(np.mean(labels[rows] == labels[cols]))
    if gdf is not None:
        from esda.shape import isoperimetric_quotient
        diss = gdf.assign(_c=labels).dissolve("_c")
        out["IPQ"] = float(np.mean(isoperimetric_quotient(diss.geometry)))
    return out

