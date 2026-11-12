"""Единый стиль графиков и карт проекта.

Цвета: однотонная синяя шкала для величин (хороплеты), пара синий/красный с серой серединой для
полярности (LISA, локальные эффекты), фиксированный порядок категориальных цветов для кластеров.
Если кластеров больше трех, номер кластера подписывается на карте: различение не держится на одном цвете.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
EDGE = "#ffffff"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)
BLUE, RED, NEUTRAL = "#256abf", "#d03b3b", "#f0efec"
LISA_COLORS = {"High-High": "#d03b3b", "Low-Low": "#256abf", "Low-High": "#9ec5f4",
               "High-Low": "#f2a39a", "незначимо": "#e8e7e2"}

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False,
    "text.color": INK, "lines.linewidth": 2,
})


def _off(ax, title=None):
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left")


def choropleth(ax, gdf, col, k=5, title=None, fmt="{:.1f}"):
    """Хороплет по естественным интервалам Фишера–Дженкса; легенда с границами классов."""
    import mapclassify
    cl = mapclassify.FisherJenks(gdf[col].to_numpy(), k=k)
    colors = [SEQ(x) for x in np.linspace(0.08, 1, k)]
    gdf.plot(ax=ax, color=[colors[b] for b in cl.yb], edgecolor=EDGE, linewidth=0.25)
    lo = gdf[col].min()
    handles = []
    for i, hi in enumerate(cl.bins):
        handles.append(Patch(facecolor=colors[i], edgecolor="none",
                             label=f"{fmt.format(lo)} – {fmt.format(hi)}  ({int((cl.yb == i).sum())})"))
        lo = hi
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, title=None, handlelength=1.2)
    _off(ax, title)
    return cl


def lisa_map(ax, gdf, lisa, alpha=0.05, title=None, labels=True):
    """Карта кластеров LISA: квадрант диаграммы Морана, только значимые при p_sim < alpha."""
    q = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
    cat = np.array([q[v] if p < alpha else "незначимо" for v, p in zip(lisa.q, lisa.p_sim)])
    gdf.plot(ax=ax, color=[LISA_COLORS[c] for c in cat], edgecolor=EDGE, linewidth=0.25)
    present = [c for c in LISA_COLORS if (cat == c).any()]
    ax.legend(handles=[Patch(facecolor=LISA_COLORS[c], label=f"{c} ({int((cat == c).sum())})") for c in present],
              loc="lower left", fontsize=7.5, handlelength=1.2)
    if labels:
        for (_, r), c in zip(gdf.iterrows(), cat):
            if c != "незначимо":
                pt = r.geometry.representative_point()
                ax.annotate(short_name(r["name"]), (pt.x, pt.y), fontsize=5.5, ha="center", color=INK2)
    _off(ax, title)
    return cat


def cluster_map(ax, gdf, labels, title=None, names=None, annotate_ids=True):
    """Карта разбиения: категориальные цвета в фиксированном порядке + номер кластера на карте."""
    labels = np.asarray(labels)
    ids = sorted(np.unique(labels))
    color = {c: CAT[i % len(CAT)] for i, c in enumerate(ids)}
    gdf.plot(ax=ax, color=[color[c] for c in labels], edgecolor=EDGE, linewidth=0.3)
    if annotate_ids:
        for (_, r), c in zip(gdf.iterrows(), labels):
            if r.geometry.area > 4e10 or len(ids) > 3:
                pt = r.geometry.representative_point()
                ax.annotate(str(c), (pt.x, pt.y), fontsize=6, ha="center", va="center",
                            color="#ffffff" if color[c] not in ("#eda100",) else INK, weight="bold")
    lab = names or {c: f"Кластер {c}" for c in ids}
    ax.legend(handles=[Patch(facecolor=color[c], label=f"{lab[c]} ({int((labels == c).sum())})") for c in ids],
              loc="lower left", fontsize=7.5, handlelength=1.2)
    _off(ax, title)
    return color


def short_name(n: str) -> str:
    rep = [("Республика ", ""), (" область", ""), (" автономный округ", " АО"), ("автономная", "АО"),
           (" без автономных округов", "*"), (" без автономного округа", "*"), ("г. ", ""), (" край", " кр."),
           ("Ханты-Мансийский АО – Югра", "ХМАО"), ("Ямало-Ненецкий АО", "ЯНАО"), (" Республика", ""),
           ("Северная Осетия – Алания", "Сев. Осетия")]
    for a, b in rep:
        n = n.replace(a, b)
    return n
