import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import streamlit as st


# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Consumo de Cerveja",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------
# Data loading / preprocessing
# ----------------------------
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and type-cast the dataset."""
    df = df.copy()

    # Standardize column names minimally (keep originals, just strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Parse date
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    # Columns expected
    numeric_cols = [
        "Temperatura Media (C)",
        "Temperatura Minima (C)",
        "Temperatura Maxima (C)",
        "Precipitacao (mm)",
        "Final de Semana",
        "Consumo de cerveja (litros)",
    ]

    # Replace decimal comma with dot for object columns, then convert.
    # Importante: NÃO remover '.' indiscriminadamente, pois algumas colunas (ex.: consumo)
    # já vêm com ponto como separador decimal.
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure Final de Semana is 0/1 if possible
    if "Final de Semana" in df.columns:
        df["Final de Semana"] = df["Final de Semana"].round().astype("Int64")

    return df


def try_load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Try to read dataset from upload; otherwise try local file."""
    if uploaded_file is not None:
        try:
            return _read_csv_from_bytes(uploaded_file.getvalue())
        except Exception as e:
            st.sidebar.error(f"Falha ao ler o CSV enviado: {e}")
            return None

    # Try local relative path first
    for path in ("Consumo_cerveja.csv", "./data/Consumo_cerveja.csv"):
        try:
            return _read_csv_from_path(path)
        except Exception:
            continue

    return None


def apply_filters(
    df: pd.DataFrame,
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
    dropna_essential: bool,
) -> pd.DataFrame:
    df_f = df.copy()

    # Date filter
    if "Data" in df_f.columns:
        start, end = date_range
        mask = (df_f["Data"] >= pd.to_datetime(start)) & (df_f["Data"] <= pd.to_datetime(end))
        df_f = df_f.loc[mask]

    # Optionally drop NAs in essential columns
    essential_cols = [
        "Data",
        "Temperatura Media (C)",
        "Precipitacao (mm)",
        "Final de Semana",
        "Consumo de cerveja (litros)",
    ]
    existing_essential = [c for c in essential_cols if c in df_f.columns]

    if dropna_essential and existing_essential:
        df_f = df_f.dropna(subset=existing_essential)

    return df_f


# ----------------------------
# Plot helpers
# ----------------------------
def plot_scatter_with_trendline(
    x: pd.Series,
    y: pd.Series,
    xlabel: str,
    ylabel: str,
    title: str,
):
    """Scatter plot with linear trendline (polyfit degree 1)."""
    data = pd.DataFrame({"x": x, "y": y}).dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data["x"], data["y"], alpha=0.7)

    slope = intercept = np.nan
    if len(data) >= 2:
        slope, intercept = np.polyfit(data["x"].to_numpy(), data["y"].to_numpy(), 1)
        xs = np.linspace(data["x"].min(), data["x"].max(), 100)
        ys = slope * xs + intercept
        ax.plot(xs, ys)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    return fig, slope, intercept, data


def safe_pearsonr(x: pd.Series, y: pd.Series):
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) < 2:
        return np.nan, np.nan, len(data)
    try:
        r, p = pearsonr(data["x"], data["y"])
        return float(r), float(p), len(data)
    except Exception:
        return np.nan, np.nan, len(data)


def plot_boxplot_by_group(df: pd.DataFrame, value_col: str, group_col: str, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column=value_col, by=group_col, ax=ax, grid=False)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    fig.suptitle("")  # remove automatic 'Boxplot grouped by ...'
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_monthly_line(monthly_mean: pd.Series, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(monthly_mean.index, monthly_mean.values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Mês")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(range(1, 13)))
    ax.grid(True, alpha=0.3)
    return fig


# ----------------------------
# UI
# ----------------------------
st.title("🍺 Análise Exploratória (EDA) — Consumo de Cerveja")
# st.caption(
#     "Objetivo: explorar relações entre clima, fim de semana e sazonalidade no consumo de cerveja, "
#     "com base no dataset `Consumo_cerveja.csv`."
# )


# Data source (apenas arquivo local)
df_raw = try_load_data(None)
if df_raw is None:
    st.warning(
        "Não encontrei o arquivo `Consumo_cerveja.csv` na pasta do app.\n\n"
        "➡️ Coloque `Consumo_cerveja.csv` na mesma pasta do `app.py`."
    )
    st.stop()

# Preprocess
try:
    df = preprocess(df_raw)
except Exception as e:
    st.error(f"Erro no pré-processamento do dataset: {e}")
    st.stop()


# Filtros fixos: usa todo o período disponível, remove nulos essenciais, threshold de chuva = 0
if "Data" not in df.columns or df["Data"].isna().all():
    st.error("A coluna 'Data' não pôde ser convertida para datetime.")
    st.stop()

min_date = df["Data"].min().date()
max_date = df["Data"].max().date()
start_date, end_date = min_date, max_date
dropna_essential = True
rain_max = float(np.nanmax(df["Precipitacao (mm)"].to_numpy())) if "Precipitacao (mm)" in df.columns else 0.0
rain_max = 0.0 if (not np.isfinite(rain_max)) else rain_max
slider_max = float(rain_max) if rain_max > 0 else 10.0
rain_threshold = 0.0

# Apply filters
try:
    df_f = apply_filters(df, (pd.Timestamp(start_date), pd.Timestamp(end_date)), dropna_essential)
except Exception as e:
    st.error(f"Erro ao aplicar filtros: {e}")
    st.stop()

if df_f.empty:
    st.warning("O dataset está vazio após o pré-processamento.")
    st.stop()

# Tabs
aba_geral, aba_q1, aba_q2, aba_q3, aba_q4 = st.tabs(
    ["Visão Geral", "1) Temperatura x Consumo", "2) Fim de semana x Dia útil", "3) Chuva x Consumo", "4) Sazonalidade"]
)


# ----------------------------
# Visão Geral
# ----------------------------
with aba_geral:
    st.subheader("Visão Geral")

    c1, c2, c3, c4 = st.columns(4)

    periodo_txt = f"{df_f['Data'].min().date()} → {df_f['Data'].max().date()}"
    media_consumo = float(df_f["Consumo de cerveja (litros)"].mean()) if "Consumo de cerveja (litros)" in df_f.columns else np.nan
    media_temp = float(df_f["Temperatura Media (C)"].mean()) if "Temperatura Media (C)" in df_f.columns else np.nan

    c1.metric("Linhas (após filtros)", f"{len(df_f):,}".replace(",", "."))
    c2.metric("Período", periodo_txt)
    c3.metric("Média de consumo (L)", f"{media_consumo:.2f}" if np.isfinite(media_consumo) else "-")
    c4.metric("Média temp. média (°C)", f"{media_temp:.2f}" if np.isfinite(media_temp) else "-")

    st.markdown(
        """
**O que olhar aqui:**
- Os KPIs refletem o dataset **já filtrado** (datas + remoção opcional de nulos).
- Use a tabela de *describe* para entender escala e dispersão das variáveis.
- Em *Qualidade dos dados*, confira rapidamente onde há valores ausentes.
"""
    )

    main_cols = [
        "Temperatura Media (C)",
        "Temperatura Minima (C)",
        "Temperatura Maxima (C)",
        "Precipitacao (mm)",
        "Final de Semana",
        "Consumo de cerveja (litros)",
    ]
    main_cols = [c for c in main_cols if c in df_f.columns]

    with st.expander("📊 Estatísticas descritivas (df.describe)", expanded=True):
        st.dataframe(df_f[main_cols].describe().T)

    # with st.expander("🧹 Qualidade dos dados (nulos por coluna)"):
    #     nulls = df_f.isna().sum().sort_values(ascending=False)
    #     st.dataframe(nulls.to_frame("nulos"))

    with st.expander("🔎 Amostra do dataset"):
        st.dataframe(df_f.head(20))


# ----------------------------
# Q1: Temperatura x Consumo
# ----------------------------
with aba_q1:
    st.subheader("1) Existe relação entre temperatura média e consumo?")

    left, right = st.columns([1.2, 0.8], vertical_alignment="top")

    if "Temperatura Media (C)" not in df_f.columns or "Consumo de cerveja (litros)" not in df_f.columns:
        st.error("Colunas necessárias não encontradas no dataset.")
    else:
        fig, slope, intercept, data_xy = plot_scatter_with_trendline(
            df_f["Temperatura Media (C)"],
            df_f["Consumo de cerveja (litros)"],
            xlabel="Temperatura Média (°C)",
            ylabel="Consumo de Cerveja (litros)",
            title="Temperatura Média vs Consumo (com linha de tendência)",
        )

        r, p, n = safe_pearsonr(df_f["Temperatura Media (C)"], df_f["Consumo de cerveja (litros)"])

        with left:
            st.pyplot(fig, use_container_width=True)

        with right:
            st.markdown(
                """
**Como interpretar:**
- Cada ponto é um dia.
- A linha é uma tendência linear (aproximação) para ajudar a ver a direção geral.
- A correlação de Pearson (*r*) mede força/direção da relação linear: valores próximos de **+1** indicam relação positiva, **-1** negativa e **0** pouca relação linear.
- O *p-value* ajuda a avaliar evidência estatística (quanto menor, maior evidência contra a hipótese de correlação zero), mas **não prova causalidade**.
"""
            )

            st.markdown("#### Principais números")
            st.write(f"**N (pares válidos):** {n}")
            st.write(f"**Correlação de Pearson (r):** {r:.4f}" if np.isfinite(r) else "**Correlação de Pearson (r):** -")
            st.write(f"**p-value:** {p:.4g}" if np.isfinite(p) else "**p-value:** -")

            if np.isfinite(slope) and np.isfinite(intercept):
                st.write(f"**Tendência linear:** consumo ≈ {slope:.3f}·temp + {intercept:.3f}")

            with st.expander("📌 Ver dados usados no gráfico"):
                st.dataframe(data_xy.rename(columns={"x": "Temperatura Media (C)", "y": "Consumo (L)"}))


# ----------------------------
# Q2: Fim de semana x Dia útil
# ----------------------------
with aba_q2:
    st.subheader("2) O consumo é diferente em fim de semana vs dia útil?")

    if "Final de Semana" not in df_f.columns or "Consumo de cerveja (litros)" not in df_f.columns:
        st.error("Colunas necessárias não encontradas no dataset.")
    else:
        df_q2 = df_f.copy()
        df_q2["Tipo de dia"] = np.where(df_q2["Final de Semana"].fillna(0).astype(int) == 1, "Fim de semana", "Dia útil")

        fig = plot_boxplot_by_group(
            df_q2,
            value_col="Consumo de cerveja (litros)",
            group_col="Tipo de dia",
            title="Distribuição do consumo por tipo de dia",
            ylabel="Consumo de Cerveja (litros)",
        )

        group_stats = (
            df_q2.groupby("Tipo de dia")["Consumo de cerveja (litros)"]
            .agg(["count", "mean", "std", "min", "median", "max"])
            .sort_index()
        )

        mean_weekend = group_stats.loc["Fim de semana", "mean"] if "Fim de semana" in group_stats.index else np.nan
        mean_weekday = group_stats.loc["Dia útil", "mean"] if "Dia útil" in group_stats.index else np.nan
        diff_means = mean_weekend - mean_weekday if (np.isfinite(mean_weekend) and np.isfinite(mean_weekday)) else np.nan

        left, right = st.columns([1.1, 0.9], vertical_alignment="top")
        with left:
            st.pyplot(fig, use_container_width=True)

        with right:
            st.markdown(
                """
**Como interpretar:**
- O boxplot resume a distribuição do consumo em cada grupo.
- A linha no meio da caixa é a **mediana**.
- A caixa cobre o intervalo interquartil (25%–75%).
- Pontos mais distantes podem indicar valores extremos.
"""
            )

            st.markdown("#### Principais números")
            if np.isfinite(diff_means):
                st.write(f"**Diferença de médias (fim de semana − dia útil):** {diff_means:.2f} L")
            else:
                st.write("**Diferença de médias:** -")

            with st.expander("📋 Estatísticas por grupo (count/mean/std/min/median/max)", expanded=True):
                st.dataframe(group_stats.style.format({"mean": "{:.2f}", "std": "{:.2f}", "min": "{:.2f}", "median": "{:.2f}", "max": "{:.2f}"}))


# ----------------------------
# Q3: Chuva x Consumo
# ----------------------------
with aba_q3:
    st.subheader("3) Chuva (precipitação) afeta o consumo?")

    if "Precipitacao (mm)" not in df_f.columns or "Consumo de cerveja (litros)" not in df_f.columns:
        st.error("Colunas necessárias não encontradas no dataset.")
    else:
        df_q3 = df_f.copy()
        df_q3["Tem chuva?"] = np.where(df_q3["Precipitacao (mm)"].fillna(0) > rain_threshold, "Com chuva", "Sem chuva")

        fig_box = plot_boxplot_by_group(
            df_q3,
            value_col="Consumo de cerveja (litros)",
            group_col="Tem chuva?",
            title=f"Consumo por condição de chuva (threshold: > {rain_threshold:.1f} mm)",
            ylabel="Consumo de Cerveja (litros)",
        )

        fig_scatter, _, _, data_xy = plot_scatter_with_trendline(
            df_q3["Precipitacao (mm)"],
            df_q3["Consumo de cerveja (litros)"],
            xlabel="Precipitação (mm)",
            ylabel="Consumo de Cerveja (litros)",
            title="Precipitação vs Consumo (scatter)",
        )

        stats_rain = (
            df_q3.groupby("Tem chuva?")["Consumo de cerveja (litros)"]
            .agg(["count", "mean", "std"])
            .sort_index()
        )

        mean_rain = stats_rain.loc["Com chuva", "mean"] if "Com chuva" in stats_rain.index else np.nan
        mean_no_rain = stats_rain.loc["Sem chuva", "mean"] if "Sem chuva" in stats_rain.index else np.nan
        diff = mean_rain - mean_no_rain if (np.isfinite(mean_rain) and np.isfinite(mean_no_rain)) else np.nan

        r_prec, p_prec, n_prec = safe_pearsonr(df_q3["Precipitacao (mm)"], df_q3["Consumo de cerveja (litros)"])

        left, right = st.columns([1, 1], vertical_alignment="top")
        with left:
            st.pyplot(fig_box, use_container_width=True)

        with right:
            st.pyplot(fig_scatter, use_container_width=True)

        st.markdown(
            """
**Como interpretar:**
- O **boxplot** compara a distribuição de consumo em dias *com chuva* vs *sem chuva* (de acordo com o threshold definido).
- O **scatter** ajuda a ver se existe alguma tendência do consumo variar conforme a precipitação aumenta.
- Mesmo com diferenças entre médias, lembre-se: isso descreve o que ocorreu nos dados e **não prova causalidade**.
"""
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Média (Sem chuva)", f"{mean_no_rain:.2f}" if np.isfinite(mean_no_rain) else "-")
        c2.metric("Média (Com chuva)", f"{mean_rain:.2f}" if np.isfinite(mean_rain) else "-")
        c3.metric("Diferença (Com − Sem)", f"{diff:.2f}" if np.isfinite(diff) else "-")

        with st.expander("📋 Estatísticas por condição (count/mean/std)", expanded=False):
            st.dataframe(stats_rain.style.format({"mean": "{:.2f}", "std": "{:.2f}"}))

        with st.expander("📌 Correlação Precipitação x Consumo (Pearson)", expanded=False):
            st.write(f"**N (pares válidos):** {n_prec}")
            st.write(f"**r:** {r_prec:.4f}" if np.isfinite(r_prec) else "**r:** -")
            st.write(f"**p-value:** {p_prec:.4g}" if np.isfinite(p_prec) else "**p-value:** -")

        with st.expander("🔎 Dados usados no scatter"):
            st.dataframe(data_xy.rename(columns={"x": "Precipitacao (mm)", "y": "Consumo (L)"}))


# ----------------------------
# Q4: Sazonalidade
# ----------------------------
with aba_q4:
    st.subheader("4) Existe sazonalidade (variação por mês) no consumo?")

    if "Data" not in df_f.columns or "Consumo de cerveja (litros)" not in df_f.columns:
        st.error("Colunas necessárias não encontradas no dataset.")
    else:
        df_q4 = df_f.copy()
        df_q4["Mes"] = df_q4["Data"].dt.month

        monthly_mean = (
            df_q4.groupby("Mes")["Consumo de cerveja (litros)"]
            .mean()
            .reindex(range(1, 13))
        )

        fig = plot_monthly_line(
            monthly_mean,
            title="Consumo médio por mês",
            ylabel="Consumo de Cerveja (litros)",
        )

        left, right = st.columns([1.15, 0.85], vertical_alignment="top")
        with left:
            st.pyplot(fig, use_container_width=True)

        with right:
            st.markdown(
                """
**Como interpretar:**
- O gráfico mostra a **média de consumo** em cada mês (1 a 12) no período filtrado.
- Picos e vales sugerem possível sazonalidade, mas confirme olhando o tamanho da amostra por mês.
"""
            )

            # Top/bottom months
            ranking = monthly_mean.dropna().sort_values(ascending=False)
            top3 = ranking.head(3).to_frame("Consumo médio (L)")
            bottom3 = ranking.sort_values(ascending=True).head(3).to_frame("Consumo médio (L)")

            st.markdown("#### Top 3 meses")
            st.dataframe(top3.style.format({"Consumo médio (L)": "{:.2f}"}))

            st.markdown("#### Bottom 3 meses")
            st.dataframe(bottom3.style.format({"Consumo médio (L)": "{:.2f}"}))

        with st.expander("📋 Tabela completa (média por mês)"):
            st.dataframe(monthly_mean.to_frame("Consumo médio (L)").style.format("{:.2f}"))

        with st.expander("📌 Contagem de dias por mês"):
            counts = df_q4["Mes"].value_counts().reindex(range(1, 13)).fillna(0).astype(int)
            st.dataframe(counts.to_frame("dias"))