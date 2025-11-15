import os
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

# ---------------------------------------------
# CONFIGURATION GÉNÉRALE
# ---------------------------------------------
DATA_PATH = "graphcast_maroc_all.nc"

st.set_page_config(
    page_title="🌍 GraphCast Maroc — Visualisation",
    page_icon="🌤️",
    layout="wide"
)

# ---------------------------------------------
# PALETTE TURBO SMOOTH
# (dégradé plus doux que Turbo classique)
# ---------------------------------------------
TURBO_SMOOTH = [
    "#30123b", "#3a309d", "#4145eb", "#2b83ff", "#58d9ff",
    "#90ffdd", "#c2ff72", "#ffe400", "#ffb200", "#ff5d00", "#c40000"
]

# ---------------------------------------------
# NOMS COMPLETS DES VARIABLES
# ---------------------------------------------
VARIABLE_NAMES = {
    "u10": "Vent zonal à 10 m",
    "v10": "Vent méridional à 10 m",
    "gh": "Hauteur géopotentielle",
    "prmsl": "Pression au niveau de la mer",
    "q": "Humidité spécifique",
    "t": "Température",
    "u": "Vent zonal (pression)",
    "v": "Vent méridional (pression)",
    "w": "Vitesse verticale"
}

# ---------------------------------------------
# CHARGEMENT DU DATASET
# ---------------------------------------------
@st.cache_resource
def load_dataset(path):
    try:
        ds = xr.open_dataset(path)

        # Recadrage Maroc
        ds = ds.sel(
            longitude=slice(-20, 10),
            latitude=slice(40, 15)
        )

        return ds
    except Exception as e:
        st.error(f"Erreur dataset : {e}")
        return None

dataset = load_dataset(DATA_PATH)
if dataset is None:
    st.stop()

# ---------------------------------------------
# INITIALISATION SESSION STATE
# ---------------------------------------------
def init_state():
    if "time_index" not in st.session_state:
        st.session_state.time_index = 0

    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False

    if "animation_mode" not in st.session_state:
        st.session_state.animation_mode = "none"

    if "animation_speed" not in st.session_state:
        st.session_state.animation_speed = 0.3

    if "disable_sidebar" not in st.session_state:
        st.session_state.disable_sidebar = False

init_state()

# ---------------------------------------------
# ANIMATION
# ---------------------------------------------
def run_animation(max_time):
    st.session_state.time_index += 1

    # Boucle continue
    if st.session_state.animation_mode == "continue":
        if st.session_state.time_index > max_time:
            st.session_state.time_index = 0

    # Mode simple
    if st.session_state.animation_mode == "simple":
        if st.session_state.time_index > max_time:
            st.session_state.time_index = max_time
            st.session_state.is_playing = False
            st.session_state.disable_sidebar = False
            return

    time.sleep(st.session_state.animation_speed)
    st.rerun()

# ---------------------------------------------
# SIDEBAR
# ---------------------------------------------
if st.session_state.disable_sidebar:
    st.sidebar.markdown(
        "<style>.sidebar-content * {pointer-events:none; opacity:0.4;}</style>",
        unsafe_allow_html=True
    )
    st.sidebar.info("⏳ Animation en cours… paramètres verrouillés")

st.sidebar.header("⚙️ Paramètres")

# Liste des vraies variables (sans unknown)
VALID_VARS = [v for v in dataset.data_vars if v in VARIABLE_NAMES]

selected_var = st.sidebar.selectbox(
    "Variable",
    VALID_VARS,
    format_func=lambda k: VARIABLE_NAMES[k]
)

var_full_name = VARIABLE_NAMES[selected_var]

# Gestion valid_time
time_dim = "valid_time"
max_time = dataset.dims[time_dim] - 1

st.session_state.time_index = st.sidebar.slider(
    "Temps",
    0, max_time,
    st.session_state.time_index
)

# Variables pression vs surface
var_dims = dataset[selected_var].dims
is_pressure_var = "isobaricInhPa" in var_dims

pressure_level = None
if is_pressure_var:
    levels = dataset["isobaricInhPa"].values
    pressure_level = st.sidebar.selectbox("Niveau de pression (hPa)", levels)

# Animation settings
st.sidebar.subheader("🎞️ Animation temporelle")

mode = st.sidebar.selectbox(
    "Mode",
    ["none", "simple", "continue"],
    format_func=lambda x: {"none":"Aucune","simple":"Une fois","continue":"Continue"}[x]
)

st.session_state.animation_mode = mode

st.session_state.animation_speed = st.sidebar.slider(
    "Vitesse (s/frame)",
    0.05, 2.0,
    st.session_state.animation_speed
)

if mode != "none":
    if not st.session_state.is_playing:
        if st.sidebar.button("▶️ Play"):
            st.session_state.is_playing = True
            st.session_state.disable_sidebar = True
            st.rerun()
    else:
        if st.sidebar.button("⏸️ Pause"):
            st.session_state.is_playing = False
            st.session_state.disable_sidebar = False

# ---------------------------------------------
# COLORBAR HTML — TURBO SMOOTH (fine)
# ---------------------------------------------
def colorbar_html(min_val, max_val, title):
    gradient = "linear-gradient(to right, " + ",".join(TURBO_SMOOTH) + ")"
    return f"""
        <div style="
            position: absolute;
            bottom: 20px;
            right: 30px;
            padding: 6px;
            background: rgba(255,255,255,0.8);
            border-radius: 8px;
            font-size: 13px;
            z-index: 9999;
        ">
            <b>{title}</b><br>
            <div style="width: 220px; height: 8px; background: {gradient}; border-radius: 4px;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 12px;">
                <span>{min_val:.2f}</span>
                <span>{max_val:.2f}</span>
            </div>
        </div>
    """
# ============================================================
# ========================   BLOC 2   =========================
#      CARTE PYDECK + STATISTIQUES + GRAPHIQUES DE BASE
# ============================================================

# ---------------------------------------------
# EXTRACTION DU DATAFRAME POUR LA VISUALISATION
# ---------------------------------------------
@st.cache_data
def build_dataframe(_dataset, var_name, t_index, p_level):
    try:
        var = _dataset[var_name]

        # Sélection temporelle
        var = var.isel(valid_time=t_index)

        # Sélection pression si applicable
        if "isobaricInhPa" in var.dims and p_level is not None:
            var = var.sel(isobaricInhPa=p_level)

        # Transformation en DataFrame
        stacked = var.stack(points=("latitude", "longitude")).reset_index("points")

        df = pd.DataFrame({
            "lat": stacked["latitude"].values,
            "lon": stacked["longitude"].values,
            "value": stacked.values
        })

        # Nettoyage
        df = df.dropna(subset=["value"])
        df = df[
            df["value"].between(
                df["value"].quantile(0.01),
                df["value"].quantile(0.99)
            )
        ]

        return df
    except Exception as e:
        st.error(f"Erreur dataframe : {e}")
        return pd.DataFrame()


df = build_dataframe(dataset, selected_var, st.session_state.time_index, pressure_level)


# ---------------------------------------------
# CALCUL DES STATISTIQUES POUR LA LÉGENDE FIXE
# ---------------------------------------------
global_min = float(df["value"].min())
global_max = float(df["value"].max())

# Titre de la légende : nom complet + pression si applicable
legend_title = var_full_name
if pressure_level is not None:
    legend_title += f" – {pressure_level} hPa"

# Injection de la légende dans la page
st.markdown(
    colorbar_html(global_min, global_max, legend_title),
    unsafe_allow_html=True
)


# ---------------------------------------------
# CRÉATION DE LA CARTE
# ---------------------------------------------
st.markdown("## 🗺️ Carte Interactive")

def create_heatmap(df, radius=40, opacity=0.8):
    return pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position=["lon", "lat"],
        get_weight="value",
        radius_pixels=radius,
        opacity=opacity
    )

def create_scatter(df):
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_radius=500,
        get_fill_color="[255 * value / value, 120, 80, 200]",
        pickable=True
    )

def create_hex(df, radius=40):
    return pdk.Layer(
        "HexagonLayer",
        data=df,
        get_position=["lon", "lat"],
        radius=radius,
        elevation_scale=50,
        extruded=True,
        pickable=True
    )

# Type de visualisation
vis_type = st.sidebar.radio(
    "Type de visualisation",
    ["Heatmap", "Points", "Hexagones"]
)

# Sélection du layer
if vis_type == "Heatmap":
    layer = create_heatmap(df)
elif vis_type == "Points":
    layer = create_scatter(df)
else:
    layer = create_hex(df)

# Vue centrée sur le Maroc
mid_lat = df["lat"].mean()
mid_lon = df["lon"].mean()

view_state = pdk.ViewState(
    latitude=float(mid_lat),
    longitude=float(mid_lon),
    zoom=5,
    pitch=40 if vis_type == "Hexagones" else 0
)

# Carte PyDeck
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style="light"
)

st.pydeck_chart(deck, use_container_width=True)


# ---------------------------------------------
# MÉTRIQUES PRINCIPALES
# ---------------------------------------------
st.markdown("## 📌 Statistiques rapides")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Min", f"{df['value'].min():.2f}")
col2.metric("Max", f"{df['value'].max():.2f}")
col3.metric("Moyenne", f"{df['value'].mean():.2f}")
col4.metric("Écart-type", f"{df['value'].std():.2f}")


# ============================================================
# =====================  ONGLET STATISTIQUES  =================
# ============================================================

tab1, tab2, tab3 = st.tabs(["Visualisation", "Statistiques", "Analyses avancées"])

with tab2:

    st.markdown("### 📊 Histogramme")
    fig_hist = px.histogram(
        df,
        x="value",
        nbins=50,
        title=f"Distribution — {var_full_name}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### 📦 Box-Plot")
    fig_box = px.box(
        df,
        y="value",
        title=f"Box Plot — {var_full_name}"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### ⏱️ Série temporelle locale")

    # Sélection des coordonnées
    col_lat, col_lon = st.columns(2)
    lat_sel = col_lat.number_input("Latitude", float(df.lat.min()), float(df.lat.max()), float(mid_lat))
    lon_sel = col_lon.number_input("Longitude", float(df.lon.min()), float(df.lon.max()), float(mid_lon))

    # Extraction de la série temporelle
    def get_time_series(ds, var, lat, lon, p):
        try:
            da = ds[var]

            if p is not None and "isobaricInhPa" in da.dims:
                da = da.sel(isobaricInhPa=p)

            da = da.sel(latitude=lat, longitude=lon, method="nearest")

            times = ds["valid_time"].values
            return pd.DataFrame({
                "time": pd.to_datetime(times),
                "value": da.values
            })

        except Exception:
            return pd.DataFrame()

    ts = get_time_series(dataset, selected_var, lat_sel, lon_sel, pressure_level)

    if not ts.empty:
        fig_ts = px.line(ts, x="time", y="value", title="Évolution temporelle")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Sélectionnez un point valide dans la zone cartographiée.")

# ============================================================
# ===================   BLOC 3 — ANALYSES AVANCÉES   =========
# ============================================================

with tab3:

    st.markdown("## 📘 Analyses avancées")

    # ========================================================
    # 1) PROFIL VERTICAL (Skew-T simplifié)
    # ========================================================
    st.markdown("### 🌡️ Profil vertical (pression)")

    if is_pressure_var:
        colA, colB = st.columns(2)
        lat_v = colA.number_input("Latitude (profil)", float(df.lat.min()), float(df.lat.max()), float(mid_lat))
        lon_v = colB.number_input("Longitude (profil)", float(df.lon.min()), float(df.lon.max()), float(mid_lon))

        try:
            # Extraction du profil vertical
            da_vert = dataset[selected_var].sel(
                latitude=lat_v,
                longitude=lon_v,
                method="nearest"
            )

            values_vert = da_vert.isel(valid_time=st.session_state.time_index).values
            levels_vert = dataset["isobaricInhPa"].values

            df_vert = pd.DataFrame({
                "pressure": levels_vert,
                "value": values_vert
            }).sort_values("pressure", ascending=False)

            fig_vert = px.line(
                df_vert,
                x="value", y="pressure",
                title=f"Profil vertical — {var_full_name}",
                labels={"pressure": "Pression (hPa)", "value": "Valeur"}
            )

            fig_vert.update_yaxes(autorange="reversed")  # standard atmosphérique
            st.plotly_chart(fig_vert, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur profil vertical : {e}")

    else:
        st.info("Cette variable ne possède pas de niveaux de pression → pas de profil vertical.")


    # ========================================================
    # 2) HOVMÖLLER (temps vs latitude)
    # ========================================================
    st.markdown("### 🕒 Hovmöller (Temps vs Latitude)")

    try:
        # On prend un méridien central
        lon_hov = mid_lon

        da_hov = dataset[selected_var].sel(
            longitude=lon_hov,
            method="nearest"
        )

        if is_pressure_var and pressure_level is not None:
            da_hov = da_hov.sel(isobaricInhPa=pressure_level)

        hov_vals = da_hov.values  # dims: time, (pressure?), lat

        if is_pressure_var:
            hov_vals = hov_vals[:, :, :]  # (time, pressure, lat)
            hov_vals = hov_vals[:, 0, :]  # on prend le premier niveau (pression sélectionnée)

        times = pd.to_datetime(dataset["valid_time"].values)
        lats = dataset["latitude"].values

        hov_df = pd.DataFrame(hov_vals, columns=lats)
        hov_df["time"] = times
        hov_df = hov_df.melt(id_vars="time", var_name="latitude", value_name="value")

        fig_hov = px.imshow(
            hov_vals,
            x=lats,
            y=times,
            aspect="auto",
            color_continuous_scale="Turbo",
            title=f"Hovmöller — {var_full_name}"
        )
        fig_hov.update_layout(yaxis_title="Temps", xaxis_title="Latitude")

        st.plotly_chart(fig_hov, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur Hovmöller : {e}")


    # ========================================================
    # 3) CARTE DE CORRÉLATION
    # ========================================================
    st.markdown("### 🔗 Corrélation locale")

    colC, colD = st.columns(2)

    var_corr = colC.selectbox(
        "Variable avec laquelle corréler",
        VALID_VARS,
        format_func=lambda k: VARIABLE_NAMES[k]
    )

    # Calcul de la corrélation spatiale
    try:
        da1 = dataset[selected_var]
        da2 = dataset[var_corr]

        if "isobaricInhPa" in da1.dims and pressure_level is not None:
            da1 = da1.sel(isobaricInhPa=pressure_level)
        if "isobaricInhPa" in da2.dims and pressure_level is not None:
            da2 = da2.sel(isobaricInhPa=pressure_level)

        # Series temporelles au point
        corr_vals = []
        for la, lo in zip(df.lat, df.lon):
            ts1 = da1.sel(latitude=la, longitude=lo, method="nearest").values
            ts2 = da2.sel(latitude=la, longitude=lo, method="nearest").values

            if np.std(ts1) == 0 or np.std(ts2) == 0:
                corr_vals.append(np.nan)
            else:
                corr_vals.append(np.corrcoef(ts1, ts2)[0, 1])

        df_corr = pd.DataFrame({
            "lat": df.lat,
            "lon": df.lon,
            "corr": corr_vals
        }).dropna()

        # Carte de corrélation
        layer_corr = pdk.Layer(
            "ScatterplotLayer",
            data=df_corr,
            get_position=["lon", "lat"],
            get_radius=4000,
            get_fill_color="[255 * (corr+1)/2, 50, 255 * (1-(corr+1)/2), 200]",
            pickable=True
        )

        deck_corr = pdk.Deck(
            layers=[layer_corr],
            initial_view_state=view_state,
            map_style="light",
            tooltip={"html": "<b>Corrélation :</b> {corr}"}
        )

        st.pydeck_chart(deck_corr, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur corrélation : {e}")


