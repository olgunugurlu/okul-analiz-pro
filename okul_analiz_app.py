
import io
import re
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim, ArcGIS
from geopy.extra.rate_limiter import RateLimiter

# Opsiyonel: grafikler
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# =========================================================
# Okul Konum Analiz Pro - Birleştirilmiş "En İyi" Sürüm
# =========================================================
# if st.query_params.get("embed") != "true":
#     st.query_params["embed"] = "true"
#     st.rerun()

params = dict(st.query_params)
if params.get("embed") != "true":
    st.info("Daha temiz görünüm için uygulamayı Embed modda açabilirsiniz.")
    st.link_button("Embed modda aç", "?embed=true")

# -------------------- SAYFA AYARLARI --------------------
st.set_page_config(
    page_title="Okul Konum Analiz Pro (Best)",
    layout="wide",
    page_icon="🏫",
    initial_sidebar_state="expanded",
)

# -------------------- THEME / CSS --------------------
st.markdown(
    """
<style>
    /* Üst barı komple kapat */
    header[data-testid="stHeader"] { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }

    /* Sağ üst action butonları (senin paylaştığın div) */
    div[data-testid="stToolbarActionButton"] { display: none !important; }
    .stToolbarActionButton { display: none !important; }

    /* Bazı sürümlerde üstteki container farklı gelebiliyor */
    [data-testid="stHeaderActionElements"] { display: none !important; }

    .main .block-container { padding-top: 1.7rem; padding-bottom: 2rem; }
    h1 { font-weight: 800; }
    h2, h3 { margin-top: 1.2rem; }

    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        color: #111 !important;
        background-color: #f0f2f6 !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: 1px solid #1f77b4;
        border-radius: 10px;
        padding: 0.55rem;
        font-weight: 700;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #1565c0;
        border-color: #fff;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255,255,255,0.18);
    }

    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 800; }
    hr { margin: 1.5rem 0; border: none; height: 2px; background: linear-gradient(90deg, transparent, #1f77b4, transparent); }
</style>
""",
    unsafe_allow_html=True,
)


# -------------------- SABİTLER --------------------
TARGET_CITY = "Eskişehir"
TARGET_COUNTRY = "Türkiye"

# Rough bounding box for Eskişehir city center area (lon_min, lat_min, lon_max, lat_max)
ESKISEHIR_VIEWBOX = (29.40, 40.50, 31.50, 39.00)  # (west, north, east, south) lon/lat for Nominatim

DEPARTMENT_COLORS = {
    "Bilişim Teknolojileri": "#3498db",
    "Elektrik-Elektronik": "#2ecc71",
    "Makine": "#e74c3c",
    "Metal": "#e67e22",
    "Mobilya": "#9b59b6",
    "Tesisat": "#1abc9c",
    "Yapı": "#c0392b",
    "Enerji": "#27ae60",
    "Tasarım": "#f39c12",
    "Güzellik": "#e91e63",
    "Çocuk Gelişimi": "#8e44ad",
    "Grafik ve Fotoğraf": "#16a085",
    "Yiyecek ve İçecek Hizmetleri": "#d35400",
    "Moda Tasarım": "#c2185b",
    "Gıda Teknolojileri": "#2c3e50",
    "Kimya Teknolojileri": "#7f8c8d",
    "Diğer": "#95a5a6",
}
dept_colors = {}

ICON_COLOR_FOLIUM = {
    "curr": "black",
    "cand": "green",
    "ideal": "blue",
}

# -------------------- SESSION STATE --------------------
defaults = {
    "curr_name": "Mevcut Bina",
    "curr_lat": 39.765600,
    "curr_lon": 30.523800,
    "cand_name": "Aday Bina",
    "cand_lat": 39.754000,
    "cand_lon": 30.511500,
    "radius_slider": 3.0,
    "loaded_df": None,
    "last_updated": "",
    "analysis_mode": "🦅 Kuş Uçuşu",
    "road_factor": 1.4,
    "opt_outlier_radius": True,
    "opt_outlier_ideal": True,
    "show_ideal": True,
    "thread_geocode": True,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# -------------------- COORD VALIDATION / NORMALIZATION --------------------
def normalize_coords(lat: float, lon: float):
    """Attempt to fix common lat/lon swap mistakes for Turkey; return (lat, lon, note)."""
    note = ""
    try:
        lat_f, lon_f = float(lat), float(lon)
    except Exception:
        return lat, lon, "Koordinatlar sayısal değil."

    # Turkey rough bounds (very permissive)
    lat_ok = 34.0 <= lat_f <= 43.0
    lon_ok = 25.0 <= lon_f <= 46.0

    # If swapped (lat looks like lon and lon looks like lat), auto-swap.
    swapped_lat_ok = 34.0 <= lon_f <= 43.0
    swapped_lon_ok = 25.0 <= lat_f <= 46.0

    if (not lat_ok or not lon_ok) and swapped_lat_ok and swapped_lon_ok:
        return lon_f, lat_f, "⚠️ Lat/Lon ters girilmiş gibi görünüyor; otomatik düzeltildi."

    if not lat_ok or not lon_ok:
        note = "⚠️ Koordinatlar Türkiye aralığının dışında görünüyor (lat ~34–43, lon ~25–46)."
    return lat_f, lon_f, note


def render_map(map_obj: folium.Map, height: int = 650):
    """Responsive folium rendering (fills container width)."""
    html = map_obj.get_root().render()
    components.html(html, height=height, width=None, scrolling=False)


# -------------------- COORD NORMALIZATION --------------------
def _to_float(x, default=0.0):
    """Convert user input to float safely (handles '39,765' commas)."""
    try:
        if isinstance(x, str):
            x = x.strip().replace(",", ".")
        return float(x)
    except Exception:
        return float(default)

def normalize_latlon(lat, lon, country_hint="TR"):
    """
    Normalize (lat, lon). If user accidentally swaps, detect and swap.
    Returns: (lat_n, lon_n, swapped:bool, note:str)
    """
    lat_f = _to_float(lat, np.nan)
    lon_f = _to_float(lon, np.nan)
    swapped = False
    note = ""

    if np.isnan(lat_f) or np.isnan(lon_f):
        return lat_f, lon_f, swapped, note

    # Basic sanity: latitude must be [-90, 90], longitude [-180, 180]
    if abs(lat_f) > 90 and abs(lon_f) <= 90:
        lat_f, lon_f = lon_f, lat_f
        swapped = True
        note = "Lat/Lon sınırı nedeniyle otomatik düzeltildi."

    # Turkey-specific heuristic: TR lat ~ [35, 43], lon ~ [25, 46]
    if country_hint == "TR":
        tr_lat_ok = 35 <= lat_f <= 43
        tr_lon_ok = 25 <= lon_f <= 46
        tr_lat_ok_swapped = 35 <= lon_f <= 43
        tr_lon_ok_swapped = 25 <= lat_f <= 46
        # If current looks wrong but swapped looks right, swap
        if (not (tr_lat_ok and tr_lon_ok)) and (tr_lat_ok_swapped and tr_lon_ok_swapped):
            lat_f, lon_f = lon_f, lat_f
            swapped = True
            note = "Türkiye aralığına göre Lat/Lon ters girilmiş göründü, otomatik düzeltildi."
        # If both plausible (rare), we keep as-is.

    return float(lat_f), float(lon_f), swapped, note

# Initial normalized defaults for sidebar number_inputs
curr_lat_n, curr_lon_n, curr_swapped, curr_note = normalize_latlon(
    st.session_state.get("curr_lat"), st.session_state.get("curr_lon"), country_hint="TR"
)
cand_lat_n, cand_lon_n, cand_swapped, cand_note = normalize_latlon(
    st.session_state.get("cand_lat"), st.session_state.get("cand_lon"), country_hint="TR"
)

# -------------------- GEOCODER --------------------
@st.cache_resource
def get_geocoder_nominatim():
    geolocator = Nominatim(user_agent="okul_konum_analiz_pro_best", timeout=10)
    return RateLimiter(geolocator.geocode, min_delay_seconds=1.1, swallow_exceptions=True)

@st.cache_resource
def get_geocoder_arcgis():
    # ArcGIS (Esri) geocoder: genelde POI/kurum isimlerinde Nominatim'den daha isabetli olabilir.
    geolocator = ArcGIS(timeout=10)
    # Çok agresif rate-limit yapmayalım; yine de servisleri yormamak için küçük gecikme:
    return RateLimiter(geolocator.geocode, min_delay_seconds=0.25, swallow_exceptions=True)

_GEOCODE_NOMINATIM = get_geocoder_nominatim()
_GEOCODE_ARCGIS = get_geocoder_arcgis()


def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    s = str(text).lower()
    for word in ["mahallesi", "mah.", "mah", "(köy)", "köyü", "koyu"]:
        s = s.replace(word, "")
    return s.strip().title()

@st.cache_data(show_spinner=False, ttl=3600)
def geocode_address_cached(address: str):
    if not address or not str(address).strip():
        return None
    try:
        loc = _GEOCODE_NOMINATIM(address)
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception:
        pass
    return None

def build_full_address(addr_text: str) -> str:
    addr_text = (addr_text or "").strip()
    if not addr_text:
        return ""
    return f"{addr_text}, {TARGET_CITY}, {TARGET_COUNTRY}"

def build_mahalle_ilce_address(mahalle: str, ilce: str) -> str:
    mahalle = (mahalle or "").strip()
    ilce = (ilce or "").strip()
    if mahalle and ilce:
        return f"{mahalle}, {ilce}, {TARGET_CITY}, {TARGET_COUNTRY}"
    if ilce:
        return f"{ilce}, {TARGET_CITY}, {TARGET_COUNTRY}"
    if mahalle:
        return f"{mahalle}, {TARGET_CITY}, {TARGET_COUNTRY}"
    return ""


# -------------------- GEOCODE: SEARCH CANDIDATES (FOR ADDRESS FINDER) --------------------
@st.cache_data(show_spinner=False, ttl=3600)
def geocode_candidates_cached(query: str, limit: int = 5, provider: str = "ArcGIS"):
    """Return multiple geocoding candidates for Eskişehir.

    provider:
      - "ArcGIS" (varsayılan): kurum/POI isimlerinde daha isabetli olabilir
      - "Nominatim": OSM tabanlı
    Not: Viewbox kısıtı Nominatim'de doğrudan kullanılır; ArcGIS'te sonuçlar sonradan bbox ile filtrelenir.
    """
    q = (query or "").strip()
    if not q:
        return []

    variants = [
        q,
        f"{q}, {TARGET_CITY}",
        f"{q}, {TARGET_CITY}, {TARGET_COUNTRY}",
    ]

    results = []
    seen = set()

    # Eskişehir kaba bbox filtresi (lon_min, lat_max, lon_max, lat_min) -> burada lat/lon kontrolü yapacağız
    lon_min, lat_max, lon_max, lat_min = ESKISEHIR_VIEWBOX

    if provider == "Nominatim":
        geolocator = Nominatim(user_agent="okul_konum_analiz_pro_best_search", timeout=10)
        for v in variants:
            try:
                locs = geolocator.geocode(
                    v,
                    exactly_one=False,
                    limit=max(10, limit),
                    country_codes="tr",
                    viewbox=ESKISEHIR_VIEWBOX,
                    bounded=True,
                )
            except Exception:
                locs = None
            if not locs:
                continue
            for loc in locs:
                try:
                    lat = float(loc.latitude)
                    lon = float(loc.longitude)
                except Exception:
                    continue
                key = (round(lat, 6), round(lon, 6))
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "label": getattr(loc, "address", None) or getattr(loc, "raw", {}).get("display_name", "Sonuç"),
                        "lat": lat,
                        "lon": lon,
                    }
                )
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break
        return results[:limit]

    # ArcGIS provider
    for v in variants:
        try:
            locs = _GEOCODE_ARCGIS(v, exactly_one=False)
        except Exception:
            locs = None
        if not locs:
            continue
        # geopy ArcGIS exactly_one=False => list
        for loc in (locs if isinstance(locs, (list, tuple)) else [locs]):
            try:
                lat = float(loc.latitude)
                lon = float(loc.longitude)
            except Exception:
                continue

            # bbox filtresi
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                continue

            key = (round(lat, 6), round(lon, 6))
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "label": getattr(loc, "address", None) or getattr(loc, "raw", {}).get("display_name", "Sonuç"),
                    "lat": lat,
                    "lon": lon,
                }
            )
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    return results[:limit]


# -------------------- GOOGLE DRIVE / SHEETS LOADER --------------------
def _extract_google_sheet_id(url: str):
    if "/d/" not in url:
        return None
    return url.split("/d/")[1].split("/")[0]

def _extract_gid(url: str) -> str:
    m = re.search(r"gid=(\d+)", url or "")
    return m.group(1) if m else "0"

def _extract_drive_file_id(url: str):
    if not url:
        return None
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None

@st.cache_data(ttl=120, show_spinner=False)
def load_data_from_google_link(url: str) -> pd.DataFrame:
    if not url or not url.strip():
        raise ValueError("Link boş olamaz.")
    url = url.strip()

    if "docs.google.com/spreadsheets" in url:
        sheet_id = _extract_google_sheet_id(url)
        gid = _extract_gid(url)
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        return pd.read_csv(csv_url)

    if "drive.google.com" in url:
        file_id = _extract_drive_file_id(url)
        if not file_id:
            raise ValueError("Drive linkinden file_id çıkarılamadı.")
        dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(dl_url, timeout=30)
        r.raise_for_status()
        content = r.content
        if content[:2] == b"PK":
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
        return pd.read_csv(io.BytesIO(content))

    raise ValueError("Geçersiz link: Google Sheets veya Google Drive linki olmalı.")

# -------------------- VERİ HAZIRLAMA / GEOCODE --------------------
def _split_ilce_mahalle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "İkamet Ettiğiniz İlçe ve Mahalle" in df.columns:
        split_data = df["İkamet Ettiğiniz İlçe ve Mahalle"].astype(str).str.split(" - ", n=1, expand=True)
        if split_data.shape[1] >= 2:
            df["Ilce_Temiz"] = split_data[0].apply(clean_text)
            df["Mahalle_Temiz"] = split_data[1].apply(clean_text)
        else:
            df["Ilce_Temiz"] = ""
            df["Mahalle_Temiz"] = ""
    else:
        df["Ilce_Temiz"] = df["İkamet Ettiğiniz İlçe"].apply(clean_text) if "İkamet Ettiğiniz İlçe" in df.columns else ""
        df["Mahalle_Temiz"] = df["Mahalle Adı"].apply(clean_text) if "Mahalle Adı" in df.columns else ""

    return df

def geocode_single_address(mahalle: str, ilce: str):
    addr1 = build_mahalle_ilce_address(mahalle, ilce)
    coords = geocode_address_cached(addr1)
    if coords is None and ilce:
        addr2 = build_mahalle_ilce_address("", ilce)
        coords = geocode_address_cached(addr2)
    return (mahalle, ilce, coords)

def process_data_unique_geocode(df: pd.DataFrame, use_threads: bool = True) -> pd.DataFrame:
    df = _split_ilce_mahalle(df)

    uniq = (
        df[["Mahalle_Temiz", "Ilce_Temiz"]]
        .fillna("")
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    lat_map, lon_map = {}, {}
    prog = st.progress(0.0, text="Adresler çözülüyor (unique geocode)...")
    total = max(len(uniq), 1)

    if use_threads:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [
                ex.submit(geocode_single_address, r["Mahalle_Temiz"], r["Ilce_Temiz"])
                for _, r in uniq.iterrows()
            ]
            for i, fut in enumerate(futures):
                mah, ilce, coords = fut.result()
                key = (mah, ilce)
                if coords:
                    lat_map[key], lon_map[key] = coords
                else:
                    lat_map[key], lon_map[key] = np.nan, np.nan
                prog.progress((i + 1) / total)
    else:
        for i, r in uniq.iterrows():
            mah, ilce, coords = geocode_single_address(r["Mahalle_Temiz"], r["Ilce_Temiz"])
            key = (mah, ilce)
            if coords:
                lat_map[key], lon_map[key] = coords
            else:
                lat_map[key], lon_map[key] = np.nan, np.nan
            prog.progress((i + 1) / total)

    prog.empty()

    keys = list(zip(df["Mahalle_Temiz"].fillna("").astype(str), df["Ilce_Temiz"].fillna("").astype(str)))
    df["Enlem"] = [lat_map.get(k, np.nan) for k in keys]
    df["Boylam"] = [lon_map.get(k, np.nan) for k in keys]
    return df

# -------------------- MESAFE HESABI --------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))

def calculate_distances(df: pd.DataFrame, curr_lat, curr_lon, cand_lat, cand_lon, mode: str, road_factor: float = 1.4):
    df = df.copy()
    lat = df["Enlem"].to_numpy()
    lon = df["Boylam"].to_numpy()

    base_curr = haversine_km(curr_lat, curr_lon, lat, lon)
    base_cand = haversine_km(cand_lat, cand_lon, lat, lon)

    if mode == "🦅 Kuş Uçuşu":
        df["Mesafe_Mevcut"] = base_curr
        df["Mesafe_Aday"] = base_cand
        return df

    if mode == "🚗 Tahmini Karayolu":
        df["Mesafe_Mevcut"] = base_curr * road_factor
        df["Mesafe_Aday"] = base_cand * road_factor
        return df

    if mode == "🏙️ Manhattan":
        lat_km = np.abs(lat - curr_lat) * 111.32
        lon_km = np.abs(lon - curr_lon) * (111.32 * np.cos(np.radians((lat + curr_lat) / 2)))
        df["Mesafe_Mevcut"] = lat_km + lon_km

        lat_km2 = np.abs(lat - cand_lat) * 111.32
        lon_km2 = np.abs(lon - cand_lon) * (111.32 * np.cos(np.radians((lat + cand_lat) / 2)))
        df["Mesafe_Aday"] = lat_km2 + lon_km2
        return df

    df["Mesafe_Mevcut"] = base_curr
    df["Mesafe_Aday"] = base_cand
    return df

def calculate_smart_radius_mean(df: pd.DataFrame, remove_outliers: bool = True) -> float:
    if "Mesafe_Mevcut" not in df.columns or df.empty:
        return 3.0
    d = df["Mesafe_Mevcut"].dropna()
    if d.empty:
        return 3.0
    if remove_outliers:
        q1, q3 = d.quantile(0.25), d.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        d = d[(d >= lower) & (d <= upper)]
    val = float(d.mean()) if not d.empty else 3.0
    return round(max(0.5, min(val, 20.0)), 2)

def calculate_center_of_gravity(df: pd.DataFrame, remove_outliers: bool = True):
    d = df.dropna(subset=["Enlem", "Boylam"]).copy()
    if d.empty:
        return None

    if not remove_outliers:
        return (float(d["Enlem"].mean()), float(d["Boylam"].mean()))

    lq1, lq3 = d["Enlem"].quantile(0.25), d["Enlem"].quantile(0.75)
    bq1, bq3 = d["Boylam"].quantile(0.25), d["Boylam"].quantile(0.75)
    d = d[
        (d["Enlem"] >= lq1 - 1.5 * (lq3 - lq1)) & (d["Enlem"] <= lq3 + 1.5 * (lq3 - lq1)) &
        (d["Boylam"] >= bq1 - 1.5 * (bq3 - bq1)) & (d["Boylam"] <= bq3 + 1.5 * (bq3 - bq1))
    ]
    if d.empty:
        return None
    return (float(d["Enlem"].mean()), float(d["Boylam"].mean()))

# -------------------- LEGEND --------------------
def add_custom_legend(map_obj, colors: dict, curr_name: str, cand_name: str, ideal_shown: bool, ideal_note: str):
    """Sağ üstte okunaklı, yarı saydam legend. Metin siyah, renk sadece nokta."""
    legend_html = f"""
    <div style="
        position: absolute;
        top: 20px; right: 20px;
        width: 300px;
        background: rgba(255,255,255,0.92);
        color: #111;
        padding: 14px 14px 10px 14px;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.25);
        box-shadow: 0 8px 18px rgba(0,0,0,0.35);
        z-index: 999999;
        font-size: 13px;
        max-height: 520px;
        overflow-y: auto;
        line-height: 1.35;
    ">
        <div style="font-weight: 800; font-size: 15px; margin-bottom: 6px;">
            📊 Gösterge Paneli
        </div>
        <hr style="margin:8px 0; border: none; border-top: 1px solid rgba(0,0,0,0.18);">
        <div style="margin-bottom:8px;"><b>🏫 Okullar</b></div>
        <div style="margin-bottom:4px;"><span style="color:#111;">●</span> {curr_name}</div>
        <div style="margin-bottom:4px;"><span style="color:#2ecc71;">●</span> {cand_name}</div>
        {f'<div style="margin-bottom:4px;"><span style="color:#f1c40f;">●</span> İdeal ({ideal_note})</div>' if ideal_shown else ''}
        <div style="margin:10px 0 6px;"><b>👥 Bölümler</b></div>
    """
    for dept, color in colors.items():
        legend_html += f'<div style="margin-bottom:4px;"><span style="color:{color}; font-weight:900;">●</span> <span style="color:#111;">{dept}</span></div>'
    legend_html += "</div>"
    map_obj.get_root().html.add_child(folium.Element(legend_html))

# -------------------- TABLO --------------------
def create_comparison_table(df: pd.DataFrame, category_col: str, radius: float) -> pd.DataFrame:
    if category_col not in df.columns:
        return pd.DataFrame()

    count_col = "Zaman damgası" if "Zaman damgası" in df.columns else df.columns[0]

    grp = df.groupby(category_col).agg(
        Grup_Mevcudu=(count_col, "count"),
        Mevcut_Erisen=("Mesafe_Mevcut", lambda x: (x <= radius).sum()),
        Aday_Erisen=("Mesafe_Aday", lambda x: (x <= radius).sum()),
    ).reset_index()

    total_row = pd.DataFrame({
        category_col: ["GENEL TOPLAM"],
        "Grup_Mevcudu": [grp["Grup_Mevcudu"].sum()],
        "Mevcut_Erisen": [grp["Mevcut_Erisen"].sum()],
        "Aday_Erisen": [grp["Aday_Erisen"].sum()],
    })
    fin = pd.concat([grp, total_row], ignore_index=True)

    fin["Mevcut_Oran_%"] = (fin["Mevcut_Erisen"] / fin["Grup_Mevcudu"] * 100).fillna(0).round(1)
    fin["Aday_Oran_%"] = (fin["Aday_Erisen"] / fin["Grup_Mevcudu"] * 100).fillna(0).round(1)
    fin["Fark"] = fin["Aday_Erisen"] - fin["Mevcut_Erisen"]

    def _durum(x):
        if x > 0:
            return f"🟢 +{x}"
        if x < 0:
            return f"🔴 {x}"
        return "⚪ 0"

    fin["Durum"] = fin["Fark"].apply(_durum)
    return fin[[category_col, "Grup_Mevcudu", "Mevcut_Erisen", "Mevcut_Oran_%", "Aday_Erisen", "Aday_Oran_%", "Fark", "Durum"]]

# =========================================================
# UI
# =========================================================
st.title("🏫 Okul Konum Analiz Pro")

with st.sidebar:
    st.markdown("### 🔍 Adres Bulucu")
    

    with st.expander("📍 Koordinat Bul", expanded=False):
        st.caption("Adres aramasını adım adım doldurun. Şehir sabit: **Eskişehir**.")
        provider = st.selectbox("Konum bulma servisi", ["ArcGIS", "Nominatim"], index=0)
        st.caption("İpucu: Okul adı + mahalle + ilçe yazmak isabeti artırır.")
        school_q = st.text_input("1) Okul adı / Yer adı", placeholder="Örn: Gazi MTAL")
        mahalle_q = st.text_input("2) Mahalle (opsiyonel)", placeholder="Örn: Akarbaşı")
        ilce_q = st.text_input("3) İlçe (opsiyonel)", placeholder="Örn: Odunpazarı / Tepebaşı")
        st.text_input("4) Şehir", value=TARGET_CITY, disabled=True)

        def _compose_query(school, mahalle, ilce):
            parts = [school, mahalle, ilce, TARGET_CITY, TARGET_COUNTRY]
            parts = [p.strip() for p in parts if str(p or '').strip()]
            return ", ".join(parts)

        query = _compose_query(school_q, mahalle_q, ilce_q)

        colA, colB = st.columns([1,1])
        if colA.button("Ara", width="stretch", disabled=not (school_q or "").strip()):
            with st.spinner("Koordinat aranıyor (Eskişehir içinde)..."):
                candidates = geocode_candidates_cached(query, limit=6, provider=provider)

                # Fallback: daha gevşek arama (mahalle/ilçe boş bırakılmışsa)
                if not candidates and (ilce_q or mahalle_q):
                    candidates = geocode_candidates_cached(_compose_query(school_q, "", "",), limit=6, provider=provider)

                # Fallback: eski davranış (en son çare)
                if not candidates:
                    loc = _GEOCODE_NOMINATIM(build_full_address(school_q))
                    if loc:
                        candidates = [{
                            "label": getattr(loc, "address", None) or getattr(loc, "raw", {}).get("display_name", "Sonuç"),
                            "lat": float(loc.latitude),
                            "lon": float(loc.longitude),
                        }]

                st.session_state["found_candidates"] = candidates or []
                st.session_state["found_name"] = (school_q or query).strip() or "Konum"
                if candidates:
                    st.success(f"{len(candidates)} sonuç bulundu. Aşağıdan seçin.")
                else:
                    st.error("Sonuç bulunamadı. Okul adı + ilçe/mahalle ekleyerek tekrar deneyin.")

        # Sonuç seçimi
        candidates = st.session_state.get("found_candidates", [])
        if candidates:
            labels = [c["label"] for c in candidates]
            sel = st.selectbox("Sonuç Seç", labels, index=0)
            chosen = next((c for c in candidates if c["label"] == sel), candidates[0])

            st.session_state["found_lat"] = float(chosen["lat"])
            st.session_state["found_lon"] = float(chosen["lon"])

            # Kısa isim: okul + (ilçe/mahalle) varsa ekle
            nm_parts = [school_q.strip()] if (school_q or "").strip() else ["Bulunan Konum"]
            if (ilce_q or "").strip() or (mahalle_q or "").strip():
                nm_parts.append(f"{(ilce_q or '').strip()} {(mahalle_q or '').strip()}".strip())
            st.session_state["found_name"] = " - ".join([p for p in nm_parts if p])

            st.code(f"{st.session_state['found_lat']:.6f}, {st.session_state['found_lon']:.6f}")

            # Mini doğrulama haritası
            try:
                _mini = folium.Map(
                    location=[st.session_state["found_lat"], st.session_state["found_lon"]],
                    zoom_start=15,
                    tiles="CartoDB positron",
                )
                folium.Marker(
                    [float(st.session_state["found_lat"]), float(st.session_state["found_lon"])],
                    tooltip="Seçilen konum",
                    icon=folium.Icon(color="blue", icon="map-marker"),
                ).add_to(_mini)
                render_map(_mini, height=260)
            except Exception:
                pass

            c1, c2 = st.columns(2)
            if c1.button("Mevcut'a aktar", width="stretch"):
                st.session_state.update({
                    "curr_lat": float(st.session_state["found_lat"]),
                    "curr_lon": float(st.session_state["found_lon"]),
                    "curr_name": st.session_state.get("found_name", "Mevcut Bina"),
                })
                st.rerun()
            if c2.button("Aday'a aktar", width="stretch"):
                st.session_state.update({
                    "cand_lat": float(st.session_state["found_lat"]),
                    "cand_lon": float(st.session_state["found_lon"]),
                    "cand_name": st.session_state.get("found_name", "Aday Bina"),
                })
                st.rerun()
    st.divider()

    st.markdown("### 📂 Veri Kaynağı")
    src = st.radio("Tip", ["Google Link", "Dosya Yükle"], label_visibility="collapsed")

    if src == "Dosya Yükle":
        up = st.file_uploader("Excel/CSV seç", type=["xlsx", "xls", "csv"])
        if up:
            try:
                if up.name.lower().endswith(".csv"):
                    st.session_state["loaded_df"] = pd.read_csv(up)
                else:
                    st.session_state["loaded_df"] = pd.read_excel(up)
                st.session_state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("Yüklendi")
            except Exception as e:
                st.error(str(e))
    else:
        lnk = st.text_input("Google Sheets/Drive Link")
        c1, c2 = st.columns(2)
        if c1.button("Çek", disabled=not (lnk or "").strip(), width="stretch"):
            try:
                st.session_state["loaded_df"] = load_data_from_google_link(lnk)
                st.session_state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("Çekildi")
            except Exception as e:
                st.error(str(e))
        if c2.button("Yenile", disabled=not (lnk or "").strip(), width="stretch"):
            st.cache_data.clear()
            st.rerun()

    if st.session_state["last_updated"]:
        st.caption(f"Son güncelleme: {st.session_state['last_updated']}")

    st.divider()
    st.markdown("### ⚙️ Geocode / Performans")
    st.session_state["thread_geocode"] = st.checkbox("Hızlı (Thread) Geocode", value=st.session_state["thread_geocode"])
    st.caption("Not: Nominatim rate-limit nedeniyle aşırı hız artmaz; ama UI daha akıcı olur.")

    st.divider()
    with st.expander("📍 Okul Koordinatları", expanded=True):

        st.session_state["curr_name"] = st.text_input("Mevcut Ad", st.session_state["curr_name"])
        st.session_state["curr_lat"] = st.number_input("Mevcut Lat", value=curr_lat_n, format="%.6f")
        st.session_state["curr_lon"] = st.number_input("Mevcut Lon", value=curr_lon_n, format="%.6f")
        st.markdown("---")
        st.session_state["cand_name"] = st.text_input("Aday Ad", st.session_state["cand_name"])
        st.session_state["cand_lat"] = st.number_input("Aday Lat", value=cand_lat_n, format="%.6f")
        st.session_state["cand_lon"] = st.number_input("Aday Lon", value=cand_lon_n, format="%.6f")
    
    st.markdown("---")

    st.markdown(
        """
        <div style="
            text-align: center;
            font-size: 13px;
            color: #dddddd;
            line-height: 1.6;
            padding-top: 10px;
        ">
            <b>İletişim</b><br>
            📧 <a href="mailto:olgunugurlu@gmail.com" style="color:#4da3ff;">
                olgunugurlu@gmail.com
            </a><br>
            🏫 Gazi Mesleki ve Teknik Anadolu Lisesi
        </div>
        """,
        unsafe_allow_html=True
    )


# Re-normalize after user inputs (applies to all calculations & maps)
curr_lat_n, curr_lon_n, curr_swapped, curr_note = normalize_latlon(
    st.session_state.get("curr_lat"), st.session_state.get("curr_lon"), country_hint="TR"
)
cand_lat_n, cand_lon_n, cand_swapped, cand_note = normalize_latlon(
    st.session_state.get("cand_lat"), st.session_state.get("cand_lon"), country_hint="TR"
)

if curr_swapped or cand_swapped:
    notes = " ".join([n for n in [curr_note, cand_note] if n])
    st.warning(f"Koordinatlar otomatik olarak normalize edildi. {notes}")


# -------------------- DATA GUARD --------------------
if st.session_state["loaded_df"] is None:
    st.info("👈 Analize başlamak için sol menüden veri yükleyin veya Google link girin.")
    st.stop()

raw_df = st.session_state["loaded_df"]

# -------------------- VERİ İŞLEME --------------------
st.markdown("## 1) Veri işleme")
if "Enlem" not in raw_df.columns or "Boylam" not in raw_df.columns:
    with st.spinner("Adresler çözümleniyor (unique geocode)..."):
        full_df = process_data_unique_geocode(raw_df, use_threads=st.session_state["thread_geocode"])
else:
    full_df = _split_ilce_mahalle(raw_df.copy())

ok = int(full_df["Enlem"].notna().sum()) if "Enlem" in full_df.columns else 0
tot = int(len(full_df))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam", tot)
c2.metric("Geocode Başarılı", ok)
c3.metric("Geocode Başarısız", tot - ok)
c4.metric("Başarı Oranı", f"%{(ok / tot * 100) if tot else 0:.1f}")

if ok < tot:
    st.warning("Bulunamayan adresler (NaN) haritaya ve ısı haritasına dahil edilmez.")

# -------------------- LEGEND / RENKLER --------------------
# Legend'da sadece okulda (veride) geçen bölümler gösterilsin.
_present = []
if "Alan / Dal" in full_df.columns:
    _present = sorted({str(x).strip() for x in full_df["Alan / Dal"].dropna().tolist() if str(x).strip()})
# Renk eşle: tanımsızlar "Diğer" rengine düşer
dept_colors = {d: DEPARTMENT_COLORS.get(d, DEPARTMENT_COLORS["Diğer"]) for d in _present} if _present else {"Diğer": DEPARTMENT_COLORS["Diğer"]}


# -------------------- AYARLAR --------------------
st.markdown("## 2) Analiz ayarları")
c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    mode = st.selectbox("Uzaklık Modu", ["🦅 Kuş Uçuşu", "🚗 Tahmini Karayolu", "🏙️ Manhattan"])
with c2:
    rf = st.slider("Yol Katsayısı", 1.1, 2.5, float(st.session_state["road_factor"]), 0.05) if mode == "🚗 Tahmini Karayolu" else float(st.session_state["road_factor"])
with c3:
    col_out1, col_out2, col_out3 = st.columns(3)
    with col_out1:
        out_rad = st.checkbox("Akıllı Yarıçap (IQR)", value=st.session_state["opt_outlier_radius"])
    with col_out2:
        out_ideal = st.checkbox("İdeal (IQR)", value=st.session_state["opt_outlier_ideal"])
    with col_out3:
        show_ideal = st.checkbox("İdeal Göster", value=st.session_state["show_ideal"])

st.session_state["analysis_mode"] = mode
st.session_state["road_factor"] = rf
st.session_state["opt_outlier_radius"] = out_rad
st.session_state["opt_outlier_ideal"] = out_ideal
st.session_state["show_ideal"] = show_ideal

# Mesafeleri hesapla
full_df = calculate_distances(
    full_df,
    curr_lat_n,
    curr_lon_n,
    cand_lat_n,
    cand_lon_n,
    mode,
    road_factor=float(rf),
)

# İdeal nokta
ideal = calculate_center_of_gravity(full_df, remove_outliers=out_ideal)
ideal_note = "Ağırlık merkezi" if ideal else "—"

sug_rad = calculate_smart_radius_mean(full_df, remove_outliers=out_rad)
c_r1, c_r2 = st.columns([1, 4])
with c_r1:
    if st.button("🤖 Önerilen", width="stretch"):
        st.session_state["radius_slider"] = float(sug_rad)
        st.rerun()
with c_r2:
    radius = st.slider("Yarıçap (KM)", 0.5, 20.0, float(st.session_state["radius_slider"]), 0.25)

st.session_state["radius_slider"] = radius

# -------------------- FİLTRE --------------------
st.markdown("## 3) Filtre")
if "Alan / Dal" in full_df.columns:
    dept_list = ["TÜMÜ"] + sorted([x for x in full_df["Alan / Dal"].dropna().unique().tolist() if str(x).strip() != ""])
else:
    dept_list = ["TÜMÜ"]
sel_dept = st.selectbox("Bölüm Filtrele", dept_list)

df_display = full_df.copy() if sel_dept == "TÜMÜ" else full_df[full_df["Alan / Dal"] == sel_dept].copy()
df_map = df_display.dropna(subset=["Enlem", "Boylam"]).copy()

st.caption(f"Filtreli kayıt: **{len(df_display)}** | Haritada kullanılabilen: **{len(df_map)}**")

# -------------------- HARİTALAR --------------------

st.markdown("## 4) Haritalar")

st.caption(
    f"📍 Mevcut: {st.session_state['curr_name']} → {float(st.session_state['curr_lat']):.6f}, {float(st.session_state['curr_lon']):.6f} | "
    f"⭐ Aday: {st.session_state['cand_name']} → {float(st.session_state['cand_lat']):.6f}, {float(st.session_state['cand_lon']):.6f}"
)


# Koordinatları normalize et (lat/lon ters girildiyse otomatik düzeltir)
curr_lat_n, curr_lon_n, curr_note = normalize_coords(st.session_state["curr_lat"], st.session_state["curr_lon"])
cand_lat_n, cand_lon_n, cand_note = normalize_coords(st.session_state["cand_lat"], st.session_state["cand_lon"])
if curr_note:
    st.warning(f"Mevcut okul: {curr_note}")
    st.session_state["curr_lat"], st.session_state["curr_lon"] = curr_lat_n, curr_lon_n
if cand_note:
    st.warning(f"Aday okul: {cand_note}")
    st.session_state["cand_lat"], st.session_state["cand_lon"] = cand_lat_n, cand_lon_n

mid = [
    (curr_lat_n + cand_lat_n) / 2,
    (curr_lon_n + cand_lon_n) / 2,
]

# 4.1 Noktasal harita (üstte)
m = folium.Map(location=mid, zoom_start=13, tiles="CartoDB positron")

if not df_map.empty:
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in df_map.iterrows():
        dept = row.get("Alan / Dal", "Bilinmiyor")
        color = dept_colors.get(dept, dept_colors.get("Diğer", "#3498db"))
        folium.CircleMarker(
            location=[float(row["Enlem"]), float(row["Boylam"])],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=f"<b>{dept}</b><br>{row.get('Ilce_Temiz','')} / {row.get('Mahalle_Temiz','')}",
        ).add_to(marker_cluster)

    # Okullar + yarıçap
    folium.Marker(
        [curr_lat_n, curr_lon_n],
        tooltip=st.session_state["curr_name"],
        icon=folium.Icon(color=ICON_COLOR_FOLIUM["curr"], icon="home"),
    ).add_to(m)
    folium.Marker(
        [cand_lat_n, cand_lon_n],
        tooltip=st.session_state["cand_name"],
        icon=folium.Icon(color=ICON_COLOR_FOLIUM["cand"], icon="star"),
    ).add_to(m)

    folium.Circle(
        location=[curr_lat_n, curr_lon_n],
        radius=radius * 1000,
        color=ICON_COLOR_FOLIUM["curr"],
        fill=False,
    ).add_to(m)
    folium.Circle(
        location=[cand_lat_n, cand_lon_n],
        radius=radius * 1000,
        color=ICON_COLOR_FOLIUM["cand"],
        fill=False,
        dash_array="5,5",
    ).add_to(m)

    if ideal and show_ideal:
        folium.Marker(
            list(ideal),
            tooltip=f"İdeal ({ideal_note})",
            icon=folium.Icon(color=ICON_COLOR_FOLIUM["ideal"], icon="crosshairs"),
        ).add_to(m)

    add_custom_legend(
        m,
        dept_colors,
        st.session_state["curr_name"],
        st.session_state["cand_name"],
        ideal_shown=bool(ideal and show_ideal),
        ideal_note=ideal_note,
    )

render_map(m, height=650)

# 4.2 Isı haritası (alta)
st.markdown("### 🔥 Isı Haritası (Yoğunluk)")
mh = folium.Map(location=mid, zoom_start=13, tiles="CartoDB positron")

if not df_map.empty:
    heat_df = df_map[["Enlem", "Boylam"]].copy()
    heat_df["Enlem"] = pd.to_numeric(heat_df["Enlem"], errors="coerce")
    heat_df["Boylam"] = pd.to_numeric(heat_df["Boylam"], errors="coerce")
    heat_df = heat_df.dropna()

    heat_points = heat_df[["Enlem", "Boylam"]].astype(float).values.tolist()
    if heat_points:
        HeatMap(
            heat_points,
            radius=22,
            blur=18,
            min_opacity=0.40,
            max_zoom=18,
        ).add_to(mh)

    folium.Marker(
        [curr_lat_n, curr_lon_n],
        tooltip=st.session_state["curr_name"],
        icon=folium.Icon(color=ICON_COLOR_FOLIUM["curr"], icon="home"),
    ).add_to(mh)
    folium.Marker(
        [cand_lat_n, cand_lon_n],
        tooltip=st.session_state["cand_name"],
        icon=folium.Icon(color=ICON_COLOR_FOLIUM["cand"], icon="star"),
    ).add_to(mh)

    if ideal and show_ideal:
        folium.CircleMarker(
            location=list(ideal),
            radius=9,
            color=ICON_COLOR_FOLIUM["ideal"],
            fill=True,
            fill_opacity=1,
            tooltip=f"İdeal ({ideal_note})",
        ).add_to(mh)

render_map(mh, height=650)

# 4.3 Bölüm -> Mahalle dağılım haritası (en altta)
st.markdown("### 🧩 Bölümlerin Mahalle Dağılımı")
md = folium.Map(location=mid, zoom_start=12, tiles="CartoDB positron")

# Okul noktaları (bu haritada da görünsün)
folium.Marker(
    [float(curr_lat_n), float(curr_lon_n)],
    tooltip=st.session_state["curr_name"],
    icon=folium.Icon(color=ICON_COLOR_FOLIUM["curr"], icon="home"),
).add_to(md)
folium.Marker(
    [float(cand_lat_n), float(cand_lon_n)],
    tooltip=st.session_state["cand_name"],
    icon=folium.Icon(color=ICON_COLOR_FOLIUM["cand"], icon="star"),
).add_to(md)
if ideal and show_ideal:
    folium.Marker(
        list(ideal),
        tooltip=f"İdeal ({ideal_note})",
        icon=folium.Icon(color=ICON_COLOR_FOLIUM["ideal"], icon="crosshairs"),
    ).add_to(md)


if (not df_map.empty) and ("Mahalle_Temiz" in df_map.columns) and ("Alan / Dal" in df_map.columns):
    tmp = df_map.copy()
    tmp["Enlem"] = pd.to_numeric(tmp["Enlem"], errors="coerce")
    tmp["Boylam"] = pd.to_numeric(tmp["Boylam"], errors="coerce")
    tmp = tmp.dropna(subset=["Enlem", "Boylam"])

    grp = (
        tmp.groupby(["Ilce_Temiz", "Mahalle_Temiz", "Alan / Dal"], dropna=False)
        .agg(Adet=("Alan / Dal", "size"), Enlem=("Enlem", "mean"), Boylam=("Boylam", "mean"))
        .reset_index()
    )

    for dept in sorted(grp["Alan / Dal"].dropna().unique().tolist()):
        fg = folium.FeatureGroup(name=str(dept), show=(sel_dept != "TÜMÜ" and dept == sel_dept))
        sub = grp[grp["Alan / Dal"] == dept]
        color = dept_colors.get(dept, dept_colors.get("Diğer", "#3498db"))

        for _, r in sub.iterrows():
            count = int(r["Adet"])
            rad = max(5, min(22, int(4 + math.sqrt(count) * 3)))

            ilce = (r.get("Ilce_Temiz") or "").strip()
            mah = (r.get("Mahalle_Temiz") or "").strip()

            folium.CircleMarker(
                location=[float(r["Enlem"]), float(r["Boylam"])],
                radius=rad,
                color=color,
                fill=True,
                fill_opacity=0.65,
                tooltip=f"{dept} | {ilce} / {mah} | {count} öğrenci",
            ).add_to(fg)

        fg.add_to(md)

    folium.LayerControl(collapsed=False).add_to(md)

render_map(md, height=700)

# -------------------- TABLOLAR --------------------
st.markdown("## 5) Analiz tabloları")
tabs = st.tabs(["🛠️ Bölüm", "🎓 Sınıf", "👫 Cinsiyet"])
with tabs[0]:
    st.dataframe(create_comparison_table(df_display, "Alan / Dal", radius), width="stretch", hide_index=True)
with tabs[1]:
    st.dataframe(create_comparison_table(df_display, "Sınıf Seviyesi", radius), width="stretch", hide_index=True)
with tabs[2]:
    st.dataframe(create_comparison_table(df_display, "Cinsiyetiniz", radius), width="stretch", hide_index=True)

# -------------------- GRAFİKLER --------------------
st.markdown("## 6) Mahalle ve Bölüm yoğunluk (grafikler)")
if not _HAS_PLOTLY:
    st.info("Plotly yüklü değilse grafikler gösterilmez. (pip install plotly)")
else:
    if "Mahalle_Temiz" in df_display.columns:
        mahalle_counts = df_display["Mahalle_Temiz"].value_counts().reset_index()
        mahalle_counts.columns = ["Mahalle", "Öğrenci Sayısı"]
        mahalle_counts = mahalle_counts[mahalle_counts["Mahalle"].astype(str).str.strip() != ""]
        top_20 = mahalle_counts.head(20)

        g1, g2 = st.columns(2)

        with g1:
            st.markdown("#### 🏘️ En yoğun 20 mahalle")
            if not top_20.empty:
                fig = px.bar(top_20, x="Öğrenci Sayısı", y="Mahalle", orientation="h", text="Öğrenci Sayısı")
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Gösterilecek mahalle yok.")

        with g2:
            st.markdown("#### 🏢 Mahalle bazlı bölüm dağılımı (Top 10 mahalle)")
            if "Alan / Dal" in df_display.columns and not top_20.empty:
                top_10_names = top_20.head(10)["Mahalle"].tolist()
                df_top10 = df_display[df_display["Mahalle_Temiz"].isin(top_10_names)].copy()
                grp2 = df_top10.groupby(["Mahalle_Temiz", "Alan / Dal"]).size().reset_index(name="Sayı")
                if not grp2.empty:
                    fig2 = px.bar(
                        grp2,
                        x="Mahalle_Temiz",
                        y="Sayı",
                        color="Alan / Dal",
                        barmode="stack",
                        color_discrete_map=DEPARTMENT_COLORS,
                    )
                    fig2.update_layout(xaxis_title="Mahalle", yaxis_title="Öğrenci Sayısı")
                    st.plotly_chart(fig2, width="stretch")
                else:
                    st.info("Top 10 mahalle için bölüm kırılımı oluşmadı.")
            else:
                st.info("Bölüm verisi yok veya mahalle listesi boş.")
    else:
        st.info("Mahalle_Temiz üretilemediği için grafikler gösterilemiyor.")

# -------------------- İNDİRME --------------------
st.markdown("## 7) İndirme / Raporlama")

c_dl1, c_dl2 = st.columns(2)

with c_dl1:
    try:
        map_html = m.get_root().render().encode("utf-8")
        st.download_button("🌍 Noktasal Harita (.html)", data=map_html, file_name="harita_noktasal.html", mime="text/html")
    except Exception:
        st.caption("Noktasal harita indirilemedi (harita oluşmadı).")
    try:
        heat_html = mh.get_root().render().encode("utf-8")
        st.download_button("🔥 Isı Haritası (.html)", data=heat_html, file_name="harita_isi.html", mime="text/html")
    except Exception:
        st.caption("Isı haritası indirilemedi (harita oluşmadı).")

with c_dl2:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_display.to_excel(writer, sheet_name="Veri", index=False)
        create_comparison_table(df_display, "Alan / Dal", radius).to_excel(writer, sheet_name="Bolum", index=False)
        create_comparison_table(df_display, "Sınıf Seviyesi", radius).to_excel(writer, sheet_name="Sinif", index=False)
        create_comparison_table(df_display, "Cinsiyetiniz", radius).to_excel(writer, sheet_name="Cinsiyet", index=False)

        if "Mahalle_Temiz" in df_display.columns:
            df_display["Mahalle_Temiz"].value_counts().reset_index().rename(
                columns={"index": "Mahalle", "Mahalle_Temiz": "Sayı"}
            ).to_excel(writer, sheet_name="Mahalle_Ozet", index=False)


        # Bölüm x İlçe x Mahalle kırılımı
        if all(c in df_display.columns for c in ["Ilce_Temiz", "Mahalle_Temiz", "Alan / Dal"]):
            bolum_mahalle = (
                df_display
                .groupby(["Ilce_Temiz", "Mahalle_Temiz", "Alan / Dal"], dropna=False)
                .size()
                .reset_index(name="Öğrenci Sayısı")
            )
            # Boş mahalle/ilçe değerlerini ele
            bolum_mahalle["Ilce_Temiz"] = bolum_mahalle["Ilce_Temiz"].fillna("").astype(str)
            bolum_mahalle["Mahalle_Temiz"] = bolum_mahalle["Mahalle_Temiz"].fillna("").astype(str)
            bolum_mahalle["Alan / Dal"] = bolum_mahalle["Alan / Dal"].fillna("Bilinmiyor").astype(str)

            bolum_mahalle = bolum_mahalle[
                (bolum_mahalle["Ilce_Temiz"].str.strip() != "") |
                (bolum_mahalle["Mahalle_Temiz"].str.strip() != "")
            ].sort_values(["Ilce_Temiz", "Mahalle_Temiz", "Öğrenci Sayısı"], ascending=[True, True, False])

            bolum_mahalle.to_excel(writer, sheet_name="Bolum_Ilce_Mahalle", index=False)

        ozet = pd.DataFrame(
            {
                "Metri̇k": ["Toplam Kayıt", "Geocode Başarılı", "Geocode Başarısız", "Yarıçap (km)", "Mod", "Bölüm Filtresi"],
                "Değer": [
                    len(df_display),
                    int(df_display["Enlem"].notna().sum()) if "Enlem" in df_display.columns else 0,
                    int(df_display["Enlem"].isna().sum()) if "Enlem" in df_display.columns else 0,
                    radius,
                    mode,
                    sel_dept,
                ],
            }
        )
        ozet.to_excel(writer, sheet_name="Ozet", index=False)

    st.download_button(
        "📊 Raporu İndir (.xlsx)",
        data=buf.getvalue(),
        file_name="okul_konum_analiz_rapor.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )


# -------------------- BÖLÜM BAZINDA MAHALLE ÖNERİSİ --------------------
st.markdown("## 8) Bölüm bazında mahalle analizi ve muhit önerisi")

base_for_dept = full_df.dropna(subset=["Enlem", "Boylam"]).copy()

if base_for_dept.empty or "Alan / Dal" not in base_for_dept.columns:
    st.info("Bölüm bazında öneri üretmek için 'Alan / Dal' ve koordinat bilgisi gerekli.")
else:
    # Hedef nokta: ideal varsa onu, yoksa aday okulu referans al
    target_lat, target_lon = (ideal if ideal else (cand_lat_n, cand_lon_n))
    target_name = "İdeal Nokta" if ideal else st.session_state["cand_name"]

    st.caption(f"Öneri mantığı: Her bölüm için mahallelerin öğrenci yoğunluğu ve {target_name} noktasına yakınlığı birlikte değerlendirilir.")

    # Önce mahalle-bölüm bazında özet
    dept_grp = (
        base_for_dept.groupby(["Alan / Dal", "Ilce_Temiz", "Mahalle_Temiz"], dropna=False)
        .agg(
            Öğrenci_Sayısı=("Alan / Dal", "size"),
            Enlem=("Enlem", "mean"),
            Boylam=("Boylam", "mean"),
        )
        .reset_index()
    )

    # Boş mahalle/ilçe satırlarını temizle
    dept_grp["Ilce_Temiz"] = dept_grp["Ilce_Temiz"].fillna("").astype(str)
    dept_grp["Mahalle_Temiz"] = dept_grp["Mahalle_Temiz"].fillna("").astype(str)
    dept_grp = dept_grp[
        (dept_grp["Ilce_Temiz"].str.strip() != "") | (dept_grp["Mahalle_Temiz"].str.strip() != "")
    ].copy()

    # Mesafe (km) hesapla
    dept_grp["Hedefe_Mesafe_km"] = haversine_km(
        target_lat, target_lon,
        dept_grp["Enlem"].astype(float).to_numpy(),
        dept_grp["Boylam"].astype(float).to_numpy()
    )

    # Puanlama (0-1 normalize): yoğunluk yüksek + hedefe yakın
    def _score_block(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        max_c = float(g["Öğrenci_Sayısı"].max()) if len(g) else 1.0
        max_d = float(g["Hedefe_Mesafe_km"].max()) if len(g) else 1.0
        g["Yoğunluk_N"] = (g["Öğrenci_Sayısı"] / (max_c if max_c else 1.0)).clip(0, 1)
        g["Yakınlık_N"] = (1 - (g["Hedefe_Mesafe_km"] / (max_d if max_d else 1.0))).clip(0, 1)
        # Ağırlıklar: yoğunluk %70, yakınlık %30
        g["Skor"] = (0.70 * g["Yoğunluk_N"] + 0.30 * g["Yakınlık_N"]).round(4)
        return g

    # Her bölüm için öneri
    depts = sorted([d for d in dept_grp["Alan / Dal"].dropna().unique().tolist() if str(d).strip() != ""])

    for dept in depts:
        g = dept_grp[dept_grp["Alan / Dal"] == dept]
        if g.empty:
            continue
        g = _score_block(g).sort_values(["Skor", "Öğrenci_Sayısı"], ascending=[False, False])

        best = g.iloc[0]
        best_ilce = best["Ilce_Temiz"]
        best_mah = best["Mahalle_Temiz"]
        best_cnt = int(best["Öğrenci_Sayısı"])
        best_dist = float(best["Hedefe_Mesafe_km"])
        best_score = float(best["Skor"])

        with st.expander(f"🏢 {dept} — önerilen muhit: **{best_ilce} / {best_mah}**", expanded=False):
            st.markdown(
                f"**Öneri:** Bu bölüm için en uygun muhit olarak **{best_ilce} / {best_mah}** öne çıkıyor "
                f"(**{best_cnt} öğrenci**, {target_name}’a **{best_dist:.2f} km**, skor **{best_score:.3f}**)."
            )

            view = g.head(15)[
                ["Ilce_Temiz", "Mahalle_Temiz", "Öğrenci_Sayısı", "Hedefe_Mesafe_km", "Skor"]
            ].rename(
                columns={
                    "Ilce_Temiz": "İlçe",
                    "Mahalle_Temiz": "Mahalle",
                    "Öğrenci_Sayısı": "Öğrenci Sayısı",
                    "Hedefe_Mesafe_km": f"{target_name} Mesafe (km)",
                    "Skor": "Skor",
                }
            )
            st.dataframe(view, width="stretch", hide_index=True)

            st.caption("Not: Skor, yoğunluk (%70) + hedefe yakınlık (%30) bileşimidir. İstersen ağırlıkları değiştirebiliriz.")


# -------------------- AĞIRLIK MERKEZİNE GÖRE ÖNERİ + MİNİ HARİTALAR --------------------
st.markdown("## 9) Bölüm dağılımının ağırlık merkezine göre muhit önerisi + mini haritalar")

st.caption(
    "Bu bölümde öneri, sadece en kalabalık mahalleye göre değil; her bölümün öğrenci noktalarının oluşturduğu "
    "**ağırlık merkezi** (dağılımın ortası) esas alınarak yapılır. Önerilen muhit, bölümün ağırlık merkezine "
    "en yakın mahalle(ler) arasından seçilir."
)




base_for_dept2 = full_df.dropna(subset=["Enlem", "Boylam"]).copy()

if base_for_dept2.empty or "Alan / Dal" not in base_for_dept2.columns:
    st.info("Ağırlık merkezi önerisi için 'Alan / Dal' ve koordinat bilgisi gerekli.")
else:
    # Bölüm listesi (veride gerçekten olanlar)
    _dept_series = base_for_dept2["Alan / Dal"].fillna("Bilinmiyor").astype(str)
    dept_list_real = sorted([d for d in _dept_series.unique().tolist() if d.strip()])

    # Yardımcı: bölüm ağırlık merkezi (IQR ile isteğe bağlı temizleme)
    def dept_center(df_dept: pd.DataFrame, remove_outliers: bool = True):
        d = df_dept.dropna(subset=["Enlem", "Boylam"]).copy()
        if d.empty:
            return None

        d["Enlem"] = pd.to_numeric(d["Enlem"], errors="coerce")
        d["Boylam"] = pd.to_numeric(d["Boylam"], errors="coerce")
        d = d.dropna(subset=["Enlem", "Boylam"]).copy()
        if d.empty:
            return None

        if not remove_outliers:
            return (float(d["Enlem"].mean()), float(d["Boylam"].mean()))

        # Lat/Lon için IQR kırpma
        lq1, lq3 = d["Enlem"].quantile(0.25), d["Enlem"].quantile(0.75)
        bq1, bq3 = d["Boylam"].quantile(0.25), d["Boylam"].quantile(0.75)
        d = d[
            (d["Enlem"] >= lq1 - 1.5 * (lq3 - lq1)) & (d["Enlem"] <= lq3 + 1.5 * (lq3 - lq1)) &
            (d["Boylam"] >= bq1 - 1.5 * (bq3 - bq1)) & (d["Boylam"] <= bq3 + 1.5 * (bq3 - bq1))
        ]
        if d.empty:
            return None
        return (float(d["Enlem"].mean()), float(d["Boylam"].mean()))

    # Mahalle özetini bölüm bazında çıkar (centroid + count)
    grp_dept_mahalle = (
        base_for_dept2.groupby(["Alan / Dal", "Ilce_Temiz", "Mahalle_Temiz"], dropna=False)
        .agg(
            Öğrenci_Sayısı=("Alan / Dal", "size"),
            Enlem=("Enlem", "mean"),
            Boylam=("Boylam", "mean"),
        )
        .reset_index()
    )

    grp_dept_mahalle["Ilce_Temiz"] = grp_dept_mahalle["Ilce_Temiz"].fillna("").astype(str)
    grp_dept_mahalle["Mahalle_Temiz"] = grp_dept_mahalle["Mahalle_Temiz"].fillna("").astype(str)
    grp_dept_mahalle["Alan / Dal"] = grp_dept_mahalle["Alan / Dal"].fillna("Bilinmiyor").astype(str)

    grp_dept_mahalle = grp_dept_mahalle[
        (grp_dept_mahalle["Ilce_Temiz"].str.strip() != "") | (grp_dept_mahalle["Mahalle_Temiz"].str.strip() != "")
    ].copy()

    # Görsel: her bölüm için mini harita + öneri
    for dept in dept_list_real:
        d_all = base_for_dept2[base_for_dept2["Alan / Dal"].fillna("Bilinmiyor").astype(str) == dept].copy()
        center = dept_center(d_all, remove_outliers=True)
        if center is None:
            continue

        c_lat, c_lon = center

        g = grp_dept_mahalle[grp_dept_mahalle["Alan / Dal"] == dept].copy()
        if g.empty:
            continue

        # Mesafe: mahalle centroid -> bölüm ağırlık merkezi
        g["Merkeze_Mesafe_km"] = haversine_km(
            c_lat, c_lon,
            pd.to_numeric(g["Enlem"], errors="coerce").to_numpy(),
            pd.to_numeric(g["Boylam"], errors="coerce").to_numpy(),
        )
        g = g.dropna(subset=["Merkeze_Mesafe_km"]).copy()
        if g.empty:
            continue

        # Min örnek eşiği: çok düşük sayılı uç mahalleleri azalt
        max_cnt = int(g["Öğrenci_Sayısı"].max()) if len(g) else 1
        min_cnt = max(2, int(round(max_cnt * 0.10)))  # max'ın %10'u, en az 2
        g_pref = g[g["Öğrenci_Sayısı"] >= min_cnt].copy()
        g_pick = (g_pref if not g_pref.empty else g).sort_values(["Merkeze_Mesafe_km", "Öğrenci_Sayısı"], ascending=[True, False])

        best = g_pick.iloc[0]
        best_ilce = str(best["Ilce_Temiz"])
        best_mah = str(best["Mahalle_Temiz"])
        best_cnt = int(best["Öğrenci_Sayısı"])
        best_dist = float(best["Merkeze_Mesafe_km"])

        # Mini harita
        color = DEPARTMENT_COLORS.get(dept, DEPARTMENT_COLORS.get("Diğer", "#3498db"))

        with st.expander(f"🧭 {dept} — ağırlık merkezine göre öneri: **{best_ilce} / {best_mah}**", expanded=False):
            st.markdown(
                f"**Ağırlık Merkezi:** `{c_lat:.6f}, {c_lon:.6f}`  \n"
                f"**Öneri:** **{best_ilce} / {best_mah}** (bu bölümden **{best_cnt} öğrenci**), "
                f"bölüm merkezine **{best_dist:.2f} km** uzaklıkta.  \n"
                f"**Not:** Çok düşük sayılı mahalleler (eşik: `{min_cnt}`) varsa, öneri öncelikle bu eşik üzerindeki mahallelerden seçilir."
            )

            # Tablo: en yakın 12 mahalle
            view = (
                g.sort_values(["Merkeze_Mesafe_km", "Öğrenci_Sayısı"], ascending=[True, False])
                .head(12)
                [["Ilce_Temiz", "Mahalle_Temiz", "Öğrenci_Sayısı", "Merkeze_Mesafe_km"]]
                .rename(columns={
                    "Ilce_Temiz": "İlçe",
                    "Mahalle_Temiz": "Mahalle",
                    "Öğrenci_Sayısı": "Öğrenci Sayısı",
                    "Merkeze_Mesafe_km": "Bölüm Merkezine Mesafe (km)",
                })
            )
            st.dataframe(view, width="stretch", hide_index=True)

            mini = folium.Map(location=[c_lat, c_lon], zoom_start=12, tiles="CartoDB positron")

            # Bölüm merkezi işareti
            folium.Marker(
                [c_lat, c_lon],
                tooltip=f"{dept} ağırlık merkezi",
                icon=folium.Icon(color="blue", icon="crosshairs"),
            ).add_to(mini)

            # Okullar
            folium.Marker(
                [curr_lat_n, curr_lon_n],
                tooltip=st.session_state["curr_name"],
                icon=folium.Icon(color=ICON_COLOR_FOLIUM["curr"], icon="home"),
            ).add_to(mini)
            folium.Marker(
                [cand_lat_n, cand_lon_n],
                tooltip=st.session_state["cand_name"],
                icon=folium.Icon(color=ICON_COLOR_FOLIUM["cand"], icon="star"),
            ).add_to(mini)

            # Mahalle balonları
            for _, r in g.iterrows():
                cnt = int(r["Öğrenci_Sayısı"])
                rad = max(5, min(22, int(4 + math.sqrt(cnt) * 3)))
                ilce = str(r.get("Ilce_Temiz", ""))
                mah = str(r.get("Mahalle_Temiz", ""))
                lat = float(r["Enlem"])
                lon = float(r["Boylam"])

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=rad,
                    color=color,
                    fill=True,
                    fill_opacity=0.45,
                    tooltip=f"{ilce} / {mah} | {cnt} öğrenci",
                ).add_to(mini)

            # Önerilen muhit işareti
            folium.Marker(
                [float(best["Enlem"]), float(best["Boylam"])],
                tooltip=f"Öneri: {best_ilce} / {best_mah}",
                icon=folium.Icon(color="green", icon="ok-sign"),
            ).add_to(mini)

            render_map(mini, height=460)