import io
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ==================== SAYFA AYARLARI ====================
st.set_page_config(
    page_title="Okul Konum Analiz Pro",
    layout="wide",
    page_icon="🏫",
    initial_sidebar_state="expanded",
)

# Modern CSS Stilleri
st.markdown(
    """
<style>
    /* Ana stil iyileştirmeleri */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Başlık stilleri */
    h1 {
        color: #1f77b4;
        font-weight: 700;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
    }

    h2, h3 {
        color: #2c3e50;
        margin-top: 1.5rem;
    }
    /* ================= SIDEBAR DARK THEME ================= */

/* Sidebar ana arka plan */
section[data-testid="stSidebar"] {
    background-color: #0b0b0b !important;
}

/* Sidebar içindeki tüm yazılar */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Sidebar başlıkları */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #ffffff !important;
}

/* Input – text, number */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background-color: #1c1c1c !important;
    color: #ffffff !important;
    border: 1px solid #333333 !important;
    border-radius: 6px;
}

/* Selectbox */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #1c1c1c !important;
    color: #ffffff !important;
    border: 1px solid #333333 !important;
}

/* Slider */
section[data-testid="stSidebar"] .stSlider > div {
    color: #ffffff !important;
}

/* Checkbox & radio label */
section[data-testid="stSidebar"] label {
    color: #ffffff !important;
}

/* Butonlar (mavi vurgulu) */
section[data-testid="stSidebar"] .stButton button {
    background-color: #1f77b4 !important;
    color: #ffffff !important;
    border-radius: 8px;
    font-weight: 600;
}

/* Divider */
section[data-testid="stSidebar"] hr {
    background: linear-gradient(90deg, transparent, #1f77b4, transparent);
    height: 2px;
    border: none;
}

/* Expander başlıkları */
section[data-testid="stSidebar"] summary {
    color: #ffffff !important;
}

    /* Metric kartları */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
    }

    /* Info box'lar */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
    }

    /* Tablo stilleri */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Tab stilleri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        background-color: #e9ecef;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }

    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #1f77b4, transparent);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==================== SABITLER ====================
TARGET_CITY = "Eskişehir"
TARGET_COUNTRY = "Türkiye"

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
    "Diğer": "#95a5a6",
}

# ==================== SESSION STATE ====================
defs = {
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
}
for k, v in defs.items():
    st.session_state.setdefault(k, v)

# ==================== GEOCODER ====================
@st.cache_resource
def get_geocoder():
    geolocator = Nominatim(user_agent="okul_konum_analiz_pro_v2", timeout=10)
    return RateLimiter(geolocator.geocode, min_delay_seconds=1.1, swallow_exceptions=True)

_GEOCODE = get_geocoder()


def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    s = str(text).lower()
    for w in ["mahallesi", "mah.", "mah", "(köy)", "köyü", "koyu"]:
        s = s.replace(w, "")
    return s.strip().title()


@st.cache_data(show_spinner=False, ttl=3600)
def geocode_address_cached(address: str):
    if not address or not str(address).strip():
        return None
    try:
        loc = _GEOCODE(address)
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception:
        return None
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


# ==================== GOOGLE DRIVE / SHEETS ====================
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
        if not sheet_id:
            raise ValueError("Google Sheets ID bulunamadı.")
        gid = _extract_gid(url)
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        return pd.read_csv(csv_url)

    if "drive.google.com" in url:
        file_id = _extract_drive_file_id(url)
        if not file_id:
            raise ValueError("Google Drive dosya ID bulunamadı.")
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        r = requests.get(download_url, timeout=30)
        r.raise_for_status()
        content = r.content

        if content[:2] == b"PK":
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
        return pd.read_csv(io.BytesIO(content))

    raise ValueError("Link Google Sheets veya Google Drive linki olmalı.")


# ==================== VERİ İŞLEME - PARALEL ====================
def geocode_single_address(mahalle: str, ilce: str) -> tuple:
    addr1 = build_mahalle_ilce_address(mahalle, ilce)
    coords = geocode_address_cached(addr1)

    if coords is None and ilce:
        addr2 = build_mahalle_ilce_address("", ilce)
        coords = geocode_address_cached(addr2)

    return (mahalle, ilce, coords)


def process_data_unique_geocode(df: pd.DataFrame) -> pd.DataFrame:
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

    uniq = (
        df[["Mahalle_Temiz", "Ilce_Temiz"]]
        .fillna("")
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    lat_map, lon_map = {}, {}
    prog = st.progress(0.0, text="🔍 Adresler çözülüyor...")
    total = max(len(uniq), 1)

    # RateLimiter + Nominatim için düşük worker daha stabil
    max_workers = 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(geocode_single_address, r["Mahalle_Temiz"], r["Ilce_Temiz"]) for _, r in uniq.iterrows()]
        for i, fut in enumerate(futures):
            mahalle, ilce, coords = fut.result()
            key = (mahalle, ilce)
            if coords is None:
                lat_map[key] = np.nan
                lon_map[key] = np.nan
            else:
                lat_map[key] = coords[0]
                lon_map[key] = coords[1]
            prog.progress((i + 1) / total)

    prog.empty()

    keys = list(zip(df["Mahalle_Temiz"].fillna("").astype(str), df["Ilce_Temiz"].fillna("").astype(str)))
    df["Enlem"] = [lat_map.get(k, np.nan) for k in keys]
    df["Boylam"] = [lon_map.get(k, np.nan) for k in keys]
    return df


# ==================== MESAFE HESAPLAMA ====================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def calculate_distances(df: pd.DataFrame, curr_lat, curr_lon, cand_lat, cand_lon, mode: str, road_factor: float = 1.4):
    df = df.copy()
    lat, lon = df["Enlem"].to_numpy(), df["Boylam"].to_numpy()

    base_curr = haversine_km(curr_lat, curr_lon, lat, lon)
    base_cand = haversine_km(cand_lat, cand_lon, lat, lon)

    if mode == "🦅 Kuş Uçuşu":
        df["Mesafe_Mevcut"] = base_curr
        df["Mesafe_Aday"] = base_cand
    elif mode == "🚗 Tahmini Karayolu":
        df["Mesafe_Mevcut"] = base_curr * road_factor
        df["Mesafe_Aday"] = base_cand * road_factor
    elif mode == "🏙️ Manhattan":
        lat_km = np.abs(lat - curr_lat) * 111.32
        lon_km = np.abs(lon - curr_lon) * (111.32 * np.cos(np.radians((lat + curr_lat) / 2)))
        df["Mesafe_Mevcut"] = lat_km + lon_km

        lat_km2 = np.abs(lat - cand_lat) * 111.32
        lon_km2 = np.abs(lon - cand_lon) * (111.32 * np.cos(np.radians((lat + cand_lat) / 2)))
        df["Mesafe_Aday"] = lat_km2 + lon_km2
    else:
        df["Mesafe_Mevcut"] = base_curr
        df["Mesafe_Aday"] = base_cand

    return df


def calculate_smart_radius_mean(df: pd.DataFrame, basis_col: str = "Mesafe_Mevcut", remove_outliers: bool = True) -> float:
    if basis_col not in df.columns or df.empty:
        return 3.0
    d = df[basis_col].dropna()
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

    lat_q1, lat_q3 = d["Enlem"].quantile(0.25), d["Enlem"].quantile(0.75)
    lon_q1, lon_q3 = d["Boylam"].quantile(0.25), d["Boylam"].quantile(0.75)
    lat_iqr, lon_iqr = lat_q3 - lat_q1, lon_q3 - lon_q1

    clean = d[
        (d["Enlem"] >= lat_q1 - 1.5 * lat_iqr) & (d["Enlem"] <= lat_q3 + 1.5 * lat_iqr) &
        (d["Boylam"] >= lon_q1 - 1.5 * lon_iqr) & (d["Boylam"] <= lon_q3 + 1.5 * lon_iqr)
    ]

    if clean.empty:
        return (float(d["Enlem"].mean()), float(d["Boylam"].mean()))

    return (float(clean["Enlem"].mean()), float(clean["Boylam"].mean()))


# ==================== HARİTA LEGEND ====================
def add_custom_legend(map_obj, title, colors, curr_name, cand_name, ideal_shown=False, ideal_note=""):
    legend_html = f"""
    <div style="
        position: fixed;
        top: 20px;
        right: 20px;
        width: 320px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        opacity: 0.96;
        color: #2c3e50;
        padding: 16px;
        border-radius: 12px;
        border: 2px solid #1f77b4;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        z-index: 9999;
        font-family: 'Segoe UI', sans-serif;
        font-size: 13px;
        max-height: 520px;
        overflow-y: auto;
    ">
        <div style="
            font-weight: 700;
            font-size: 16px;
            margin-bottom: 12px;
            color: #1f77b4;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 8px;
        ">
            {title}
        </div>

        <div style="margin-bottom: 12px; padding-bottom: 8px;">
            <div style="font-weight: 600; margin-bottom: 6px; color: #34495e;">🏫 Okullar</div>

            <div style="margin: 4px 0; padding: 6px; border-radius: 6px; background: #f8f9fa;">
                <span style="color: #000; font-weight: 700;">●</span>
                <span style="color:#000; font-weight: 600;"> {curr_name}</span>
            </div>

            <div style="margin: 4px 0; padding: 6px; border-radius: 6px; background: #f8f9fa;">
                <span style="color: #2ecc71; font-weight: 700;">●</span>
                <span style="color:#2ecc71; font-weight: 600;"> {cand_name}</span>
            </div>
    """

    if ideal_shown:
        legend_html += f"""
            <div style="margin: 4px 0; padding: 6px; border-radius: 6px; background: #e3f2fd;">
                <span style="color: #1f77b4; font-weight: 700;">●</span>
                <span style="color:#1f77b4; font-weight: 600;"> İdeal Okul Konumu</span>
                <div style="font-size: 11px; color: #1f77b4; margin-left: 16px;">({ideal_note})</div>
            </div>
        """

    legend_html += """
        </div>

        <div>
            <div style="font-weight: 600; margin-bottom: 8px; color: #34495e;">👥 Bölümler</div>
    """

    for dept, color in colors.items():
        legend_html += f"""
            <div style="margin: 3px 0; padding: 6px; border-radius: 6px; background: #f8f9fa;">
                <span style="color: {color}; font-weight: 700;">●</span>
                <span style="color: {color}; font-weight: 600;"> {dept}</span>
            </div>
        """

    legend_html += """
        </div>
    </div>
    """

    map_obj.get_root().html.add_child(folium.Element(legend_html))


def create_comparison_table(df, category_col, radius):
    if category_col not in df.columns:
        return pd.DataFrame()

    count_col = "Zaman damgası" if "Zaman damgası" in df.columns else df.columns[0]

    grouped = df.groupby(category_col).agg(
        Toplam_Ogrenci=(count_col, "count"),
        Mevcut_Erisen=("Mesafe_Mevcut", lambda x: (x <= radius).sum()),
        Aday_Erisen=("Mesafe_Aday", lambda x: (x <= radius).sum()),
    ).reset_index()

    total_row = pd.DataFrame({
        category_col: ["📊 GENEL TOPLAM"],
        "Toplam_Ogrenci": [grouped["Toplam_Ogrenci"].sum()],
        "Mevcut_Erisen": [grouped["Mevcut_Erisen"].sum()],
        "Aday_Erisen": [grouped["Aday_Erisen"].sum()],
    })

    final_data = pd.concat([grouped, total_row], ignore_index=True)
    final_data["Mevcut_%"] = (final_data["Mevcut_Erisen"] / final_data["Toplam_Ogrenci"] * 100).fillna(0).round(1)
    final_data["Aday_%"] = (final_data["Aday_Erisen"] / final_data["Toplam_Ogrenci"] * 100).fillna(0).round(1)
    final_data["Fark"] = final_data["Aday_Erisen"] - final_data["Mevcut_Erisen"]

    def durum(fark):
        if fark > 0:
            return f"🟢 +{fark} öğrenci (Kazanım)"
        if fark < 0:
            return f"🔴 {fark} öğrenci (Kayıp)"
        return "⚪ Değişim Yok"

    final_data["Analiz Sonucu"] = final_data["Fark"].apply(durum)

    final_df = final_data[[category_col, "Toplam_Ogrenci", "Mevcut_Erisen", "Mevcut_%", "Aday_Erisen", "Aday_%", "Fark", "Analiz Sonucu"]].copy()
    final_df.columns = [category_col, "Toplam", "Mevcut Erişen", "Mevcut %", "Aday Erişen", "Aday %", "Fark", "Detay"]
    return final_df


# ==================== ANA ARAYÜZ ====================
st.title("🏫 Okul Konum Analizi Pro")
st.markdown("**Veri odaklı karar destek sistemi**")

# ==================== SİDEBAR ====================
with st.sidebar:
    st.markdown("### 🔍 Adres Bulucu")

    with st.expander("📍 Koordinat Bul", expanded=False):
        addr_text = st.text_input("Adres / Okul Adı", placeholder="Örn: Gazi MTAL, Eskişehir")

        if st.button("🎯 Ara", disabled=not addr_text.strip(), use_container_width=True):
            with st.spinner("Aranıyor..."):
                query = build_full_address(addr_text)
                loc = _GEOCODE(query)
                if loc:
                    st.session_state["found_lat"] = loc.latitude
                    st.session_state["found_lon"] = loc.longitude
                    st.session_state["found_name"] = addr_text
                else:
                    st.error("❌ Adres bulunamadı")

        if "found_lat" in st.session_state:
            st.success(f"✅ ({st.session_state.found_lat:.6f}, {st.session_state.found_lon:.6f})")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("➡️ Mevcut", use_container_width=True):
                    st.session_state["curr_lat"] = st.session_state["found_lat"]
                    st.session_state["curr_lon"] = st.session_state["found_lon"]
                    st.session_state["curr_name"] = st.session_state.get("found_name", "Mevcut")
                    st.rerun()
            with c2:
                if st.button("➡️ Aday", use_container_width=True):
                    st.session_state["cand_lat"] = st.session_state["found_lat"]
                    st.session_state["cand_lon"] = st.session_state["found_lon"]
                    st.session_state["cand_name"] = st.session_state.get("found_name", "Aday")
                    st.rerun()

    st.divider()

    st.markdown("### 📂 Veri Kaynağı")
    source_method = st.radio("Yöntem:", ["📊 Google Link", "📁 Dosya Yükle"], label_visibility="collapsed")

    if source_method == "📁 Dosya Yükle":
        uploaded = st.file_uploader("Excel/CSV dosyası seç", type=["xlsx", "xls", "csv"])
        if uploaded:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    st.session_state["loaded_df"] = pd.read_csv(uploaded)
                else:
                    st.session_state["loaded_df"] = pd.read_excel(uploaded, engine="openpyxl")
                st.session_state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("✅ Veri yüklendi")
            except Exception as e:
                st.error(f"❌ Hata: {e}")
    else:
        link = st.text_input("Google Drive/Sheets linki", placeholder="Herkese açık link yapıştır")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📥 Çek", disabled=not link.strip(), use_container_width=True):
                try:
                    with st.spinner("İndiriliyor..."):
                        df_tmp = load_data_from_google_link(link.strip())
                        st.session_state["loaded_df"] = df_tmp
                        st.session_state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("✅ Alındı")
                except Exception as e:
                    st.error(f"❌ {e}")
        with c2:
            if st.button("🔄 Yenile", disabled=not link.strip(), use_container_width=True):
                st.cache_data.clear()
                try:
                    with st.spinner("Yenileniyor..."):
                        df_tmp = load_data_from_google_link(link.strip())
                        st.session_state["loaded_df"] = df_tmp
                        st.session_state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("✅ Yenilendi")
                except Exception as e:
                    st.error(f"❌ {e}")

    if st.session_state["last_updated"]:
        st.caption(f"🕐 Son güncelleme: {st.session_state['last_updated']}")

    st.divider()

    st.markdown("### 📍 Okul Koordinatları")

    with st.expander("🏢 Mevcut Okul", expanded=True):
        st.session_state["curr_name"] = st.text_input("Okul Adı", value=st.session_state["curr_name"], key="curr_name_input")
        st.session_state["curr_lat"] = st.number_input("Enlem", value=float(st.session_state["curr_lat"]), format="%.6f", key="curr_lat_input")
        st.session_state["curr_lon"] = st.number_input("Boylam", value=float(st.session_state["curr_lon"]), format="%.6f", key="curr_lon_input")

    with st.expander("🌟 Aday Okul", expanded=True):
        st.session_state["cand_name"] = st.text_input("Okul Adı", value=st.session_state["cand_name"], key="cand_name_input")
        st.session_state["cand_lat"] = st.number_input("Enlem", value=float(st.session_state["cand_lat"]), format="%.6f", key="cand_lat_input")
        st.session_state["cand_lon"] = st.number_input("Boylam", value=float(st.session_state["cand_lon"]), format="%.6f", key="cand_lon_input")


# ==================== MAIN CONTENT ====================
if st.session_state["loaded_df"] is None:
    st.info("👈 **Başlamak için** sol menüden veri yükleyin")
    st.markdown(
        """
### 🚀 Hızlı Başlangıç
1. Sidebar'dan veri kaynağı seçin
2. Google Sheets/Drive linki veya dosya yükleyin
3. Okul koordinatlarını girin
4. Analiz sonuçlarını görüntüleyin
"""
    )
    st.stop()

raw_df = st.session_state["loaded_df"]

# ==================== VERİ İŞLEME ====================
with st.spinner("⏳ Veriler işleniyor..."):
    full_df = process_data_unique_geocode(raw_df)

ok_cnt = int(full_df["Enlem"].notna().sum())
total_cnt = len(full_df)
success_rate = (ok_cnt / total_cnt * 100) if total_cnt else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("📊 Toplam Kayıt", total_cnt)
with c2:
    st.metric("✅ Başarılı", ok_cnt)
with c3:
    st.metric("❌ Başarısız", total_cnt - ok_cnt)
with c4:
    st.metric("📈 Başarı Oranı", f"%{success_rate:.1f}")

if ok_cnt < total_cnt:
    st.warning(f"⚠️ {total_cnt - ok_cnt} adres çözümlenemedi ve haritada gösterilmeyecek")

st.divider()

# ==================== ANALİZ AYARLARI ====================
st.markdown("### ⚙️ Analiz Ayarları")

c1, c2, c3 = st.columns([2, 2, 3])

with c1:
    analysis_mode = st.selectbox(
        "📏 Uzaklık Modu",
        ["🦅 Kuş Uçuşu", "🚗 Tahmini Karayolu", "🏙️ Manhattan"],
        help="Mesafe hesaplama yöntemi",
    )
    st.session_state["analysis_mode"] = analysis_mode

with c2:
    if analysis_mode == "🚗 Tahmini Karayolu":
        st.session_state["road_factor"] = st.slider(
            "🛣️ Yol Katsayısı",
            1.1,
            2.5,
            float(st.session_state["road_factor"]),
            0.05,
            help="Yolların kıvrımlılığı (1.4 = %40 daha uzun)",
        )

with c3:
    st.markdown("#### 🧹 Aykırı Değerler")

    # Kullanıcı isterse aykırı değerleri DAHİL edebilir (temizleme kapalı)
    ca, cb = st.columns(2)

    with ca:
        radius_mode = st.radio(
            "🎯 Akıllı Yarıçap (Smart Radius)",
            ["Aykırıları hariç tut (önerilen)", "Aykırıları dahil et"],
            index=0 if st.session_state["opt_outlier_radius"] else 1,
        )

    with cb:
        ideal_mode = st.radio(
            "🎯 İdeal Konum (Ağırlık Merkezi)",
            ["Aykırıları hariç tut (önerilen)", "Aykırıları dahil et"],
            index=0 if st.session_state["opt_outlier_ideal"] else 1,
        )

    if st.button("✅ Aykırı Değer Ayarlarını Uygula", use_container_width=True):
        st.session_state["opt_outlier_radius"] = (radius_mode.startswith("Aykırıları hariç"))
        st.session_state["opt_outlier_ideal"] = (ideal_mode.startswith("Aykırıları hariç"))
        st.rerun()

# Mesafe hesaplama
full_df = calculate_distances(
    full_df,
    st.session_state["curr_lat"],
    st.session_state["curr_lon"],
    st.session_state["cand_lat"],
    st.session_state["cand_lon"],
    st.session_state["analysis_mode"],
    road_factor=float(st.session_state["road_factor"]),
)

suggested = calculate_smart_radius_mean(
    full_df,
    basis_col="Mesafe_Mevcut",
    remove_outliers=st.session_state["opt_outlier_radius"],
)

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("🤖 Önerilen Yarıçapı Uygula", use_container_width=True):
        st.session_state["radius_slider"] = suggested
        st.rerun()
with c2:
    st.info(f"💡 **Önerilen yarıçap:** {suggested} km (Öğrenci dağılımına göre hesaplandı)")

st.session_state["radius_slider"] = st.slider(
    "📐 Analiz Yarıçapı (km)",
    0.5,
    15.0,
    float(st.session_state["radius_slider"]),
    0.5,
    help="Bu yarıçap içindeki öğrenciler 'erişebilir' sayılır",
)

radius = float(st.session_state["radius_slider"])

st.divider()

# ==================== FİLTRELEME ====================
st.markdown("### 🔍 Veri Filtreleme")

dept_list = ["🎯 TÜM BÖLÜMLER"]
if "Alan / Dal" in full_df.columns:
    dept_list += sorted(full_df["Alan / Dal"].dropna().unique().tolist())

selected_dept = st.selectbox("Bölüm Seç", dept_list, help="Analizi belirli bir bölüme daraltın")

if selected_dept == "🎯 TÜM BÖLÜMLER":
    df_display = full_df.copy()
else:
    df_display = full_df[full_df["Alan / Dal"] == selected_dept].copy()

st.caption(f"📌 **{len(df_display)}** kayıt görüntüleniyor")

df_map = df_display.dropna(subset=["Enlem", "Boylam"]).copy()

st.divider()

# ==================== HARİTALAR ====================
st.markdown("### 🗺️ İnteraktif Haritalar")

mid_lat = (st.session_state["curr_lat"] + st.session_state["cand_lat"]) / 2
mid_lon = (st.session_state["curr_lon"] + st.session_state["cand_lon"]) / 2

# Bu değişkenler raporlama bölümünde de kullanılır
m = None
m_heat = None
heat_data = []
ideal_coords = None
ideal_note = ""

# İdeal konum kontrolü (her iki harita için ortak)
show_ideal = st.checkbox("🎯 İdeal konumu haritalarda göster", value=st.session_state["show_ideal"])
st.session_state["show_ideal"] = show_ideal

if not df_map.empty:
    ideal_coords = calculate_center_of_gravity(df_map, remove_outliers=st.session_state["opt_outlier_ideal"])
    ideal_note = "Aykırılar hariç" if st.session_state["opt_outlier_ideal"] else "Tümü dahil"

# --------- 1) Noktasal Harita (üstte) ---------
st.markdown("#### 📍 Noktasal Harita")

m = folium.Map(location=[mid_lat, mid_lon], zoom_start=12, tiles="CartoDB positron")
mc = MarkerCluster(name="Öğrenciler").add_to(m)

used_colors = {}
if "Alan / Dal" not in df_map.columns:
    df_map["Alan / Dal"] = "Diğer"

for _, row in df_map.iterrows():
    dept = row.get("Alan / Dal", "Diğer") or "Diğer"
    color = DEPARTMENT_COLORS.get(dept, DEPARTMENT_COLORS["Diğer"])
    used_colors[dept] = color
    folium.CircleMarker(
        [float(row["Enlem"]), float(row["Boylam"])],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=f"<b>{dept}</b><br>{row.get('Ilce_Temiz','')} / {row.get('Mahalle_Temiz','')}",
        tooltip=f"{dept}",
    ).add_to(mc)

# Okullar
folium.Marker(
    [st.session_state["curr_lat"], st.session_state["curr_lon"]],
    tooltip=st.session_state["curr_name"],
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(m)
folium.Circle(
    [st.session_state["curr_lat"], st.session_state["curr_lon"]],
    radius=radius * 1000,
    color="black",
    fill=False,
    weight=2,
    dash_array="5,5",
).add_to(m)

folium.Marker(
    [st.session_state["cand_lat"], st.session_state["cand_lon"]],
    tooltip=st.session_state["cand_name"],
    icon=folium.Icon(color="green", icon="star", prefix="fa"),
).add_to(m)
folium.Circle(
    [st.session_state["cand_lat"], st.session_state["cand_lon"]],
    radius=radius * 1000,
    color="green",
    fill=False,
    weight=2,
    dash_array="5,5",
).add_to(m)

# İdeal konum
if ideal_coords and show_ideal:
    st.caption(f"🎯 İdeal Konum ({ideal_note}): {ideal_coords[0]:.6f}, {ideal_coords[1]:.6f}")
    folium.CircleMarker(
        location=list(ideal_coords),
        radius=10,
        color="#1f77b4",
        fill=True,
        fill_opacity=1,
        tooltip=f"İdeal Konum ({ideal_note})",
        popup=f"<b>İdeal Konum</b><br>{ideal_coords[0]:.6f}, {ideal_coords[1]:.6f}",
    ).add_to(m)

add_custom_legend(
    m,
    title="📊 Gösterge Paneli",
    colors=used_colors,
    curr_name=st.session_state["curr_name"],
    cand_name=st.session_state["cand_name"],
    ideal_shown=(show_ideal and ideal_coords is not None),
    ideal_note=ideal_note,
)

st_folium(m, width=None, height=720, returned_objects=[])

st.divider()

# --------- 2) Isı Haritası (altta) ---------
st.markdown("#### 🔥 Isı Haritası")

heat_data = (
    df_map[["Enlem", "Boylam"]]
    .dropna()
    .astype(float)
    .values
    .tolist()
)

if heat_data:
    m_heat = folium.Map(location=[mid_lat, mid_lon], zoom_start=12, tiles="CartoDB positron")
    HeatMap(
        heat_data,
        radius=18,
        blur=12,
        min_opacity=0.5,
        gradient={0.0: "#3498db", 0.5: "#f39c12", 1.0: "#e74c3c"},
    ).add_to(m_heat)

    folium.Marker(
        [st.session_state["curr_lat"], st.session_state["curr_lon"]],
        tooltip=st.session_state["curr_name"],
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m_heat)
    folium.Marker(
        [st.session_state["cand_lat"], st.session_state["cand_lon"]],
        tooltip=st.session_state["cand_name"],
        icon=folium.Icon(color="green", icon="star", prefix="fa"),
    ).add_to(m_heat)

    if ideal_coords and show_ideal:
        folium.CircleMarker(
            list(ideal_coords),
            radius=10,
            color="#1f77b4",
            fill=True,
            fill_opacity=1,
            tooltip=f"İdeal Konum ({ideal_note})",
        ).add_to(m_heat)

    st_folium(m_heat, width=None, height=720, returned_objects=[])
else:
    st.info("ℹ️ Isı haritası için yeterli koordinat bulunamadı (Enlem/Boylam boş).")
st.divider()

# ==================== KARŞILAŞTIRMA TABLOLARI ====================
st.markdown(f"### 📊 Erişim Analizi ({radius:.2f} km)")

tab1, tab2, tab3 = st.tabs(["👫 Cinsiyet Analizi", "🎓 Sınıf Analizi", "🛠️ Bölüm Analizi"])

with tab1:
    tbl = create_comparison_table(df_display, "Cinsiyetiniz", radius)
    if not tbl.empty:
        st.dataframe(tbl, width="stretch", hide_index=True)
    else:
        st.info("Cinsiyet kolonu bulunamadı")

with tab2:
    tbl = create_comparison_table(df_display, "Sınıf Seviyesi", radius)
    if not tbl.empty:
        st.dataframe(tbl, width="stretch", hide_index=True)
    else:
        st.info("Sınıf kolonu bulunamadı")

with tab3:
    tbl = create_comparison_table(df_display, "Alan / Dal", radius)
    if not tbl.empty:
        st.dataframe(tbl, width="stretch", hide_index=True)
    else:
        st.info("Bölüm kolonu bulunamadı")

st.divider()

# ==================== RAPORLAMA ====================
st.markdown("### 📥 Rapor İndirme")

c1, c2 = st.columns(2)

with c1:
    st.markdown("#### 🗺️ Haritalar")
    if m is not None:
        map_html = m.get_root().render().encode("utf-8")
        st.download_button(
            "📍 Noktasal Harita (.html)",
            data=map_html,
            file_name=f"harita_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True,
        )

    if heat_data and m_heat is not None:
        heat_html = m_heat.get_root().render().encode("utf-8")
        st.download_button(
            "🔥 Isı Haritası (.html)",
            data=heat_html,
            file_name=f"isi_haritasi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True,
        )

with c2:
    st.markdown("#### 📊 Excel Raporu")
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df_display.to_excel(writer, index=False, sheet_name="Filtreli Veri")

        summary = pd.DataFrame(
            {
                "Metrik": [
                    "Toplam Kayıt",
                    "Geocode Başarılı",
                    "Geocode Başarısız",
                    "Başarı Oranı (%)",
                    "Analiz Yarıçapı (km)",
                    "Analiz Modu",
                    "Önerilen Yarıçap (km)",
                    "Analiz Tarihi",
                ],
                "Değer": [
                    len(df_display),
                    int(df_display["Enlem"].notna().sum()),
                    int(df_display["Enlem"].isna().sum()),
                    f"{success_rate:.1f}",
                    radius,
                    st.session_state["analysis_mode"],
                    suggested,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ],
            }
        )
        summary.to_excel(writer, index=False, sheet_name="Özet")

        create_comparison_table(df_display, "Cinsiyetiniz", radius).to_excel(writer, index=False, sheet_name="Cinsiyet")
        create_comparison_table(df_display, "Sınıf Seviyesi", radius).to_excel(writer, index=False, sheet_name="Sınıf")
        create_comparison_table(df_display, "Alan / Dal", radius).to_excel(writer, index=False, sheet_name="Bölüm")

    st.download_button(
        "📊 Tam Rapor İndir (.xlsx)",
        data=excel_buffer.getvalue(),
        file_name=f"rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.divider()
st.caption("🏫 Okul Konum Analiz Pro v2.0 | Gelişmiş Coğrafi Analiz Sistemi")
