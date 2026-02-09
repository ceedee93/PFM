import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta, time
import calendar
from io import BytesIO
from copy import deepcopy
import json
import re
import hashlib

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Portfolio Szenario Tool – Strom & Gas",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 12px 16px;
        color: white;
    }
    div[data-testid="stMetric"] label { color: #a0aec0 !important; font-size: 0.85rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e; border-radius: 8px 8px 0 0;
        padding: 6px 14px; color: #a0aec0; font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] { background: #0f3460 !important; color: white !important; }
    .alert-box { padding: 10px; border-radius: 8px; margin: 5px 0; }
    .alert-green { background: #1a4731; border: 1px solid #48bb78; color: #c6f6d5; }
    .alert-red { background: #4a1a1a; border: 1px solid #fc8181; color: #fed7d7; }
    .alert-yellow { background: #4a3a1a; border: 1px solid #ecc94b; color: #fefcbf; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
PEAK_START = 8
PEAK_END = 20
GAS_DAY_START = 6
QUARTER_MONTHS = {1: [1,2,3], 2: [4,5,6], 3: [7,8,9], 4: [10,11,12]}
SUMMER_MONTHS = [4,5,6,7,8,9]
WINTER_MONTHS = [10,11,12,1,2,3]

# ═══════════════════════════════════════════════════════════════
# AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════
def log_action(action: str, details: str = ""):
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []
    st.session_state.audit_log.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "details": details[:200],
    })

# ═══════════════════════════════════════════════════════════════
# HOLIDAYS & DST
# ═══════════════════════════════════════════════════════════════
def easter_date(year):
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)

def get_feiertage(year):
    e = easter_date(year)
    holidays = {
        date(year,1,1), e-timedelta(2), e, e+timedelta(1),
        date(year,5,1), e+timedelta(39), e+timedelta(49), e+timedelta(50),
        date(year,10,3), date(year,12,25), date(year,12,26),
    }
    # Reformationstag (bundesweit seit 2017 in einigen Ländern, hier optional)
    return holidays

def get_dst_transitions(year):
    mar31 = date(year, 3, 31)
    spring = mar31 - timedelta(days=(mar31.weekday()+1)%7)
    oct31 = date(year, 10, 31)
    fall = oct31 - timedelta(days=(oct31.weekday()+1)%7)
    return {"spring": spring, "fall": fall}

def get_hours_in_day(d, year):
    dst = get_dst_transitions(year)
    if d == dst["spring"]: return 23
    if d == dst["fall"]: return 25
    return 24

def get_hours_month(year, month):
    feiertage = get_feiertage(year)
    days = calendar.monthrange(year, month)[1]
    base_h = peak_h = 0
    for day in range(1, days+1):
        d = date(year, month, day)
        dh = get_hours_in_day(d, year)
        base_h += dh
        if d.weekday() < 5 and d not in feiertage:
            peak_h += (PEAK_END - PEAK_START)
    return {"base": base_h, "peak": peak_h, "offpeak": base_h - peak_h, "days": days}

def get_quarter_hours(year, q):
    t = {"base":0,"peak":0,"offpeak":0,"days":0}
    for m in QUARTER_MONTHS[q]:
        h = get_hours_month(year, m)
        for k in t: t[k] += h[k]
    return t

# ═══════════════════════════════════════════════════════════════
# TIMESERIES ALIGNMENT
# ═══════════════════════════════════════════════════════════════
class TimeSeriesAligner:
    @staticmethod
    def detect_alignment(df, ts_col=None, commodity="Strom"):
        result = {"alignment":"unknown","resolution":"unknown",
                  "gas_day_start":GAS_DAY_START,"ts_col":ts_col,"needs_conversion":False}
        if df is None or len(df)==0: return result
        if ts_col is None:
            candidates = [c for c in df.columns if any(
                kw in c.lower() for kw in ["zeit","time","timestamp","datum","date","von","bis"])]
            ts_col = candidates[0] if candidates else df.columns[0]
        result["ts_col"] = ts_col
        try:
            ts = pd.to_datetime(df[ts_col], dayfirst=True, errors="coerce").dropna()
        except: return result
        if len(ts)<2: return result

        median_diff = ts.diff().dropna().median()
        if median_diff <= pd.Timedelta(minutes=20): result["resolution"]="15min"
        elif median_diff <= pd.Timedelta(minutes=75): result["resolution"]="60min"
        elif median_diff <= pd.Timedelta(hours=25): result["resolution"]="daily"
        else: result["resolution"]="other"

        first = ts.iloc[0]
        if result["resolution"] in ("15min","60min"):
            if first.hour==0 and first.minute==0: result["alignment"]="left"
            elif first.hour==0 and first.minute==15: result["alignment"]="right"; result["needs_conversion"]=True
            elif first.hour==1 and first.minute==0: result["alignment"]="right"; result["needs_conversion"]=True
            elif first.hour in (6,7) and first.minute==0:
                result["alignment"]="left"; result["gas_day_start"]=first.hour
            elif first.hour in (6,7) and first.minute==15:
                result["alignment"]="right"; result["gas_day_start"]=first.hour
                result["needs_conversion"]=True
            else: result["alignment"]="left"
        elif result["resolution"]=="daily":
            result["alignment"]="left"
            if first.hour in (6,7): result["gas_day_start"]=first.hour
        return result

    @staticmethod
    def normalize(df, info):
        if not info.get("needs_conversion"): return df
        r = df.copy()
        tc = info["ts_col"]; res = info["resolution"]
        try: r[tc] = pd.to_datetime(r[tc], dayfirst=True, errors="coerce")
        except: return r
        shift = {"15min": pd.Timedelta(minutes=15), "60min": pd.Timedelta(hours=1),
                 "daily": pd.Timedelta(days=1)}
        if res in shift: r[tc] -= shift[res]
        return r

# ═══════════════════════════════════════════════════════════════
# FILE READER
# ═══════════════════════════════════════════════════════════════
class FileReader:
    @staticmethod
    def read(f):
        if f is None: return None
        name = f.name.lower()
        if name.endswith((".xlsx",".xls")): return FileReader._excel(f)
        if name.endswith(".csv"): return FileReader._csv(f)
        st.error("❌ Nur CSV/XLSX"); return None

    @staticmethod
    def _excel(f):
        try:
            df = pd.read_excel(f, engine="openpyxl")
            return FileReader._clean(df)
        except Exception as e: st.error(f"❌ Excel: {e}"); return None

    @staticmethod
    def _csv(f):
        try:
            content = f.read(); f.seek(0)
            for enc in ["utf-8","latin-1","cp1252","iso-8859-1"]:
                try: text = content.decode(enc); break
                except: continue
            else: text = content.decode("utf-8", errors="replace"); enc="utf-8"
            header = text.split("\n")[0] if text else ""
            counts = {";":header.count(";"), ",":header.count(","),
                      "\t":header.count("\t"), "|":header.count("|")}
            sep = max(counts, key=counts.get) if max(counts.values())>0 else ";"
            dec = "," if sep==";" else "."
            f.seek(0)
            try: df = pd.read_csv(f, sep=sep, decimal=dec, encoding=enc)
            except: f.seek(0); df = pd.read_csv(f, sep=sep, decimal=".", encoding=enc)
            if len(df.columns)<=1 and sep!=",":
                f.seek(0); df = pd.read_csv(f, sep=",", decimal=".", encoding=enc)
            return FileReader._clean(df)
        except Exception as e: st.error(f"❌ CSV: {e}"); return None

    @staticmethod
    def _clean(df):
        if df is None: return None
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all").reset_index(drop=True)
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    test = df[col].astype(str).str.replace(",",".").str.strip()
                    num = pd.to_numeric(test, errors="coerce")
                    if num.notna().sum() > len(df)*0.5:
                        if not any(kw in col.lower() for kw in ["dat","zeit","time","stamp","von","bis"]):
                            df[col] = num
                except: pass
        return df

# ═══════════════════════════════════════════════════════════════
# HPFC PROCESSOR
# ═══════════════════════════════════════════════════════════════
class HPFCProcessor:
    @staticmethod
    def process(df, ts_col, price_col, year, commodity="Strom"):
        r = df.copy()
        r[ts_col] = pd.to_datetime(r[ts_col], dayfirst=True, errors="coerce")
        r[price_col] = pd.to_numeric(r[price_col], errors="coerce")
        r = r.dropna(subset=[ts_col, price_col])
        r = r[r[ts_col].dt.year == year]
        if len(r)==0: return None
        r["month"] = r[ts_col].dt.month
        r["hour"] = r[ts_col].dt.hour
        r["date"] = r[ts_col].dt.date
        feiertage = get_feiertage(year)
        r["is_peak"] = r.apply(lambda x: x["date"] not in feiertage and
            pd.Timestamp(x["date"]).weekday()<5 and PEAK_START<=x["hour"]<PEAK_END, axis=1)
        rows = []
        for m in range(1,13):
            md = r[r["month"]==m]
            if len(md)==0: continue
            hi = get_hours_month(year, m)
            ml = f"M{m:02d}-{year}"
            q = (m-1)//3+1
            base_p = md[price_col].mean()
            rows.append({"Produkt":ml,"Typ":"Monat","Commodity":commodity,
                "Lasttyp":"Base","Jahr":year,"Monat":m,"Quartal":q,
                "Stunden":hi["base"],"Preis_EUR_MWh":round(base_p,2)})
            if commodity=="Strom":
                pk = md[md["is_peak"]]
                pp = pk[price_col].mean() if len(pk)>0 else base_p*1.15
                rows.append({"Produkt":ml,"Typ":"Monat","Commodity":commodity,
                    "Lasttyp":"Peak","Jahr":year,"Monat":m,"Quartal":q,
                    "Stunden":hi["peak"],"Preis_EUR_MWh":round(pp,2)})
                op = md[~md["is_peak"]]
                opp = op[price_col].mean() if len(op)>0 else base_p*0.85
                rows.append({"Produkt":ml,"Typ":"Monat","Commodity":commodity,
                    "Lasttyp":"Offpeak","Jahr":year,"Monat":m,"Quartal":q,
                    "Stunden":hi["offpeak"],"Preis_EUR_MWh":round(opp,2)})
        return pd.DataFrame(rows) if rows else None

    # ═══════════════════════════════════════════════════════════════
    # PRODUCT STRUCTURE
    # ═══════════════════════════════════════════════════════════════
    def get_product_structure(year, commodity):
        rows = []
        lts = ["Base", "Peak", "Offpeak"] if commodity == "Strom" else ["Base"]
        for m in range(1, 13):
            h = get_hours_month(year, m)
            ml = f"M{m:02d}-{year}";
            q = (m - 1) // 3 + 1
            for lt in lts:
                rows.append({"Produkt": ml, "Typ": "Monat", "Commodity": commodity,
                             "Lasttyp": lt, "Jahr": year, "Monat": m, "Quartal": q,
                             "Stunden": h[lt.lower()], "Preis_EUR_MWh": 0.0})
        for q in range(1, 5):
            qh = get_quarter_hours(year, q)
            for lt in lts:
                rows.append({"Produkt": f"Q{q}-{year}", "Typ": "Quartal", "Commodity": commodity,
                             "Lasttyp": lt, "Jahr": year, "Monat": None, "Quartal": q,
                             "Stunden": qh[lt.lower()], "Preis_EUR_MWh": 0.0})
        if commodity == "Gas":
            sh = sum(get_hours_month(year, m)["base"] for m in SUMMER_MONTHS)
            rows.append({"Produkt": f"Summer-{year}", "Typ": "Season", "Commodity": "Gas",
                         "Lasttyp": "Base", "Jahr": year, "Monat": None, "Quartal": None, "Stunden": sh,
                         "Preis_EUR_MWh": 0.0})
            wh = sum(get_hours_month(year, m)["base"] for m in [10, 11, 12]) + \
                 sum(get_hours_month(year + 1, m)["base"] for m in [1, 2, 3])
            rows.append({"Produkt": f"Winter-{year}/{year + 1}", "Typ": "Season", "Commodity": "Gas",
                         "Lasttyp": "Base", "Jahr": year, "Monat": None, "Quartal": None, "Stunden": wh,
                         "Preis_EUR_MWh": 0.0})
        for lt in lts:
            ym = [r for r in rows if r["Typ"] == "Monat" and r["Lasttyp"] == lt]
            th = sum(r["Stunden"] for r in ym)
            rows.append({"Produkt": f"Cal-{year}", "Typ": "Jahr", "Commodity": commodity,
                         "Lasttyp": lt, "Jahr": year, "Monat": None, "Quartal": None, "Stunden": th,
                         "Preis_EUR_MWh": 0.0})
        return pd.DataFrame(rows)

    def build_default_pfc(year, commodity, base_price=80.0, peak_price=95.0):
        s = get_product_structure(year, commodity)
        # Saisonale Variation
        seasonal = {1: 1.12, 2: 1.08, 3: 0.98, 4: 0.92, 5: 0.88, 6: 0.85,
                    7: 0.87, 8: 0.90, 9: 0.95, 10: 1.02, 11: 1.08, 12: 1.15}
        for idx, row in s.iterrows():
            m = row.get("Monat")
            factor = seasonal.get(m, 1.0) if m else 1.0
            if row["Lasttyp"] == "Peak":
                s.at[idx, "Preis_EUR_MWh"] = round(peak_price * factor, 2)
            elif row["Lasttyp"] == "Offpeak":
                hi = get_hours_month(year, m) if m else {"base": 1, "peak": 1, "offpeak": 1}
                if hi["offpeak"] > 0:
                    op = (base_price * factor * hi["base"] - peak_price * factor * hi["peak"]) / hi["offpeak"]
                    s.at[idx, "Preis_EUR_MWh"] = round(max(op, 0), 2)
                else:
                    s.at[idx, "Preis_EUR_MWh"] = round(base_price * factor * 0.7, 2)
            else:
                s.at[idx, "Preis_EUR_MWh"] = round(base_price * factor, 2)
        return recalculate_aggregates(s)

    def recalculate_aggregates(pfc):
        r = pfc.copy()
        months = r[r["Typ"] == "Monat"]
        commodity = r["Commodity"].iloc[0] if "Commodity" in r.columns else "Strom"
        for lt in months["Lasttyp"].unique():
            ltm = months[months["Lasttyp"] == lt]
            if len(ltm) == 0 or ltm["Stunden"].sum() == 0: continue
            for q in range(1, 5):
                qm = ltm[ltm["Quartal"] == q]
                if len(qm) > 0 and qm["Stunden"].sum() > 0:
                    wp = (qm["Preis_EUR_MWh"] * qm["Stunden"]).sum() / qm["Stunden"].sum()
                    mask = (r["Typ"] == "Quartal") & (r["Quartal"] == q) & (r["Lasttyp"] == lt)
                    r.loc[mask, "Preis_EUR_MWh"] = round(wp, 4)
            twh = (ltm["Preis_EUR_MWh"] * ltm["Stunden"]).sum()
            th = ltm["Stunden"].sum()
            if th > 0: r.loc[(r["Typ"] == "Jahr") & (r["Lasttyp"] == lt), "Preis_EUR_MWh"] = round(twh / th, 4)
            if commodity == "Gas" and lt == "Base":
                for season, qlist in [("Summer", [2, 3]), ("Winter", [4])]:
                    sm = ltm[ltm["Quartal"].isin(qlist)]
                    if len(sm) > 0 and sm["Stunden"].sum() > 0:
                        sp = (sm["Preis_EUR_MWh"] * sm["Stunden"]).sum() / sm["Stunden"].sum()
                        r.loc[(r["Typ"] == "Season") & (r["Produkt"].str.contains(season)), "Preis_EUR_MWh"] = round(sp,
                                                                                                                     4)
        return r

    def recalculate_offpeak(pfc):
        r = pfc.copy()
        months = r[r["Typ"] == "Monat"]
        for m in months["Monat"].dropna().unique():
            md = months[months["Monat"] == m]
            b = md[md["Lasttyp"] == "Base"];
            p = md[md["Lasttyp"] == "Peak"];
            o = md[md["Lasttyp"] == "Offpeak"]
            if len(b) > 0 and len(p) > 0 and len(o) > 0:
                bp, bh = b.iloc[0]["Preis_EUR_MWh"], b.iloc[0]["Stunden"]
                pp, ph = p.iloc[0]["Preis_EUR_MWh"], p.iloc[0]["Stunden"]
                oh = o.iloc[0]["Stunden"]
                if oh > 0:
                    op = (bp * bh - pp * ph) / oh
                    mask = (r["Typ"] == "Monat") & (r["Monat"] == m) & (r["Lasttyp"] == "Offpeak")
                    r.loc[mask, "Preis_EUR_MWh"] = round(max(op, 0), 4)
        return r

    def validate_pfc(pfc):
        issues = []
        if pfc is None or len(pfc) == 0: return ["PFC leer"]
        months = pfc[pfc["Typ"] == "Monat"]
        for m in range(1, 13):
            if len(months[months["Monat"] == m]) == 0: issues.append(f"⚠️ Monat {m} fehlt")
        zeros = pfc[pfc["Preis_EUR_MWh"] == 0]
        if len(zeros) > 0: issues.append(f"⚠️ {len(zeros)} Produkte mit Preis=0")
        negs = pfc[pfc["Preis_EUR_MWh"] < 0]
        if len(negs) > 0: issues.append(f"⚠️ {len(negs)} negative Preise")
        # Base/Peak/Offpeak Konsistenz
        for m in range(1, 13):
            md = months[months["Monat"] == m]
            b = md[md["Lasttyp"] == "Base"];
            p = md[md["Lasttyp"] == "Peak"];
            o = md[md["Lasttyp"] == "Offpeak"]
            if len(b) > 0 and len(p) > 0 and len(o) > 0:
                bp, bh = b.iloc[0]["Preis_EUR_MWh"], b.iloc[0]["Stunden"]
                pp, ph = p.iloc[0]["Preis_EUR_MWh"], p.iloc[0]["Stunden"]
                op, oh = o.iloc[0]["Preis_EUR_MWh"], o.iloc[0]["Stunden"]
                if bh > 0:
                    implied = (pp * ph + op * oh) / bh
                    if abs(bp - implied) > 0.5:
                        issues.append(f"⚠️ M{m:02d}: Base≠Ø(Peak/Offpeak), Δ={abs(bp - implied):.2f}")
        return issues

    # ═══════════════════════════════════════════════════════════════
    # POSITION EXPANSION
    # ═══════════════════════════════════════════════════════════════
    def expand_to_months(positions, year):
        if positions is None or len(positions) == 0:
            return pd.DataFrame(columns=["Produkt", "Lasttyp", "Menge_MW", "Preis_EUR_MWh", "Original_Produkt"])
        expanded = []
        for _, pos in positions.iterrows():
            produkt = str(pos.get("Produkt", "")).strip()
            typ = str(pos.get("Typ", "")).strip()
            lasttyp = str(pos.get("Lasttyp", "Base")).strip()
            menge = float(pos.get("Menge_MW", 0))
            preis = float(pos.get("Preis_EUR_MWh", 0))
            targets = []
            if typ == "Monat" or produkt.startswith("M"):
                targets = [produkt]
            elif typ == "Quartal" or produkt.startswith("Q"):
                for qi in range(1, 5):
                    if f"Q{qi}" in produkt: targets = [f"M{m:02d}-{year}" for m in QUARTER_MONTHS[qi]]; break
            elif typ == "Jahr" or produkt.startswith("Cal"):
                targets = [f"M{m:02d}-{year}" for m in range(1, 13)]
            elif "Summer" in produkt or "Sommer" in produkt:
                targets = [f"M{m:02d}-{year}" for m in SUMMER_MONTHS]
            elif "Winter" in produkt:
                targets = [f"M{m:02d}-{year}" for m in WINTER_MONTHS]
            for mt in targets:
                expanded.append({"Produkt": mt, "Lasttyp": lasttyp, "Menge_MW": menge,
                                 "Preis_EUR_MWh": preis, "Original_Produkt": produkt})
        return pd.DataFrame(expanded) if expanded else pd.DataFrame(
            columns=["Produkt", "Lasttyp", "Menge_MW", "Preis_EUR_MWh", "Original_Produkt"])

    # ═══════════════════════════════════════════════════════════════
    # PNL ENGINE
    # ═══════════════════════════════════════════════════════════════
    def calculate_portfolio(pfc, beschaffung=None, absatz=None, offene_position=None):
        if pfc is None: return None
        year = int(pfc["Jahr"].dropna().iloc[0])
        result = pfc[pfc["Typ"] == "Monat"].copy().reset_index(drop=True)
        for c in ["Beschaffung_MW", "Beschaffung_Preis", "Absatz_MW", "Absatz_Preis", "Offene_Position_MW"]:
            result[c] = 0.0
        if offene_position is not None and len(offene_position) > 0:
            op_exp = expand_to_months(offene_position, year)
            for idx, row in result.iterrows():
                m = op_exp[(op_exp["Produkt"] == row["Produkt"]) & (op_exp["Lasttyp"] == row["Lasttyp"])]
                if len(m) > 0: result.at[idx, "Offene_Position_MW"] = m["Menge_MW"].sum()
        else:
            if beschaffung is not None and len(beschaffung) > 0:
                be = expand_to_months(beschaffung, year)
                for idx, row in result.iterrows():
                    m = be[(be["Produkt"] == row["Produkt"]) & (be["Lasttyp"] == row["Lasttyp"])]
                    if len(m) > 0:
                        tmw = m["Menge_MW"].sum()
                        result.at[idx, "Beschaffung_MW"] = tmw
                        if tmw != 0: result.at[idx, "Beschaffung_Preis"] = (m["Menge_MW"] * m[
                            "Preis_EUR_MWh"]).sum() / tmw
            if absatz is not None and len(absatz) > 0:
                ae = expand_to_months(absatz, year)
                for idx, row in result.iterrows():
                    m = ae[(ae["Produkt"] == row["Produkt"]) & (ae["Lasttyp"] == row["Lasttyp"])]
                    if len(m) > 0:
                        tmw = m["Menge_MW"].sum()
                        result.at[idx, "Absatz_MW"] = tmw
                        if tmw != 0: result.at[idx, "Absatz_Preis"] = (m["Menge_MW"] * m["Preis_EUR_MWh"]).sum() / tmw
            result["Offene_Position_MW"] = result["Absatz_MW"] - result["Beschaffung_MW"]
        result["Marktpreis"] = result["Preis_EUR_MWh"]
        result["Offene_Position_MWh"] = result["Offene_Position_MW"] * result["Stunden"]
        result["Beschaffung_MWh"] = result["Beschaffung_MW"] * result["Stunden"]
        result["Absatz_MWh"] = result["Absatz_MW"] * result["Stunden"]
        # PnL
        gedeckt = np.minimum(np.abs(result["Beschaffung_MW"]), np.abs(result["Absatz_MW"]))
        result["PnL_realisiert_EUR"] = np.where(
            (result["Beschaffung_MW"] > 0) & (result["Absatz_MW"] > 0),
            gedeckt * result["Stunden"] * (result["Absatz_Preis"] - result["Beschaffung_Preis"]), 0)
        avg_cost = np.where(result["Beschaffung_MW"] > 0, result["Beschaffung_Preis"], result["Marktpreis"])
        result["PnL_unrealisiert_EUR"] = result["Offene_Position_MW"] * result["Stunden"] * (
                    result["Marktpreis"] - avg_cost)
        result["PnL_gesamt_EUR"] = result["PnL_realisiert_EUR"] + result["PnL_unrealisiert_EUR"]
        result["Deckungsgrad_%"] = np.where(result["Absatz_MW"] > 0,
                                            np.clip(result["Beschaffung_MW"] / result["Absatz_MW"] * 100, 0,
                                                    999.9).round(1),
                                            np.where(result["Beschaffung_MW"] > 0, 999.9, 0.0))
        result["MtM_EUR"] = result["Offene_Position_MW"] * result["Stunden"] * result["Marktpreis"]
        return result

    # ═══════════════════════════════════════════════════════════════
    # PNL ATTRIBUTION (Volume, Price, Timing Effects)
    # ═══════════════════════════════════════════════════════════════
    def pnl_attribution(portfolio, benchmark_pfc=None):
        """Zerlege PnL in Volumen-, Preis- und Mix-Effekt."""
        if portfolio is None or len(portfolio) == 0: return None
        attr = portfolio[["Produkt", "Lasttyp", "Monat", "Stunden",
                          "Beschaffung_MW", "Beschaffung_Preis", "Absatz_MW", "Absatz_Preis", "Marktpreis"]].copy()
        # Volumeneffekt: ΔVolumen × Benchmark-Preis
        attr["Volume_Effect_EUR"] = (attr["Absatz_MW"] - attr["Beschaffung_MW"]) * attr["Stunden"] * attr["Marktpreis"]
        # Preiseffekt: Volumen × ΔPreis
        attr["Price_Effect_EUR"] = np.where(attr["Beschaffung_MW"] > 0,
                                            attr["Beschaffung_MW"] * attr["Stunden"] * (
                                                        attr["Marktpreis"] - attr["Beschaffung_Preis"]), 0)
        # Margin-Effekt (Absatz vs Markt)
        attr["Margin_Effect_EUR"] = np.where(attr["Absatz_MW"] > 0,
                                             attr["Absatz_MW"] * attr["Stunden"] * (
                                                         attr["Absatz_Preis"] - attr["Marktpreis"]), 0)
        return attr

    # ═══════════════════════════════════════════════════════════════
    # SCENARIO ENGINE
    # ═══════════════════════════════════════════════════════════════
    def apply_scenario(pfc, shift_pct=0.0, shift_abs=0.0, individual_changes=None):
        sc = pfc.copy()
        if shift_pct != 0: sc["Preis_EUR_MWh"] *= (1 + shift_pct / 100)
        if shift_abs != 0: sc["Preis_EUR_MWh"] += shift_abs
        if individual_changes:
            for (prod, lt), new_p in individual_changes.items():
                mask = (sc["Produkt"] == prod) & (sc["Lasttyp"] == lt)
                if mask.sum() == 0: continue
                old_p = pfc.loc[mask, "Preis_EUR_MWh"].iloc[0]
                sc.loc[mask, "Preis_EUR_MWh"] = new_p
                if old_p > 0:
                    ratio = new_p / old_p
                    row = sc.loc[mask].iloc[0];
                    typ = row["Typ"]
                    if typ == "Jahr":
                        sc.loc[sc["Typ"].isin(["Quartal", "Monat", "Season"]) & (
                                    sc["Lasttyp"] == lt), "Preis_EUR_MWh"] *= ratio
                    elif typ == "Quartal":
                        sc.loc[(sc["Typ"] == "Monat") & (sc["Quartal"] == row["Quartal"]) & (
                                    sc["Lasttyp"] == lt), "Preis_EUR_MWh"] *= ratio
                    elif typ == "Season":
                        ms = SUMMER_MONTHS if "Summer" in prod else WINTER_MONTHS if "Winter" in prod else []
                        sc.loc[(sc["Typ"] == "Monat") & (sc["Monat"].isin(ms)) & (
                                    sc["Lasttyp"] == lt), "Preis_EUR_MWh"] *= ratio
        return recalculate_aggregates(sc)

    # ═══════════════════════════════════════════════════════════════
    # VAR CALCULATION
    # ═══════════════════════════════════════════════════════════════
    def calculate_var(portfolio, confidence=0.95, holding_days=1, volatility_pct=2.0, method="parametric"):
        if portfolio is None or len(portfolio) == 0: return {"VaR": 0, "CVaR": 0}
        exposure = (portfolio["Offene_Position_MW"].abs() * portfolio["Stunden"] * portfolio["Marktpreis"]).sum()
        if exposure == 0: return {"VaR": 0, "CVaR": 0, "exposure": 0}
        dv = volatility_pct / 100
        if method == "parametric":
            z = scipy_stats.norm.ppf(confidence) if scipy_stats else {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}.get(
                confidence, 1.645)
            var = z * dv * np.sqrt(holding_days) * exposure
            cvar = var * 1.25
        else:
            np.random.seed(42)
            rets = np.random.normal(0, dv, (10000, holding_days))
            cum = np.sum(rets, axis=1)
            pnl = exposure * cum
            var = -np.percentile(pnl, (1 - confidence) * 100)
            losses = pnl[pnl < -var]
            cvar = -losses.mean() if len(losses) > 0 else var
        return {"VaR": round(var, 2), "CVaR": round(cvar, 2), "exposure": round(exposure, 2),
                "confidence": confidence, "holding_days": holding_days, "volatility": volatility_pct}

    # ═══════════════════════════════════════════════════════════════
    # TECHNICAL INDICATORS (Bollinger, RSI, MACD, Momentum)
    # ═══════════════════════════════════════════════════════════════
    def calc_bollinger(prices, window=20, num_std=2):
        s = pd.Series(prices)
        ma = s.rolling(window).mean()
        std = s.rolling(window).std()
        return {"ma": ma, "upper": ma + num_std * std, "lower": ma - num_std * std}

    def calc_rsi(prices, period=14):
        s = pd.Series(prices)
        delta = s.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calc_macd(prices, fast=12, slow=26, signal=9):
        s = pd.Series(prices)
        ema_fast = s.ewm(span=fast).mean()
        ema_slow = s.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    def calc_momentum(prices, period=10):
        s = pd.Series(prices)
        return s - s.shift(period)

    # ═══════════════════════════════════════════════════════════════
    # BENCHMARK COMPARISON
    # ═══════════════════════════════════════════════════════════════
    def benchmark_comparison(portfolio, benchmark_pfc):
        """Vergleiche Ist-Beschaffung vs. Benchmark (z.B. EEX Settlement)."""
        if portfolio is None or benchmark_pfc is None: return None
        result = portfolio.copy()
        bm_months = benchmark_pfc[benchmark_pfc["Typ"] == "Monat"]
        result["Benchmark_Preis"] = 0.0
        for idx, row in result.iterrows():
            bm = bm_months[(bm_months["Produkt"] == row["Produkt"]) & (bm_months["Lasttyp"] == row["Lasttyp"])]
            if len(bm) > 0: result.at[idx, "Benchmark_Preis"] = bm.iloc[0]["Preis_EUR_MWh"]
        result["Benchmark_Diff_EUR_MWh"] = result["Beschaffung_Preis"] - result["Benchmark_Preis"]
        result["Benchmark_PnL_EUR"] = np.where(result["Beschaffung_MW"] > 0,
                                               -result["Benchmark_Diff_EUR_MWh"] * result["Beschaffung_MW"] * result[
                                                   "Stunden"], 0)
        return result

    # ═══════════════════════════════════════════════════════════════
    # TRANCHE PLANNING
    # ═══════════════════════════════════════════════════════════════
    def calculate_tranche_status(tranches, target_mw, year, commodity):
        """Berechne Beschaffungsfortschritt aus Tranchenliste."""
        if not tranches: return {"procured_mw": 0, "remaining_mw": target_mw, "avg_price": 0,
                                 "coverage_pct": 0, "tranches": [], "n_tranches": 0}
        total_mw = sum(t["menge_mw"] for t in tranches)
        avg_price = sum(t["menge_mw"] * t["preis"] for t in tranches) / total_mw if total_mw > 0 else 0
        return {"procured_mw": round(total_mw, 2), "remaining_mw": round(target_mw - total_mw, 2),
                "avg_price": round(avg_price, 2),
                "coverage_pct": round(total_mw / target_mw * 100, 1) if target_mw > 0 else 0,
                "tranches": tranches, "n_tranches": len(tranches)}

    # ═══════════════════════════════════════════════════════════════
    # SPREAD ANALYSIS (Spark, Dark, Clean)
    # ═══════════════════════════════════════════════════════════════
    def calc_spreads(strom_base, gas_base, co2_price, efficiency_gas=0.5, efficiency_coal=0.38,
                     co2_factor_gas=0.202, co2_factor_coal=0.337):
        """
        Spark Spread = Strompreis - Gaspreis/η_gas
        Dark Spread = Strompreis - Kohlepreis/η_coal  (Kohle wird aus Gas approx.)
        Clean Spark = Spark - CO2 × EF/η
        """
        spark = strom_base - gas_base / efficiency_gas
        clean_spark = spark - co2_price * co2_factor_gas / efficiency_gas
        # Approximate dark spread (assuming coal ~ gas*0.6 for simplicity)
        coal_approx = gas_base * 0.6
        dark = strom_base - coal_approx / efficiency_coal
        clean_dark = dark - co2_price * co2_factor_coal / efficiency_coal
        return {"spark": round(spark, 2), "clean_spark": round(clean_spark, 2),
                "dark": round(dark, 2), "clean_dark": round(clean_dark, 2)}

    # ═══════════════════════════════════════════════════════════════
    # TRADE SIMULATOR
    # ═══════════════════════════════════════════════════════════════
    def simulate_trade(pfc, portfolio, produkt, lasttyp, menge_mw, preis, richtung, kontrahent=""):
        result = portfolio.copy()
        year = int(pfc["Jahr"].dropna().iloc[0])
        if produkt.startswith("Cal"):
            affected = [f"M{m:02d}-{year}" for m in range(1, 13)]
        elif produkt.startswith("Q"):
            q = int(re.search(r"Q(\d)", produkt).group(1));
            affected = [f"M{m:02d}-{year}" for m in QUARTER_MONTHS[q]]
        elif produkt.startswith("M"):
            affected = [produkt]
        elif "Summer" in produkt:
            affected = [f"M{m:02d}-{year}" for m in SUMMER_MONTHS]
        elif "Winter" in produkt:
            affected = [f"M{m:02d}-{year}" for m in WINTER_MONTHS]
        else:
            return result
        for idx, row in result.iterrows():
            if row["Produkt"] in affected and row["Lasttyp"] == lasttyp:
                if richtung == "Kauf":
                    old_mw = row["Beschaffung_MW"];
                    old_p = row["Beschaffung_Preis"]
                    new_mw = old_mw + menge_mw
                    result.at[idx, "Beschaffung_MW"] = new_mw
                    result.at[idx, "Beschaffung_Preis"] = (
                                (old_mw * old_p + menge_mw * preis) / new_mw) if new_mw else preis
                else:
                    old_mw = row["Absatz_MW"];
                    old_p = row["Absatz_Preis"]
                    new_mw = old_mw + menge_mw
                    result.at[idx, "Absatz_MW"] = new_mw
                    result.at[idx, "Absatz_Preis"] = ((old_mw * old_p + menge_mw * preis) / new_mw) if new_mw else preis
                result.at[idx, "Offene_Position_MW"] = result.at[idx, "Absatz_MW"] - result.at[idx, "Beschaffung_MW"]
        # Recalc PnL
        gedeckt = np.minimum(np.abs(result["Beschaffung_MW"]), np.abs(result["Absatz_MW"]))
        result["PnL_realisiert_EUR"] = np.where((result["Beschaffung_MW"] > 0) & (result["Absatz_MW"] > 0),
                                                gedeckt * result["Stunden"] * (
                                                            result["Absatz_Preis"] - result["Beschaffung_Preis"]), 0)
        avg_cost = np.where(result["Beschaffung_MW"] > 0, result["Beschaffung_Preis"], result["Marktpreis"])
        result["PnL_unrealisiert_EUR"] = result["Offene_Position_MW"] * result["Stunden"] * (
                    result["Marktpreis"] - avg_cost)
        result["PnL_gesamt_EUR"] = result["PnL_realisiert_EUR"] + result["PnL_unrealisiert_EUR"]
        result["Offene_Position_MWh"] = result["Offene_Position_MW"] * result["Stunden"]
        result["Deckungsgrad_%"] = np.where(result["Absatz_MW"] > 0,
                                            np.clip(result["Beschaffung_MW"] / result["Absatz_MW"] * 100, 0,
                                                    999.9).round(1),
                                            np.where(result["Beschaffung_MW"] > 0, 999.9, 0.0))
        return result

    # ═══════════════════════════════════════════════════════════════
    # EXCEL EXPORT (MULTI-SHEET)
    # ═══════════════════════════════════════════════════════════════
    def export_excel(portfolio, pfc, commodity, year, trade_history=None, tranches=None):
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            if portfolio is not None:
                portfolio.to_excel(w, sheet_name="PnL_Detail", index=False)
                summary = pd.DataFrame({"KPI": [
                    "PnL Gesamt", "PnL Realisiert", "PnL Unrealisiert",
                    "Max Offene Pos (MW)", "Ø Deckungsgrad",
                    "Beschaffung MWh", "Absatz MWh", "Offene Pos MWh"],
                    "Wert": [portfolio["PnL_gesamt_EUR"].sum(), portfolio["PnL_realisiert_EUR"].sum(),
                             portfolio["PnL_unrealisiert_EUR"].sum(), portfolio["Offene_Position_MW"].abs().max(),
                             portfolio.loc[portfolio["Absatz_MW"] > 0, "Deckungsgrad_%"].mean() if (
                                         portfolio["Absatz_MW"] > 0).any() else 0,
                             portfolio["Beschaffung_MWh"].sum(), portfolio["Absatz_MWh"].sum(),
                             portfolio["Offene_Position_MWh"].sum()]})
                summary.to_excel(w, sheet_name="Summary", index=False)
            if pfc is not None: pfc.to_excel(w, sheet_name="PFC", index=False)
            if trade_history: pd.DataFrame(trade_history).to_excel(w, sheet_name="Trade_History", index=False)
            if tranches: pd.DataFrame(tranches).to_excel(w, sheet_name="Tranchen", index=False)
        return buf.getvalue()

    # ═══════════════════════════════════════════════════════════════
    #  MAIN APPLICATION
    # ═══════════════════════════════════════════════════════════════
    def main():
        # ─── SESSION STATE ───
        defaults = {
            "pfc": None, "portfolio": None, "commodity": "Strom", "year": 2026,
            "beschaffung": None, "absatz": None, "offene_position": None,
            "trade_history": [], "benchmark_pfc": None, "tranches": [],
            "price_alerts": [], "audit_log": [], "co2_price": 70.0,
            "historical_prices": None, "spot_prices": None,
        }
        for k, v in defaults.items():
            if k not in st.session_state: st.session_state[k] = v

        # ─── SIDEBAR ───
        with st.sidebar:
            st.title("⚡ Portfolio Tool")
            st.markdown("---")
            commodity = st.selectbox("Commodity", ["Strom", "Gas"], key="sb_commodity")
            year = st.number_input("Lieferjahr", 2024, 2035, st.session_state.year, key="sb_year")
            st.session_state.commodity = commodity
            st.session_state.year = year

            st.markdown("---")
            st.markdown("### 📊 Status")
            pfc = st.session_state.pfc
            port = st.session_state.portfolio
            if pfc is not None:
                issues = validate_pfc(pfc)
                if issues:
                    st.markdown(f'<span style="color:#ecc94b">⚠️ PFC: {len(issues)} Hinweise</span>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#48bb78">✅ PFC geladen</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#fc8181">❌ Keine PFC</span>', unsafe_allow_html=True)
            if port is not None:
                pnl = port["PnL_gesamt_EUR"].sum()
                color = "#48bb78" if pnl >= 0 else "#fc8181"
                st.markdown(f'<span style="color:{color}">💰 PnL: {pnl:,.0f} €</span>', unsafe_allow_html=True)
            st.markdown(f"📝 Trades: {len(st.session_state.trade_history)}")
            st.markdown(f"🔄 Tranchen: {len(st.session_state.tranches)}")

            # Price Alerts
            if st.session_state.price_alerts and pfc is not None:
                st.markdown("### ⚠️ Preis-Alerts")
                for alert in st.session_state.price_alerts:
                    prod = alert["produkt"];
                    lt = alert["lasttyp"]
                    row = pfc[(pfc["Produkt"] == prod) & (pfc["Lasttyp"] == lt)]
                    if len(row) > 0:
                        p = row.iloc[0]["Preis_EUR_MWh"]
                        if p >= alert.get("upper", 99999):
                            st.markdown(
                                f'<div class="alert-box alert-red">🔴 {prod} {lt}: {p:.2f} ≥ {alert["upper"]:.2f}</div>',
                                unsafe_allow_html=True)
                        elif p <= alert.get("lower", 0):
                            st.markdown(
                                f'<div class="alert-box alert-green">🟢 {prod} {lt}: {p:.2f} ≤ {alert["lower"]:.2f}</div>',
                                unsafe_allow_html=True)

        # ─── MAIN TABS ───
        st.title(f"⚡ Portfolio Szenario Tool – {commodity} {year}")

        tabs = st.tabs([
            "📊 Marktdaten & PFC",
            "📈 Technische Analyse",
            "📋 Positionen",
            "🔄 Tranchenbeschaffung",
            "💰 PnL Übersicht",
            "🎯 PnL Attribution",
            "📊 Benchmark",
            "🎲 Szenario-Analyse",
            "🛒 Beschaffungssimulation",
            "⚡ Spot & Residual",
            "🌍 CO₂ & Spreads",
            "📉 VaR & Risiko",
            "💼 Budget-Planung",
            "⚠️ Preis-Alerts",
            "📑 Audit & Export",
        ])

        # ══════════════════════════════════════════════════════════
        # TAB 0: MARKTDATEN & PFC
        # ══════════════════════════════════════════════════════════
        with tabs[0]:
            st.subheader("📊 Marktdaten & PFC (Price Forward Curve)")
            method = st.radio("Eingabemethode", ["📁 PFC-Datei hochladen", "⌨️ Default-PFC generieren",
                                                 "📈 HPFC hochladen (stündlich/viertelstündlich)"], horizontal=True,
                              key="pfc_method")

            if method == "📁 PFC-Datei hochladen":
                f = st.file_uploader("PFC CSV/XLSX", type=["csv", "xlsx"], key="pfc_upload")
                if f:
                    df = FileReader.read(f)
                    if df is not None:
                        st.dataframe(df.head(20), use_container_width=True)
                        align = TimeSeriesAligner.detect_alignment(df, commodity=commodity)
                        if align["needs_conversion"]:
                            st.info(f"🔄 Rechtsbündig erkannt ({align['resolution']}). Wird normalisiert.")
                            df = TimeSeriesAligner.normalize(df, align)
                        # Check if this is hourly/quarterly data or product-level
                        if align["resolution"] in ("15min", "60min"):
                            st.info("📈 Stündliche/viertelstündliche Daten erkannt → HPFC-Aggregation")
                            cols = [c for c in df.columns if c != align["ts_col"]]
                            pcol = st.selectbox("Preisspalte", cols, key="hpfc_pcol")
                            if st.button("HPFC aggregieren", key="btn_hpfc"):
                                agg = HPFCProcessor.process(df, align["ts_col"], pcol, year, commodity)
                                if agg is not None:
                                    full = get_product_structure(year, commodity)
                                    for idx, row in full.iterrows():
                                        m = agg[(agg["Produkt"] == row["Produkt"]) & (agg["Lasttyp"] == row["Lasttyp"])]
                                        if len(m) > 0: full.at[idx, "Preis_EUR_MWh"] = m.iloc[0]["Preis_EUR_MWh"]
                                    full = recalculate_aggregates(full)
                                    if commodity == "Strom": full = recalculate_offpeak(full)
                                    st.session_state.pfc = full
                                    log_action("PFC geladen", "HPFC-Aggregation")
                                    st.success("✅ HPFC aggregiert und PFC erstellt!")
                        else:
                            struct = get_product_structure(year, commodity)
                            merged = merge_pfc_with_structure(df, struct)
                            merged = recalculate_aggregates(merged)
                            if commodity == "Strom": merged = recalculate_offpeak(merged)
                            st.session_state.pfc = merged
                            log_action("PFC geladen", "Datei-Upload")
                            st.success("✅ PFC geladen!")

            elif method == "⌨️ Default-PFC generieren":
                c1, c2 = st.columns(2)
                bp = c1.number_input("Base-Preis €/MWh", 10.0, 500.0, 80.0, key="def_base")
                pp = c2.number_input("Peak-Preis €/MWh", 10.0, 600.0, 95.0,
                                     key="def_peak") if commodity == "Strom" else bp
                if st.button("Default-PFC generieren", key="btn_def_pfc"):
                    st.session_state.pfc = build_default_pfc(year, commodity, bp, pp)
                    log_action("PFC generiert", f"Base={bp}, Peak={pp}")
                    st.success("✅ Default-PFC mit saisonaler Variation erstellt!")

            elif method == "📈 HPFC hochladen (stündlich/viertelstündlich)":
                f = st.file_uploader("HPFC CSV/XLSX", type=["csv", "xlsx"], key="hpfc_upload")
                if f:
                    df = FileReader.read(f)
                    if df is not None:
                        align = TimeSeriesAligner.detect_alignment(df, commodity=commodity)
                        st.info(
                            f"Erkannt: {align['alignment']}, {align['resolution']}, Gas-Day: {align['gas_day_start']}:00")
                        if align["needs_conversion"]:
                            df = TimeSeriesAligner.normalize(df, align)
                            st.info("✅ Auf linksbündig normalisiert")
                        st.dataframe(df.head(10), use_container_width=True)
                        cols = [c for c in df.columns if c != align.get("ts_col", df.columns[0])]
                        pcol = st.selectbox("Preisspalte", cols, key="hpfc2_pcol")
                        if st.button("Aggregieren", key="btn_hpfc2"):
                            agg = HPFCProcessor.process(df, align["ts_col"], pcol, year, commodity)
                            if agg is not None:
                                full = get_product_structure(year, commodity)
                                for idx, row in full.iterrows():
                                    m = agg[(agg["Produkt"] == row["Produkt"]) & (agg["Lasttyp"] == row["Lasttyp"])]
                                    if len(m) > 0: full.at[idx, "Preis_EUR_MWh"] = m.iloc[0]["Preis_EUR_MWh"]
                                full = recalculate_aggregates(full)
                                if commodity == "Strom": full = recalculate_offpeak(full)
                                st.session_state.pfc = full
                                log_action("PFC geladen", "HPFC-Upload")
                                st.success("✅ PFC erstellt!")

            # Display & Edit PFC
            pfc = st.session_state.pfc
            if pfc is not None:
                st.markdown("---")
                issues = validate_pfc(pfc)
                if issues:
                    with st.expander(f"⚠️ {len(issues)} Validierungshinweise"):
                        for i in issues: st.warning(i)

                st.markdown("### PFC Übersicht")
                display_cols = ["Produkt", "Typ", "Lasttyp", "Stunden", "Preis_EUR_MWh"]
                existing = [c for c in display_cols if c in pfc.columns]
                for typ in ["Jahr", "Quartal", "Season", "Monat"]:
                    d = pfc[pfc["Typ"] == typ]
                    if len(d) > 0:
                        st.markdown(f"**{typ}e:**")
                        st.dataframe(d[existing].reset_index(drop=True), use_container_width=True,
                                     height=min(35 * len(d) + 38, 400))

                # Manual edit
                with st.expander("✏️ Manuell bearbeiten"):
                    prods = pfc[pfc["Typ"] == "Monat"]["Produkt"].unique()
                    edit_prod = st.selectbox("Produkt", prods, key="edit_prod")
                    lts = pfc[pfc["Produkt"] == edit_prod]["Lasttyp"].unique()
                    edit_lt = st.selectbox("Lasttyp", lts, key="edit_lt")
                    cur = pfc[(pfc["Produkt"] == edit_prod) & (pfc["Lasttyp"] == edit_lt)]
                    if len(cur) > 0:
                        new_p = st.number_input("Neuer Preis €/MWh", 0.0, 1000.0, float(cur.iloc[0]["Preis_EUR_MWh"]),
                                                key="edit_price")
                        if st.button("Aktualisieren", key="btn_edit_pfc"):
                            mask = (pfc["Produkt"] == edit_prod) & (pfc["Lasttyp"] == edit_lt)
                            pfc.loc[mask, "Preis_EUR_MWh"] = new_p
                            pfc = recalculate_aggregates(pfc)
                            if commodity == "Strom": pfc = recalculate_offpeak(pfc)
                            st.session_state.pfc = pfc
                            log_action("PFC bearbeitet", f"{edit_prod} {edit_lt}: {new_p}")
                            st.rerun()

                # PFC Chart
                months = pfc[pfc["Typ"] == "Monat"]
                if len(months) > 0:
                    fig = go.Figure()
                    for lt in months["Lasttyp"].unique():
                        ltd = months[months["Lasttyp"] == lt].sort_values("Monat")
                        fig.add_trace(go.Bar(x=ltd["Produkt"], y=ltd["Preis_EUR_MWh"], name=lt))
                    fig.update_layout(title="PFC – Monatspreise", barmode="group", height=400,
                                      yaxis_title="€/MWh", xaxis_title="Produkt")
                    st.plotly_chart(fig, use_container_width=True)

        # ══════════════════════════════════════════════════════════
        # TAB 1: TECHNISCHE ANALYSE
        # ══════════════════════════════════════════════════════════
        with tabs[1]:
            st.subheader("📈 Technische Analyse")
            st.info("Laden Sie historische Preisdaten hoch oder nutzen Sie die PFC als Basis.")
            source = st.radio("Datenquelle", ["📁 Historische Preise hochladen", "📊 PFC-Monate als Zeitreihe"],
                              horizontal=True, key="ta_source")

            prices_series = None
            labels = None

            if source == "📁 Historische Preise hochladen":
                f = st.file_uploader("Historische Preise CSV/XLSX", type=["csv", "xlsx"], key="hist_upload")
                if f:
                    df = FileReader.read(f)
                    if df is not None:
                        st.dataframe(df.head(10), use_container_width=True)
                        ncols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if ncols:
                            pcol = st.selectbox("Preisspalte", ncols, key="ta_pcol")
                            prices_series = df[pcol].dropna().values
                            labels = list(range(len(prices_series)))
                            st.session_state.historical_prices = df
            else:
                pfc = st.session_state.pfc
                if pfc is not None:
                    lt = st.selectbox("Lasttyp", pfc["Lasttyp"].unique(), key="ta_lt")
                    months = pfc[(pfc["Typ"] == "Monat") & (pfc["Lasttyp"] == lt)].sort_values("Monat")
                    if len(months) > 0:
                        prices_series = months["Preis_EUR_MWh"].values
                        labels = months["Produkt"].tolist()

            if prices_series is not None and len(prices_series) >= 5:
                c1, c2 = st.columns(2)

                # Bollinger
                boll = calc_bollinger(prices_series, min(20, len(prices_series) // 2 + 1))
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=labels, y=prices_series, name="Preis", line=dict(color="#3182ce")))
                fig_bb.add_trace(go.Scatter(x=labels, y=boll["ma"], name="MA", line=dict(color="#ecc94b", dash="dash")))
                fig_bb.add_trace(
                    go.Scatter(x=labels, y=boll["upper"], name="Upper Band", line=dict(color="#fc8181", dash="dot")))
                fig_bb.add_trace(
                    go.Scatter(x=labels, y=boll["lower"], name="Lower Band", line=dict(color="#48bb78", dash="dot"),
                               fill="tonexty", fillcolor="rgba(72,187,120,0.1)"))
                fig_bb.update_layout(title="Bollinger Bänder", height=350, yaxis_title="€/MWh")
                c1.plotly_chart(fig_bb, use_container_width=True)

                # RSI
                rsi = calc_rsi(prices_series, min(14, len(prices_series) // 2))
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=labels, y=rsi, name="RSI", line=dict(color="#805ad5")))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Überkauft (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Überverkauft (30)")
                fig_rsi.update_layout(title="RSI (Relative Strength Index)", height=350, yaxis_title="RSI")
                c2.plotly_chart(fig_rsi, use_container_width=True)

                # MACD
                macd = calc_macd(prices_series, min(12, len(prices_series) // 3 + 1),
                                 min(26, len(prices_series) // 2 + 1), min(9, len(prices_series) // 3))
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=labels, y=macd["macd"], name="MACD", line=dict(color="#3182ce")))
                fig_macd.add_trace(go.Scatter(x=labels, y=macd["signal"], name="Signal", line=dict(color="#e53e3e")))
                colors = ["#48bb78" if v >= 0 else "#fc8181" for v in macd["histogram"]]
                fig_macd.add_trace(go.Bar(x=labels, y=macd["histogram"], name="Histogram", marker_color=colors))
                fig_macd.update_layout(title="MACD", height=350, yaxis_title="MACD")
                c1.plotly_chart(fig_macd, use_container_width=True)

                # Momentum
                mom = calc_momentum(prices_series, min(10, len(prices_series) // 2))
                fig_mom = go.Figure()
                mom_colors = ["#48bb78" if v >= 0 else "#fc8181" for v in mom]
                fig_mom.add_trace(go.Bar(x=labels, y=mom, name="Momentum", marker_color=mom_colors))
                fig_mom.add_hline(y=0, line_color="white")
                fig_mom.update_layout(title="Momentum", height=350, yaxis_title="Momentum")
                c2.plotly_chart(fig_mom, use_container_width=True)

                # Signal Summary
                st.markdown("### 📊 Signal-Zusammenfassung")
                last_rsi = rsi.dropna().iloc[-1] if len(rsi.dropna()) > 0 else 50
                last_macd_h = macd["histogram"].dropna().iloc[-1] if len(macd["histogram"].dropna()) > 0 else 0
                last_price = prices_series[-1]
                last_upper = boll["upper"].dropna().iloc[-1] if len(boll["upper"].dropna()) > 0 else last_price
                last_lower = boll["lower"].dropna().iloc[-1] if len(boll["lower"].dropna()) > 0 else last_price

                signals = []
                if last_rsi > 70:
                    signals.append("🔴 RSI überkauft → Verkaufssignal")
                elif last_rsi < 30:
                    signals.append("🟢 RSI überverkauft → Kaufsignal")
                else:
                    signals.append(f"🟡 RSI neutral ({last_rsi:.1f})")
                if last_macd_h > 0:
                    signals.append("🟢 MACD positiv → Aufwärtstrend")
                else:
                    signals.append("🔴 MACD negativ → Abwärtstrend")
                if last_price >= last_upper:
                    signals.append("🔴 Preis am oberen Bollinger Band")
                elif last_price <= last_lower:
                    signals.append("🟢 Preis am unteren Bollinger Band")
                for s in signals: st.markdown(f"- {s}")
            else:
                st.info("Mindestens 5 Datenpunkte für technische Analyse erforderlich.")

        # ══════════════════════════════════════════════════════════
        # TAB 2: POSITIONEN
        # ══════════════════════════════════════════════════════════
        with tabs[2]:
            st.subheader("📋 Positionen")
            pos_method = st.radio("Eingabe", ["📁 Offene Position direkt", "📊 Beschaffung + Absatz getrennt"],
                                  horizontal=True, key="pos_method")

            if pos_method == "📁 Offene Position direkt":
                f = st.file_uploader("Offene Positionen CSV/XLSX", type=["csv", "xlsx"], key="op_upload")
                if f:
                    df = FileReader.read(f)
                    if df is not None:
                        st.dataframe(df, use_container_width=True)
                        st.session_state.offene_position = df
                        log_action("Offene Positionen geladen")
                        st.success("✅ Geladen!")
                st.markdown("#### Oder manuell eingeben:")
                with st.form("manual_op"):
                    prods = [f"Cal-{year}"] + [f"Q{q}-{year}" for q in range(1, 5)] + \
                            [f"M{m:02d}-{year}" for m in range(1, 13)]
                    mp = st.selectbox("Produkt", prods)
                    mlt = st.selectbox("Lasttyp", ["Base", "Peak", "Offpeak"] if commodity == "Strom" else ["Base"])
                    mmw = st.number_input("Menge MW", -500.0, 500.0, 10.0)
                    mt = "Monat" if mp.startswith("M") else "Quartal" if mp.startswith("Q") else "Jahr"
                    if st.form_submit_button("Hinzufügen"):
                        row = {"Produkt": mp, "Typ": mt, "Lasttyp": mlt, "Menge_MW": mmw, "Preis_EUR_MWh": 0}
                        if st.session_state.offene_position is None:
                            st.session_state.offene_position = pd.DataFrame([row])
                        else:
                            st.session_state.offene_position = pd.concat(
                                [st.session_state.offene_position, pd.DataFrame([row])], ignore_index=True)
                        log_action("Position manuell", "f{mp} {mlt} {mmw}MW")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📥 Beschaffung**")
                    f = st.file_uploader("Beschaffung", type=["csv", "xlsx"], key="besch_upload")
                    if f:
                        df = FileReader.read(f)
                        if df is not None: st.session_state.beschaffung = df; st.success("✅")
                with c2:
                    st.markdown("**📤 Absatz/Vertrieb**")
                    f = st.file_uploader("Absatz", type=["csv", "xlsx"], key="abs_upload")
                    if f:
                        df = FileReader.read(f)
                        if df is not None: st.session_state.absatz = df; st.success("✅")

            if st.button("🔄 Portfolio berechnen", key="btn_calc_port"):
                pfc = st.session_state.pfc
                if pfc is not None:
                    port = calculate_portfolio(pfc, st.session_state.beschaffung,
                                               st.session_state.absatz, st.session_state.offene_position)
                    st.session_state.portfolio = port
                    log_action("Portfolio berechnet")
                    st.success("✅ Portfolio berechnet!")
                else:
                    st.error("❌ Bitte zuerst PFC laden!")

        # ══════════════════════════════════════════════════════════
        # TAB 3: TRANCHENBESCHAFFUNG
        # ══════════════════════════════════════════════════════════
        with tabs[3]:
            st.subheader("🔄 Tranchenbeschaffung")
            st.markdown("Planen und verfolgen Sie die schrittweise Beschaffung über Tranchen.")

            c1, c2, c3 = st.columns(3)
            target_mw = c1.number_input("Ziel-Menge (MW)", 0.0, 1000.0, 50.0, key="tranche_target")
            n_planned = c2.number_input("Geplante Anzahl Tranchen", 1, 50, 10, key="tranche_n")
            tranche_lt = c3.selectbox("Lasttyp", ["Base", "Peak"] if commodity == "Strom" else ["Base"],
                                      key="tranche_lt")

            # Add tranche
            st.markdown("#### ➕ Neue Tranche buchen")
            with st.form("new_tranche"):
                tc1, tc2, tc3, tc4 = st.columns(4)
                t_date = tc1.date_input("Datum", date.today())
                t_mw = tc2.number_input("Menge MW", 0.1, 500.0, target_mw / max(n_planned, 1))
                t_price = tc3.number_input("Preis €/MWh", 0.0, 500.0, 80.0)
                t_kontr = tc4.text_input("Kontrahent", "")
                t_prod = st.selectbox("Produkt", [f"Cal-{year}"] + [f"Q{q}-{year}" for q in range(1, 5)] +
                                      [f"M{m:02d}-{year}" for m in range(1, 13)], key="tranche_prod")
                if st.form_submit_button("Tranche buchen"):
                    st.session_state.tranches.append({
                        "datum": str(t_date), "produkt": t_prod, "lasttyp": tranche_lt,
                        "menge_mw": t_mw, "preis": t_price, "kontrahent": t_kontr})
                    log_action("Tranche gebucht", f"{t_prod} {t_mw}MW @{t_price}")
                    st.rerun()

            # Status
            status = calculate_tranche_status(st.session_state.tranches, target_mw, year, commodity)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Beschafft (MW)", f"{status['procured_mw']:.1f}")
            c2.metric("Offen (MW)", f"{status['remaining_mw']:.1f}")
            c3.metric("Ø Preis", f"{status['avg_price']:.2f} €/MWh")
            c4.metric("Deckung", f"{status['coverage_pct']:.1f}%")

            # Progress chart
            if st.session_state.tranches:
                tdf = pd.DataFrame(st.session_state.tranches)
                tdf["datum"] = pd.to_datetime(tdf["datum"])
                tdf = tdf.sort_values("datum")
                tdf["cum_mw"] = tdf["menge_mw"].cumsum()
                tdf["cum_avg_price"] = (tdf["menge_mw"] * tdf["preis"]).cumsum() / tdf["cum_mw"]

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=tdf["datum"], y=tdf["cum_mw"], name="Kum. MW",
                                         fill="tozeroy", fillcolor="rgba(49,130,206,0.3)", line=dict(color="#3182ce")))
                fig.add_hline(y=target_mw, line_dash="dash", line_color="red", annotation_text=f"Ziel: {target_mw} MW")
                fig.add_trace(go.Scatter(x=tdf["datum"], y=tdf["cum_avg_price"], name="Ø Preis",
                                         line=dict(color="#ecc94b", dash="dot")), secondary_y=True)
                fig.update_layout(title="Beschaffungsfortschritt", height=400)
                fig.update_yaxes(title_text="MW", secondary_y=False)
                fig.update_yaxes(title_text="€/MWh", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    tdf[["datum", "produkt", "lasttyp", "menge_mw", "preis", "kontrahent", "cum_mw", "cum_avg_price"]],
                    use_container_width=True)

                if st.button("↩️ Letzte Tranche rückgängig", key="undo_tranche"):
                    st.session_state.tranches.pop()
                    log_action("Tranche rückgängig")
                    st.rerun()

        # ══════════════════════════════════════════════════════════
        # TAB 4: PNL ÜBERSICHT
        # ══════════════════════════════════════════════════════════
        with tabs[4]:
            st.subheader("💰 PnL Übersicht")
            port = st.session_state.portfolio
            if port is None:
                st.warning("⚠️ Bitte zuerst Portfolio berechnen (Tab Positionen).")
            else:
                # KPIs
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("PnL Gesamt", f"{port['PnL_gesamt_EUR'].sum():,.0f} €")
                c2.metric("Realisiert", f"{port['PnL_realisiert_EUR'].sum():,.0f} €")
                c3.metric("Unrealisiert", f"{port['PnL_unrealisiert_EUR'].sum():,.0f} €")
                c4.metric("Max Offene Pos", f"{port['Offene_Position_MW'].abs().max():.1f} MW")
                avg_dk = port.loc[port["Absatz_MW"] > 0, "Deckungsgrad_%"].mean() if (
                            port["Absatz_MW"] > 0).any() else 0
                c5.metric("Ø Deckungsgrad", f"{avg_dk:.1f}%")

                # PnL Waterfall
                months = port.sort_values("Monat")
                base = months[months["Lasttyp"] == "Base"] if commodity == "Strom" else months
                fig_wf = go.Figure(go.Waterfall(x=base["Produkt"], y=base["PnL_gesamt_EUR"],
                                                connector={"line": {"color": "rgb(63,63,63)"}},
                                                increasing={"marker": {"color": "#48bb78"}},
                                                decreasing={"marker": {"color": "#fc8181"}},
                                                totals={"marker": {"color": "#4299e1"}}))
                fig_wf.update_layout(title="PnL Waterfall (Base)", height=400, yaxis_title="€")
                st.plotly_chart(fig_wf, use_container_width=True)

                c1, c2 = st.columns(2)
                # Open Position Chart
                fig_op = go.Figure()
                for lt in months["Lasttyp"].unique():
                    ltd = months[months["Lasttyp"] == lt].sort_values("Monat")
                    colors = ["#48bb78" if v >= 0 else "#fc8181" for v in ltd["Offene_Position_MW"]]
                    fig_op.add_trace(
                        go.Bar(x=ltd["Produkt"], y=ltd["Offene_Position_MW"], name=lt, marker_color=colors))
                fig_op.update_layout(title="Offene Position (MW)", height=350, barmode="group", yaxis_title="MW")
                c1.plotly_chart(fig_op, use_container_width=True)

                # Coverage
                fig_dk = go.Figure()
                for lt in months["Lasttyp"].unique():
                    ltd = months[months["Lasttyp"] == lt].sort_values("Monat")
                    fig_dk.add_trace(go.Bar(x=ltd["Produkt"], y=ltd["Deckungsgrad_%"], name=lt))
                fig_dk.add_hline(y=100, line_dash="dash", line_color="white", annotation_text="100%")
                fig_dk.update_layout(title="Deckungsgrad (%)", height=350, barmode="group", yaxis_title="%")
                c2.plotly_chart(fig_dk, use_container_width=True)

                st.dataframe(port[["Produkt", "Lasttyp", "Beschaffung_MW", "Beschaffung_Preis",
                                   "Absatz_MW", "Absatz_Preis", "Offene_Position_MW", "Marktpreis",
                                   "PnL_realisiert_EUR", "PnL_unrealisiert_EUR", "PnL_gesamt_EUR", "Deckungsgrad_%"]],
                             use_container_width=True)

        # ══════════════════════════════════════════════════════════
        # TAB 5: PNL ATTRIBUTION
        # ══════════════════════════════════════════════════════════
        with tabs[5]:
            st.subheader("🎯 PnL Attribution")
            port = st.session_state.portfolio
            if port is None:
                st.warning("⚠️ Portfolio berechnen erforderlich.")
            else:
                attr = pnl_attribution(port)
                if attr is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Volumen-Effekt", f"{attr['Volume_Effect_EUR'].sum():,.0f} €")
                    c2.metric("Preis-Effekt", f"{attr['Price_Effect_EUR'].sum():,.0f} €")
                    c3.metric("Margin-Effekt", f"{attr['Margin_Effect_EUR'].sum():,.0f} €")

                    # Stacked bar by month
                    base_attr = attr[attr["Lasttyp"] == "Base"].sort_values(
                        "Monat") if commodity == "Strom" else attr.sort_values("Monat")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=base_attr["Produkt"], y=base_attr["Volume_Effect_EUR"], name="Volumen",
                                         marker_color="#3182ce"))
                    fig.add_trace(go.Bar(x=base_attr["Produkt"], y=base_attr["Price_Effect_EUR"], name="Preis",
                                         marker_color="#ecc94b"))
                    fig.add_trace(go.Bar(x=base_attr["Produkt"], y=base_attr["Margin_Effect_EUR"], name="Margin",
                                         marker_color="#48bb78"))
                    fig.update_layout(title="PnL Attribution pro Monat", barmode="relative", height=400,
                                      yaxis_title="€")
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(
                        attr[["Produkt", "Lasttyp", "Volume_Effect_EUR", "Price_Effect_EUR", "Margin_Effect_EUR"]],
                        use_container_width=True)

        # ══════════════════════════════════════════════════════════
        # TAB 6: BENCHMARK
        # ══════════════════════════════════════════════════════════
        with tabs[6]:
            st.subheader("📊 Benchmark-Vergleich")
            st.markdown("Vergleichen Sie Ihre Ist-Beschaffungspreise mit einem Benchmark (z.B. EEX Settlement).")

            bm_method = st.radio("Benchmark", ["📁 Benchmark-PFC hochladen", "📊 Default-Benchmark (=aktuelle PFC)"],
                                 horizontal=True, key="bm_method")

            if bm_method == "📁 Benchmark-PFC hochladen":
                f = st.file_uploader("Benchmark CSV/XLSX", type=["csv", "xlsx"], key="bm_upload")
                if f:
                    df = FileReader.read(f)
                    if df is not None:
                        struct = get_product_structure(year, commodity)
                        bm = merge_pfc_with_structure(df, struct)
                        bm = recalculate_aggregates(bm)
                        st.session_state.benchmark_pfc = bm
                        st.success("✅ Benchmark geladen!")
            else:
                if st.session_state.pfc is not None:
                    st.session_state.benchmark_pfc = st.session_state.pfc.copy()
                    st.info("PFC als Benchmark gesetzt.")

            port = st.session_state.portfolio
            bm = st.session_state.benchmark_pfc
            if port is not None and bm is not None:
                comp = benchmark_comparison(port, bm)
                if comp is not None:
                    c1, c2 = st.columns(2)
                    total_bm_pnl = comp["Benchmark_PnL_EUR"].sum()
                    avg_diff = comp.loc[comp["Beschaffung_MW"] > 0, "Benchmark_Diff_EUR_MWh"].mean() if (
                                comp["Beschaffung_MW"] > 0).any() else 0
                    c1.metric("Benchmark PnL", f"{total_bm_pnl:,.0f} €",
                              delta=f"{'besser' if total_bm_pnl > 0 else 'schlechter'} als Benchmark")
                    c2.metric("Ø Abweichung", f"{avg_diff:.2f} €/MWh")

                    base = comp[comp["Lasttyp"] == "Base"].sort_values(
                        "Monat") if commodity == "Strom" else comp.sort_values("Monat")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=base["Produkt"], y=base["Beschaffung_Preis"], name="Ist-Beschaffung",
                                         marker_color="#3182ce"))
                    fig.add_trace(
                        go.Bar(x=base["Produkt"], y=base["Benchmark_Preis"], name="Benchmark", marker_color="#ecc94b"))
                    fig.update_layout(title="Beschaffung vs. Benchmark", barmode="group", height=400,
                                      yaxis_title="€/MWh")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Portfolio und Benchmark erforderlich.")

        # ══════════════════════════════════════════════════════════
        # TAB 7: SZENARIO-ANALYSE
        # ══════════════════════════════════════════════════════════
        with tabs[7]:
            st.subheader("🎲 Szenario-Analyse")
            pfc = st.session_state.pfc
            if pfc is None:
                st.warning("⚠️ PFC erforderlich.");
                return

            st.markdown("### Parallele Verschiebung")
            c1, c2 = st.columns(2)
            shift_pct = c1.slider("Prozentual (%)", -50.0, 50.0, 0.0, 0.5, key="sc_pct")
            shift_abs = c2.slider("Absolut (€/MWh)", -50.0, 50.0, 0.0, 0.5, key="sc_abs")

            # Individual changes
            st.markdown("### Individuelle Preisänderungen")
            ind_changes = {}
            with st.expander("➕ Individuelle Änderungen"):
                prods = pfc["Produkt"].unique().tolist()
                sc_prod = st.selectbox("Produkt", prods, key="sc_prod")
                sc_lt = st.selectbox("Lasttyp", pfc[pfc["Produkt"] == sc_prod]["Lasttyp"].unique(), key="sc_lt")
                cur = pfc[(pfc["Produkt"] == sc_prod) & (pfc["Lasttyp"] == sc_lt)]
                cur_p = cur.iloc[0]["Preis_EUR_MWh"] if len(cur) > 0 else 0
                new_p = st.number_input(f"Neuer Preis (aktuell: {cur_p:.2f})", 0.0, 1000.0, float(cur_p),
                                        key="sc_new_p")
                if st.button("Änderung hinzufügen", key="btn_sc_add"):
                    ind_changes[(sc_prod, sc_lt)] = new_p

            if st.button("🎲 Szenario berechnen", key="btn_sc_calc"):
                sc_pfc = apply_scenario(pfc, shift_pct, shift_abs, ind_changes if ind_changes else None)
                sc_port = calculate_portfolio(sc_pfc, st.session_state.beschaffung,
                                              st.session_state.absatz, st.session_state.offene_position)

                if sc_port is not None:
                    port = st.session_state.portfolio
                    base_pnl = port["PnL_gesamt_EUR"].sum() if port is not None else 0
                    sc_pnl = sc_port["PnL_gesamt_EUR"].sum()
                    delta = sc_pnl - base_pnl

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Basis PnL", f"{base_pnl:,.0f} €")
                    c2.metric("Szenario PnL", f"{sc_pnl:,.0f} €")
                    c3.metric("Delta", f"{delta:,.0f} €", delta=f"{delta:,.0f}")

                    # Price comparison
                    orig_m = pfc[pfc["Typ"] == "Monat"]
                    sc_m = sc_pfc[sc_pfc["Typ"] == "Monat"]
                    for lt in orig_m["Lasttyp"].unique():
                        o = orig_m[orig_m["Lasttyp"] == lt].sort_values("Monat")
                        s = sc_m[sc_m["Lasttyp"] == lt].sort_values("Monat")
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(x=o["Produkt"], y=o["Preis_EUR_MWh"], name="Basis", line=dict(color="#3182ce")))
                        fig.add_trace(go.Scatter(x=s["Produkt"], y=s["Preis_EUR_MWh"], name="Szenario",
                                                 line=dict(color="#e53e3e", dash="dash")))
                        fig.update_layout(title=f"Preisvergleich – {lt}", height=300, yaxis_title="€/MWh")
                        st.plotly_chart(fig, use_container_width=True)

                # Sensitivity Table
                st.markdown("### 📊 Sensitivitätstabelle")
                shifts = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
                sens_data = []
                for s in shifts:
                    sp = apply_scenario(pfc, shift_pct=s)
                    sp_port = calculate_portfolio(sp, st.session_state.beschaffung,
                                                  st.session_state.absatz, st.session_state.offene_position)
                    pnl = sp_port["PnL_gesamt_EUR"].sum() if sp_port is not None else 0
                    sens_data.append({"Shift_%": s, "PnL_EUR": round(pnl, 0)})
                sens_df = pd.DataFrame(sens_data)
                fig_sens = go.Figure(go.Bar(x=sens_df["Shift_%"], y=sens_df["PnL_EUR"],
                                            marker_color=["#48bb78" if v >= 0 else "#fc8181" for v in
                                                          sens_df["PnL_EUR"]]))
                fig_sens.update_layout(title="Sensitivität: PnL vs. Preisshift", height=350,
                                       xaxis_title="Preisshift (%)", yaxis_title="PnL (€)")
                st.plotly_chart(fig_sens, use_container_width=True)

        # ══════════════════════════════════════════════════════════
        # TAB 8: BESCHAFFUNGSSIMULATION
        # ══════════════════════════════════════════════════════════
        with tabs[8]:
            st.subheader("🛒 Beschaffungssimulation")
            port = st.session_state.portfolio
            pfc = st.session_state.pfc
            if port is None or pfc is None:
                st.warning("⚠️ PFC und Portfolio erforderlich.")
            else:
                with st.form("trade_form"):
                    c1, c2, c3 = st.columns(3)
                    prods = pfc["Produkt"].unique().tolist()
                    t_prod = c1.selectbox("Produkt", prods)
                    t_lt = c2.selectbox("Lasttyp", pfc[pfc["Produkt"] == t_prod]["Lasttyp"].unique())
                    t_dir = c3.selectbox("Richtung", ["Kauf", "Verkauf"])
                    c4, c5, c6 = st.columns(3)
                    t_mw = c4.number_input("Menge MW", 0.1, 500.0, 5.0)
                    cur_p = pfc[(pfc["Produkt"] == t_prod) & (pfc["Lasttyp"] == t_lt)]
                    def_p = float(cur_p.iloc[0]["Preis_EUR_MWh"]) if len(cur_p) > 0 else 80.0
                    t_price = c5.number_input("Preis €/MWh", 0.0, 1000.0, def_p)
                    t_kontr = c6.text_input("Kontrahent", "EEX")

                    if st.form_submit_button("📝 Trade buchen"):
                        new_port = simulate_trade(pfc, port, t_prod, t_lt, t_mw, t_price, t_dir, t_kontr)
                        st.session_state.portfolio = new_port
                        st.session_state.trade_history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "produkt": t_prod, "lasttyp": t_lt, "richtung": t_dir,
                            "menge_mw": t_mw, "preis": t_price, "kontrahent": t_kontr})
                        log_action("Trade", f"{t_dir} {t_mw}MW {t_prod} @{t_price} ({t_kontr})")
                        st.rerun()

                if st.session_state.trade_history:
                    st.markdown("### 📜 Trade-Historie")
                    thdf = pd.DataFrame(st.session_state.trade_history)
                    st.dataframe(thdf, use_container_width=True)
                    # By counterparty
                    if "kontrahent" in thdf.columns:
                        kontr_sum = thdf.groupby("kontrahent").agg(
                            Trades=("menge_mw", "count"),
                            Volumen_MW=("menge_mw", "sum"),
                            Ø_Preis=("preis", "mean")).reset_index()
                        st.markdown("**Kontrahenten-Übersicht:**")
                        st.dataframe(kontr_sum, use_container_width=True)
                    if st.button("↩️ Letzten Trade rückgängig", key="undo_trade"):
                        if st.session_state.trade_history:
                            st.session_state.trade_history.pop()
                            # Recalc portfolio from scratch
                            port = calculate_portfolio(pfc, st.session_state.beschaffung,
                                                       st.session_state.absatz, st.session_state.offene_position)
                            for t in st.session_state.trade_history:
                                port = simulate_trade(pfc, port, t["produkt"], t["lasttyp"],
                                                      t["menge_mw"], t["preis"], t["richtung"])
                            st.session_state.portfolio = port
                            log_action("Trade rückgängig")
                            st.rerun()

        # ══════════════════════════════════════════════════════════
        # TAB 9: SPOT & RESIDUAL
        # ══════════════════════════════════════════════════════════
        with tabs[9]:
            st.subheader("⚡ Spot & Residualposition")
            port = st.session_state.portfolio
            if port is None:
                st.warning("⚠️ Portfolio erforderlich.")
            else:
                st.markdown(
                    "Die Residualposition ist der Anteil, der nicht über Terminprodukte gedeckt ist und am Spotmarkt beschafft/verkauft werden muss.")
                residual = port[["Produkt", "Lasttyp", "Monat", "Offene_Position_MW", "Offene_Position_MWh", "Stunden",
                                 "Marktpreis"]].copy()
                residual["Spot_Kosten_EUR"] = residual["Offene_Position_MW"] * residual["Stunden"] * residual[
                    "Marktpreis"]

                c1, c2, c3 = st.columns(3)
                total_res_mwh = residual["Offene_Position_MWh"].sum()
                total_spot_cost = residual["Spot_Kosten_EUR"].sum()
                c1.metric("Residual (MWh)", f"{total_res_mwh:,.0f}")
                c2.metric("Spot-Kosten (€)", f"{total_spot_cost:,.0f}")
                avg_spot = total_spot_cost / total_res_mwh if total_res_mwh != 0 else 0
                c3.metric("Ø Spot-Preis", f"{avg_spot:.2f} €/MWh")

                # Spot price simulation
                st.markdown("### 🎲 Spot-Preis-Simulation")
                spot_dev = st.slider("Spot-Abweichung vom Marktpreis (%)", -30.0, 30.0, 0.0, 1.0, key="spot_dev")
                sim_res = residual.copy()
                sim_res["Sim_Spot_Preis"] = sim_res["Marktpreis"] * (1 + spot_dev / 100)
                sim_res["Sim_Spot_Kosten"] = sim_res["Offene_Position_MW"] * sim_res["Stunden"] * sim_res[
                    "Sim_Spot_Preis"]

                fig = go.Figure()
                base_res = sim_res[sim_res["Lasttyp"] == "Base"].sort_values(
                    "Monat") if commodity == "Strom" else sim_res.sort_values("Monat")
                fig.add_trace(go.Bar(x=base_res["Produkt"], y=base_res["Spot_Kosten_EUR"], name="Basis Spot",
                                     marker_color="#3182ce"))
                fig.add_trace(go.Bar(x=base_res["Produkt"], y=base_res["Sim_Spot_Kosten"], name="Simuliert",
                                     marker_color="#e53e3e"))
                fig.update_layout(title="Spot-Kosten: Basis vs. Simulation", barmode="group", height=400,
                                  yaxis_title="€")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(sim_res[["Produkt", "Lasttyp", "Offene_Position_MW", "Offene_Position_MWh",
                                      "Marktpreis", "Spot_Kosten_EUR", "Sim_Spot_Preis", "Sim_Spot_Kosten"]],
                             use_container_width=True)

        # ══════════════════════════════════════════════════════════
        # TAB 10: CO2 & SPREADS
        # ══════════════════════════════════════════════════════════
        with tabs[10]:
            st.subheader("🌍 CO₂ & Spread-Analyse")
            pfc = st.session_state.pfc
            if pfc is None:
                st.warning("⚠️ PFC erforderlich.")
            else:
                c1, c2, c3 = st.columns(3)
                co2 = c1.number_input("CO₂-Preis (€/t)", 0.0, 200.0, st.session_state.co2_price, key="co2_input")
                eff_gas = c2.number_input("Wirkungsgrad Gas (%)", 20.0, 70.0, 50.0, key="eff_gas") / 100
                eff_coal = c3.number_input("Wirkungsgrad Kohle (%)", 20.0, 50.0, 38.0, key="eff_coal") / 100
                st.session_state.co2_price = co2

                months = pfc[(pfc["Typ"] == "Monat") & (pfc["Lasttyp"] == "Base")].sort_values("Monat")
                # Get gas prices if available (from a separate gas PFC or approximation)
                gas_prices = months["Preis_EUR_MWh"].values * 0.35  # Approx if no gas PFC

                st.info(
                    "💡 Gas-Preise werden approximiert (35% des Strompreises). Für exakte Werte bitte separate Gas-PFC laden.")

                spread_data = []
                for idx, row in months.iterrows():
                    gp = row["Preis_EUR_MWh"] * 0.35
                    sp = calc_spreads(row["Preis_EUR_MWh"], gp, co2, eff_gas, eff_coal)
                    sp["Produkt"] = row["Produkt"];
                    sp["Monat"] = row["Monat"]
                    sp["Strom_Base"] = row["Preis_EUR_MWh"];
                    sp["Gas_approx"] = gp
                    spread_data.append(sp)
                sdf = pd.DataFrame(spread_data)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ø Spark Spread", f"{sdf['spark'].mean():.2f} €/MWh")
                c2.metric("Ø Clean Spark", f"{sdf['clean_spark'].mean():.2f} €/MWh")
                c3.metric("Ø Dark Spread", f"{sdf['dark'].mean():.2f} €/MWh")
                c4.metric("Ø Clean Dark", f"{sdf['clean_dark'].mean():.2f} €/MWh")

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=sdf["Produkt"], y=sdf["spark"], name="Spark Spread", line=dict(color="#ecc94b")))
                fig.add_trace(
                    go.Scatter(x=sdf["Produkt"], y=sdf["clean_spark"], name="Clean Spark", line=dict(color="#48bb78")))
                fig.add_trace(
                    go.Scatter(x=sdf["Produkt"], y=sdf["dark"], name="Dark Spread", line=dict(color="#fc8181")))
                fig.add_trace(
                    go.Scatter(x=sdf["Produkt"], y=sdf["clean_dark"], name="Clean Dark", line=dict(color="#805ad5")))
                fig.add_hline(y=0, line_dash="dash", line_color="white")
                fig.update_layout(title="Spread-Analyse pro Monat", height=400, yaxis_title="€/MWh")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(sdf[["Produkt", "Strom_Base", "Gas_approx", "spark", "clean_spark", "dark", "clean_dark"]],
                             use_container_width=True)

        # ══════════════════════════════════════════════════════════
        # TAB 11: VAR & RISIKO
        # ══════════════════════════════════════════════════════════
        with tabs[11]:
            st.subheader("📉 Value at Risk & Risiko")
            port = st.session_state.portfolio
            if port is None:
                st.warning("⚠️ Portfolio erforderlich.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                conf = c1.selectbox("Konfidenzniveau", [0.90, 0.95, 0.99], 1, key="var_conf")
                hold = c2.number_input("Halteperiode (Tage)", 1, 30, 1, key="var_hold")
                vol = c3.number_input("Tägliche Volatilität (%)", 0.1, 20.0, 2.0, key="var_vol")
                method = c4.selectbox("Methode", ["parametric", "historisch"], key="var_method")

                var_result = calculate_var(port, conf, hold, vol, method)
                c1, c2, c3 = st.columns(3)
                c1.metric("Value at Risk", f"{var_result['VaR']:,.0f} €")
                c2.metric("CVaR / Expected Shortfall", f"{var_result['CVaR']:,.0f} €")
                c3.metric("Exposure", f"{var_result.get('exposure', 0):,.0f} €")

                # Risk Heatmap
                st.markdown("### 🗺️ Risiko-Heatmap")
                shifts = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
                months_list = port.sort_values("Monat")["Produkt"].unique()
                base_only = port[port["Lasttyp"] == "Base"].sort_values(
                    "Monat") if commodity == "Strom" else port.sort_values("Monat")
                heatmap_data = np.zeros((len(shifts), len(base_only)))
                for i, s in enumerate(shifts):
                    for j, (_, row) in enumerate(base_only.iterrows()):
                        shifted_price = row["Marktpreis"] * (1 + s / 100)
                        heatmap_data[i, j] = row["Offene_Position_MW"] * row["Stunden"] * (
                                    shifted_price - row["Marktpreis"])

                fig_hm = go.Figure(go.Heatmap(z=heatmap_data, x=base_only["Produkt"].values,
                                              y=[f"{s:+d}%" for s in shifts], colorscale="RdYlGn", zmid=0,
                                              text=np.round(heatmap_data, 0), texttemplate="%{text:,.0f}"))
                fig_hm.update_layout(title="PnL-Änderung (€) bei Preisshift", height=400,
                                     xaxis_title="Monat", yaxis_title="Preisshift")
                st.plotly_chart(fig_hm, use_container_width=True)

                # VaR Sensitivity
                st.markdown("### 📊 VaR-Sensitivität (Volatilität)")
                vols = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
                var_sens = [calculate_var(port, conf, hold, v, method)["VaR"] for v in vols]
                fig_vs = go.Figure(go.Scatter(x=vols, y=var_sens, mode="lines+markers",
                                              line=dict(color="#e53e3e"), marker=dict(size=8)))
                fig_vs.update_layout(title=f"VaR bei verschiedenen Volatilitäten ({conf * 100:.0f}% KN, {hold}d)",
                                     height=300, xaxis_title="Tägliche Volatilität (%)", yaxis_title="VaR (€)")
                st.plotly_chart(fig_vs, use_container_width=True)

        # ══════════════════════════════════════════════════════════
        # TAB 12: BUDGET-PLANUNG
        # ══════════════════════════════════════════════════════════
        with tabs[12]:
            st.subheader("💼 Budget-Planung")
            pfc = st.session_state.pfc
            port = st.session_state.portfolio
            if pfc is None:
                st.warning("⚠️ PFC erforderlich.")
            else:
                st.markdown("Planen Sie Ihr Energiekosten-Budget auf Basis der aktuellen PFC und Positionen.")
                c1, c2 = st.columns(2)
                budget_vol = c1.number_input("Jahresverbrauch (MWh)", 1000, 10000000, 100000, key="budget_vol")
                budget_lt = c2.selectbox("Hauptlasttyp", ["Base", "Peak+Offpeak-Mix"], key="budget_lt")

                months_pfc = pfc[(pfc["Typ"] == "Monat") & (pfc["Lasttyp"] == "Base")].sort_values("Monat")
                if len(months_pfc) > 0:
                    total_hours = months_pfc["Stunden"].sum()
                    avg_mw = budget_vol / total_hours if total_hours > 0 else 0

                    budget_data = []
                    for _, row in months_pfc.iterrows():
                        mwh = avg_mw * row["Stunden"]
                        cost = mwh * row["Preis_EUR_MWh"]
                        budget_data.append({"Monat": row["Produkt"], "MWh": round(mwh, 0),
                                            "Preis": round(row["Preis_EUR_MWh"], 2), "Kosten_EUR": round(cost, 0)})
                    bdf = pd.DataFrame(budget_data)

                    c1, c2, c3 = st.columns(3)
                    total_cost = bdf["Kosten_EUR"].sum()
                    avg_price = total_cost / budget_vol if budget_vol > 0 else 0
                    c1.metric("Jahresbudget", f"{total_cost:,.0f} €")
                    c2.metric("Ø Preis", f"{avg_price:.2f} €/MWh")
                    c3.metric("Ø Leistung", f"{avg_mw:.1f} MW")

                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(
                        go.Bar(x=bdf["Monat"], y=bdf["Kosten_EUR"], name="Budget (€)", marker_color="#3182ce"))
                    fig.add_trace(go.Scatter(x=bdf["Monat"], y=bdf["Preis"], name="Preis (€/MWh)",
                                             line=dict(color="#ecc94b")), secondary_y=True)
                    fig.update_layout(title="Monatsbudget", height=400)
                    fig.update_yaxes(title_text="€", secondary_y=False)
                    fig.update_yaxes(title_text="€/MWh", secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)

                    # Budget scenarios
                    st.markdown("### 📊 Budget-Szenarien")
                    sc_shifts = [-20, -10, 0, 10, 20]
                    budget_scenarios = []
                    for s in sc_shifts:
                        sc_cost = total_cost * (1 + s / 100)
                        budget_scenarios.append({"Szenario": f"{s:+d}%", "Budget_EUR": round(sc_cost, 0),
                                                 "Ø_Preis": round(avg_price * (1 + s / 100), 2)})
                    st.dataframe(pd.DataFrame(budget_scenarios), use_container_width=True)

                    st.dataframe(bdf, use_container_width=True)

        # ══════════════════════════════════════════════════════════
        # TAB 13: PREIS-ALERTS
        # ══════════════════════════════════════════════════════════
        with tabs[13]:
            st.subheader("⚠️ Preis-Alerts & Limits")
            pfc = st.session_state.pfc
            if pfc is None:
                st.warning("⚠️ PFC erforderlich.")
            else:
                with st.form("alert_form"):
                    c1, c2, c3, c4 = st.columns(4)
                    a_prod = c1.selectbox("Produkt", pfc["Produkt"].unique(), key="alert_prod")
                    a_lt = c2.selectbox("Lasttyp", pfc[pfc["Produkt"] == a_prod]["Lasttyp"].unique(), key="alert_lt")
                    a_upper = c3.number_input("Obergrenze €/MWh", 0.0, 1000.0, 100.0, key="alert_upper")
                    a_lower = c4.number_input("Untergrenze €/MWh", 0.0, 1000.0, 50.0, key="alert_lower")
                    if st.form_submit_button("Alert hinzufügen"):
                        st.session_state.price_alerts.append({"produkt": a_prod, "lasttyp": a_lt,
                                                              "upper": a_upper, "lower": a_lower})
                        log_action("Preis-Alert", f"{a_prod} {a_lt}: {a_lower}-{a_upper}")
                        st.rerun()

                if st.session_state.price_alerts:
                    st.markdown("### Aktive Alerts")
                    for i, alert in enumerate(st.session_state.price_alerts):
                        row = pfc[(pfc["Produkt"] == alert["produkt"]) & (pfc["Lasttyp"] == alert["lasttyp"])]
                        current = row.iloc[0]["Preis_EUR_MWh"] if len(row) > 0 else 0
                        status = "🟢 OK"
                        if current >= alert["upper"]:
                            status = "🔴 OBEN"
                        elif current <= alert["lower"]:
                            status = "🟡 UNTEN"
                        st.markdown(f"{status} | **{alert['produkt']}** {alert['lasttyp']}: "
                                    f"Aktuell {current:.2f} | Limits: {alert['lower']:.2f} – {alert['upper']:.2f}")

                    if st.button("🗑️ Alle Alerts löschen", key="clear_alerts"):
                        st.session_state.price_alerts = []
                        st.rerun()

                    # Alert visualization
                    months = pfc[(pfc["Typ"] == "Monat") & (pfc["Lasttyp"] == "Base")].sort_values("Monat")
                    if len(months) > 0 and st.session_state.price_alerts:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=months["Produkt"], y=months["Preis_EUR_MWh"],
                                                 name="Marktpreis", line=dict(color="#3182ce", width=3)))
                        for alert in st.session_state.price_alerts:
                            if alert["lasttyp"] == "Base":
                                fig.add_hline(y=alert["upper"], line_dash="dash", line_color="red",
                                              annotation_text=f"Max {alert['produkt']}")
                                fig.add_hline(y=alert["lower"], line_dash="dash", line_color="green",
                                              annotation_text=f"Min {alert['produkt']}")
                        fig.update_layout(title="Preis-Alert Visualisierung", height=400, yaxis_title="€/MWh")
                        st.plotly_chart(fig, use_container_width=True)

        # ══════════════════════════════════════════════════════════
        # TAB 14: AUDIT & EXPORT
        # ══════════════════════════════════════════════════════════
        with tabs[14]:
            st.subheader("📑 Audit Trail & Export")

            # Audit Log
            if st.session_state.audit_log:
                st.markdown("### 📋 Audit Trail")
                adf = pd.DataFrame(st.session_state.audit_log)
                st.dataframe(adf, use_container_width=True)
            else:
                st.info("Noch keine Aktionen protokolliert.")

            # Excel Export
            st.markdown("### 📥 Excel-Export")
            port = st.session_state.portfolio
            pfc = st.session_state.pfc
            if st.button("📥 Vollständigen Report exportieren", key="btn_export"):
                if pfc is not None:
                    xlsx = export_excel(port, pfc, commodity, year,
                                        st.session_state.trade_history, st.session_state.tranches)
                    st.download_button("💾 Download Excel", data=xlsx,
                                       file_name=f"Portfolio_Report_{commodity}_{year}.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.error("❌ Keine Daten zum Exportieren.")

            # JSON State Export
            if st.button("💾 Session als JSON sichern", key="btn_json"):
                state = {
                    "commodity": commodity, "year": year,
                    "trade_history": st.session_state.trade_history,
                    "tranches": st.session_state.tranches,
                    "price_alerts": st.session_state.price_alerts,
                    "audit_log": st.session_state.audit_log,
                }
                st.download_button("Download JSON", data=json.dumps(state, indent=2, default=str),
                                   file_name=f"session_{commodity}_{year}.json", mime="application/json")

    def merge_pfc_with_structure(pfc_data, structure):
        merged = structure.copy()
        price_col = None
        for c in pfc_data.columns:
            cl = c.lower().replace(" ", "").replace("_", "")
            if any(kw in cl for kw in ["preis", "price", "eurmwh", "eur/mwh", "€/mwh"]):
                price_col = c;
                break
        if price_col is None:
            ncols = pfc_data.select_dtypes(include=[np.number]).columns
            price_col = ncols[-1] if len(ncols) > 0 else None
        if price_col is None: return merged
        prod_col = None
        for c in pfc_data.columns:
            if any(kw in c.lower() for kw in ["produkt", "product", "contract"]): prod_col = c; break
        if prod_col is None: prod_col = pfc_data.columns[0]
        lt_col = None
        for c in pfc_data.columns:
            if any(kw in c.lower() for kw in ["lasttyp", "loadtype", "last", "type"]) and c != prod_col:
                lt_col = c;
                break
        if lt_col:
            pm = {}
            for _, row in pfc_data.iterrows():
                try:
                    pm[(str(row[prod_col]).strip(), str(row[lt_col]).strip())] = float(row[price_col])
                except:
                    continue
            for idx, row in merged.iterrows():
                key = (row["Produkt"], row["Lasttyp"])
                if key in pm: merged.at[idx, "Preis_EUR_MWh"] = pm[key]
        else:
            pm = {}
            for _, row in pfc_data.iterrows():
                try:
                    pm[str(row[prod_col]).strip()] = float(row[price_col])
                except:
                    continue
            for idx, row in merged.iterrows():
                if row["Produkt"] in pm: merged.at[idx, "Preis_EUR_MWh"] = pm[row["Produkt"]]
        return merged

    # ═══════════════════════════════════════════════════════════════
    # RUN
    # ═══════════════════════════════════════════════════════════════
    if __name__ == "__main__":
        main()