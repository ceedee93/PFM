import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# KONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="PV & BESS Projektsimulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 12px 16px;
        border-radius: 8px;
    }
    div[data-testid="stMetric"] label {font-size: 0.85rem; color: #495057;}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# DATENKLASSEN FÜR SAUBERE PARAMETERÜBERGABE
# ============================================================================
@dataclass
class PlantConfig:
    """Physische Anlagenparameter."""
    pv_mwp: float = 14.0
    grid_limit_mw: float = 11.5
    bess_power_mw: float = 5.0
    bess_capacity_mwh: float = 15.0
    bess_active: bool = True
    # Wirkungsgrade
    inverter_efficiency: float = 0.97  # DC/AC
    bess_rte: float = 0.88  # Round-Trip AC-AC
    bess_min_soc_pct: float = 0.05  # 5% DoD-Reserve
    bess_max_soc_pct: float = 0.95  # 95% max SoC
    # Degradation p.a.
    pv_degradation: float = 0.005
    bess_degradation: float = 0.025
    bess_augment_threshold: float = 0.70  # SoH Schwelle


@dataclass
class FinanceConfig:
    """Finanzparameter."""
    # CAPEX
    capex_pv_eur_kwp: float = 650.0
    capex_bess_eur_kwh: float = 280.0
    capex_bess_eur_kw: float = 80.0
    capex_infra_eur: float = 250_000.0  # Netz, Wege, Planung
    # OPEX (jährlich)
    opex_pv_eur_kwp: float = 12.0
    opex_bess_eur_kwh: float = 5.0
    pacht_eur_ha: float = 3_000.0
    pacht_ha: float = 14.0  # ca. 1 ha/MWp
    versicherung_pct: float = 0.004  # 0.4% vom CAPEX
    direktvermarktung_eur_mwh: float = 3.5
    # Finanzierung
    eigenkapital_pct: float = 0.25
    fremdkapital_zins: float = 0.045
    tilgung_jahre: int = 18
    # Steuer
    koerperschaftssteuer: float = 0.15
    gewerbesteuer: float = 0.14  # Hebesatz abhängig
    soli: float = 0.055  # auf KSt
    afa_pv_jahre: int = 20
    afa_bess_jahre: int = 10
    # Inflation
    preis_eskalation: float = 0.01
    opex_eskalation: float = 0.02
    pacht_eskalation: float = 0.015
    # Sonstiges
    project_lifetime: int = 25
    wacc: float = 0.06
    restwert_pct: float = 0.05


# ============================================================================
# DATEN-ENGINE
# ============================================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_profiles(n_hours: int = 8760, seed: int = 42) -> pd.DataFrame:
    """
    Erzeugt realistische PV- und Preisprofile.

    PV: Basierend auf Sonnenstand-Approximation für ~48°N (Süddeutschland).
    Preise: Basierend auf typischen Day-Ahead-Mustern mit Duck-Curve-Effekt.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    day_of_year = t / 24.0
    hour_of_day = t % 24

    # --- PV Profil ---
    # Tageslänge variiert mit Jahreszeit
    declination = 23.45 * np.sin(np.radians((284 + day_of_year) * 360 / 365))
    latitude = 48.0
    hour_angle = (hour_of_day - 12) * 15  # Grad

    # Sonnenhöhe (vereinfacht)
    sin_elevation = (
        np.sin(np.radians(latitude)) * np.sin(np.radians(declination))
        + np.cos(np.radians(latitude))
        * np.cos(np.radians(declination))
        * np.cos(np.radians(hour_angle))
    )
    sin_elevation = np.maximum(0, sin_elevation)

    # Wolken/Wetter-Faktor
    # Sommer klarer, Winter trüber
    seasonal_clarity = 0.55 + 0.25 * np.sin(
        np.radians((day_of_year - 80) * 360 / 365)
    )
    daily_weather = rng.beta(3, 1.5, n_hours) * seasonal_clarity

    pv_cf = sin_elevation * daily_weather
    # Normierung auf realistischen Kapazitätsfaktor (~11-12% für Süddeutschland)
    annual_cf = pv_cf.sum() / n_hours
    target_cf = 0.115
    if annual_cf > 0:
        pv_cf = pv_cf * (target_cf / annual_cf)
    pv_cf = np.clip(pv_cf, 0, 1)

    # --- Strompreis Profil ---
    # Basis: Tagesverlauf
    base_shape = 50 + 25 * np.sin(np.radians((hour_of_day - 6) * 360 / 24))
    # Saisonalität (Winter teurer)
    seasonal_price = -15 * np.sin(np.radians((day_of_year - 30) * 360 / 365))
    # Duck-Curve: PV drückt Mittagspreise
    solar_cannibalization = -pv_cf * 180
    # Volatilität
    noise = rng.normal(0, 12, n_hours)
    # Peaks (zufällige Preisspitzen)
    spikes = rng.exponential(5, n_hours) * (rng.random(n_hours) > 0.97)

    prices = base_shape + seasonal_price + solar_cannibalization + noise + spikes
    # Negative Preise möglich (realistisch)
    prices = np.maximum(-50, prices)

    return pd.DataFrame(
        {"pv_capacity_factor": pv_cf, "market_price_eur_mwh": prices},
        index=pd.RangeIndex(n_hours, name="hour"),
    )


def load_uploaded_data(uploaded_file, file_type: str) -> Optional[pd.DataFrame]:
    """Robustes Laden von CSV/Excel mit Fehlerbehandlung."""
    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        # Zeitstempel suchen und parsen
        time_candidates = [
            c
            for c in df.columns
            if any(k in c.lower() for k in ["date", "time", "zeit", "timestamp"])
        ]
        if time_candidates:
            col = time_candidates[0]
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            df = df.dropna(subset=[col])
            df = df.set_index(col).sort_index()

        return df

    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")
        return None


# ============================================================================
# DISPATCH ENGINE (Kernstück – physikalisch korrekt)
# ============================================================================
def run_hourly_dispatch(
    pv_cf: np.ndarray,
    prices: np.ndarray,
    plant: PlantConfig,
) -> Dict[str, np.ndarray]:
    """
    Stundenbasierter Dispatch mit korrekter Speicherphysik.

    Strategie (Prioritätenreihenfolge):
    1. Clipping-Prevention: Überschuss laden
    2. Negative Preise: Laden statt einspeisen / abregeln
    3. Preisarbitrage: Laden bei günstig, entladen bei teuer
    4. Peak-Shaving: Restkapazität für Preis-Peaks nutzen

    Physik:
    - Lade-Effizienz = sqrt(RTE)  → Verlust beim Laden (AC→DC→Batterie)
    - Entlade-Effizienz = sqrt(RTE) → Verlust beim Entladen (Batterie→DC→AC)
    - SoC-Limits werden eingehalten (min/max)
    """
    n = len(pv_cf)
    eta_charge = np.sqrt(plant.bess_rte)  # ~0.938
    eta_discharge = np.sqrt(plant.bess_rte)  # ~0.938

    usable_min = plant.bess_capacity_mwh * plant.bess_min_soc_pct
    usable_max = plant.bess_capacity_mwh * plant.bess_max_soc_pct

    # Ergebnis-Arrays
    pv_gen_ac = pv_cf * plant.pv_mwp * plant.inverter_efficiency  # MWh (1h Intervall)
    grid_export = np.zeros(n)
    bess_charge = np.zeros(n)  # positiv = laden (MWh AC-seitig)
    bess_discharge = np.zeros(n)  # positiv = entladen (MWh AC-seitig)
    clipped = np.zeros(n)
    soc = np.zeros(n + 1)
    soc[0] = usable_min  # Start leer

    if not plant.bess_active or plant.bess_capacity_mwh <= 0:
        # Ohne Speicher: einfaches Clipping
        grid_export = np.minimum(pv_gen_ac, plant.grid_limit_mw)
        clipped = np.maximum(0, pv_gen_ac - plant.grid_limit_mw)
        return {
            "pv_gen_ac": pv_gen_ac,
            "grid_export": grid_export,
            "bess_charge": bess_charge,
            "bess_discharge": bess_discharge,
            "clipped": clipped,
            "soc": soc[:n],
            "revenue": grid_export * prices,
        }

    # Preisstatistik für Schwellwerte (gleitend wäre besser, aber Tagesdurchschnitt ok)
    # Wir berechnen tägliche Perzentile
    daily_prices = prices.reshape(-1, 24) if n >= 24 else prices.reshape(1, -1)
    daily_p30 = np.percentile(daily_prices, 30, axis=1)
    daily_p70 = np.percentile(daily_prices, 70, axis=1)

    for t in range(n):
        day_idx = min(t // 24, len(daily_p30) - 1)
        price_low = daily_p30[day_idx]
        price_high = daily_p70[day_idx]

        p_pv = pv_gen_ac[t]
        p = prices[t]
        current_soc = soc[t]

        # Verfügbarer Platz / Energie
        space_mwh = usable_max - current_soc  # MWh im Speicher
        available_mwh = current_soc - usable_min

        max_charge_ac = min(
            plant.bess_power_mw, space_mwh / eta_charge if eta_charge > 0 else 0
        )
        max_discharge_ac = min(
            plant.bess_power_mw, available_mwh * eta_discharge
        )

        charge_ac = 0.0
        discharge_ac = 0.0

        # --- SCHRITT 1: Clipping Prevention ---
        excess_over_grid = p_pv - plant.grid_limit_mw
        if excess_over_grid > 0:
            # Überschuss in Batterie
            charge_from_clip = min(excess_over_grid, max_charge_ac)
            charge_ac += charge_from_clip
            max_charge_ac -= charge_from_clip
            remaining_clip = excess_over_grid - charge_from_clip
            clipped[t] = remaining_clip
            p_available_for_grid = plant.grid_limit_mw
        else:
            p_available_for_grid = p_pv

        # --- SCHRITT 2: Negative Preise → Laden statt einspeisen ---
        if p < 0 and p_available_for_grid > 0 and max_charge_ac > 0:
            charge_from_neg = min(p_available_for_grid, max_charge_ac)
            charge_ac += charge_from_neg
            max_charge_ac -= charge_from_neg
            p_available_for_grid -= charge_from_neg

        # --- SCHRITT 3: Arbitrage ---
        elif p < price_low and max_charge_ac > 0 and p_available_for_grid > 0:
            # Laden: PV-Strom nehmen statt einzuspeisen (wenn Preis niedrig)
            fraction = min(0.5, (price_low - p) / (price_low + 1))  # Anteil
            charge_arb = min(p_available_for_grid * fraction, max_charge_ac)
            charge_ac += charge_arb
            max_charge_ac -= charge_arb
            p_available_for_grid -= charge_arb

        elif p > price_high and max_discharge_ac > 0:
            # Entladen: Ins Netz einspeisen (wenn Platz im Netz)
            grid_headroom = plant.grid_limit_mw - p_available_for_grid
            if grid_headroom > 0:
                discharge_ac = min(grid_headroom, max_discharge_ac)

        # --- SoC Update ---
        soc_after = current_soc + charge_ac * eta_charge - (
            discharge_ac / eta_discharge if eta_discharge > 0 else 0
        )
        soc[t + 1] = np.clip(soc_after, usable_min, usable_max)

        # Tatsächliche Ladung/Entladung basierend auf SoC-Clipping
        actual_delta_soc = soc[t + 1] - current_soc
        if actual_delta_soc > 0:  # Netto geladen
            charge_ac = actual_delta_soc / eta_charge
            discharge_ac = 0
        elif actual_delta_soc < 0:  # Netto entladen
            discharge_ac = abs(actual_delta_soc) * eta_discharge
            charge_ac = 0

        bess_charge[t] = charge_ac
        bess_discharge[t] = discharge_ac

        # --- Grid Export ---
        grid_export[t] = min(
            p_available_for_grid - charge_ac + bess_charge[t]
            if charge_ac > bess_charge[t]
            else p_available_for_grid
            - bess_charge[t]
            + charge_from_clip
            if excess_over_grid > 0
            else p_available_for_grid,
            plant.grid_limit_mw,
        )
        # Vereinfachte, korrekte Grid-Export-Berechnung
        total_to_grid = p_pv - clipped[t] - bess_charge[t] + bess_discharge[t]
        grid_export[t] = np.clip(total_to_grid, 0, plant.grid_limit_mw)

    revenue = grid_export * prices

    return {
        "pv_gen_ac": pv_gen_ac,
        "grid_export": grid_export,
        "bess_charge": bess_charge,
        "bess_discharge": bess_discharge,
        "clipped": clipped,
        "soc": soc[:n],
        "revenue": revenue,
    }


# ============================================================================
# FINANZMODELL (Jahresbasiert, mit AfA und Fremdkapital)
# ============================================================================
def calculate_financials(
    dispatch_year1: Dict[str, np.ndarray],
    plant: PlantConfig,
    finance: FinanceConfig,
) -> Dict:
    """
    25-Jahres-Finanzmodell mit:
    - Linearer AfA (PV: 20J, BESS: 10J)
    - Annuitätendarlehen
    - Korrekte Steuerberechnung (KSt + GewSt + Soli)
    - Batterie-Augmentation
    - Pacht, Versicherung, Direktvermarktung
    """
    years = finance.project_lifetime

    # --- CAPEX ---
    capex_pv = plant.pv_mwp * 1000 * finance.capex_pv_eur_kwp
    capex_bess = 0
    if plant.bess_active:
        capex_bess = (
            plant.bess_capacity_mwh * 1000 * finance.capex_bess_eur_kwh
            + plant.bess_power_mw * 1000 * finance.capex_bess_eur_kw
        )
    capex_total = capex_pv + capex_bess + finance.capex_infra_eur

    # --- FINANZIERUNG ---
    fremdkapital = capex_total * (1 - finance.eigenkapital_pct)
    eigenkapital = capex_total * finance.eigenkapital_pct

    # Annuität berechnen
    zins = finance.fremdkapital_zins
    n_tilg = finance.tilgung_jahre
    if zins > 0 and n_tilg > 0 and fremdkapital > 0:
        annuitaet = fremdkapital * (zins * (1 + zins) ** n_tilg) / (
            (1 + zins) ** n_tilg - 1
        )
    else:
        annuitaet = fremdkapital / max(n_tilg, 1)

    # --- AfA ---
    afa_pv_annual = capex_pv / finance.afa_pv_jahre
    afa_bess_annual = capex_bess / finance.afa_bess_jahre if finance.afa_bess_jahre > 0 else 0

    # Steuersatz kombiniert
    kst = finance.koerperschaftssteuer
    gewst = finance.gewerbesteuer
    soli_rate = finance.soli
    combined_tax = kst * (1 + soli_rate) + gewst  # ~30.175%

    # --- JAHRES-ARRAYS ---
    revenue = np.zeros(years)
    opex = np.zeros(years)
    pacht = np.zeros(years)
    versicherung_cost = np.zeros(years)
    dvm_cost = np.zeros(years)  # Direktvermarktung
    afa = np.zeros(years)
    zinsaufwand = np.zeros(years)
    tilgung = np.zeros(years)
    ebt = np.zeros(years)
    tax = np.zeros(years)
    net_income = np.zeros(years)
    cashflow_equity = np.zeros(years + 1)  # +1 für Jahr 0
    augmentation_cost = np.zeros(years)

    cashflow_equity[0] = -eigenkapital

    # Restschuld
    restschuld = fremdkapital

    # Batterie SoH
    bess_soh = 1.0

    # Basis-Erlös Jahr 1
    base_revenue = dispatch_year1["revenue"].sum()
    base_energy = dispatch_year1["grid_export"].sum()

    for y in range(years):
        yr = y + 1  # Betriebsjahr

        # --- PV Degradation ---
        pv_deg_factor = (1 - plant.pv_degradation) ** y

        # --- Batterie Degradation & Augmentation ---
        if plant.bess_active and plant.bess_capacity_mwh > 0:
            if bess_soh < plant.bess_augment_threshold:
                # Augmentation: Kosten = 60% des originalen BESS CAPEX (Preisverfall)
                cost_aug = capex_bess * 0.6 * ((1 - 0.03) ** y)  # 3% Preisverfall/Jahr
                augmentation_cost[y] = cost_aug
                bess_soh = 1.0
            else:
                bess_soh *= 1 - plant.bess_degradation

        # --- ERLÖSE ---
        # Skalierung: PV-Degradation + BESS-SoH + Preis-Eskalation
        energy_factor = pv_deg_factor  # Vereinfacht: Batterie-Degradation reduziert Arbitrage
        price_factor = (1 + finance.preis_eskalation) ** y
        revenue[y] = base_revenue * energy_factor * price_factor

        # --- OPEX ---
        infl = (1 + finance.opex_eskalation) ** y
        opex[y] = plant.pv_mwp * 1000 * finance.opex_pv_eur_kwp * infl
        if plant.bess_active:
            opex[y] += plant.bess_capacity_mwh * 1000 * finance.opex_bess_eur_kwh * infl

        # Pacht
        pacht[y] = (
            finance.pacht_eur_ha
            * finance.pacht_ha
            * (1 + finance.pacht_eskalation) ** y
        )

        # Versicherung
        versicherung_cost[y] = capex_total * finance.versicherung_pct

        # Direktvermarktung
        dvm_cost[y] = base_energy * pv_deg_factor * finance.direktvermarktung_eur_mwh

        # --- AfA ---
        afa_pv_y = afa_pv_annual if yr <= finance.afa_pv_jahre else 0
        afa_bess_y = afa_bess_annual if yr <= finance.afa_bess_jahre else 0
        afa[y] = afa_pv_y + afa_bess_y

        # --- FREMDKAPITAL ---
        if restschuld > 0 and yr <= n_tilg:
            zinsaufwand[y] = restschuld * zins
            tilgung[y] = annuitaet - zinsaufwand[y]
            if tilgung[y] > restschuld:
                tilgung[y] = restschuld
            restschuld -= tilgung[y]
            restschuld = max(0, restschuld)
        else:
            zinsaufwand[y] = 0
            tilgung[y] = 0

        # --- GEWINN/VERLUST ---
        total_opex = opex[y] + pacht[y] + versicherung_cost[y] + dvm_cost[y]
        ebitda = revenue[y] - total_opex
        ebt[y] = ebitda - afa[y] - zinsaufwand[y]
        tax[y] = max(0, ebt[y] * combined_tax)  # Verlustvortrag vereinfacht ignoriert
        net_income[y] = ebt[y] - tax[y]

        # --- CASHFLOW (Equity) ---
        # CF = Netto-Einkommen + AfA (non-cash) - Tilgung - Augmentation
        cashflow_equity[yr] = net_income[y] + afa[y] - tilgung[y] - augmentation_cost[y]

    # Restwert in letztem Jahr
    cashflow_equity[-1] += capex_total * finance.restwert_pct

    # --- KPIs ---
    project_cashflows = np.copy(cashflow_equity)
    project_cashflows[0] = -capex_total  # Projekt-IRR: Gesamtinvest

    try:
        irr_project = npf.irr(project_cashflows)
    except Exception:
        irr_project = np.nan

    try:
        irr_equity = npf.irr(cashflow_equity)
    except Exception:
        irr_equity = np.nan

    npv_project = npf.npv(finance.wacc, project_cashflows)
    npv_equity = npf.npv(finance.wacc, cashflow_equity)

    # Payback
    cumulative = np.cumsum(cashflow_equity)
    payback_idx = np.where(cumulative > 0)[0]
    payback_year = payback_idx[0] if len(payback_idx) > 0 else years + 1

    # LCOE
    total_energy = sum(
        base_energy * (1 - plant.pv_degradation) ** y for y in range(years)
    )
    total_cost_pv = capex_total + sum(
        (opex[y] + pacht[y] + versicherung_cost[y])
        / (1 + finance.wacc) ** (y + 1)
        for y in range(years)
    )
    lcoe = (total_cost_pv / total_energy) * 1000 if total_energy > 0 else 0

    return {
        "capex_total": capex_total,
        "capex_pv": capex_pv,
        "capex_bess": capex_bess,
        "eigenkapital": eigenkapital,
        "fremdkapital": fremdkapital,
        "irr_project": irr_project,
        "irr_equity": irr_equity,
        "npv_project": npv_project,
        "npv_equity": npv_equity,
        "payback_year": payback_year,
        "lcoe": lcoe,
        "cashflow_equity": cashflow_equity,
        "project_cashflows": project_cashflows,
        "revenue": revenue,
        "opex": opex,
        "pacht": pacht,
        "versicherung": versicherung_cost,
        "dvm_cost": dvm_cost,
        "afa": afa,
        "zinsaufwand": zinsaufwand,
        "tilgung": tilgung,
        "tax": tax,
        "net_income": net_income,
        "augmentation": augmentation_cost,
        "annual_energy_mwh": base_energy,
    }


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("⚙️ Projektparameter")

    # --- Anlage ---
    st.header("1. Anlage")
    pv_mwp = st.number_input(
        "PV-Leistung (MWp)", min_value=0.5, max_value=100.0, value=14.0, step=0.5
    )
    grid_limit = st.number_input(
        "Netzanschluss (MW)",
        min_value=0.5,
        max_value=100.0,
        value=11.5,
        step=0.5,
        help="Maximale Einspeiseleistung am Netzverknüpfungspunkt",
    )

    ratio = pv_mwp / grid_limit if grid_limit > 0 else 0
    ratio_color = "🟢" if ratio < 1.3 else ("🟡" if ratio < 1.5 else "🔴")
    st.caption(f"{ratio_color} Überbauungsfaktor: {ratio:.2f}")

    st.header("2. Speicher")
    bess_on = st.checkbox("Speicher aktivieren", value=True)
    if bess_on:
        col_bp, col_bc = st.columns(2)
        with col_bp:
            bess_p = st.slider("Leistung (MW)", 0.5, 20.0, 5.0, 0.5)
        with col_bc:
            bess_c = st.slider("Kapazität (MWh)", 1.0, 60.0, 15.0, 1.0)
        dur = bess_c / bess_p if bess_p > 0 else 0
        st.caption(f"⏱️ Dauer: {dur:.1f}h | C-Rate: {1/dur if dur > 0 else 0:.2f}C")
        bess_rte = st.slider(
            "Round-Trip-Efficiency (%)", 80, 95, 88, help="AC-AC Wirkungsgrad"
        )
    else:
        bess_p, bess_c, bess_rte = 0, 0, 88

    st.header("3. CAPEX")
    c_pv = st.number_input("PV (€/kWp)", value=650.0, step=10.0)
    c_bess_kwh = st.number_input("BESS Energie (€/kWh)", value=280.0, step=10.0)
    c_bess_kw = st.number_input("BESS Leistung (€/kW)", value=80.0, step=10.0)
    c_infra = st.number_input("Infrastruktur (€)", value=250_000.0, step=10_000.0)

    st.header("4. Finanzierung")
    ek_pct = st.slider("Eigenkapitalquote (%)", 10, 100, 25, 5) / 100
    fk_zins = st.slider("Fremdkapitalzins (%)", 1.0, 8.0, 4.5, 0.25) / 100
    tilg_j = st.slider("Tilgungsdauer (Jahre)", 5, 25, 18)

    st.header("5. Daten")
    data_mode = st.radio(
        "Quelle", ["🔧 Synthetisch (Demo)", "📁 Eigene Datei"], index=0
    )

    df_profiles = None
    if data_mode == "📁 Eigene Datei":
        ufile = st.file_uploader("CSV oder Excel", type=["csv", "xlsx"])
        if ufile:
            ftype = ufile.name.rsplit(".", 1)[-1].lower()
            df_raw = load_uploaded_data(ufile, ftype)
            if df_raw is not None:
                st.success(f"✅ {len(df_raw)} Zeilen geladen")
                c_pv_col = st.selectbox(
                    "PV-Kapazitätsfaktor (0-1)", df_raw.columns, index=0
                )
                c_pr_col = st.selectbox(
                    "Marktpreis (€/MWh)", df_raw.columns, index=min(1, len(df_raw.columns) - 1)
                )
                df_profiles = pd.DataFrame(
                    {
                        "pv_capacity_factor": pd.to_numeric(
                            df_raw[c_pv_col], errors="coerce"
                        ).fillna(0),
                        "market_price_eur_mwh": pd.to_numeric(
                            df_raw[c_pr_col], errors="coerce"
                        ).ffill().fillna(50),
                    }
                )

    if df_profiles is None:
        df_profiles = generate_synthetic_profiles()


# ============================================================================
# HAUPT-DASHBOARD
# ============================================================================
st.title("⚡ PV & Speicher – Projektbewertung")
st.markdown(
    "Stundenbasierte Dispatch-Simulation mit vollständigem Finanzmodell "
    "(AfA, Fremdkapital, Steuern, Augmentation)"
)

# Konfigurationen zusammenbauen
plant = PlantConfig(
    pv_mwp=pv_mwp,
    grid_limit_mw=grid_limit,
    bess_power_mw=bess_p,
    bess_capacity_mwh=bess_c,
    bess_active=bess_on,
    bess_rte=bess_rte / 100,
)
fin = FinanceConfig(
    capex_pv_eur_kwp=c_pv,
    capex_bess_eur_kwh=c_bess_kwh,
    capex_bess_eur_kw=c_bess_kw,
    capex_infra_eur=c_infra,
    eigenkapital_pct=ek_pct,
    fremdkapital_zins=fk_zins,
    tilgung_jahre=tilg_j,
)

# ============================================================================
# SIMULATION AUSFÜHREN
# ============================================================================
if st.button("🚀 Simulation starten", type="primary", use_container_width=True):
    with st.spinner("Dispatch-Simulation läuft..."):
        pv_cf = df_profiles["pv_capacity_factor"].values[:8760]
        prices = df_profiles["market_price_eur_mwh"].values[:8760]

        # Auffüllen falls < 8760
        if len(pv_cf) < 8760:
            st.warning(
                f"Nur {len(pv_cf)} Stunden vorhanden, "
                f"fülle auf 8760 auf (Wiederholung)."
            )
            repeats = int(np.ceil(8760 / len(pv_cf)))
            pv_cf = np.tile(pv_cf, repeats)[:8760]
            prices = np.tile(prices, repeats)[:8760]

        # Dispatch
        dispatch = run_hourly_dispatch(pv_cf, prices, plant)

        # Vergleich: PV only
        plant_no_bess = PlantConfig(
            pv_mwp=pv_mwp,
            grid_limit_mw=grid_limit,
            bess_power_mw=0,
            bess_capacity_mwh=0,
            bess_active=False,
        )
        dispatch_no_bess = run_hourly_dispatch(pv_cf, prices, plant_no_bess)

    with st.spinner("Finanzmodell berechnen..."):
        fin_results = calculate_financials(dispatch, plant, fin)

        fin_no_bess = FinanceConfig(
            capex_pv_eur_kwp=c_pv,
            capex_bess_eur_kwh=0,
            capex_bess_eur_kw=0,
            capex_infra_eur=c_infra,
            eigenkapital_pct=ek_pct,
            fremdkapital_zins=fk_zins,
            tilgung_jahre=tilg_j,
        )
        fin_results_no_bess = calculate_financials(
            dispatch_no_bess, plant_no_bess, fin_no_bess
        )

    # ================================================================
    # ERGEBNISSE ANZEIGEN
    # ================================================================
    st.markdown("---")
    st.header("📊 Ergebnisse")

    # --- KPI Vergleich ---
    st.subheader("Kennzahlen-Vergleich")
    c1, c2, c3, c4, c5 = st.columns(5)

    irr_diff = (
        (fin_results["irr_equity"] - fin_results_no_bess["irr_equity"]) * 100
        if not (np.isnan(fin_results["irr_equity"]) or np.isnan(fin_results_no_bess["irr_equity"]))
        else 0
    )

    c1.metric(
        "CAPEX Gesamt",
        f"{fin_results['capex_total']/1e6:.2f} Mio €",
        f"davon BESS: {fin_results['capex_bess']/1e6:.2f} Mio €",
    )
    c2.metric(
        "Equity IRR (mit BESS)" if bess_on else "Equity IRR",
        f"{fin_results['irr_equity']*100:.1f} %"
        if not np.isnan(fin_results["irr_equity"])
        else "n/a",
        f"{irr_diff:+.1f} pp vs. ohne BESS" if bess_on else None,
    )
    c3.metric(
        "Equity IRR (ohne BESS)",
        f"{fin_results_no_bess['irr_equity']*100:.1f} %"
        if not np.isnan(fin_results_no_bess["irr_equity"])
        else "n/a",
    )
    c4.metric(
        "Payback (Equity)",
        f"{fin_results['payback_year']} Jahre",
    )
    c5.metric(
        "LCOE",
        f"{fin_results['lcoe']:.1f} €/MWh",
    )

    # Zweite KPI-Reihe
    c6, c7, c8, c9 = st.columns(4)
    c6.metric(
        "Jahresertrag (Jahr 1)",
        f"{fin_results['annual_energy_mwh']:,.0f} MWh",
    )
    c7.metric(
        "Erlöse (Jahr 1)",
        f"{fin_results['revenue'][0]/1e6:.2f} Mio €",
    )
    c8.metric(
        "Abregelung (ohne BESS)",
        f"{dispatch_no_bess['clipped'].sum():,.0f} MWh/a",
    )
    c9.metric(
        "Abregelung (mit BESS)" if bess_on else "Abregelung",
        f"{dispatch['clipped'].sum():,.0f} MWh/a",
        f"{dispatch['clipped'].sum() - dispatch_no_bess['clipped'].sum():,.0f} MWh"
        if bess_on
        else None,
        delta_color="inverse",
    )

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💰 Cashflow", "⚡ Dispatch", "📈 Sensitivität", "📋 Detaildaten"]
    )

    with tab1:
        col_cf1, col_cf2 = st.columns([3, 2])

        with col_cf1:
            years_arr = np.arange(len(fin_results["cashflow_equity"]))
            cum_cf = np.cumsum(fin_results["cashflow_equity"])

            fig_cf = go.Figure()
            colors = [
                "#d32f2f" if v < 0 else "#388e3c"
                for v in fin_results["cashflow_equity"]
            ]
            fig_cf.add_trace(
                go.Bar(
                    x=years_arr,
                    y=fin_results["cashflow_equity"] / 1e6,
                    marker_color=colors,
                    name="Equity CF",
                    opacity=0.7,
                )
            )
            fig_cf.add_trace(
                go.Scatter(
                    x=years_arr,
                    y=cum_cf / 1e6,
                    mode="lines+markers",
                    name="Kumuliert",
                    line=dict(color="#1565c0", width=2.5),
                    marker=dict(size=4),
                )
            )
            fig_cf.add_hline(
                y=0,
                line_dash="dot",
                line_color="gray",
                annotation_text="Break-Even",
            )
            fig_cf.update_layout(
                title="Equity-Cashflow über Projektlaufzeit",
                xaxis_title="Jahr",
                yaxis_title="Mio €",
                template="plotly_white",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_cf, use_container_width=True)

        with col_cf2:
            # Kosten-/Erlösstruktur Jahr 1
            yr1_data = {
                "Position": [
                    "Erlöse",
                    "Betriebs-OPEX",
                    "Pacht",
                    "Versicherung",
                    "Direktverm.",
                    "Zinsen",
                    "Steuern",
                ],
                "Betrag (T€)": [
                    fin_results["revenue"][0] / 1e3,
                    -fin_results["opex"][0] / 1e3,
                    -fin_results["pacht"][0] / 1e3,
                    -fin_results["versicherung"][0] / 1e3,
                    -fin_results["dvm_cost"][0] / 1e3,
                    -fin_results["zinsaufwand"][0] / 1e3,
                    -fin_results["tax"][0] / 1e3,
                ],
            }
            fig_waterfall = go.Figure(
                go.Waterfall(
                    x=yr1_data["Position"],
                    y=yr1_data["Betrag (T€)"],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#388e3c"}},
                    decreasing={"marker": {"color": "#d32f2f"}},
                    totals={"marker": {"color": "#1565c0"}},
                )
            )
            fig_waterfall.update_layout(
                title="Erlös-/Kostenstruktur (Jahr 1)",
                yaxis_title="T€",
                template="plotly_white",
                height=450,
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

    with tab2:
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            # Beispielwoche: Sommer
            week_start = st.slider(
                "Startstunde der Woche", 0, 8760 - 168, 4000, 24
            )
            week_end = week_start + 168
            sl = slice(week_start, week_end)

            fig_dispatch = go.Figure()
            fig_dispatch.add_trace(
                go.Scatter(
                    y=dispatch["pv_gen_ac"][sl],
                    name="PV Erzeugung (AC)",
                    fill="tozeroy",
                    fillcolor="rgba(255,193,7,0.3)",
                    line=dict(color="#ffa000"),
                )
            )
            fig_dispatch.add_trace(
                go.Scatter(
                    y=dispatch["grid_export"][sl],
                    name="Netzeinspeisung",
                    line=dict(color="#1565c0", width=2),
                )
            )
            if bess_on:
                fig_dispatch.add_trace(
                    go.Scatter(
                        y=dispatch["bess_charge"][sl],
                        name="BESS Laden",
                        line=dict(color="#43a047", dash="dot"),
                    )
                )
                fig_dispatch.add_trace(
                    go.Scatter(
                        y=dispatch["bess_discharge"][sl],
                        name="BESS Entladen",
                        line=dict(color="#e53935", dash="dot"),
                    )
                )
            fig_dispatch.add_hline(
                y=grid_limit,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Netzlimit {grid_limit} MW",
            )
            fig_dispatch.update_layout(
                title="Dispatch Wochenprofil",
                xaxis_title="Stunde",
                yaxis_title="MW",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig_dispatch, use_container_width=True)

        with col_d2:
            if bess_on:
                fig_soc = go.Figure()
                fig_soc.add_trace(
                    go.Scatter(
                        y=dispatch["soc"][sl],
                        name="SoC",
                        fill="tozeroy",
                        fillcolor="rgba(21,101,192,0.2)",
                        line=dict(color="#1565c0"),
                    )
                )
                fig_soc.add_hline(
                    y=plant.bess_capacity_mwh * plant.bess_max_soc_pct,
                    line_dash="dot",
                    line_color="orange",
                    annotation_text="Max SoC",
                )
                fig_soc.update_layout(
                    title="Speicher-Füllstand (SoC)",
                    xaxis_title="Stunde",
                    yaxis_title="MWh",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig_soc, use_container_width=True)
            else:
                fig_prices = px.histogram(
                    x=prices,
                    nbins=80,
                    title="Strompreisverteilung",
                    labels={"x": "€/MWh", "y": "Stunden"},
                )
                fig_prices.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_prices, use_container_width=True)

        # Dauerlinien
        st.subheader("Dauerlinien")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            sorted_pv = np.sort(dispatch["pv_gen_ac"])[::-1]
            sorted_grid = np.sort(dispatch["grid_export"])[::-1]
            fig_dl = go.Figure()
            fig_dl.add_trace(
                go.Scatter(y=sorted_pv, name="PV Erzeugung", line=dict(color="#ffa000"))
            )
            fig_dl.add_trace(
                go.Scatter(
                    y=sorted_grid,
                    name="Netzeinspeisung",
                    line=dict(color="#1565c0"),
                )
            )
            fig_dl.add_hline(y=grid_limit, line_dash="dash", line_color="red")
            fig_dl.update_layout(
                title="Dauerlinie Leistung",
                xaxis_title="Stunden (sortiert)",
                yaxis_title="MW",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig_dl, use_container_width=True)

        with col_dl2:
            sorted_prices = np.sort(prices)[::-1]
            fig_pdl = go.Figure()
            fig_pdl.add_trace(
                go.Scatter(
                    y=sorted_prices,
                    name="Marktpreis",
                    line=dict(color="#7b1fa2"),
                )
            )
            fig_pdl.update_layout(
                title="Preisdauerlinie",
                xaxis_title="Stunden (sortiert)",
                yaxis_title="€/MWh",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig_pdl, use_container_width=True)

    with tab3:
        st.subheader("Schnelle Sensitivität")
        st.markdown(
            "Variiere einen Parameter und sieh die Auswirkung auf den Equity-IRR."
        )

        sens_param = st.selectbox(
            "Parameter",
            ["BESS Kapazität (MWh)", "PV Leistung (MWp)", "Strompreis-Skalierung (%)"],
        )

        sens_results = []

        if sens_param == "BESS Kapazität (MWh)":
            vals = np.arange(0, 41, 5)
            for v in vals:
                p_test = PlantConfig(
                    pv_mwp=pv_mwp,
                    grid_limit_mw=grid_limit,
                    bess_power_mw=min(bess_p, v / 3) if v > 0 else 0,
                    bess_capacity_mwh=v,
                    bess_active=v > 0,
                    bess_rte=bess_rte / 100,
                )
                d = run_hourly_dispatch(pv_cf, prices, p_test)
                f = calculate_financials(d, p_test, fin)
                sens_results.append(
                    {"x": v, "irr": f["irr_equity"] * 100 if not np.isnan(f["irr_equity"]) else 0}
                )

        elif sens_param == "PV Leistung (MWp)":
            vals = np.arange(5, 25, 1)
            for v in vals:
                p_test = PlantConfig(
                    pv_mwp=v,
                    grid_limit_mw=grid_limit,
                    bess_power_mw=bess_p,
                    bess_capacity_mwh=bess_c,
                    bess_active=bess_on,
                    bess_rte=bess_rte / 100,
                )
                d = run_hourly_dispatch(pv_cf, prices, p_test)
                f = calculate_financials(d, p_test, fin)
                sens_results.append(
                    {"x": v, "irr": f["irr_equity"] * 100 if not np.isnan(f["irr_equity"]) else 0}
                )

        elif sens_param == "Strompreis-Skalierung (%)":
            vals = np.arange(60, 160, 10)
            for v in vals:
                scaled_prices = prices * (v / 100)
                d = run_hourly_dispatch(pv_cf, scaled_prices, plant)
                f = calculate_financials(d, plant, fin)
                sens_results.append(
                    {"x": v, "irr": f["irr_equity"] * 100 if not np.isnan(f["irr_equity"]) else 0}
                )

        if sens_results:
            df_sens = pd.DataFrame(sens_results)
            fig_sens = px.line(
                df_sens,
                x="x",
                y="irr",
                markers=True,
                title=f"Sensitivität: Equity-IRR vs. {sens_param}",
                labels={"x": sens_param, "irr": "Equity IRR (%)"},
            )
            fig_sens.add_hline(
                y=fin.wacc * 100,
                line_dash="dash",
                line_color="red",
                annotation_text=f"WACC {fin.wacc*100:.0f}%",
            )
            fig_sens.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_sens, use_container_width=True)

    with tab4:
        st.subheader("Jährliche Detailübersicht")
        detail_df = pd.DataFrame(
            {
                "Jahr": np.arange(1, fin.project_lifetime + 1),
                "Erlöse (€)": fin_results["revenue"],
                "OPEX (€)": fin_results["opex"],
                "Pacht (€)": fin_results["pacht"],
                "Versicherung (€)": fin_results["versicherung"],
                "DirektVerm. (€)": fin_results["dvm_cost"],
                "AfA (€)": fin_results["afa"],
                "Zinsen (€)": fin_results["zinsaufwand"],
                "Tilgung (€)": fin_results["tilgung"],
                "Steuern (€)": fin_results["tax"],
                "Augmentation (€)": fin_results["augmentation"],
                "Netto-Einkommen (€)": fin_results["net_income"],
                "Equity CF (€)": fin_results["cashflow_equity"][1:],
                "Kumuliert (€)": np.cumsum(fin_results["cashflow_equity"])[1:],
            }
        )

        st.dataframe(
            detail_df.style.format("{:,.0f}"),
            use_container_width=True,
            height=600,
        )

        # Download
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            detail_df.to_excel(writer, sheet_name="Jahresübersicht", index=False)

            # Stündliche Dispatch-Daten
            hourly_df = pd.DataFrame(
                {
                    "Stunde": np.arange(8760),
                    "PV_AC_MW": dispatch["pv_gen_ac"],
                    "Grid_Export_MW": dispatch["grid_export"],
                    "BESS_Charge_MW": dispatch["bess_charge"],
                    "BESS_Discharge_MW": dispatch["bess_discharge"],
                    "Clipped_MW": dispatch["clipped"],
                    "SoC_MWh": dispatch["soc"],
                    "Preis_EUR_MWh": prices,
                    "Erlös_EUR": dispatch["revenue"],
                }
            )
            hourly_df.to_excel(writer, sheet_name="Stündlich", index=False)

        st.download_button(
            "📥 Ergebnisse als Excel herunterladen",
            data=output.getvalue(),
            file_name="pv_bess_simulation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    # Landing Page
    st.info(
        "👈 Parameter in der Seitenleiste einstellen, dann **'Simulation starten'** klicken."
    )

    st.markdown("---")
    st.markdown(
        """
        ### Was dieses Tool kann

        | Feature | Detail |
        |---------|--------|
        | **Dispatch** | Stundenbasiert (8.760h), Clipping-Prevention + Preisarbitrage |
        | **Speicherphysik** | Korrekte RTE-Aufteilung (Laden/Entladen), SoC-Limits, Degradation |
        | **Finanzmodell** | AfA, Fremdkapital (Annuität), KSt+GewSt+Soli, Pacht, Versicherung |
        | **Augmentation** | Automatischer Batterietausch bei SoH < 70% |
        | **Vergleich** | PV-only vs. PV+BESS automatisch nebeneinander |
        | **Sensitivität** | Ein-Parameter-Variation mit IRR-Grafik |
        | **Export** | Excel-Download mit Jahres- UND Stundendaten |
        """
    )

# Footer
st.markdown("---")
st.caption(
    "Projektsimulator v2.0 | Modellbasierte Indikation – keine Finanz- oder Anlageberatung. "
    "Für bankfähige Gutachten sind zertifizierte Ertragsprognosen (P50/P90) erforderlich."
)