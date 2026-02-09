"""
🔋 Stadtwerke Hintertupfingen: PV & Speicher Investment Modell
Professionelles Erlös- und Cashflow-Modell für Freiflächen-PV mit Batteriespeicher
"""

import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. FINANCIAL MODEL CLASS
# ---------------------------------------------------------------------------

class FinancialModel:
    """
    Core financial model for PV + BESS project evaluation.
    Handles dispatch optimization and cashflow calculations.
    """
    
    def __init__(self, inputs: dict):
        self.i = inputs
        
    def validate_and_resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure hourly data for full year (8760 hours)."""
        df = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Resample to hourly if needed
        if len(df) != 8760:
            df = df.resample('h').mean().interpolate()
            
        # Fill any remaining gaps
        df = df.fillna(method='ffill').fillna(0)
        
        return df.head(8760)  # Ensure exactly 8760 rows
    
    def run_dispatch_optimization(self, pv_mw: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simple rule-based dispatch: 
        - Charge battery when clipping would occur
        - Discharge during high price hours
        """
        n_hours = len(pv_mw)
        grid_limit = self.i['grid_limit_mw']
        bess_power = self.i['bess_power_mw']
        bess_cap = self.i['bess_capacity_mwh']
        rte = self.i['bess_rte']
        
        grid_export = np.zeros(n_hours)
        bat_flow = np.zeros(n_hours)  # Positive = charging, Negative = discharging
        soc = np.zeros(n_hours)
        clipping = np.zeros(n_hours)
        
        current_soc = 0.0
        
        for h in range(n_hours):
            pv = pv_mw[h]
            price = prices[h]
            
            # Calculate clipping potential
            excess = max(0, pv - grid_limit)
            
            # Charging logic: store excess that would be clipped
            charge_potential = min(excess, bess_power, (bess_cap - current_soc))
            
            # Discharging logic: discharge when price is above daily average
            # Simple heuristic: discharge if price > 70th percentile of day
            day_start = (h // 24) * 24
            day_end = min(day_start + 24, n_hours)
            day_prices = prices[day_start:day_end]
            price_threshold = np.percentile(day_prices, 70) if len(day_prices) > 0 else 60
            
            discharge_potential = 0
            if price > price_threshold and current_soc > 0 and pv < grid_limit:
                available_grid_capacity = grid_limit - pv
                discharge_potential = min(
                    available_grid_capacity,
                    bess_power,
                    current_soc * np.sqrt(rte)  # Account for RTE on discharge
                )
            
            # Determine final action
            if charge_potential > 0:
                bat_flow[h] = charge_potential
                current_soc += charge_potential * np.sqrt(rte)  # RTE loss on charge
                grid_export[h] = min(pv - charge_potential, grid_limit)
                clipping[h] = max(0, pv - charge_potential - grid_limit)
            elif discharge_potential > 0:
                bat_flow[h] = -discharge_potential
                current_soc -= discharge_potential / np.sqrt(rte)  # RTE loss on discharge
                grid_export[h] = min(pv + discharge_potential, grid_limit)
                clipping[h] = 0
            else:
                grid_export[h] = min(pv, grid_limit)
                clipping[h] = max(0, pv - grid_limit)
            
            current_soc = max(0, min(bess_cap, current_soc))
            soc[h] = current_soc
            
        return grid_export, bat_flow, soc, clipping
    
    def calculate_cashflow(self, grid_export: np.ndarray, prices: np.ndarray) -> Tuple[pd.DataFrame, float]:
        """
        Build annual cashflow model with:
        - Revenue from grid export
        - OPEX (fixed + variable)
        - Depreciation (linear)
        - Debt service (annuity)
        - Taxes
        - FCFE calculation
        """
        years = self.i['project_years']
        capex = self.i['capex_total']
        opex_y1 = self.i['opex_total_year_1']
        dv_cost = self.i['dv_cost_eur_mwh']
        
        debt_share = self.i['debt_share_percent'] / 100.0
        rate = self.i['interest_rate_percent'] / 100.0
        n_periods = self.i['loan_term_years']
        
        # Equity / Debt Split
        debt_amount = capex * debt_share
        equity_amount = capex * (1 - debt_share)
        
        # Annuity calculation
        if rate > 0 and n_periods > 0:
            annuity = debt_amount * (rate * (1 + rate)**n_periods) / ((1 + rate)**n_periods - 1)
        else:
            annuity = debt_amount / max(n_periods, 1)
        
        remaining_debt = debt_amount
        
        # Depreciation (linear over 20 years for PV)
        depr_years = 20
        annual_depreciation = capex / depr_years
        
        # Base year revenue calculation
        base_revenue_mwh = grid_export.sum()  # MWh in Year 1
        base_price_avg = np.mean(prices)
        
        # Initialize DataFrame
        columns = ['Revenue', 'OPEX', 'EBITDA', 'Depreciation', 'EBIT', 
                   'Interest', 'EBT', 'Tax', 'Net_Income', 'Principal', 'FCF_Equity', 'DSCR']
        cf_df = pd.DataFrame(index=range(0, years + 1), columns=columns, dtype=float)
        
        # Year 0: Investment
        cf_df.loc[0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -equity_amount, 0]
        
        for y in range(1, years + 1):
            # Degradation
            pv_deg_factor = (1 - self.i['deg_pv_percent']/100) ** (y - 1)
            bess_deg_factor = (1 - self.i['deg_bess_percent']/100) ** (y - 1)
            
            # Price escalation
            price_factor = (1 + self.i['infl_electricity_percent']/100) ** (y - 1)
            opex_factor = (1 + self.i['infl_opex_percent']/100) ** (y - 1)
            
            # Revenue (simplified: assume proportional degradation)
            annual_mwh = base_revenue_mwh * pv_deg_factor
            avg_price = base_price_avg * price_factor
            rev = annual_mwh * (avg_price - dv_cost)
            
            # OPEX
            opex = opex_y1 * opex_factor
            
            # EBITDA
            ebitda = rev - opex
            
            # Depreciation (only for first 20 years)
            depreciation = annual_depreciation if y <= depr_years else 0
            
            # EBIT
            ebit = ebitda - depreciation
            
            # Debt Service
            interest = 0
            principal = 0
            if y <= n_periods and remaining_debt > 0:
                interest = remaining_debt * rate
                principal = annuity - interest
                remaining_debt -= principal
            
            ebt = ebit - interest
            
            # Tax
            tax = max(0, ebt * (self.i['tax_rate_percent'] / 100.0))
            
            # Net Income
            net_income = ebt - tax
            
            # FCF Equity = Net Income + Depreciation - Principal
            fcf_equity = net_income + depreciation - principal
            
            # DSCR Check
            dscr = ebitda / (interest + principal) if (interest + principal) > 0 else 99.9
            
            # Store
            cf_df.loc[y] = [rev, -opex, ebitda, -depreciation, ebit, 
                           -interest, ebt, -tax, net_income, -principal, fcf_equity, dscr]
        
        return cf_df, equity_amount


# ---------------------------------------------------------------------------
# 2. DATA PARSING FUNCTIONS
# ---------------------------------------------------------------------------

def parse_pvgis_file(uploaded_file) -> Optional[np.ndarray]:
    """
    Special parser for PVGIS CSV/Excel export with German number format.
    Analyzes history (e.g., 2005-2023) and creates a representative P50 year.
    """
    try:
        # 1. Read file (try different separators)
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', 
                           skiprows=10, skipfooter=10, encoding='utf-8')
        except:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', skiprows=10, skipfooter=10)
            except:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file)
        
        # 2. Identify columns
        time_col = next((c for c in df.columns if 'time' in c.lower() or 
                        'date' in c.lower() or 'zeit' in c.lower()), None)
        power_col = next((c for c in df.columns if c.strip() == 'P' or 
                         'power' in c.lower() or 'pv' in c.lower()), None)
        
        if not time_col:
            time_col = df.columns[0]
        if not power_col:
            power_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # 3. German -> English number conversion
        if df[power_col].dtype == object:
            df[power_col] = (df[power_col].astype(str)
                           .str.replace('.', '', regex=False)
                           .str.replace(',', '.', regex=False))
        
        df[power_col] = pd.to_numeric(df[power_col], errors='coerce')
        
        # 4. Convert Watt to MW
        max_val = df[power_col].max()
        if max_val > 10000:  # Likely in Watts
            df['mw_out'] = df[power_col] / 1_000_000.0
        elif max_val > 100:  # Likely in kW
            df['mw_out'] = df[power_col] / 1_000.0
        else:
            df['mw_out'] = df[power_col]
        
        # 5. Parse timestamp
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col, 'mw_out'])
        df.set_index(time_col, inplace=True)
        
        # 6. Create P50 representative year
        st.info(f"Analyzing historical dataset from {df.index.year.min()} to {df.index.year.max()}...")
        
        # Group by (month, day, hour) and take mean
        df_p50 = df.groupby([df.index.month, df.index.day, df.index.hour])['mw_out'].mean().reset_index()
        df_p50.columns = ['month', 'day', 'hour', 'mw']
        
        # Remove Feb 29 for non-leap year
        df_p50 = df_p50[~((df_p50['month'] == 2) & (df_p50['day'] == 29))]
        
        # Create datetime index for target year
        try:
            p50_series = pd.Series(df_p50['mw'].values, index=pd.to_datetime(
                dict(year=2025, month=df_p50['month'], day=df_p50['day'], hour=df_p50['hour'])
            ))
            full_idx = pd.date_range('2025-01-01', '2025-12-31 23:00', freq='h')
            p50_series = p50_series.reindex(full_idx).interpolate().fillna(0)
            final_mw = p50_series.values
        except:
            # Fallback: just use the values directly
            final_mw = df_p50['mw'].values[:8760]
            if len(final_mw) < 8760:
                final_mw = np.pad(final_mw, (0, 8760 - len(final_mw)), mode='constant')
        
        # 7. Normalize to peak
        peak_p50 = np.max(final_mw)
        if peak_p50 > 0:
            normalized_profile = final_mw / peak_p50
        else:
            normalized_profile = final_mw
        
        return normalized_profile
    
    except Exception as e:
        st.error(f"Error parsing PVGIS file: {str(e)}")
        return None


def generate_synthetic_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic PV and price profiles for demo mode."""
    t = np.arange(8760)
    
    # PV: Bell curve during day, seasonal variation
    hour_of_day = t % 24
    day_of_year = t // 24
    
    # Solar elevation proxy
    solar_factor = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
    solar_factor[hour_of_day < 6] = 0
    solar_factor[hour_of_day > 18] = 0
    
    # Seasonal variation (higher in summer)
    seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # Random weather variation
    weather_factor = np.random.uniform(0.7, 1.0, 8760)
    
    pv_profile = solar_factor * seasonal_factor * weather_factor
    pv_profile = pv_profile / pv_profile.max()  # Normalize
    
    # Price: Duck curve + seasonal variation
    # Higher prices in morning/evening, lower during solar peak
    base_price = 60
    morning_peak = 20 * np.exp(-((hour_of_day - 8) ** 2) / 8)
    evening_peak = 25 * np.exp(-((hour_of_day - 19) ** 2) / 8)
    solar_depression = -30 * pv_profile
    seasonal_price = 10 * np.sin(2 * np.pi * (day_of_year - 30) / 365)  # Winter higher
    noise = np.random.normal(0, 8, 8760)
    
    price_profile = base_price + morning_peak + evening_peak + solar_depression + seasonal_price + noise
    price_profile = np.clip(price_profile, 0, 200)  # Realistic bounds
    
    return pv_profile, price_profile


# ---------------------------------------------------------------------------
# 3. STREAMLIT UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="PV & Speicher Investment Modell",
        page_icon="🔋",
        layout="wide"
    )
    
    st.title("🔋 Stadtwerke Hintertupfingen: PV & Speicher Investment Modell")
    st.markdown("Professionelles Erlös- und Cashflow-Modell für Freiflächen-PV mit Batteriespeicher (Clipping-Optimierung).")
    
    # --- INPUT SECTION (SIDEBAR) ---
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        
        with st.expander("1. Anlagentechnik", expanded=True):
            pv_mw = st.number_input("PV Leistung (MWp)", 5.0, 50.0, 14.0, step=0.5)
            grid_mw = st.number_input("Netzanschluss (MW)", 1.0, 50.0, 11.5, step=0.5,
                                     help="Grenze am Netzverknüpfungspunkt")
            bess_mw = st.number_input("Speicher Leistung (MW)", 0.0, 20.0, 6.0, step=0.5)
            bess_mwh = st.number_input("Speicher Kapazität (MWh)", 0.0, 100.0, 12.0, step=1.0)
            rte = st.number_input("Systemwirkungsgrad (%)", 80.0, 98.0, 90.0, step=1.0) / 100.0
        
        with st.expander("2. Investition (CAPEX) & OPEX"):
            capex_pv = st.number_input("CAPEX PV (€/kWp)", 300, 1200, 600, step=50)
            capex_bess_kwh = st.number_input("CAPEX BESS (€/kWh)", 50, 800, 250, step=25)
            capex_bess_kw = st.number_input("CAPEX BESS Leistung (€/kW)", 0, 400, 80, step=10)
            capex_grid = st.number_input("Netzanschluss/Trafo (€)", 0, 2000000, 250000, step=50000)
            
            opex_pv_kwp = st.number_input("OPEX PV (€/kWp/a)", 5.0, 30.0, 12.0, step=1.0)
            opex_bess_mwh = st.number_input("OPEX BESS (€/MWh/a)", 0.0, 8000.0, 1500.0, step=100.0)
            dv_cost = st.number_input("Direktvermarktung (€/MWh)", 0.0, 10.0, 2.5, step=0.5)
        
        with st.expander("3. Finanzierung & Steuern"):
            debt_share = st.slider("Fremdkapitalanteil (%)", 0, 100, 80)
            interest_rate = st.number_input("Zins (%)", 0.0, 15.0, 4.5, step=0.25)
            loan_term = st.number_input("Kreditlaufzeit (Jahre)", 5, 30, 15)
            tax_rate = st.number_input("Steuersatz (KSt+GewSt %)", 0.0, 50.0, 30.0, step=1.0)
            wacc_req = st.number_input("WACC Ziel (%)", 0.0, 20.0, 6.0, step=0.5)
        
        with st.expander("4. Marktszenarien & Degradation"):
            infl_el = st.number_input("Strompreissteigerung (%/a)", -5.0, 10.0, 1.0, step=0.5)
            infl_opex = st.number_input("Inflation OPEX (%/a)", 0.0, 8.0, 2.0, step=0.5)
            deg_pv = st.number_input("PV Degradation (%/a)", 0.0, 2.0, 0.5, step=0.1)
            deg_bess = st.number_input("BESS Degradation (%/a)", 0.0, 5.0, 1.5, step=0.25)
        
        with st.expander("5. Daten Import"):
            data_source = st.radio("Quelle PV-Daten:", 
                                  ["Synthetisch (Demo)", "PVGIS Datei uploaden"])
            
            pv_profile_norm = None
            price_profile_series = None
            
            if data_source == "PVGIS Datei uploaden":
                pvgis_file = st.file_uploader("PVGIS CSV/Excel (Stundendaten)", 
                                             type=["csv", "xlsx", "xls"])
                if pvgis_file:
                    with st.spinner("Analysiere PVGIS Daten & berechne P50-Jahr..."):
                        pv_profile_norm = parse_pvgis_file(pvgis_file)
                    
                    if pv_profile_norm is not None:
                        st.success("✅ PVGIS Daten erfolgreich konvertiert!")
                        st.line_chart(pv_profile_norm[:168])  # First week
                        st.caption("Vorschau: Normiertes Profil (Erste 7 Tage)")
                
                price_file = st.file_uploader("Strompreise (Optional)", 
                                             type=["csv", "xlsx"])
                if price_file:
                    try:
                        if price_file.name.endswith('csv'):
                            df_price = pd.read_csv(price_file)
                        else:
                            df_price = pd.read_excel(price_file)
                        price_col = next((c for c in df_price.columns 
                                         if 'price' in c.lower() or 'preis' in c.lower()), 
                                        df_price.columns[-1])
                        price_profile_series = df_price[price_col].values[:8760]
                        st.success("✅ Preisdaten geladen!")
                    except Exception as e:
                        st.error(f"Fehler: {e}")
    
    # --- PREPARE DATA ---
    # Fallback to synthetic if no upload
    if pv_profile_norm is None:
        if data_source == "PVGIS Datei uploaden":
            st.warning("⚠️ Bitte PVGIS Datei hochladen oder auf 'Synthetisch' wechseln.")
            pv_profile_norm, price_profile_series = generate_synthetic_data()
        else:
            pv_profile_norm, price_profile_series = generate_synthetic_data()
    
    if price_profile_series is None:
        _, price_profile_series = generate_synthetic_data()
    
    # Ensure correct length
    if len(pv_profile_norm) < 8760:
        pv_profile_norm = np.pad(pv_profile_norm, (0, 8760 - len(pv_profile_norm)), mode='edge')
    if len(price_profile_series) < 8760:
        price_profile_series = np.pad(price_profile_series, (0, 8760 - len(price_profile_series)), mode='edge')
    
    pv_profile_norm = pv_profile_norm[:8760]
    price_profile_series = price_profile_series[:8760]
    
    # Scale PV profile to installed capacity
    pv_series_mw = pv_profile_norm * pv_mw
    
    # Create index
    idx = pd.date_range("2025-01-01", periods=8760, freq="h")
    df_hourly = pd.DataFrame({
        'pv': pv_profile_norm,
        'price': price_profile_series
    }, index=idx)
    
    # --- CALCULATE CAPEX/OPEX ---
    total_capex = (pv_mw * 1000 * capex_pv) + \
                  (bess_mwh * 1000 * capex_bess_kwh) + \
                  (bess_mw * 1000 * capex_bess_kw) + \
                  capex_grid
    
    total_opex_y1 = (pv_mw * 1000 * opex_pv_kwp) + (bess_mwh * opex_bess_mwh)
    
    # Display CAPEX breakdown
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Investitionsübersicht")
    st.sidebar.markdown(f"**CAPEX PV:** {pv_mw * 1000 * capex_pv / 1e6:.2f} Mio. €")
    st.sidebar.markdown(f"**CAPEX BESS:** {(bess_mwh * 1000 * capex_bess_kwh + bess_mw * 1000 * capex_bess_kw) / 1e6:.2f} Mio. €")
    st.sidebar.markdown(f"**Netzanschluss:** {capex_grid / 1e6:.2f} Mio. €")
    st.sidebar.markdown(f"**Gesamt CAPEX:** {total_capex / 1e6:.2f} Mio. €")
    st.sidebar.markdown(f"**OPEX Jahr 1:** {total_opex_y1 / 1e3:.1f} T€/a")
    
    # --- PREPARE MODEL INPUTS ---
    inputs = {
        'project_years': 25,
        'grid_limit_mw': grid_mw,
        'bess_power_mw': bess_mw,
        'bess_capacity_mwh': bess_mwh,
        'bess_rte': rte,
        'deg_pv_percent': deg_pv,
        'deg_bess_percent': deg_bess,
        'capex_total': total_capex,
        'opex_total_year_1': total_opex_y1,
        'dv_cost_eur_mwh': dv_cost,
        'debt_share_percent': debt_share,
        'interest_rate_percent': interest_rate,
        'loan_term_years': loan_term,
        'tax_rate_percent': tax_rate,
        'infl_electricity_percent': infl_el,
        'infl_opex_percent': infl_opex
    }
    
    model = FinancialModel(inputs)
    
    # --- RUN SIMULATION ---
    if st.button("🚀 Simulation starten", type="primary", use_container_width=True):
        with st.spinner("Optimiere Dispatch und berechne Cashflow-Wasserfall..."):
            
            # A. Dispatch Engine
            grid_export, bat_flow, soc, clipping = model.run_dispatch_optimization(
                pv_series_mw, price_profile_series
            )
            
            # B. Financial Engine
            cf_df, equity_invest = model.calculate_cashflow(grid_export, price_profile_series)
            
            # C. KPIs
            equity_cashflows = cf_df['FCF_Equity'].values.copy()
            
            try:
                irr_eq = npf.irr(equity_cashflows)
                if np.isnan(irr_eq):
                    irr_eq = 0.0
            except:
                irr_eq = 0.0
            
            try:
                npv_eq = npf.npv(wacc_req/100, equity_cashflows)
            except:
                npv_eq = 0.0
            
            total_clipped_mwh = clipping.sum()
            
            st.session_state['results'] = {
                'cf': cf_df,
                'irr': irr_eq,
                'npv': npv_eq,
                'invest': equity_invest,
                'clipping': total_clipped_mwh,
                'total_capex': total_capex,
                'dispatch': pd.DataFrame({
                    'PV': pv_series_mw,
                    'Grid_Export': grid_export,
                    'SoC': soc,
                    'Bat_Flow': bat_flow,
                    'Clipping': clipping,
                    'Price': price_profile_series
                }, index=idx)
            }
    
    # --- DASHBOARD OUTPUT ---
    if 'results' in st.session_state:
        res = st.session_state['results']
        
        st.markdown("---")
        st.subheader("📊 Ergebnisübersicht")
        
        # 1. Executive Summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Eigenkapital Invest",
            f"{res['invest']/1e6:.2f} Mio. €",
            f"CAPEX: {res['total_capex']/1e6:.2f} Mio. €"
        )
        col2.metric(
            "Equity IRR (post-tax)",
            f"{res['irr']*100:.2f} %",
            delta=f"{(res['irr']*100 - wacc_req):.1f}% vs. WACC" if res['irr'] > 0 else None
        )
        col3.metric(
            "Equity NPV",
            f"{res['npv']/1e6:.2f} Mio. €",
            help=f"Diskontiert mit {wacc_req}%"
        )
        col4.metric(
            "Verlorenes Clipping",
            f"{res['clipping']:.1f} MWh/a",
            help="Energie die wegen Netzlimit + vollem Speicher abgeregelt wurde"
        )
        
        # Additional KPIs
        col5, col6, col7, col8 = st.columns(4)
        total_export = res['dispatch']['Grid_Export'].sum()
        avg_price_achieved = (res['dispatch']['Grid_Export'] * res['dispatch']['Price']).sum() / total_export if total_export > 0 else 0
        
        col5.metric("Jahreseinspeisung", f"{total_export:.0f} MWh/a")
        col6.metric("Ø Erlöspreis", f"{avg_price_achieved:.1f} €/MWh")
        col7.metric("Volllaststunden PV", f"{total_export / pv_mw:.0f} h")
        col8.metric("Min. DSCR", f"{res['cf']['DSCR'].iloc[1:loan_term+1].min():.2f}x")
        
        # 2. Charts
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Cashflow & Rendite",
            "🔬 Sensitivitätsanalyse",
            "🔌 Dispatch & Technik",
            "📑 Rohdaten"
        ])
        
        with tab1:
            st.subheader("Cashflow Wasserfall (Equity)")
            cf_plot = res['cf']
            
            fig_cf = go.Figure()
            fig_cf.add_trace(go.Bar(
                x=cf_plot.index,
                y=cf_plot['FCF_Equity']/1e6,
                name='Free Cash Flow to Equity',
                marker_color=['red' if x < 0 else 'green' for x in cf_plot['FCF_Equity']]
            ))
            fig_cf.add_trace(go.Scatter(
                x=cf_plot.index,
                y=cf_plot['FCF_Equity'].cumsum()/1e6,
                name='Kumuliert',
                mode='lines+markers',
                line=dict(color='blue', width=2)
            ))
            fig_cf.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_cf.update_layout(
                xaxis_title="Jahr",
                yaxis_title="Mio. €",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_cf, use_container_width=True)
            
            # Revenue breakdown
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Jahres-GuV Struktur (Jahr 1)")
                guv_data = {
                    'Position': ['Umsatz', 'OPEX', 'EBITDA', 'AfA', 'EBIT', 'Zinsen', 'EBT', 'Steuern', 'Jahresüberschuss'],
                    'Wert': [
                        cf_plot.loc[1, 'Revenue'],
                        cf_plot.loc[1, 'OPEX'],
                        cf_plot.loc[1, 'EBITDA'],
                        cf_plot.loc[1, 'Depreciation'],
                        cf_plot.loc[1, 'EBIT'],
                        cf_plot.loc[1, 'Interest'],
                        cf_plot.loc[1, 'EBT'],
                        cf_plot.loc[1, 'Tax'],
                        cf_plot.loc[1, 'Net_Income']
                    ]
                }
                df_guv = pd.DataFrame(guv_data)
                df_guv['Wert (T€)'] = df_guv['Wert'] / 1000
                st.dataframe(df_guv[['Position', 'Wert (T€)']].style.format({'Wert (T€)': '{:.1f}'}))
            
            with col_b:
                st.subheader("DSCR Entwicklung")
                dscr_data = cf_plot['DSCR'].iloc[1:loan_term+1].clip(0, 5)
                fig_dscr = go.Figure()
                fig_dscr.add_trace(go.Scatter(
                    x=dscr_data.index,
                    y=dscr_data.values,
                    mode='lines+markers',
                    name='DSCR',
                    line=dict(color='purple')
                ))
                fig_dscr.add_hline(y=1.2, line_dash="dash", line_color="red",
                                  annotation_text="Min. Covenant (1.2x)")
                fig_dscr.update_layout(
                    xaxis_title="Jahr",
                    yaxis_title="DSCR",
                    yaxis_range=[0, max(3, dscr_data.max() + 0.5)]
                )
                st.plotly_chart(fig_dscr, use_container_width=True)
        
        with tab2:
            st.subheader("Tornado Chart: Einfluss auf NPV")
            
            # Simplified sensitivity estimates
            base_npv = res['npv']
            
            sensitivities = [
                ('CAPEX +10%', -total_capex * 0.10 * (1-debt_share/100)),
                ('CAPEX -10%', total_capex * 0.10 * (1-debt_share/100)),
                ('Strompreis +10%', res['cf']['Revenue'].sum() * 0.10 * 0.4),
                ('Strompreis -10%', -res['cf']['Revenue'].sum() * 0.10 * 0.4),
                ('OPEX +20%', -total_opex_y1 * 15 * 0.20 * 0.6),
                ('OPEX -20%', total_opex_y1 * 15 * 0.20 * 0.6),
                ('Zins +1%', -total_capex * debt_share/100 * 0.01 * loan_term * 0.5),
                ('Zins -1%', total_capex * debt_share/100 * 0.01 * loan_term * 0.5),
            ]
            
            df_sens = pd.DataFrame(sensitivities, columns=['Szenario', 'NPV_Delta'])
            df_sens['NPV_Delta_Mio'] = df_sens['NPV_Delta'] / 1e6
            df_sens['Typ'] = ['Negativ' if d < 0 else 'Positiv' for d in df_sens['NPV_Delta']]
            
            fig_tor = px.bar(
                df_sens,
                x='NPV_Delta_Mio',
                y='Szenario',
                color='Typ',
                orientation='h',
                color_discrete_map={'Negativ': 'red', 'Positiv': 'green'},
                title="Einfluss auf NPV (Approximation)"
            )
            fig_tor.add_vline(x=0, line_color="black")
            fig_tor.update_layout(xaxis_title="Δ NPV (Mio. €)", yaxis_title="")
            st.plotly_chart(fig_tor, use_container_width=True)
            
            st.caption("*Hinweis: Approximierte Sensitivitäten. Für exakte Werte wäre eine vollständige Neuberechnung erforderlich.*")
            
            # Breakeven Analysis
            st.subheader("Breakeven Analyse")
            col_be1, col_be2 = st.columns(2)
            
            with col_be1:
                if res['irr'] > 0:
                    # Approximate breakeven price
                    current_rev = res['cf']['Revenue'].sum()
                    breakeven_rev = current_rev - res['npv'] / 0.4  # Rough approximation
                    price_ratio = breakeven_rev / current_rev if current_rev > 0 else 1
                    breakeven_price = np.mean(price_profile_series) * price_ratio
                    
                    st.metric(
                        "Breakeven Strompreis",
                        f"{breakeven_price:.1f} €/MWh",
                        f"bei WACC {wacc_req}%"
                    )
                else:
                    st.warning("IRR konnte nicht berechnet werden")
            
            with col_be2:
                # Payback period
                cumsum = res['cf']['FCF_Equity'].cumsum()
                payback_idx = cumsum[cumsum > 0].first_valid_index()
                if payback_idx:
                    st.metric("Payback Periode (Equity)", f"{payback_idx} Jahre")
                else:
                    st.metric("Payback Periode (Equity)", "> 25 Jahre")
        
        with tab3:
            st.subheader("Dispatch Analyse")
            
            # Week selector
            week_options = {
                'Januar (Winter)': (0, 168),
                'April (Frühling)': (2160, 2328),
                'Juli (Sommer)': (4344, 4512),
                'Oktober (Herbst)': (6552, 6720)
            }
            
            selected_week = st.selectbox("Woche auswählen:", list(week_options.keys()))
            start_h, end_h = week_options[selected_week]
            
            df_disp = res['dispatch'].iloc[start_h:end_h]
            
            # Main dispatch chart
            fig_d = go.Figure()
            
            # PV Area
            fig_d.add_trace(go.Scatter(
                x=df_disp.index,
                y=df_disp['PV'],
                name='PV Erzeugung',
                fill='tozeroy',
                line=dict(color='gold', width=0),
                fillcolor='rgba(255, 215, 0, 0.5)'
            ))
            
            # Grid Export
            fig_d.add_trace(go.Scatter(
                x=df_disp.index,
                y=df_disp['Grid_Export'],
                name='Netzeinspeisung',
                line=dict(color='green', width=2)
            ))
            
            # Clipping
            fig_d.add_trace(go.Scatter(
                x=df_disp.index,
                y=df_disp['Clipping'],
                name='Clipping',
                fill='tozeroy',
                line=dict(color='red', width=0),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
            
            # Battery SoC on secondary axis
            fig_d.add_trace(go.Scatter(
                x=df_disp.index,
                y=df_disp['SoC'],
                name='Speicherstand (MWh)',
                line=dict(color='blue', width=2, dash='dot'),
                yaxis='y2'
            ))
            
            # Grid limit line
            fig_d.add_hline(y=grid_mw, line_dash="dash", 
                          annotation_text=f"Netzlimit ({grid_mw} MW)",
                          line_color="red")
            
            fig_d.update_layout(
                title=f"Dispatch: {selected_week}",
                xaxis_title="Zeit",
                yaxis=dict(title="Leistung [MW]", side='left'),
                yaxis2=dict(title="Speicher [MWh]", overlaying='y', side='right'),
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_d, use_container_width=True)
            
            # Statistics
            col_s1, col_s2, col_s3 = st.columns(3)
            
            with col_s1:
                st.markdown("**Wochenstatistik:**")
                st.write(f"- PV Erzeugung: {df_disp['PV'].sum():.1f} MWh")
                st.write(f"- Netzeinspeisung: {df_disp['Grid_Export'].sum():.1f} MWh")
                st.write(f"- Clipping: {df_disp['Clipping'].sum():.1f} MWh")
            
            with col_s2:
                st.markdown("**Speichernutzung:**")
                cycles = abs(df_disp['Bat_Flow']).sum() / (2 * bess_mwh) if bess_mwh > 0 else 0
                st.write(f"- Zyklen diese Woche: {cycles:.1f}")
                st.write(f"- Max. SoC: {df_disp['SoC'].max():.1f} MWh")
                st.write(f"- Ø SoC: {df_disp['SoC'].mean():.1f} MWh")
            
            with col_s3:
                st.markdown("**Preise:**")
                st.write(f"- Ø Preis: {df_disp['Price'].mean():.1f} €/MWh")
                st.write(f"- Max. Preis: {df_disp['Price'].max():.1f} €/MWh")
                st.write(f"- Min. Preis: {df_disp['Price'].min():.1f} €/MWh")
            
            # Duration curves
            st.subheader("Jahresdauerlinien")
            
            col_dc1, col_dc2 = st.columns(2)
            
            with col_dc1:
                sorted_pv = np.sort(res['dispatch']['PV'].values)[::-1]
                sorted_export = np.sort(res['dispatch']['Grid_Export'].values)[::-1]
                
                fig_dc = go.Figure()
                fig_dc.add_trace(go.Scatter(
                    x=list(range(8760)),
                    y=sorted_pv,
                    name='PV Erzeugung',
                    line=dict(color='gold')
                ))
                fig_dc.add_trace(go.Scatter(
                    x=list(range(8760)),
                    y=sorted_export,
                    name='Netzeinspeisung',
                    line=dict(color='green')
                ))
                fig_dc.add_hline(y=grid_mw, line_dash="dash", line_color="red")
                fig_dc.update_layout(
                    title="Leistungs-Dauerlinie",
                    xaxis_title="Stunden",
                    yaxis_title="MW"
                )
                st.plotly_chart(fig_dc, use_container_width=True)
            
            with col_dc2:
                sorted_prices = np.sort(res['dispatch']['Price'].values)[::-1]
                
                fig_price_dc = go.Figure()
                fig_price_dc.add_trace(go.Scatter(
                    x=list(range(8760)),
                    y=sorted_prices,
                    name='Spotpreis',
                    line=dict(color='purple'),
                    fill='tozeroy',
                    fillcolor='rgba(128, 0, 128, 0.2)'
                ))
                fig_price_dc.update_layout(
                    title="Preis-Dauerlinie",
                    xaxis_title="Stunden",
                    yaxis_title="€/MWh"
                )
                st.plotly_chart(fig_price_dc, use_container_width=True)
        
        with tab4:
            st.subheader("Cashflow Rohdaten")
            
            # Format dataframe for display
            display_cf = res['cf'].copy()
            for col in display_cf.columns:
                if col != 'DSCR':
                    display_cf[col] = display_cf[col] / 1000  # Convert to T€
            
            st.dataframe(
                display_cf.style.format({
                    'Revenue': '{:,.1f}',
                    'OPEX': '{:,.1f}',
                    'EBITDA': '{:,.1f}',
                    'Depreciation': '{:,.1f}',
                    'EBIT': '{:,.1f}',
                    'Interest': '{:,.1f}',
                    'EBT': '{:,.1f}',
                    'Tax': '{:,.1f}',
                    'Net_Income': '{:,.1f}',
                    'Principal': '{:,.1f}',
                    'FCF_Equity': '{:,.1f}',
                    'DSCR': '{:.2f}'
                }),
                use_container_width=True
            )
            st.caption("Alle Werte außer DSCR in T€")
            
            # Download button
            csv = res['cf'].to_csv()
            st.download_button(
                label="📥 Cashflow als CSV herunterladen",
                data=csv,
                file_name="cashflow_analysis.csv",
                mime="text/csv"
            )
            
            st.subheader("Stündliche Dispatch-Daten (Auszug)")
            st.dataframe(res['dispatch'].head(100).style.format('{:.2f}'))

if __name__ == "__main__":
    main()
