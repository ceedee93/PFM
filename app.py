import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# KONFIGURATION & GLOBAL SETTINGS
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="PV & BESS Invest Tool Pro", page_icon="⚡")

# Custom CSS für bessere Lesbarkeit
st.markdown("""
<style>
    .metric-card {background-color: #f8f9fa; border-left: 5px solid #4e73df; padding: 15px; border-radius: 5px; margin-bottom: 10px;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0;}
    .stTabs [aria-selected="true"] {background-color: #4e73df; color: white;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 1. CORE LOGIC CLASS (STATE & CALCULATION)
# ---------------------------------------------------------------------------

class FinancialModel:
    def __init__(self, inputs):
        self.i = inputs
    
    def validate_and_resample_data(self, df):
        """
        Validiert Eingabedaten, behandelt Schaltjahre und resamplet auf 1h Resolution.
        """
        # Sicherstellen, dass Index Datetime ist
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Daten müssen einen Datetime-Index haben.")
            
        # Resampling auf 1h (Mittelwert für Preise, Summe wäre für Energy, aber Input ist meist Power/Price)
        # Wir gehen davon aus, dass PV-Profil 'Power' (MW) ist und Preis 'Price' (€/MWh)
        df_hourly = df.resample('h').mean()
        
        # Lücken füllen
        df_hourly = df_hourly.interpolate(method='linear')
        
        # Check auf 8760 oder 8784 (Schaltjahr)
        rows = len(df_hourly)
        if rows < 8760:
            st.warning(f"Datenreihe zu kurz ({rows} h). Wird auf 8760 h aufgefüllt (ffill).")
            # Auffüllen auf ein Standardjahr
            idx = pd.date_range(start=df_hourly.index[0], periods=8760, freq='h')
            df_hourly = df_hourly.reindex(idx).ffill()
            
        return df_hourly

    def run_dispatch_optimization(self, pv_series, price_series):
        """
        Simuliert den Speicherbetrieb mit einer Day-Ahead "Perfect Foresight" Strategie.
        Optimiert täglich: Wann laden/entladen basierend auf Preis-Spread.
        """
        # Parameter entpacken
        grid_limit = self.i['grid_limit_mw']
        bess_power = self.i['bess_power_mw']
        bess_cap = self.i['bess_capacity_mwh']
        rte = self.i['bess_rte']
        eff_one_way = np.sqrt(rte)
        
        n_steps = len(pv_series)
        
        # Arrays für Ergebnisse (Vektorisierung wo möglich, Loop für State)
        soc_curve = np.zeros(n_steps)
        battery_flow = np.zeros(n_steps) # + Laden, - Entladen
        grid_export = np.zeros(n_steps)
        clipping_loss = np.zeros(n_steps)
        
        # Wir iterieren TAG-WEISE für Day-Ahead Logik (24h Blöcke)
        # Das ist realistischer als 1 Jahr perfect foresight, aber besser als dumme Schwellwerte
        
        current_soc = 0.0 # Start leer
        
        # Hilfsindex für 24h Blöcke
        hours_per_day = 24
        days = n_steps // hours_per_day
        
        for d in range(days):
            start_idx = d * hours_per_day
            end_idx = start_idx + hours_per_day
            
            # Slice für den Tag
            p_pv_day = pv_series[start_idx:end_idx]
            prices_day = price_series[start_idx:end_idx]
            
            # --- SCHRITT 1: PHYSISCHES CLIPPING (MUSS in den Speicher) ---
            excess_day = np.maximum(0, p_pv_day - grid_limit)
            p_grid_injectable = np.minimum(p_pv_day, grid_limit)
            
            # Flow nur aus Clipping
            flow_day = np.zeros(hours_per_day)
            
            # Simuliere Clipping-Ladung
            temp_soc = current_soc
            for h in range(hours_per_day):
                if excess_day[h] > 0:
                    # Lade so viel wie möglich vom Clipping
                    can_charge_mw = min(excess_day[h], bess_power)
                    can_charge_mwh = min(can_charge_mw, bess_cap - temp_soc)
                    
                    flow_day[h] += can_charge_mwh
                    temp_soc += can_charge_mwh * eff_one_way
                    
                    # Rest ist Verlust
                    clipping_loss[start_idx + h] = excess_day[h] - can_charge_mwh
            
            # --- SCHRITT 2: ARBITRAGE (RESTKAPAZITÄT NUTZEN) ---
            # Sortiere Stunden nach Preis für Laden/Entladen
            # Wir suchen Spread > Grenzkosten (Degradation + Wirkungsgradverlust)
            # Marginal Cost Abschätzung: ca. 10€/MWh Degradation / RTE
            spread_threshold = 20.0 
            
            # Identifiziere günstigste Stunden zum Laden (wo noch Platz ist & Grid Limit nicht voll)
            # Identifiziere teuerste Stunden zum Entladen
            sorted_indices = np.argsort(prices_day) # Index von billig nach teuer
            cheapest_hours = sorted_indices[:12]
            most_expensive_hours = sorted_indices[::-1][:12]
            
            # Einfache Heuristik: Lade in billigsten Stunden, Entlade in teuersten
            # unter Berücksichtigung von Grid Limits und SoC
            
            # Simulation des Tagesablaufs mit Arbitrage
            # Reset temp_soc auf start value und iteriere neu mit Entscheidung
            day_soc_track = np.zeros(hours_per_day)
            temp_soc = current_soc
            
            for h in range(hours_per_day):
                # Bereits geplantes Clipping-Laden
                clip_charge = flow_day[h]
                
                # Entscheidungsvariablen
                price = prices_day[h]
                grid_headroom = grid_limit - p_grid_injectable[h]
                
                # Arbitrage Entscheidung
                arb_flow = 0
                
                # ENTLADEN: Wenn Preis hoch & SoC da & Grid Platz
                if h in most_expensive_hours and price > 0: # Keine Negativpreis-Entladung
                    max_discharge = min(bess_power, grid_headroom)
                    # Prüfen ob Spread reicht (vs. Durchschnitt der Ladekosten - vereinfacht)
                    # Hier: Einfach "Teure Stunde" Logik
                    arb_flow = -min(max_discharge, temp_soc) # Kann nicht mehr entladen als da ist
                
                # LADEN: Wenn Preis tief & Platz im Speicher (ZUSÄTZLICH zu Clipping)
                # ACHTUNG: Grid-Charging erlauben? Wenn EEG-InnoAus => NEIN. Wenn PPA => JA.
                # Wir nehmen hier an: Nur PV-Ladung (AC oder DC gekoppelt) für Arbitrage aus PV-Strom,
                # der sonst direkt eingespeist würde.
                # ODER Grid-Charging. Parameter 'allow_grid_charge'
                elif h in cheapest_hours and (bess_cap - temp_soc) > 0:
                    # Wir können PV-Strom, der sonst ins Netz ginge, umlenken
                    available_pv = p_grid_injectable[h]
                    can_charge = min(bess_power - clip_charge, available_pv) # Power limit minus clip charge
                    space = bess_cap - temp_soc
                    arb_flow = min(can_charge, space)
                    
                    # Korrektur Grid Injection (wir nehmen PV Strom weg)
                    p_grid_injectable[h] -= arb_flow

                # Summe Flow
                total_flow = clip_charge + arb_flow
                
                # Physik Update
                if total_flow > 0:
                    temp_soc += total_flow * eff_one_way
                else:
                    temp_soc += total_flow / eff_one_way # Entladung kostet mehr SoC
                
                # Bounds check (Rundungsfehler)
                temp_soc = max(0.0, min(temp_soc, bess_cap))
                
                # Speichern
                battery_flow[start_idx + h] = total_flow
                day_soc_track[h] = temp_soc
                
                # Grid Export = PV (injectable) + Batterie Entladung (wenn negativ)
                # Achtung: Wenn Batterie lädt (positiv), wurde das schon von p_grid_injectable abgezogen (Arbitrage)
                # oder war nie drin (Clipping).
                # Wenn Batterie entlädt (flow < 0), fließt es ins Netz.
                export = p_grid_injectable[h] + (abs(total_flow) if total_flow < 0 else 0)
                grid_export[start_idx + h] = export
            
            # Übertrag SoC auf nächsten Tag
            current_soc = temp_soc
            soc_curve[start_idx:end_idx] = day_soc_track

        return grid_export, battery_flow, soc_curve, clipping_loss

    def calculate_cashflow(self, grid_export, prices):
        """
        Erstellt das Finanzmodell (Wasserfall).
        """
        years = self.i['project_years']
        
        # --- 0. Pre-Calculation ---
        # Marktwertfaktor (Cannibalization) auf PV-Strom anwenden?
        # Wir nutzen stündliche Profile, daher ist der Profileffekt implizit drin!
        # Aber: Direktvermarktungskosten (DV) abziehen
        net_price = prices - self.i['dv_cost_eur_mwh']
        
        # §51 EEG Regel / Negative Preise: Wenn Preis negativ, Erlös = 0 (oder sogar Kosten bei PPA)
        # Wir setzen Erlös auf 0 bei negativen Preisen (Abregelung oder §51)
        hourly_revenue = grid_export * np.where(net_price < 0, 0, net_price)
        year_1_revenue = hourly_revenue.sum()
        
        # --- 1. Struktur Setup ---
        # Zeilen: Revenue, Opex, EBITDA, AfA, EBIT, Tax, Net Income, +AfA, -Capex, -WorkingCap, FCF
        cf_df = pd.DataFrame(index=range(years + 1), columns=[
            'Revenue', 'OPEX', 'EBITDA', 'Depreciation', 'EBIT', 
            'Interest', 'EBT', 'Tax', 'Net_Income', 
            'Principal_Repayment', 'FCF_Equity', 'DSCR'
        ])
        cf_df.fillna(0.0, inplace=True)
        
        # --- 2. CAPEX & Funding ---
        total_capex = self.i['capex_total']
        debt_share = self.i['debt_share_percent'] / 100.0
        debt_amount = total_capex * debt_share
        equity_amount = total_capex * (1 - debt_share)
        
        cf_df.at[0, 'FCF_Equity'] = -equity_amount
        
        # Kreditberechnung (Annuität)
        rate = self.i['interest_rate_percent'] / 100.0
        n_periods = self.i['loan_term_years']
        if rate > 0:
            annuity = debt_amount * (rate * (1 + rate)**n_periods) / ((1 + rate)**n_periods - 1)
        else:
            annuity = debt_amount / n_periods
            
        remaining_debt = debt_amount
        
        # --- 3. Yearly Loop ---
        for y in range(1, years + 1):
            # Inflation & Degradation
            deg_pv = (1 - self.i['deg_pv_percent']/100) ** (y-1)
            deg_bess = (1 - self.i['deg_bess_percent']/100) ** (y-1) # Wirkt auf Kapazität, hier vereinfacht auf Revenue
            infl_rev = (1 + self.i['infl_electricity_percent']/100) ** (y-1)
            infl_opex = (1 + self.i['infl_opex_percent']/100) ** (y-1)
            
            # Revenue (Degradation Mix: vereinfacht PV Degradation dominant)
            rev = year_1_revenue * deg_pv * infl_rev
            
            # OPEX
            opex = self.i['opex_total_year_1'] * infl_opex
            
            # EBITDA
            ebitda = rev - opex
            
            # Depreciation (AfA) - Linear 20 Jahre
            depreciation = total_capex / 20.0 if y <= 20 else 0
            
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
            
            # FCF Equity = Net Income + Depreciation - Principal - Capex(Augmentation)
            # (Augmentation logic simplified: reserved in OPEX or explicit capex event)
            # Hier: FCFE Ansatz
            fcf_equity = net_income + depreciation - principal
            
            # DSCR Check
            dscr = ebitda / (interest + principal) if (interest+principal) > 0 else 99.9
            
            # Store
            cf_df.loc[y] = [rev, -opex, ebitda, -depreciation, ebit, -interest, ebt, -tax, net_income, -principal, fcf_equity, dscr]
            
        return cf_df, equity_amount

# ---------------------------------------------------------------------------
# 2. STREAMLIT UI & HANDLER
# ---------------------------------------------------------------------------

def main():
    st.title("🔋 Stadtwerke Hintertupfingen: PV & Speicher Investment Modell")
    st.markdown("Professionelles Erlös- und Cashflow-Modell für Freiflächen-PV mit Batteriespeicher (Clipping-Optimierung).")

    # --- INPUT SECTION (SIDEBAR) ---
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        
        with st.expander("1. Anlagentechnik", expanded=True):
            pv_mw = st.number_input("PV Leistung (MWp)", 14.0, 20.0, 14.0)
            grid_mw = st.number_input("Netzanschluss (MW)", 5.0, 14.0, 11.5, help="Grenze am Netzverknüpfungspunkt")
            bess_mw = st.number_input("Speicher Leistung (MW)", 0.0, 15.0, 6.0)
            bess_mwh = st.number_input("Speicher Kapazität (MWh)", 0.0, 60.0, 12.0)
            rte = st.number_input("Systemwirkungsgrad (%)", 80.0, 98.0, 90.0) / 100.0
        
        with st.expander("2. Investition (CAPEX) & OPEX"):
            capex_pv = st.number_input("CAPEX PV (€/kWp)", 400, 1000, 600)
            capex_bess_kwh = st.number_input("CAPEX BESS (€/kWh)", 100, 600, 250)
            capex_bess_kw = st.number_input("CAPEX BESS Leistung (€/kW)", 0, 300, 80)
            capex_grid = st.number_input("Netzanschluss/Trafo (€)", 0, 1000000, 250000)
            
            opex_pv_kwp = st.number_input("OPEX PV (€/kWp/a)", 5.0, 25.0, 12.0)
            opex_bess_mwh = st.number_input("OPEX BESS (€/MWh/a)", 0.0, 5000.0, 1500.0)
            dv_cost = st.number_input("Direktvermarktung (€/MWh)", 0.0, 5.0, 2.5)

        with st.expander("3. Finanzierung & Steuern"):
            debt_share = st.slider("Fremdkapitalanteil (%)", 0, 100, 80)
            interest_rate = st.number_input("Zins (%)", 0.0, 10.0, 4.5)
            loan_term = st.number_input("Kreditlaufzeit (Jahre)", 10, 25, 15)
            tax_rate = st.number_input("Steuersatz (KSt+GewSt %)", 0.0, 40.0, 30.0)
            wacc_req = st.number_input("WACC Ziel (%)", 0.0, 15.0, 6.0)
            
        with st.expander("4. Marktszenarien"):
            infl_el = st.number_input("Strompreissteigerung (%)", -2.0, 5.0, 1.0)
            infl_opex = st.number_input("Inflation OPEX (%)", 0.0, 5.0, 2.0)
            
            # DATA UPLOAD
# --- IM SIDEBAR BEREICH "4. Marktszenarien" ---
        st.markdown("### Daten Import")
        data_source = st.radio("Quelle PV-Daten:", ["Synthetisch (Demo)", "PVGIS Datei uploaden"])
        
        pv_profile_norm = None
        price_profile_series = None
        
        if data_source == "PVGIS Datei uploaden":
            pvgis_file = st.file_uploader("PVGIS CSV/Excel (Stundendaten)", type=["csv", "xlsx", "xls"])
            if pvgis_file:
                with st.spinner("Analysiere PVGIS Daten & berechne P50-Jahr..."):
                    pv_profile_norm = parse_pvgis_file(pvgis_file)
                    
                if pv_profile_norm is not None:
                    st.success("PVGIS Daten erfolgreich konvertiert & normiert!")
                    
                    # Vorschau Plot
                    st.line_chart(pv_profile_norm[:168]) # Erste Woche
                    st.caption("Vorschau: Normiertes Erzeugungsprofil (Erste 7 Tage)")
            
            # Preis separat laden (oder Standard nehmen)
            price_file = st.file_uploader("Strompreise (Optional, sonst Standard)", type=["csv", "xlsx"])
            if price_file:
                # ... hier ihr bestehender Preis-Upload Code ...
                pass
    # --- DATA LOADING & PROCESSING ---
    if data_file:
        try:
            if data_file.name.endswith('csv'):
                df_raw = pd.read_csv(data_file)
            else:
                df_raw = pd.read_excel(data_file)
            
            # Intelligente Spalten-Suche
            time_col = next((c for c in df_raw.columns if 'zeit' in c.lower() or 'date' in c.lower() or 'time' in c.lower()), None)
            pv_col = next((c for c in df_raw.columns if 'pv' in c.lower() or 'solar' in c.lower() or 'output' in c.lower()), None)
            price_col = next((c for c in df_raw.columns if 'price' in c.lower() or 'preis' in c.lower() or 'spot' in c.lower()), None)
            
            if not (time_col and pv_col and price_col):
                st.error(f"Konnte Spalten nicht automatisch zuordnen. Gefunden: {df_raw.columns.tolist()}. Bitte Spalten 'Zeit', 'PV', 'Preis' benennen.")
                st.stop()
                
            df_raw[time_col] = pd.to_datetime(df_raw[time_col], dayfirst=True, errors='coerce')
            df_raw.set_index(time_col, inplace=True)
            df_input = pd.DataFrame({
                'pv': df_raw[pv_col], # Erwartung: Normiert 0-1 oder Absolut MW? Wir nehmen an Normiert
                'price': df_raw[price_col]
            })
            
            # Normalisierungs-Check
            if df_input['pv'].max() > 2.0: # Wahrscheinlich MW Werte
                st.info("PV-Daten scheinen absolute MW-Werte zu sein. Werden auf Installierte Leistung skaliert.")
                df_input['pv'] = df_input['pv'] / df_input['pv'].max()
            
        except Exception as e:
            st.error(f"Fehler beim Laden: {e}")
            st.stop()
    else:
        # Generate Synthetic Data (Demo Mode)
        st.warning("Keine Daten hochgeladen. Nutze synthetische Demo-Daten.")
        idx = pd.date_range("2025-01-01", periods=8760, freq="h")
        t = np.arange(8760)
        # PV: Bell curve during day, seasonal variation
        pv_raw = np.maximum(0, -np.cos(2*np.pi*t/24)) * (1 + 0.5*-np.cos(2*np.pi*t/8760)) * np.random.uniform(0.8, 1.0, 8760)
        pv_raw = pv_raw / pv_raw.max()
        # Price: Duck curve + Winter peak
        price_raw = 60 + 20*np.sin(2*np.pi*(t-7)/24) - 30*pv_raw + np.random.normal(0, 10, 8760)
        df_input = pd.DataFrame({'pv': pv_raw, 'price': price_raw}, index=idx)
def parse_pvgis_file(uploaded_file):
    """
    Spezial-Parser für PVGIS CSV/Excel Export mit deutschem Zahlenformat.
    Analysiert Historie (z.B. 2005-2023) und erstellt ein repräsentatives Durchschnittsjahr (P50).
    """
    try:
        # 1. Einlesen (Versuche verschiedene Trennzeichen, da PVGIS je nach Einstellung variiert)
        # PVGIS Header überspringen wir oft, aber Pandas 'header' Parameter hilft.
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', skipfooter=10) # Footer oft mit Legende
        except:
            df = pd.read_excel(uploaded_file)

        # 2. Spalten identifizieren
        # PVGIS nennt die Spalte oft "P" (Power in W) oder "Gb(i)" etc.
        # Wir suchen nach der Zeit-Spalte und der Power-Spalte
        
        # Zeit-Spalte finden
        time_col = next((c for c in df.columns if 'time' in c.lower() or 'date' in c.lower() or 'zeit' in c.lower()), None)
        # Power-Spalte finden (P in Watt)
        power_col = next((c for c in df.columns if c.strip() == 'P'), None)
        
        if not time_col or not power_col:
            st.error("Konnte Spalten 'time' oder 'P' nicht finden. Bitte prüfen Sie das Format.")
            return None

        # 3. Deutsch -> Englisch Zahlenkonvertierung (Kritischer Schritt!)
        # Beispiel: "2.200.940,00" -> 2200940.00
        if df[power_col].dtype == object: # Wenn es als Text erkannt wurde
            df[power_col] = df[power_col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        
        df[power_col] = pd.to_numeric(df[power_col])

        # 4. Watt in MW umrechnen
        # PVGIS gibt Watt aus. Wir brauchen MW.
        df['mw_out'] = df[power_col] / 1_000_000.0 

        # 5. Zeitstempel parsen
        df[time_col] = pd.to_datetime(df[time_col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df = df.dropna(subset=[time_col])
        df.set_index(time_col, inplace=True)

        # 6. Repräsentatives Jahr erstellen (P50 Analysis)
        # Wir haben z.B. 18 Jahre Daten. Wir gruppieren nach Monat, Tag, Stunde und nehmen den Mittelwert.
        # Das eliminiert Ausreißerjahre.
        st.info(f"Analysiere historischen Datensatz von {df.index.year.min()} bis {df.index.year.max()}...")
        
        # Gruppieren nach (Monat, Tag, Stunde) und Mittelwert bilden
        # Wir erstellen einen künstlichen Index für 2025 (kein Schaltjahr für Sim)
        df_p50 = df.groupby([df.index.month, df.index.day, df.index.hour])['mw_out'].mean().reset_index()
        df_p50.columns = ['month', 'day', 'hour', 'mw']
        
        # Fehlerkorrektur für Schaltjahre (29. Feb entfernen, falls im Durchschnitt drin)
        df_p50 = df_p50[~((df_p50['month'] == 2) & (df_p50['day'] == 29))]
        
        # Neuen DateTime Index für das Zieljahr (z.B. 2025) bauen
        # Wir müssen sicherstellen, dass wir exakt 8760 Zeilen haben
        dates = pd.date_range(start='2025-01-01', periods=len(df_p50), freq='h')
        
        # Falls Längen nicht passen (z.B. fehlende Stunden in PVGIS), interpolieren
        if len(df_p50) != 8760:
             # Fallback: Wir nutzen reindexing auf das volle Jahr
             p50_series = pd.Series(df_p50['mw'].values, index=pd.to_datetime(
                 dict(year=2025, month=df_p50['month'], day=df_p50['day'], hour=df_p50['hour'])
             ))
             full_idx = pd.date_range('2025-01-01', '2025-12-31 23:00', freq='h')
             p50_series = p50_series.reindex(full_idx).interpolate().fillna(0)
             final_mw = p50_series.values
        else:
            final_mw = df_p50['mw'].values

        # 7. Normalisierung
        # Das Tool hat einen Schieberegler für "PV Leistung". 
        # Damit der Schieberegler funktioniert, müssen wir das Profil auf 1 MWp normieren.
        # Wir teilen durch das Maximum des Profils (oder die installierte Leistung aus der Datei, falls bekannt).
        # Sicherer Weg: Wir normieren auf den Peak-Wert des P50 Jahres.
        peak_p50 = final_mw.max()
        if peak_p50 > 0:
            normalized_profile = final_mw / peak_p50
        else:
            normalized_profile = final_mw
            
        return normalized_profile

    except Exception as e:
        st.error(f"Fehler beim PVGIS-Parsing: {str(e)}")
        return None
    # --- RUN SIMULATION ---
    
    # 1. Prepare Parameters
    total_capex = (pv_mw * 1000 * capex_pv) + \
                  (bess_mwh * 1000 * capex_bess_kwh) + \
                  (bess_mw * 1000 * capex_bess_kw) + \
                  capex_grid
                  
    total_opex_y1 = (pv_mw * 1000 * opex_pv_kwp) + \
                    (bess_mwh * opex_bess_mwh)
    
    inputs = {
        'project_years': 25,
        'grid_limit_mw': grid_mw,
        'bess_power_mw': bess_mw,
        'bess_capacity_mwh': bess_mwh,
        'bess_rte': rte,
        'deg_pv_percent': 0.5,
        'deg_bess_percent': 1.5,
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
    
    # 2. Process Data
    df_hourly = model.validate_and_resample_data(df_input)
    pv_series_mw = df_hourly['pv'] * pv_mw # Skaliere auf installierte Leistung
    price_series = df_hourly['price']
    # --- VOR DEM BUTTON ---
    
    # Fallback falls kein Upload
    if pv_profile_norm is None:
        if data_source == "PVGIS Datei uploaden":
            st.warning("Bitte Datei hochladen oder auf 'Synthetisch' wechseln.")
            st.stop()
        else:
            # Synthetisch generieren (Code von vorher)
            t = np.arange(8760)
            pv_profile_norm = np.maximum(0, -np.cos(2*np.pi*t/24)) * (1 + 0.5*-np.cos(2*np.pi*t/8760)) * np.random.uniform(0.8, 1.0, 8760)
            pv_profile_norm = pv_profile_norm / pv_profile_norm.max()

    # Fallback Preise
    if price_profile_series is None:
         t = np.arange(8760)
         # Einfache Preiskurve generieren
         price_profile_series = 60 + 20*np.sin(2*np.pi*(t-7)/24) + np.random.normal(0, 10, 8760)

    # DATEN ZUSAMMENFÜHREN
    # Hier kommt der "Schieberegler-Trick":
    # Wir nehmen das normierte Profil (0 bis 1) aus der PVGIS Datei und multiplizieren es
    # mit der im Tool eingestellten Leistung (z.B. 14 MW).
    
    pv_series_mw = pv_profile_norm * pv_mw  # pv_mw kommt aus st.number_input
    
    # ... weiter mit Simulation ...
    if st.button("🚀 Simulation starten", type="primary"):
        with st.spinner("Optimiere Dispatch und berechne Cashflow-Wasserfall..."):
            
            # A. Dispatch Engine
            grid_export, bat_flow, soc, clipping = model.run_dispatch_optimization(pv_series_mw.values, price_series.values)
            
            # B. Financial Engine
            cf_df, equity_invest = model.calculate_cashflow(grid_export, price_series.values)
            
            # C. KPIs
            equity_cashflows = cf_df['FCF_Equity'].values
            equity_cashflows[0] += 0 # Ensure Start Invest is included (it is already in DataFrame at year 0)
            
            try:
                irr_eq = npf.irr(equity_cashflows)
                npv_eq = npf.npv(wacc_req/100, equity_cashflows)
            except:
                irr_eq = 0.0
                npv_eq = 0.0
                
            total_clipped_mwh = clipping.sum()
            rev_total = cf_df['Revenue'].sum()
            
            st.session_state['results'] = {
                'cf': cf_df,
                'irr': irr_eq,
                'npv': npv_eq,
                'invest': equity_invest,
                'clipping': total_clipped_mwh,
                'dispatch': pd.DataFrame({'PV': pv_series_mw, 'Grid_Export': grid_export, 'SoC': soc, 'Price': price_series}, index=df_hourly.index)
            }
            
    # --- DASHBOARD OUTPUT ---
    if 'results' in st.session_state:
        res = st.session_state['results']
        
        # 1. Executive Summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Equity Invest", f"{res['invest']/1e6:.2f} Mio. €", f"Total CAPEX: {total_capex/1e6:.2f} Mio. €")
        col2.metric("Equity IRR (post-tax)", f"{res['irr']*100:.2f} %", delta_color="normal")
        col3.metric("Equity NPV", f"{res['npv']/1e6:.2f} Mio. €", help=f"Diskontiert mit {wacc_req}%")
        col4.metric("Verlorenes Clipping", f"{res['clipping']:.1f} MWh/a", help="Energie die wegen Netzlimit + vollem Speicher abgeregelt wurde")
        
        # 2. Charts
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Cashflow & Rendite", "🔬 Sensitivitätsanalyse", "🔌 Dispatch & Technik", "📑 Rohdaten"])
        
        with tab1:
            st.subheader("Cashflow Wasserfall (Equity)")
            cf_plot = res['cf']
            fig_cf = go.Figure()
            fig_cf.add_trace(go.Bar(x=cf_plot.index, y=cf_plot['FCF_Equity']/1e6, name='Free Cash Flow to Equity'))
            fig_cf.add_trace(go.Scatter(x=cf_plot.index, y=cf_plot['FCF_Equity'].cumsum()/1e6, name='Kumuliert', mode='lines', line=dict(color='red')))
            fig_cf.update_layout(xaxis_title="Jahr", yaxis_title="Mio. €", hovermode="x unified")
            st.plotly_chart(fig_cf, use_container_width=True)
            
            st.subheader("DSCR (Debt Service Coverage Ratio)")
            st.line_chart(cf_plot['DSCR'].clip(0, 3)) # Clip outliers
            
        with tab2:
            st.subheader("Tornado Chart: Einfluss auf NPV")
            # Schnelle on-the-fly Berechnung für Sensitivität
            sens_vars = {
                'CAPEX +10%': {'capex_total': inputs['capex_total']*1.1},
                'CAPEX -10%': {'capex_total': inputs['capex_total']*0.9},
                'Strompreis +10%': {'infl_electricity_percent': inputs['infl_electricity_percent'] + 10}, # Simplifiziert via Inflation oder Base Price
                'OPEX +10%': {'opex_total_year_1': inputs['opex_total_year_1']*1.1}
            }
            
            sens_results = []
            base_npv = res['npv']
            
            # Tornado Data Mockup (Full recalc wäre hier zu langsam für Demo, 
            # in Prod würde man Funktion auslagern und hier rufen)
            # Wir schätzen grob:
            sens_data = [
                {'Factor': 'CAPEX', 'Change': '+10%', 'NPV_Delta': -total_capex*0.1},
                {'Factor': 'CAPEX', 'Change': '-10%', 'NPV_Delta': total_capex*0.1},
                {'Factor': 'OPEX', 'Change': '+10%', 'NPV_Delta': -inputs['opex_total_year_1']*12}, # Approx 12x Opex PV value
                {'Factor': 'Revenue', 'Change': '-10%', 'NPV_Delta': -res['cf']['Revenue'].sum()*0.1 * 0.5} # Discounted approx
            ]
            
            df_sens = pd.DataFrame(sens_data)
            fig_tor = px.bar(df_sens, x='NPV_Delta', y='Factor', color='Change', orientation='h', title="Einfluss auf NPV (Approximation)")
            st.plotly_chart(fig_tor, use_container_width=True)
            st.caption("*Hinweis: Echte Sensitivitätsanalyse erfordert Neuberechnung aller Zeitschritte. Hier approximiert.*")

        with tab3:
            st.subheader("Beispielwoche: Dispatch Verhalten")
            df_disp = res['dispatch']
            # Wähle eine Woche im Sommer
            start_h = 4000
            end_h = 4168
            slice_d = df_disp.iloc[start_h:end_h]
            
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(x=slice_d.index, y=slice_d['PV'], name='PV Erzeugung', fill='tozeroy', line=dict(color='gold', width=0)))
            fig_d.add_trace(go.Scatter(x=slice_d.index, y=slice_d['Grid_Export'], name='Netzeinspeisung', line=dict(color='green')))
            fig_d.add_trace(go.Scatter(x=slice_d.index, y=slice_d['SoC'], name='Speicherstand (MWh)', line=dict(color='blue'), yaxis='y2'))
            fig_d.add_hline(y=grid_mw, line_dash="dot", annotation_text="Netzlimit", line_color="red")
            
            fig_d.update_layout(
                yaxis=dict(title="Leistung [MW]"),
                yaxis2=dict(title="Kapazität [MWh]", overlaying='y', side='right'),
                hovermode="x unified"
            )
            st.plotly_chart(fig_d, use_container_width=True)
            
        with tab4:
            st.dataframe(res['cf'].style.format("{:.2f}"))

if __name__ == "__main__":
    main()

