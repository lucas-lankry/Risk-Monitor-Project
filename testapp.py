import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import json
import os
import math

# Import Bloomberg API
try:
    import blpapi

    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False
    st.warning("⚠️ Bloomberg API non disponible. Installez blpapi: pip install blpapi")


# ============================================================================
# FONCTIONS BLOOMBERG
# ============================================================================

def start_bloomberg_session():
    """Démarre une session Bloomberg API"""
    if not BLOOMBERG_AVAILABLE:
        return None

    try:
        options = blpapi.SessionOptions()
        options.serverHost = "localhost"
        options.serverPort = 8194

        session = blpapi.Session(options)
        if not session.start():
            st.error("❌ Impossible de démarrer la session Bloomberg")
            return None
        if not session.openService("//blp/refdata"):
            st.error("❌ Impossible d'ouvrir le service Bloomberg refdata")
            return None

        return session
    except Exception as e:
        st.error(f"❌ Erreur Bloomberg: {str(e)}")
        return None

def get_bloomberg_data_batch(session, tickers, fields):
    """
    Récupère plusieurs champs Bloomberg pour plusieurs tickers en UNE SEULE requête
    """
    if session is None:
        return {}

    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")

    # Ajouter TOUS les tickers
    for ticker in tickers:
        request.append("securities", ticker)

    # Ajouter TOUS les champs
    for field in fields:
        request.append("fields", field)

    # Envoyer la requête
    session.sendRequest(request)

    # Dictionnaire pour stocker les résultats
    data = {ticker: {} for ticker in tickers}

    # Traiter la réponse
    while True:
        event = session.nextEvent()

        if event.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE):
            for msg in event:
                if msg.hasElement("securityData"):
                    security_data_array = msg.getElement("securityData")

                    for i in range(security_data_array.numValues()):
                        security_data = security_data_array.getValueAsElement(i)
                        ticker = security_data.getElementAsString("security")

                        if security_data.hasElement("fieldData"):
                            field_data = security_data.getElement("fieldData")

                            for field in fields:
                                try:
                                    if field_data.hasElement(field):
                                        value = field_data.getElementAsFloat(field)
                                        data[ticker][field] = value
                                    else:
                                        data[ticker][field] = None
                                except Exception:
                                    data[ticker][field] = None

        if event.eventType() == blpapi.Event.RESPONSE:
            break

    return data

def calculate_pnl(row, last_price):
    """
    Calcule le P&L selon la formule Excel
    D2 = Position (Size)
    Q2 = Put_Call (Type)
    G2 = Last_Price
    N2 = Strike_Px
    C2 = Settlement_Price
    O2 = Contract_Size
    """
    try:
        position = row.get('Size', row.get('Position', row.get('Position_Size', 0)))
        option_type = row.get('Type', row.get('Put_Call', '')).upper()
        strike = row.get('Strike Px', row.get('Strike_Px', row.get('Strike', 0)))
        settlement_price = row.get('Settlement Price', row.get('Settlement_Price', 0))
        contract_size = row.get('Contract_Size', row.get('Contract Size', 1000))

        # Si pas de last_price, on ne peut pas calculer
        if last_price is None or pd.isna(last_price):
            return None

        if strike == 0 or settlement_price == 0:
            return None

        # Formule Excel traduite en Python
        if position > 0:
            if option_type == 'C':
                pnl = (max(last_price - strike, 0) - settlement_price) * contract_size * position
            else:  # PUT
                pnl = (max(strike - last_price, 0) - settlement_price) * contract_size * position
        else:  # position <= 0 (short)
            if option_type == 'C':
                pnl = (settlement_price - max(last_price - strike, 0)) * contract_size * abs(position)
            else:  # PUT
                pnl = (settlement_price - max(strike - last_price, 0)) * contract_size * abs(position)

        return pnl

    except Exception as e:
        print(f"Erreur calcul P&L: {e}")
        return None

def fetch_bloomberg_greeks(df):
    """
    Récupère les Greeks et prix depuis Bloomberg pour toutes les positions
    RETOURNE UN NOUVEAU DATAFRAME sans modifier l'original
    """
    session = start_bloomberg_session()

    if session is None:
        st.warning("⚠️ Bloomberg non disponible - utilisation des calculs locaux")
        return pd.DataFrame()

    # Mapping des champs Bloomberg
    bloomberg_fields = ['BID', 'ASK', 'PX_LAST', 'IVOL_MID',
                        'OPT_DELTA', 'OPT_GAMMA', 'OPT_VEGA',
                        'OPT_THETA', 'OPT_RHO']

    tickers = df['Ticker'].tolist()

    with st.spinner(f"📡 Récupération des données Bloomberg pour {len(tickers)} positions..."):
        try:
            bloomberg_data = get_bloomberg_data_batch(session, tickers, bloomberg_fields)

            # Créer un nouveau DataFrame avec les résultats Bloomberg
            bbg_results = []

            for idx, row in df.iterrows():
                ticker = row.get('Ticker', '')
                # Utiliser 'Size' au lieu de 'Position_Size'
                position_size = row.get('Size', row.get('Position_Size', 0))
                strike = row.get('Strike Px',row.get('Strike_Px', row.get('Strike', 0)))  # MODIFIÉ - essaie différents noms
                settlement_price = row.get('Settlement Price', row.get('Settlement_Price', 0))

                if ticker in bloomberg_data:
                    last_price = bloomberg_data[ticker].get('PX_LAST', None)

                    result = {
                        'Product': ticker,
                        'Position_Size': position_size,
                        'Strike_px': strike,  # Cette colonne existera maintenant
                        'Settlement_Price': settlement_price,
                        'Bid': bloomberg_data[ticker].get('BID', None),
                        'Ask': bloomberg_data[ticker].get('ASK', None),
                        'Last_Price': bloomberg_data[ticker].get('PX_LAST', None),
                        'IV': bloomberg_data[ticker].get('IVOL_MID', None),
                        'Delta': bloomberg_data[ticker].get('OPT_DELTA', None),
                        'Gamma': bloomberg_data[ticker].get('OPT_GAMMA', None),
                        'Vega': bloomberg_data[ticker].get('OPT_VEGA', None),
                        'Theta': bloomberg_data[ticker].get('OPT_THETA', None),
                        'Rho': bloomberg_data[ticker].get('OPT_RHO', None),
                        'PnL': calculate_pnl(row, last_price)
                    }
                else:
                    result = {
                        'Product': ticker,
                        'Position_Size': position_size,
                        'Strike_px': strike,
                        'Settlement_Price': settlement_price,
                        'Bid': None,
                        'Ask': None,
                        'Last_Price': None,
                        'IV': None,
                        'Delta': None,
                        'Gamma': None,
                        'Vega': None,
                        'Theta': None,
                        'Rho': None,
                        'PnL': None
                    }

                bbg_results.append(result)

            st.success(f"✅ Données Bloomberg récupérées pour {len(tickers)} positions")
            return pd.DataFrame(bbg_results)

        except Exception as e:
            st.error(f"❌ Erreur lors de la récupération Bloomberg: {str(e)}")
            return pd.DataFrame()
        finally:
            session.stop()


# ============================================================================
# FONCTIONS DE CALCUL DES GREEKS (Black-76 et Bachelier)
# ============================================================================

def norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black76_greeks(F: float, K: float, T: float, r: float, sigma: float, option_type: str):
    """Black-76 greeks pour une option sur future."""
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "vega": None, "theta": None, "rho": None}

    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    df = math.exp(-r * T)

    Nd1 = norm_cdf(d1)
    nd1 = norm_pdf(d1)

    option_type = option_type.upper()[0]

    if option_type == "C":
        delta = df * Nd1
    else:
        delta = df * (Nd1 - 1.0)

    gamma = df * nd1 / (F * sigma * sqrtT)
    vega = df * F * nd1 * sqrtT
    theta = None
    rho = None

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def bachelier_greeks(F: float, K: float, T: float, r: float, sigma: float, option_type: str):
    """Bachelier model greeks pour une option sur future."""
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "vega": None, "theta": None, "rho": None, "price": None}

    sqrtT = math.sqrt(T)
    df = math.exp(-r * T)
    sigma_normal = sigma * F

    d = (F - K) / (sigma_normal * sqrtT)
    Nd = norm_cdf(d)
    nd = norm_pdf(d)

    option_type = option_type.upper()[0]

    if option_type == "C":
        price = df * ((F - K) * Nd + sigma_normal * sqrtT * nd)
        delta = df * Nd
    else:
        price = df * ((K - F) * norm_cdf(-d) + sigma_normal * sqrtT * nd)
        delta = df * (Nd - 1.0)

    gamma = df * nd / (sigma_normal * sqrtT)
    vega = df * sqrtT * nd
    theta = None
    rho = None

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho, "price": price}


def safe_time_to_maturity(maturity_value, today: date) -> float:
    """Convertit la maturité en T (années)."""
    if isinstance(maturity_value, (datetime, pd.Timestamp)):
        maturity_date = maturity_value.date()
    else:
        try:
            m = pd.to_datetime(maturity_value, errors="coerce", dayfirst=True)
        except Exception:
            m = pd.NaT

        if pd.isna(m):
            return 0.00001
        maturity_date = m.date()

    days = (maturity_date - today).days
    T = days / 365.0
    if T <= 0:
        T = 0.00001
    return T


def compute_b76_greeks_for_position(row, bloomberg_df, risk_free_rate=0.05, valuation_date=None):
    """Calcule les Greeks Black-76 pour une position"""
    if valuation_date is None:
        valuation_date = datetime.today().date()

    try:
        if "Maturity" not in row or pd.isna(row["Maturity"]):
            return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None}

        T = safe_time_to_maturity(row["Maturity"], valuation_date)
        F = row.get("Settlement_Price", row.get("Settlement Price", 0))
        K = row.get("Strike", row.get("Strike_Px", 0))

        if F == 0 or K == 0:
            return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None}

        # Récupérer l'IV depuis Bloomberg si disponible
        ticker = row.get('Ticker', '')
        iv = None
        if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
            bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
            iv = bbg_row.get('IV', None)

        # Si pas d'IV Bloomberg, utiliser la colonne du tableau original
        if iv is None or pd.isna(iv):
            iv = row.get("IV")

        if iv in (None, '', ' ') or pd.isna(iv):
            sigma = 0.30
        else:
            sigma = float(iv) / 100.0

        option_type = row.get("Type", row.get("Put_Call", "C"))
        # Utiliser 'Size' au lieu de 'Position_Size'
        position_size = row.get("Size", row.get("Position_Size", 1))

        greeks = black76_greeks(
            F=float(F), K=float(K), T=float(T),
            r=float(risk_free_rate), sigma=float(sigma),
            option_type=option_type
        )

        return {
            "Delta": greeks["delta"] * position_size if greeks["delta"] is not None else None,
            "Gamma": greeks["gamma"] * position_size if greeks["gamma"] is not None else None,
            "Vega": greeks["vega"] * position_size if greeks["vega"] is not None else None,
            "Theta": greeks["theta"],
            "Rho": greeks["rho"]
        }
    except Exception as e:
        print(f"Erreur calcul B76: {e}")
        return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None}


def compute_bachelier_greeks_for_position(row, bloomberg_df, risk_free_rate=0.05, valuation_date=None):
    """Calcule les Greeks Bachelier pour une position"""
    if valuation_date is None:
        valuation_date = datetime.today().date()

    try:
        if "Maturity" not in row or pd.isna(row["Maturity"]):
            return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None, "Price": None}

        T = safe_time_to_maturity(row["Maturity"], valuation_date)
        F = row.get("Settlement_Price", row.get("Settlement Price", 0))
        K = row.get("Strike", row.get("Strike_Px", 0))

        if F == 0 or K == 0:
            return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None, "Price": None}

        # Récupérer l'IV depuis Bloomberg si disponible
        ticker = row.get('Ticker', '')
        iv = None
        if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
            bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
            iv = bbg_row.get('IV', None)

        # Si pas d'IV Bloomberg, utiliser la colonne du tableau original
        if iv is None or pd.isna(iv):
            iv = row.get("IV")

        if iv in (None, '', ' ') or pd.isna(iv):
            sigma = 0.30
        else:
            sigma = float(iv) / 100.0

        option_type = row.get("Type", row.get("Put_Call", "C"))
        # Utiliser 'Size' au lieu de 'Position_Size'
        position_size = row.get("Size", row.get("Position_Size", 1))

        greeks = bachelier_greeks(
            F=float(F), K=float(K), T=float(T),
            r=float(risk_free_rate), sigma=float(sigma),
            option_type=option_type
        )

        return {
            "Delta": greeks["delta"] * position_size if greeks["delta"] is not None else None,
            "Gamma": greeks["gamma"] * position_size if greeks["gamma"] is not None else None,
            "Vega": greeks["vega"] * position_size if greeks["vega"] is not None else None,
            "Theta": greeks["theta"],
            "Rho": greeks["rho"],
            "Price": greeks["price"]
        }
    except Exception as e:
        print(f"Erreur calcul Bachelier: {e}")
        return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None, "Price": None}


# ============================================================================
# CONFIGURATION STREAMLIT
# ============================================================================

st.set_page_config(page_title="Options Risk Monitor", layout="wide", initial_sidebar_state="expanded")

st.title("Options Risk Monitor - Futures")
st.markdown("---")

# Initialiser le session state
if 'positions' not in st.session_state:
    st.session_state.positions = pd.DataFrame(columns=[
        'Ticker', 'Strike', 'Size', 'Maturity', 'Settlement_Price', 'Contract_Size', 'Type'
    ])

if 'b76_greeks' not in st.session_state:
    st.session_state.b76_greeks = pd.DataFrame()

if 'bachelier_greeks' not in st.session_state:
    st.session_state.bachelier_greeks = pd.DataFrame()

if 'bloomberg_greeks' not in st.session_state:
    st.session_state.bloomberg_greeks = pd.DataFrame()

if 'rates_data' not in st.session_state:
    st.session_state.rates_data = None

if 'show_positions' not in st.session_state:
    st.session_state.show_positions = False

if 'risk_free_rate' not in st.session_state:
    st.session_state.risk_free_rate = 0.05


# ============================================================================
# FONCTION DE CALCUL PRINCIPALE
# ============================================================================

def run_calculation():
    """Calcule les Greeks pour toutes les positions"""
    if st.session_state.positions.empty:
        st.warning("Aucune position à calculer")
        return

    # 1. RÉCUPÉRATION DES DONNÉES BLOOMBERG (sans modifier positions)
    bloomberg_df = fetch_bloomberg_greeks(st.session_state.positions)
    st.session_state.bloomberg_greeks = bloomberg_df

    # 2. CALCUL BLACK-76
    b76_results = []
    for idx, row in st.session_state.positions.iterrows():
        greeks = compute_b76_greeks_for_position(row, bloomberg_df, st.session_state.risk_free_rate)

        ticker = row.get('Ticker', '')
        position_size = row.get('Size', row.get('Position_Size', 0))
        strike = row.get('Strike Px', row.get('Strike_Px', row.get('Strike', 0)))  # MODIFIÉ
        settlement_price = row.get('Settlement Price', row.get('Settlement_Price', 0))

        # Récupérer les données de prix depuis Bloomberg
        bid = None
        ask = None
        last_price = None
        iv = None

        if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
            bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
            bid = bbg_row.get('Bid', None)
            ask = bbg_row.get('Ask', None)
            last_price = bbg_row.get('Last_Price', None)
            iv = bbg_row.get('IV', None)

        result = {
            'Product': ticker,
            'Position_Size': position_size,
            'Strike_px': strike,
            'Settlement_Price': settlement_price,
            'Bid': bid,
            'Ask': ask,
            'Last_Price': last_price,
            'IV': iv,
            'Delta': greeks['Delta'],
            'Gamma': greeks['Gamma'],
            'Vega': greeks['Vega'],
            'Theta': greeks['Theta'],
            'Rho': greeks['Rho'],
            'PnL': calculate_pnl(row, last_price)
        }
        b76_results.append(result)

    st.session_state.b76_greeks = pd.DataFrame(b76_results)

    # 3. CALCUL BACHELIER
    bach_results = []
    for idx, row in st.session_state.positions.iterrows():
        greeks = compute_bachelier_greeks_for_position(row, bloomberg_df, st.session_state.risk_free_rate)

        ticker = row.get('Ticker', '')
        position_size = row.get('Size', row.get('Position_Size', 0))
        strike = row.get('Strike Px', row.get('Strike_Px', row.get('Strike', 0)))  # MODIFIÉ
        settlement_price = row.get('Settlement Price', row.get('Settlement_Price', 0))

        # Récupérer les données de prix depuis Bloomberg
        bid = None
        ask = None
        last_price = None
        iv = None

        if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
            bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
            bid = bbg_row.get('Bid', None)
            ask = bbg_row.get('Ask', None)
            last_price = bbg_row.get('Last_Price', None)
            iv = bbg_row.get('IV', None)

        result = {
            'Product': ticker,
            'Position_Size': position_size,
            'Strike_px': strike,
            'Settlement_Price': settlement_price,
            'Bid': bid,
            'Ask': ask,
            'Last_Price': last_price,
            'IV': iv,
            'Delta': greeks['Delta'],
            'Gamma': greeks['Gamma'],
            'Vega': greeks['Vega'],
            'Theta': greeks['Theta'],
            'Rho': greeks['Rho'],
            'PnL': calculate_pnl(row, last_price)
        }
        bach_results.append(result)

    st.session_state.bachelier_greeks = pd.DataFrame(bach_results)

    st.success("✅ Greeks calculés avec succès!")

def save_data():
    data = {
        'positions': st.session_state.positions.to_dict(),
        'b76_greeks': st.session_state.b76_greeks.to_dict(),
        'bachelier_greeks': st.session_state.bachelier_greeks.to_dict(),
        'bloomberg_greeks': st.session_state.bloomberg_greeks.to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    with open('risk_monitor_data.json', 'w') as f:
        json.dump(data, f)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("Configuration")

    st.markdown("### Bloomberg Status")
    if BLOOMBERG_AVAILABLE:
        st.success("✅ Bloomberg API installée")
    else:
        st.error("❌ Bloomberg API non disponible")
        st.code("pip install blpapi", language="bash")

    mode = st.radio(
        "Mode de travail",
        ["Saisie manuelle", "Import Excel"]
    )

    if mode == "Import Excel":
        uploaded_file = st.file_uploader("Charger le fichier Excel", type=['xlsx', 'xls'])

        if uploaded_file:
            try:
                xls = pd.ExcelFile(uploaded_file)
                selected_sheet = st.selectbox("Selectionner l'onglet", xls.sheet_names)

                if st.button("Importer les donnees"):
                    st.session_state.positions = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

                    if 'B76_Greeks' in xls.sheet_names:
                        st.session_state.b76_greeks = pd.read_excel(uploaded_file, sheet_name='B76_Greeks')
                    if 'Bachelier_Greeks' in xls.sheet_names:
                        st.session_state.bachelier_greeks = pd.read_excel(uploaded_file, sheet_name='Bachelier_Greeks')
                    if 'US_Rates_Curve' in xls.sheet_names:
                        st.session_state.rates_data = pd.read_excel(uploaded_file, sheet_name='US_Rates_Curve')

                    st.success("Donnees importees avec succes")
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de l'import : {e}")

    st.markdown("---")

    st.subheader("Actions")

    if st.button("🔄 Actualiser avec Bloomberg", use_container_width=True, type="primary"):
        if not st.session_state.positions.empty:
            with st.spinner("Calcul en cours..."):
                run_calculation()
                save_data()
                st.rerun()
        else:
            st.warning("Aucune position a calculer")

    if st.button("Modifier ou Supprimer une position", use_container_width=True):
        st.session_state.show_positions = True

    st.markdown("---")

    st.subheader("Export")

    if not st.session_state.positions.empty:
        if st.button("Export Excel", use_container_width=True):
            output_file = f"risk_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                st.session_state.positions.to_excel(writer, sheet_name='Positions', index=False)

                if not st.session_state.b76_greeks.empty:
                    st.session_state.b76_greeks.to_excel(writer, sheet_name='B76_Greeks', index=False)

                if not st.session_state.bachelier_greeks.empty:
                    st.session_state.bachelier_greeks.to_excel(writer, sheet_name='Bachelier_Greeks', index=False)

                if not st.session_state.bloomberg_greeks.empty:
                    st.session_state.bloomberg_greeks.to_excel(writer, sheet_name='Bloomberg_Greeks', index=False)

                if st.session_state.rates_data is not None:
                    st.session_state.rates_data.to_excel(writer, sheet_name='US_Rates_Curve', index=False)

            st.success(f"Fichier exporte : {output_file}")

        csv = st.session_state.positions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export CSV",
            data=csv,
            file_name=f"positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================================
# AFFICHAGE PRINCIPAL
# ============================================================================

df = st.session_state.positions

# Metrics en haut
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Nombre de Positions",
            value=len(df),
        )

    with col2:
        if 'Size' in df.columns:
            total_position = df['Size'].sum()
            st.metric(
                label="Position Totale",
                value=f"{int(total_position)}",
            )

    with col3:
        if not st.session_state.b76_greeks.empty and 'Delta' in st.session_state.b76_greeks.columns:
            total_delta = st.session_state.b76_greeks['Delta'].dropna().sum()
            st.metric(
                label="Delta Total (B76)",
                value=f"{total_delta:.2f}",
                delta=f"{total_delta:.2f}" if total_delta != 0 else None
            )
        else:
            st.metric(label="Delta Total", value="0.00")

    with col4:
        if not st.session_state.b76_greeks.empty and 'Gamma' in st.session_state.b76_greeks.columns:
            total_gamma = st.session_state.b76_greeks['Gamma'].dropna().sum()
            st.metric(
                label="Gamma Total (B76)",
                value=f"{total_gamma:.4f}",
                delta=f"{total_gamma:.4f}" if total_gamma != 0 else None
            )
        else:
            st.metric(label="Gamma Total", value="0.00")

    st.markdown("---")

# Tabs pour différentes vues
if st.session_state.show_positions:
    if mode == "Saisie manuelle":
        selected_tab = st.radio("Navigation",
                                ["Nouvelle Position", "Positions", "Greeks", "Courbe des Taux", "Analyse"],
                                index=1, horizontal=True, label_visibility="collapsed")
    else:
        selected_tab = st.radio("Navigation", ["Positions", "Greeks", "Courbe des Taux", "Analyse"],
                                index=0, horizontal=True, label_visibility="collapsed")
    st.session_state.show_positions = False
else:
    if mode == "Saisie manuelle":
        selected_tab = st.radio("Navigation",
                                ["Nouvelle Position", "Positions", "Greeks", "Courbe des Taux", "Analyse"],
                                index=0, horizontal=True, label_visibility="collapsed")
    else:
        selected_tab = st.radio("Navigation", ["Positions", "Greeks", "Courbe des Taux", "Analyse"],
                                index=0, horizontal=True, label_visibility="collapsed")

if mode == "Saisie manuelle" and selected_tab == "Nouvelle Position":
    st.subheader("Ajouter une nouvelle position")

    with st.form("new_position_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            ticker = st.text_input("Ticker", value="CLG6C")
            strike = st.number_input("Strike", value=58.0, step=0.5)

        with col2:
            position_size = st.number_input("Size", value=10, step=1)
            maturity = st.text_input("Maturity", value="FEB 26")

        with col3:
            settlement_price = st.number_input("Settlement Price", value=0.0, step=0.01)
            contract_size = st.number_input("Contract Size", value=1000, step=100)

        with col4:
            option_type = st.selectbox("Type", ["C", "P"])

        submitted = st.form_submit_button("Ajouter la position", use_container_width=True)

        if submitted:
            new_row = {
                'Ticker': ticker,
                'Strike': strike,
                'Size': position_size,
                'Maturity': maturity,
                'Settlement_Price': settlement_price,
                'Contract_Size': contract_size,
                'Type': option_type
            }

            st.session_state.positions = pd.concat([
                st.session_state.positions,
                pd.DataFrame([new_row])
            ], ignore_index=True)

            save_data()
            st.success("Position ajoutee ! Cliquez sur Actualiser pour calculer les Greeks.")
            st.rerun()

if selected_tab == "Positions":
    st.subheader("Positions actuelles")

    if not df.empty:
        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            if 'Type' in df.columns:
                option_type = st.multiselect(
                    "Type d'option",
                    options=df['Type'].unique(),
                    default=df['Type'].unique()
                )
                df_filtered = df[df['Type'].isin(option_type)]
            else:
                df_filtered = df

        with col_filter2:
            if 'Maturity' in df.columns:
                maturities = st.multiselect(
                    "Maturite",
                    options=sorted(df['Maturity'].unique()),
                    default=sorted(df['Maturity'].unique())
                )
                df_filtered = df_filtered[df_filtered['Maturity'].isin(maturities)]

        st.markdown("### Tableau des Positions")
        st.dataframe(df_filtered, use_container_width=True, height=400)

        st.markdown("### Modifier ou Supprimer une position")
        st.info(
            "Editez directement les valeurs dans le tableau ci-dessous. Les modifications seront sauvegardees automatiquement.")

        edited_df = st.data_editor(
            df_filtered,
            use_container_width=True,
            height=400,
            num_rows="dynamic",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                "Size": st.column_config.NumberColumn("Size", format="%.0f"),
                "Maturity": st.column_config.TextColumn("Maturity"),
                "Settlement_Price": st.column_config.NumberColumn("Settlement Price", format="%.2f"),
                "Contract_Size": st.column_config.NumberColumn("Contract Size", format="%.0f"),
                "Type": st.column_config.SelectboxColumn("Type", options=["C", "P"])
            }
        )

        if not edited_df.equals(df_filtered):
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Sauvegarder les modifications", use_container_width=True):
                    for idx in edited_df.index:
                        if idx in df.index:
                            st.session_state.positions.loc[idx] = edited_df.loc[idx]

                    deleted_indices = set(df_filtered.index) - set(edited_df.index)
                    if deleted_indices:
                        st.session_state.positions = st.session_state.positions.drop(deleted_indices).reset_index(
                            drop=True)

                    save_data()
                    st.success("Modifications sauvegardees")
                    st.rerun()
    else:
        st.info("Aucune position enregistree. Ajoutez une position ou importez un fichier Excel.")

if selected_tab == "Greeks":
    st.subheader("Analyse des Greeks")

    # Onglet Bloomberg en premier
    st.markdown("### Bloomberg Greeks")
    if not st.session_state.bloomberg_greeks.empty:
        # Créer une copie du dataframe avec les Greeks pondérés
        display_df = st.session_state.bloomberg_greeks.copy()
        display_df['Delta'] = display_df['Delta'] * display_df['Position_Size']
        display_df['Gamma'] = display_df['Gamma'] * display_df['Position_Size']
        display_df['Vega'] = display_df['Vega'] * display_df['Position_Size']
        display_df['Theta'] = display_df['Theta'] * display_df['Position_Size']
        display_df['Rho'] = display_df['Rho'] * display_df['Position_Size']

        st.dataframe(
            display_df,
            use_container_width=True,
            height=250,
            column_config={
                "Position_Size": st.column_config.NumberColumn("Position Size", format="%.0f"),
                "Strike_px": st.column_config.NumberColumn("Strike", format="%.2f"),
                "Settlement_Price": st.column_config.NumberColumn("Settlement Price", format="%.4f"),
                "Bid": st.column_config.NumberColumn("Bid", format="%.4f"),
                "Ask": st.column_config.NumberColumn("Ask", format="%.4f"),
                "Last_Price": st.column_config.NumberColumn("Last Price", format="%.4f"),
                "IV": st.column_config.NumberColumn("IV (%)", format="%.2f"),
                "Delta": st.column_config.NumberColumn("Delta", format="%.4f"),
                "Gamma": st.column_config.NumberColumn("Gamma", format="%.6f"),
                "Vega": st.column_config.NumberColumn("Vega", format="%.4f"),
                "Theta": st.column_config.NumberColumn("Theta", format="%.4f"),
                "Rho": st.column_config.NumberColumn("Rho", format="%.4f"),
                "PnL": st.column_config.NumberColumn("P&L", format="%.2f")  # AJOUT
            }
        )

        # Afficher les totaux Bloomberg
        col1, col2, col3 = st.columns(3)
        with col1:
            total_delta_bbg = display_df['Delta'].dropna().sum()
            st.metric("Delta Total", f"{total_delta_bbg:.4f}")
        with col2:
            total_gamma_bbg = display_df['Gamma'].dropna().sum()
            st.metric("Gamma Total", f"{total_gamma_bbg:.6f}")
        with col3:
            total_vega_bbg = display_df['Vega'].dropna().sum()
            st.metric("Vega Total", f"{total_vega_bbg:.4f}")
    else:
        st.info("Aucune donnee Bloomberg. Cliquez sur 'Actualiser avec Bloomberg' pour calculer.")

    st.markdown("---")

    st.markdown("### Black-76 Model")
    if not st.session_state.b76_greeks.empty:
        # Créer une copie du dataframe avec les Greeks pondérés
        display_df_b76 = st.session_state.b76_greeks.copy()
        display_df_b76['Delta'] = display_df_b76['Delta'] * display_df_b76['Position_Size']
        display_df_b76['Gamma'] = display_df_b76['Gamma'] * display_df_b76['Position_Size']
        display_df_b76['Vega'] = display_df_b76['Vega'] * display_df_b76['Position_Size']

        st.dataframe(
            display_df_b76,
            use_container_width=True,
            height=250,
            column_config={
                "Position_Size": st.column_config.NumberColumn("Position Size", format="%.0f"),
                "Bid": st.column_config.NumberColumn("Bid", format="%.4f"),
                "Ask": st.column_config.NumberColumn("Ask", format="%.4f"),
                "Last_Price": st.column_config.NumberColumn("Last Price", format="%.4f"),
                "IV": st.column_config.NumberColumn("IV (%)", format="%.2f"),
                "Delta": st.column_config.NumberColumn("Delta", format="%.4f"),
                "Gamma": st.column_config.NumberColumn("Gamma", format="%.6f"),
                "Vega": st.column_config.NumberColumn("Vega", format="%.4f")
            }
        )

        # Afficher les totaux B76
        col1, col2, col3 = st.columns(3)
        with col1:
            total_delta_b76 = display_df_b76['Delta'].dropna().sum()
            st.metric("Delta Total", f"{total_delta_b76:.4f}")
        with col2:
            total_gamma_b76 = display_df_b76['Gamma'].dropna().sum()
            st.metric("Gamma Total", f"{total_gamma_b76:.6f}")
        with col3:
            total_vega_b76 = display_df_b76['Vega'].dropna().sum()
            st.metric("Vega Total", f"{total_vega_b76:.4f}")
    else:
        st.info("Aucune donnee B76. Cliquez sur 'Actualiser avec Bloomberg' pour calculer.")

    st.markdown("---")

    st.markdown("### Bachelier Model")
    if not st.session_state.bachelier_greeks.empty:
        # Créer une copie du dataframe avec les Greeks pondérés
        display_df_bach = st.session_state.bachelier_greeks.copy()
        display_df_bach['Delta'] = display_df_bach['Delta'] * display_df_bach['Position_Size']
        display_df_bach['Gamma'] = display_df_bach['Gamma'] * display_df_bach['Position_Size']
        display_df_bach['Vega'] = display_df_bach['Vega'] * display_df_bach['Position_Size']

        st.dataframe(
            display_df_bach,
            use_container_width=True,
            height=250,
            column_config={
                "Position_Size": st.column_config.NumberColumn("Position Size", format="%.0f"),
                "Bid": st.column_config.NumberColumn("Bid", format="%.4f"),
                "Ask": st.column_config.NumberColumn("Ask", format="%.4f"),
                "Last_Price": st.column_config.NumberColumn("Last Price", format="%.4f"),
                "IV": st.column_config.NumberColumn("IV (%)", format="%.2f"),
                "Delta": st.column_config.NumberColumn("Delta", format="%.4f"),
                "Gamma": st.column_config.NumberColumn("Gamma", format="%.6f"),
                "Vega": st.column_config.NumberColumn("Vega", format="%.4f")
            }
        )

        # Afficher les totaux Bachelier
        col1, col2, col3 = st.columns(3)
        with col1:
            total_delta_bach = display_df_bach['Delta'].dropna().sum()
            st.metric("Delta Total", f"{total_delta_bach:.4f}")
        with col2:
            total_gamma_bach = display_df_bach['Gamma'].dropna().sum()
            st.metric("Gamma Total", f"{total_gamma_bach:.6f}")
        with col3:
            total_vega_bach = display_df_bach['Vega'].dropna().sum()
            st.metric("Vega Total", f"{total_vega_bach:.4f}")
    else:
        st.info("Aucune donnee Bachelier. Cliquez sur 'Actualiser avec Bloomberg' pour calculer.")

if selected_tab == "Courbe des Taux":
    st.subheader("Courbe des Taux US Treasury")

    if st.session_state.rates_data is not None:
        rates_df = st.session_state.rates_data

        if 'Tenor' in rates_df.columns and 'Rate_%' in rates_df.columns:
            fig_rates = go.Figure()

            fig_rates.add_trace(go.Scatter(
                x=rates_df['Tenor'],
                y=rates_df['Rate_%'],
                mode='lines+markers',
                name='Yield Curve',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))

            fig_rates.update_layout(
                title='US Treasury Yield Curve',
                xaxis_title='Maturity (Years)',
                yaxis_title='Yield (%)',
                hovermode='x unified',
                height=500
            )

            st.plotly_chart(fig_rates, use_container_width=True)
            st.dataframe(rates_df, use_container_width=True)
    else:
        st.info("Aucune donnee de courbe des taux. Importez un fichier Excel contenant l'onglet 'US_Rates_Curve'.")

if selected_tab == "Analyse":
    st.subheader("Analyse de Risque")

    if not df.empty:
        analysis_col1, analysis_col2 = st.columns(2)

        with analysis_col1:
            if 'Type' in df.columns and 'Size' in df.columns:
                put_call_dist = df.groupby('Type')['Size'].sum().reset_index()
                fig_pc = px.pie(
                    put_call_dist,
                    values='Size',
                    names='Type',
                    title='Distribution Put/Call',
                    color='Type',
                    color_discrete_map={'C': '#2ecc71', 'P': '#e74c3c'}
                )
                st.plotly_chart(fig_pc, use_container_width=True)

        with analysis_col2:
            if 'Strike' in df.columns and 'Size' in df.columns:
                position_by_strike = df.groupby('Strike')['Size'].sum().reset_index()
                fig_strike = px.bar(
                    position_by_strike,
                    x='Strike',
                    y='Size',
                    title='Position par Strike',
                    labels={'Strike': 'Strike Price', 'Size': 'Position'}
                )
                st.plotly_chart(fig_strike, use_container_width=True)
    else:
        st.info("Aucune donnee disponible pour l'analyse.")

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center'>
        <p>Risk Monitor v2.1 | Derniere mise a jour : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    """,
    unsafe_allow_html=True
)