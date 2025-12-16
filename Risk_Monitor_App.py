import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm
from datetime import datetime, date
import json
import os
import math
import smtplib
import socket
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import io

# Import SendGrid
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
    import base64
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

# Import Bloomberg API
try:
    import blpapi

    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False
    st.warning("⚠️ Bloomberg API non disponible. Installez blpapi: pip install blpapi")

# Définition des tenors standard pour la courbe US Treasury
TREASURY_TICKERS = {
    '1M': 'USGG1M Index',    # 1 Month
    '3M': 'USGG3M Index',    # 3 Month
    '6M': 'USGG6M Index',    # 6 Month
    '1Y': 'USGG1YR Index',   # 1 Year
    '2Y': 'USGG2YR Index',   # 2 Year
    '3Y': 'USGG3YR Index',   # 3 Year
    '5Y': 'USGG5YR Index',   # 5 Year
    '7Y': 'USGG7YR Index',   # 7 Year
    '10Y': 'USGG10YR Index', # 10 Year
    '20Y': 'USGG20YR Index', # 20 Year
    '30Y': 'USGG30YR Index'  # 30 Year
}

# Conversion des tenors en années
TENOR_TO_YEARS = {
    '1M': 1/12, '3M': 3/12, '6M': 6/12,
    '1Y': 1, '2Y': 2, '3Y': 3, '5Y': 5,
    '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30
}
# ============================================================================
# FONCTIONS BLOOMBERG, RATES, EMAIL
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
        raw_type = str(row.get("Put/Call", row.get("Put_Call", row.get("Type", "C")))).strip().upper()
        option_type = "P" if raw_type.startswith("P") else "C"
        strike = row.get('Strike Px', row.get('Strike_Px', row.get('Strike', 0)))
        settlement_price = row.get('Settlement Price', row.get('Settlement_Price', 0))
        contract_size = row.get('Contract_Size', row.get('Contract Size', 1000))

        # Si pas de last_price, on ne peut pas calculer
        if last_price is None or pd.isna(last_price):
            return None

        if strike == 0 or settlement_price == 0:
            return None

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


def style_model_price(row):
    """
    Style pour la colonne Model Price :
    - Vert doux si Model Price < Last Price (opportunité d'achat)
    - Rouge doux si Model Price > Last Price (surévalué)
    """
    styles = [''] * len(row)

    try:
        model_price = row.get('Price', None)
        last_price = row.get('Last_Price', None)

        if model_price is not None and last_price is not None:
            if not pd.isna(model_price) and not pd.isna(last_price):
                if 'Price' in row.index:
                    price_idx = row.index.get_loc('Price')

                    if model_price < last_price:
                        # Vert désaturé
                        styles[price_idx] = (
                            'background-color: #3CB371; '   # MediumSeaGreen
                            'color: #ffffff; '
                            'font-weight: 600'
                        )
                    elif model_price > last_price:
                        # Rouge désaturé
                        styles[price_idx] = (
                            'background-color: #CD5C5C; '   # IndianRed
                            'color: #ffffff; '
                            'font-weight: 600'
                        )
                    else:
                        # Neutre gris clair
                        styles[price_idx] = (
                            'background-color: #E0E0E0; '
                            'color: #333333; '
                            'font-weight: 500'
                        )
    except Exception:
        pass

    return styles

def style_pnl_gradient(val):
    """
    Style pour les colonnes PnL avec gradient de couleur plus doux
    - Rouge foncé ➝ rose pour pertes
    - Gris pour quasi neutre
    - Vert clair ➝ vert foncé pour gains
    """
    if pd.isna(val) or val is None:
        return ''

    try:
        val = float(val)

        if val < -10000:
            # Perte très importante
            color = '#8B0000'  # DarkRed
            text_color = 'white'
        elif val < -5000:
            color = '#B22222'  # FireBrick
            text_color = 'white'
        elif val < -1000:
            color = '#CD5C5C'  # IndianRed
            text_color = 'white'
        elif val < -100:
            color = '#F4A6A6'  # Rouge très clair / rose
            text_color = '#333333'
        elif val < 0:
            color = '#F9D6D5'  # Rose très pâle
            text_color = '#333333'
        elif val == 0:
            color = '#E0E0E0'  # Gris clair
            text_color = '#333333'
        elif val < 100:
            color = '#E3F4E9'  # Vert très pâle
            text_color = '#333333'
        elif val < 1000:
            color = '#A9DFBF'  # Vert clair
            text_color = '#1B4D3E'
        elif val < 10000:
            color = '#52BE80'  # Vert moyen
            text_color = 'white'
        else:
            color = '#1E8449'  # Vert foncé
            text_color = 'white'

        return f'background-color: {color}; color: {text_color}; font-weight: 600'

    except Exception:
        return ''



def get_underlying_future_ticker(option_ticker):
    """
    Extrait le ticker du future sous-jacent depuis le ticker de l'option
    Exemple: 'CLG6C 58.50 Comdty' → 'CLG6 Comdty'
    """
    if not isinstance(option_ticker, str):
        return None

    # Supprimer 'C' ou 'P' à la fin + ' Comdty'
    parts = option_ticker.split()
    if len(parts) < 2:
        return None

    option_code = parts[0]  # 'CLG6C' ou 'CLG6P'

    # Enlever le dernier caractère (C ou P)
    if option_code and option_code[-1] in ['C', 'P']:
        future_code = option_code[:-1]  # 'CLG6'
        return f"{future_code} Comdty"

    return None

def extract_underlying_code(ticker):
    """
    Extrait le code du sous-jacent depuis le ticker de l'option
    Exemple: 'CLG6C 58.50 Comdty' → 'CL' (Crude Oil)
             'NGJ6C 2.50 Comdty' → 'NG' (Natural Gas)
    """
    if not isinstance(ticker, str):
        return None

    parts = ticker.split()
    if len(parts) < 2:
        return None

    option_code = parts[0]  # 'CLG6C'

    # Extraire les 2-3 premières lettres (code du produit)
    # Format typique : [PRODUIT][MOIS][ANNEE][C/P]
    # Ex: CLG6C → CL, NGJ6P → NG, GCZ5C → GC

    # La plupart des codes produits font 2 lettres, certains 3
    if len(option_code) >= 2:
        # Essayer 2 lettres d'abord
        return option_code[:2]

    return None


def test_bloomberg_fields(ticker):
    """
    Fonction de debug pour tester les champs Bloomberg disponibles
    """
    session = start_bloomberg_session()

    if session is None:
        st.error("Bloomberg non disponible")
        return

    try:
        st.write(f"### Test des champs pour {ticker}")

        # Liste des champs Rho possibles à tester
        rho_fields = [
            'FUT_OPT_RHO_MID',
            'FUT_OPT_RHO',
            'OPT_RHO_MID',
            'OPT_RHO',
            'RHO',
            'FUT_OPT_RHO_BID',
            'FUT_OPT_RHO_ASK'
        ]

        st.write("**Test des champs Rho:**")
        for field in rho_fields:
            try:
                result = session.bdp(ticker, field)
                st.write(f"✅ {field}: {result}")
            except:
                st.write(f"❌ {field}: Non disponible")

        # Optionnel: tester tous les Greeks
        st.write("\n**Tous les Greeks:**")
        all_greeks = ['OPT_DELTA', 'OPT_GAMMA', 'OPT_VEGA', 'OPT_THETA']
        for field in all_greeks:
            try:
                result = session.bdp(ticker, field)
                st.write(f"✅ {field}: {result}")
            except:
                st.write(f"❌ {field}: Non disponible")

    except Exception as e:
        st.error(f"Erreur: {str(e)}")
    finally:
        session.stop()

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
                        'OPT_THETA']

    tickers = df['Ticker'].tolist()

    # Extraire les tickers des futures sous-jacents
    underlying_tickers = []
    ticker_to_underlying = {}
    for ticker in tickers:
        underlying = get_underlying_future_ticker(ticker)
        if underlying and underlying not in underlying_tickers:
            underlying_tickers.append(underlying)
        ticker_to_underlying[ticker] = underlying

    with st.spinner(f" Récupération des données Bloomberg pour {len(tickers)} positions..."):
        try:
            # Récupérer les données des options
            bloomberg_data = get_bloomberg_data_batch(session, tickers, bloomberg_fields)

            # Récupérer les prix des futures sous-jacents
            underlying_prices = {}
            if underlying_tickers:
                underlying_data = get_bloomberg_data_batch(session, underlying_tickers, ['PX_LAST'])
                for fut_ticker, data in underlying_data.items():
                    underlying_prices[fut_ticker] = data.get('PX_LAST', None)

            # Créer un nouveau DataFrame avec les résultats Bloomberg
            bbg_results = []

            for idx, row in df.iterrows():
                ticker = row.get('Ticker', '')
                position_size = row.get('Size', row.get('Position_Size', 0))
                strike = row.get('Strike Px', row.get('Strike_Px', row.get('Strike', 0)))
                settlement_price = row.get('Settlement Price', row.get('Settlement_Price', 0))

                # Récupérer le prix du future sous-jacent
                underlying_ticker = ticker_to_underlying.get(ticker)
                underlying_price = underlying_prices.get(underlying_ticker, None) if underlying_ticker else None

                if ticker in bloomberg_data:
                    last_price = bloomberg_data[ticker].get('PX_LAST', None)

                    result = {
                        'Product': ticker,
                        'Position_Size': position_size,
                        'Strike_px': strike,
                        'Settlement_Price': settlement_price,
                        'Bid': bloomberg_data[ticker].get('BID', None),
                        'Ask': bloomberg_data[ticker].get('ASK', None),
                        'Last_Price': bloomberg_data[ticker].get('PX_LAST', None),
                        'Underlying_Price': underlying_price,  # ← AJOUT
                        'IV': bloomberg_data[ticker].get('IVOL_MID', None),
                        'Delta': bloomberg_data[ticker].get('OPT_DELTA', None),
                        'Gamma': bloomberg_data[ticker].get('OPT_GAMMA', None),
                        'Vega': bloomberg_data[ticker].get('OPT_VEGA', None),
                        'Theta': bloomberg_data[ticker].get('OPT_THETA', None),
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
                        'Underlying_Price': underlying_price,  # ← AJOUT
                        'IV': None,
                        'Delta': None,
                        'Gamma': None,
                        'Vega': None,
                        'Theta': None,
                        'PnL': None
                    }

                bbg_results.append(result)

            st.success(f" Données Bloomberg récupérées pour {len(tickers)} positions")
            return pd.DataFrame(bbg_results)

        except Exception as e:
            st.error(f"❌ Erreur lors de la récupération Bloomberg: {str(e)}")
            return pd.DataFrame()
        finally:
            session.stop()


def fetch_bloomberg_rates_curve():
    """
    Récupère la courbe des taux US Treasury depuis Bloomberg
    RETOURNE UN DATAFRAME avec les colonnes: Tenor, Maturity_Years, Ticker, Rate_%, Rate_Decimal
    """
    session = start_bloomberg_session()

    if session is None:
        st.warning("⚠️ Bloomberg non disponible - courbe de taux non récupérée")
        return pd.DataFrame()

    try:
        with st.spinner(f" Récupération de la courbe US Treasury ({len(TREASURY_TICKERS)} tenors)..."):
            # Récupération des taux via Bloomberg
            tickers = list(TREASURY_TICKERS.values())
            rate_data = get_bloomberg_data_batch(session, tickers, ['PX_LAST'])

            # Construction du DataFrame de la courbe
            curve_records = []
            missing_count = 0

            for tenor, ticker in TREASURY_TICKERS.items():
                rate = rate_data.get(ticker, {}).get('PX_LAST')
                years = TENOR_TO_YEARS[tenor]

                if rate is not None and not pd.isna(rate):
                    curve_records.append({
                        'Tenor': tenor,
                        'Maturity_Years': years,
                        'Ticker': ticker,
                        'Rate_%': rate,
                        'Rate_Decimal': rate / 100.0
                    })
                    print(f"  ✓ {tenor:4} ({ticker:20}) : {rate:6.3f}%")
                else:
                    missing_count += 1
                    print(f"  ⚠️ {tenor:4} ({ticker:20}) : N/A")

            if curve_records:
                rates_curve_df = pd.DataFrame(curve_records)
                st.success(f" Courbe de taux récupérée: {len(curve_records)}/{len(TREASURY_TICKERS)} points")

                if missing_count > 0:
                    st.warning(f"⚠️ {missing_count} point(s) manquant(s)")

                return rates_curve_df
            else:
                st.error("❌ Aucune donnée de taux récupérée")
                return pd.DataFrame()

    except Exception as e:
        st.error(f"❌ Erreur lors de la récupération de la courbe: {str(e)}")
        return pd.DataFrame()
    finally:
        session.stop()

def get_rate_for_T(T: float) -> float:
    """
    Renvoie le taux sans risque (décimal) correspondant à la maturité T (en années)
    en interpolant sur st.session_state.rates_data.
    Fallback sur st.session_state.risk_free_rate si la courbe n'est pas dispo.
    """
    rates_df = st.session_state.get("rates_data", None)
    if rates_df is None or rates_df.empty or T is None or T <= 0:
        return st.session_state.risk_free_rate

    # Vérifier la présence des colonnes
    if not {"Maturity_Years", "Rate_Decimal"}.issubset(rates_df.columns):
        return st.session_state.risk_free_rate

    # Trier par maturité
    df = rates_df.sort_values("Maturity_Years")

    # Clip si T est hors de la courbe
    if T <= df["Maturity_Years"].iloc[0]:
        return float(df["Rate_Decimal"].iloc[0])
    if T >= df["Maturity_Years"].iloc[-1]:
        return float(df["Rate_Decimal"].iloc[-1])

    # Interpolation linéaire
    return float(np.interp(T, df["Maturity_Years"].values, df["Rate_Decimal"].values))


def send_email_with_attachment(recipient_email, subject, body, excel_file_bytes, filename):
    """
    Envoie un email avec le fichier Excel en pièce jointe via SendGrid API
    """

    # Vérifier si SendGrid est disponible
    if not SENDGRID_AVAILABLE:
        st.error("❌ SendGrid non installé")
        st.code("pip install sendgrid", language="bash")
        return False, "❌ SendGrid non installé. Exécutez: pip install sendgrid"

    try:
        # Configuration SendGrid avec gestion d'erreur améliorée
        try:
            api_key = st.secrets["sendgrid"]["API_KEY"]
            sender_email = st.secrets["sendgrid"]["SENDER_EMAIL"]
            sender_name = st.secrets["sendgrid"].get("SENDER_NAME", "Risk Monitor")
        except KeyError as e:
            error_msg = f"❌ Configuration manquante dans secrets.toml: {str(e)}\n\n"
            error_msg += "Ajoutez dans .streamlit/secrets.toml:\n\n"
            error_msg += '[sendgrid]\n'
            error_msg += 'API_KEY = "SG.votre_cle_ici"\n'
            error_msg += 'SENDER_EMAIL = "votre.email@gmail.com"\n'
            error_msg += 'SENDER_NAME = "Risk Monitor"'
            st.error(error_msg)
            return False, error_msg

        # Validation des données
        if not api_key or len(api_key) < 20:
            return False, "❌ API Key SendGrid invalide ou trop courte"

        if not sender_email or '@' not in sender_email:
            return False, f"❌ Email expéditeur invalide: {sender_email}"

        if not recipient_email or '@' not in recipient_email:
            return False, f"❌ Email destinataire invalide: {recipient_email}"

        st.info(f"📋 Configuration validée:")
        st.write(f"- Sender: {sender_email}")
        st.write(f"- Recipient: {recipient_email}")
        st.write(f"- API Key: {api_key[:10]}...{api_key[-4:]}")

        # Créer le message
        st.info("📧 Création du message...")
        message = Mail(
            from_email=(sender_email, sender_name),
            to_emails=recipient_email,
            subject=subject,
            html_content=body.replace('\n', '<br>')
        )

        # Encoder le fichier en base64
        st.info(f"📎 Encodage du fichier ({len(excel_file_bytes)} bytes)...")
        encoded_file = base64.b64encode(excel_file_bytes).decode('utf-8')

        # Créer l'attachement avec les bons paramètres
        attached_file = Attachment(
            FileContent(encoded_file),
            FileName(filename),
            FileType('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
            Disposition('attachment')
        )
        message.attachment = attached_file

        # Envoyer l'email
        st.info("🚀 Envoi via SendGrid...")
        sg = SendGridAPIClient(api_key)

        # Ajouter un timeout et une meilleure gestion d'erreur
        try:
            response = sg.send(message)
        except Exception as send_error:
            error_detail = str(send_error)

            # Erreurs spécifiques SendGrid
            if "401" in error_detail or "Unauthorized" in error_detail:
                return False, "❌ Clé API invalide. Vérifiez votre clé sur https://app.sendgrid.com/settings/api_keys"
            elif "403" in error_detail or "Forbidden" in error_detail:
                return False, "❌ Accès refusé. Vérifiez que l'email expéditeur est vérifié dans SendGrid"
            elif "400" in error_detail:
                return False, f"❌ Requête invalide: {error_detail}"
            else:
                return False, f"❌ Erreur d'envoi: {error_detail}"

        st.info(f"📬 Réponse SendGrid: Code {response.status_code}")

        # Vérifier le succès
        if response.status_code in [200, 201, 202]:
            success_msg = f"✅ Email envoyé avec succès à {recipient_email}!\n\n"
            success_msg += f"Code: {response.status_code}\n"
            success_msg += "Vérifiez votre boîte de réception (et les spams)"
            return True, success_msg
        else:
            return False, f"❌ Erreur SendGrid (Code {response.status_code})"

    except Exception as e:
        error_msg = f"💥 Exception inattendue: {str(e)}\n\n"
        error_msg += f"Type: {type(e).__name__}"

        import traceback
        st.error(error_msg)
        st.code(traceback.format_exc())

        return False, error_msg


# ============================================================================
# FONCTIONS DE CALCUL
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
    Nd2 = norm_cdf(d2)
    nd1 = norm_pdf(d1)

    option_type = option_type.upper()[0]

    if option_type == "C":
        delta = df * Nd1
        theta = -(F * nd1 * sigma * df) / (2 * sqrtT) - r * df * (F * Nd1 - K * Nd2)
        rho = -T * df * (F * Nd1 - K * Nd2)
    else:  # PUT
        delta = df * (Nd1 - 1.0)
        theta = (-(F * nd1 * sigma * df) / (2 * sqrtT) + r * df * (K * norm_cdf(-d2) - F * norm_cdf(-d1)))
        rho = -T * df * (K * norm_cdf(-d2) - F * norm_cdf(-d1))

    gamma = df * nd1 / (F * sigma * sqrtT)
    vega = df * F * nd1 * sqrtT * 0.01

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
        # Theta pour Call
        theta = (-sigma_normal * nd * df) / (2 * sqrtT) - r * price
        # Rho pour Call
        rho = -T * price
    else:  # PUT
        price = df * ((K - F) * norm_cdf(-d) + sigma_normal * sqrtT * nd)
        delta = df * (Nd - 1.0)
        # Theta pour Put
        theta = ((-sigma_normal * nd * df) / (2 * sqrtT) - r * price)
        # Rho pour Put
        rho = -T * price

    gamma = df * nd / (sigma_normal * sqrtT)
    vega = df * sqrtT * nd

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho, "price": price}


# ============================================================================
# FONCTIONS HESTON MODEL & VOL LOCALE
# ============================================================================

def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    """
    Calculate the Heston characteristic function.
    Adapté pour les options sur futures.
    """
    a = kappa * theta
    b = kappa + lambd

    rspi = rho * sigma * phi * 1j
    d = np.sqrt((rho * sigma * phi * 1j - b) ** 2 + (phi * 1j + phi ** 2) * sigma ** 2)
    g = (b - rspi + d) / (b - rspi - d)

    d_tau = d * tau
    g_exp_d_tau = g * np.exp(np.clip(d_tau, -100, 100))

    denom_term2 = 1 - g
    denom_term2 = np.where(np.abs(denom_term2) < 1e-10, 1e-10, denom_term2)

    num_term2 = 1 - g_exp_d_tau
    num_term2 = np.where(np.abs(num_term2) < 1e-10, 1e-10, num_term2)

    exp1 = np.exp(r * phi * 1j * tau)
    term2 = S0 ** (phi * 1j) * (num_term2 / denom_term2) ** (-2 * a / sigma ** 2)

    numerator = 1 - np.exp(np.clip(d_tau, -100, 100))
    denominator = 1 - g_exp_d_tau
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)

    ratio = numerator / denominator

    exp_arg = (a * tau * (b - rspi + d) / sigma ** 2 +
               v0 * (b - rspi + d) * ratio / sigma ** 2)

    exp_arg_real = np.real(exp_arg)
    exp_arg = np.where(exp_arg_real > 100, 100 + 1j * np.imag(exp_arg), exp_arg)
    exp_arg = np.where(exp_arg_real < -100, -100 + 1j * np.imag(exp_arg), exp_arg)

    exp2 = np.exp(exp_arg)
    result = exp1 * term2 * exp2

    result = np.where(np.isfinite(result), result, 0 + 0j)

    return result

def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    """
    Price European call options using Heston model with rectangular integration.
    """
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    P, umax, N = 0, 100, 10000
    dphi = umax / N

    for i in range(1, N):
        phi = dphi * (2 * i + 1) / 2
        numerator = (np.exp(r * tau) * heston_charfunc(phi - 1j, *args) -
                     K * heston_charfunc(phi, *args))
        denominator = 1j * phi * K ** (1j * phi)
        P += dphi * numerator / denominator

    return np.real((S0 - K * np.exp(-r * tau)) / 2 + P / np.pi)

def heston_greeks(F, K, T, r, v0, kappa, theta, sigma, rho, lambd, option_type):
    """
    Calcule les Greeks Heston par différences finies.
    """
    if T <= 0 or v0 <= 0 or F <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "vega": None, "theta": None, "rho": None, "price": None}

    # Prix de base
    price = heston_price_rec(F, K, v0, kappa, theta, sigma, rho, lambd, T, r)

    # Delta (sensibilité au prix du sous-jacent)
    dF = 0.01 * F
    price_up = heston_price_rec(F + dF, K, v0, kappa, theta, sigma, rho, lambd, T, r)
    price_down = heston_price_rec(F - dF, K, v0, kappa, theta, sigma, rho, lambd, T, r)
    delta = (price_up - price_down) / (2 * dF)

    # Gamma (sensibilité du delta)
    gamma = (price_up - 2 * price + price_down) / (dF ** 2)

    # Vega (sensibilité à la volatilité initiale)
    dv = 0.01 * v0
    price_vega_up = heston_price_rec(F, K, v0 + dv, kappa, theta, sigma, rho, lambd, T, r)
    vega = (price_vega_up - price) / dv

    # Theta (sensibilité au temps)
    dT = 1 / 365  # 1 jour
    if T > dT:
        price_theta = heston_price_rec(F, K, v0, kappa, theta, sigma, rho, lambd, T - dT, r)
        theta = (price_theta - price) / dT
    else:
        theta = None

    # Rho (sensibilité au taux sans risque)
    dr = 0.01
    price_rho = heston_price_rec(F, K, v0, kappa, theta, sigma, rho, lambd, T, r + dr)
    rho_greek = (price_rho - price) / dr

    # Ajuster pour Put si nécessaire
    option_type = option_type.upper()[0]
    if option_type == "P":
        # Put-Call parity: P = C - F*e^(-rT) + K*e^(-rT)
        df = np.exp(-r * T)
        price = price - F * df + K * df
        delta = delta - df
        # Gamma et Vega sont identiques pour Put et Call

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega * 0.01,
        "theta": theta * 0.01,
        "rho": rho_greek,
        "price": price
    }


def compute_local_volatility_surface(smile_data, F, T_range, K_range):
    """
    Calcule la surface de volatilité locale à partir du smile de marché
    en utilisant la formule de Dupire

    Args:
        smile_data: dict {maturity: DataFrame(Strike, IV_Market)}
        F: Prix forward actuel
        T_range: Liste des maturités
        K_range: Liste des strikes

    Returns:
        local_vol_surface: Array 2D de volatilités locales
    """
    # Créer une grille d'IV de marché interpolée
    T_grid = []
    K_grid = []
    IV_grid = []

    for T, df_smile in smile_data.items():
        df_valid = df_smile.dropna(subset=['IV_Market'])
        if len(df_valid) > 0:
            for _, row in df_valid.iterrows():
                T_grid.append(T)
                K_grid.append(row['Strike'])
                IV_grid.append(row['IV_Market'] / 100.0)  # Convertir en décimal

    if len(T_grid) < 4:
        print("⚠️ Pas assez de points pour interpoler la surface")
        return None

    # Créer un interpolateur 2D (Strike, Maturity) -> IV
    from scipy.interpolate import griddata

    points = np.array([K_grid, T_grid]).T
    values = np.array(IV_grid)

    # Créer une grille régulière
    K_mesh, T_mesh = np.meshgrid(K_range, T_range)

    # Interpoler
    IV_surface = griddata(points, values, (K_mesh, T_mesh), method='cubic')

    # Calculer les dérivées pour Dupire
    # σ_local² = (∂C/∂T) / (½ K² ∂²C/∂K²)

    local_vol_surface = np.zeros_like(IV_surface)

    for i in range(1, len(T_range) - 1):
        for j in range(1, len(K_range) - 1):
            T = T_range[i]
            K = K_range[j]
            sigma_iv = IV_surface[i, j]

            if np.isnan(sigma_iv):
                continue

            # Approximation simple de Dupire
            # σ_local ≈ σ_BS * sqrt(1 + 2T * ∂σ/∂T + ...)

            # Dérivées par différences finies
            dSigma_dT = (IV_surface[i + 1, j] - IV_surface[i - 1, j]) / (T_range[i + 1] - T_range[i - 1])
            dSigma_dK = (IV_surface[i, j + 1] - IV_surface[i, j - 1]) / (K_range[j + 1] - K_range[j - 1])
            d2Sigma_dK2 = (IV_surface[i, j + 1] - 2 * IV_surface[i, j] + IV_surface[i, j - 1]) / (
                        (K_range[j + 1] - K_range[j]) ** 2)

            # Formule de Dupire simplifiée
            numerator = sigma_iv ** 2 + 2 * T * sigma_iv * dSigma_dT
            denominator = (1 + 2 * K * dSigma_dK * np.log(K / F) / (sigma_iv) +
                           K ** 2 * (d2Sigma_dK2 + dSigma_dK ** 2 * T))

            if denominator > 0:
                local_vol_surface[i, j] = np.sqrt(max(numerator / denominator, 0.01))
            else:
                local_vol_surface[i, j] = sigma_iv

    return local_vol_surface

def local_vol_price_pde(F, K, T, r, local_vol_func, option_type='C', N_S=100, N_T=50):
    """
    Price une option avec volatilité locale en résolvant l'EDP par Crank-Nicolson

    Args:
        F: Prix forward
        K: Strike
        T: Maturité
        r: Taux sans risque
        local_vol_func: Fonction qui retourne σ_local(S, t)
        option_type: 'C' ou 'P'
        N_S: Nombre de points en espace
        N_T: Nombre de pas de temps
    """
    # Grille de prix
    S_max = 3 * F
    S_min = 0.3 * F
    dS = (S_max - S_min) / N_S
    dt = T / N_T

    S = np.linspace(S_min, S_max, N_S + 1)

    # Condition terminale
    if option_type.upper() == 'C':
        V = np.maximum(S - K, 0) * np.exp(-r * T)
    else:
        V = np.maximum(K - S, 0) * np.exp(-r * T)

    # Résolution backward en temps
    for n in range(N_T, 0, -1):
        t = n * dt

        # Matrice tridiagonale pour Crank-Nicolson
        a = np.zeros(N_S - 1)
        b = np.zeros(N_S - 1)
        c = np.zeros(N_S - 1)
        d = np.zeros(N_S - 1)

        for i in range(1, N_S):
            sigma_local = local_vol_func(S[i], t)

            alpha = 0.25 * dt * (sigma_local ** 2 * S[i] ** 2 / dS ** 2)

            if i > 1:
                a[i - 1] = -alpha
            b[i - 1] = 1 + 2 * alpha + r * dt / 2
            if i < N_S - 1:
                c[i - 1] = -alpha

            d[i - 1] = V[i]

        # Résoudre le système tridiagonal
        V[1:N_S] = thomas_algorithm(a, b, c, d)

    # Interpoler pour obtenir le prix au strike K
    return np.interp(F, S, V)

def thomas_algorithm(a, b, c, d):
    """Résout un système tridiagonal Ax = d"""
    n = len(d)
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n - 1):
        c_prime[i] = c[i] / (b[i] - a[i - 1] * c_prime[i - 1])

    for i in range(1, n):
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / (b[i] - a[i - 1] * c_prime[i - 1])

    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


# ============================================================================

def safe_time_to_maturity(maturity_value, today: date) -> float:
    """Convertit la maturité en T (années)."""
    # Sécuriser le type de `today` au cas où une variable float globale l'écrase
    if isinstance(today, (int, float)):
        today = datetime.today().date()

    # Si c'est déjà un datetime/Timestamp
    if isinstance(maturity_value, (datetime, pd.Timestamp)):
        maturity_date = maturity_value.date()
    else:
        # Essayer de parser comme date
        try:
            m = pd.to_datetime(maturity_value, errors="coerce", dayfirst=True)
        except Exception:
            m = pd.NaT

        # Si échec, essayer format "MMM YY" (ex: "FEB 26", "MAR 26")
        if pd.isna(m) and isinstance(maturity_value, str):
            try:
                # Remplacer "FEB 26" → "FEB 2026"
                maturity_str = maturity_value.strip().upper()
                parts = maturity_str.split()
                if len(parts) == 2:
                    month_str = parts[0]
                    year_str = parts[1]

                    # Convertir année courte (26 → 2026)
                    if len(year_str) == 2:
                        year_int = int(year_str)
                        # Si année < 50, c'est 20XX, sinon 19XX
                        full_year = 2000 + year_int if year_int < 50 else 1900 + year_int
                    else:
                        full_year = int(year_str)

                    # Parser avec le mois et l'année complète
                    m = pd.to_datetime(f"{month_str} 01 {full_year}", format="%b %d %Y")
            except Exception as e:
                print(f"⚠️ Impossible de parser maturity '{maturity_value}': {e}")
                m = pd.NaT

        if pd.isna(m):
            print(f"⚠️ Maturity invalide '{maturity_value}' → T = 0.00001")
            return 0.00001

        maturity_date = m.date()

    days = (maturity_date - today).days
    T = days / 365.0

    if T <= 0:
        print(f"⚠️ Maturity {maturity_date} est passée ou aujourd'hui → T = 0.00001")
        T = 0.00001

    return T


def compute_b76_greeks_for_position(row, bloomberg_df, valuation_date=None):
    """Calcule les Greeks Black-76 pour une position"""
    ticker = row.get('Ticker', 'Unknown')

    if valuation_date is None:
        valuation_date = datetime.today().date()

    try:
        # CHECK 1 : Maturity
        if "Maturity" not in row or pd.isna(row["Maturity"]):
            print(f"❌ {ticker} : Maturity manquante ou NaN")
            return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None}

        # CHECK 2 : Calcul de T
        T = safe_time_to_maturity(row["Maturity"], valuation_date)

        # Taux sans risque en fonction de T (fallback sur risk_free_rate global)
        if st.session_state.get("rates_data") is not None:
            r_local = get_rate_for_T(T)
        else:
            r_local = st.session_state.risk_free_rate

        # CHECK 3 : F et K
        K = row.get("Strike", row.get("Strike_Px", row.get("Strike Px", 0)))
        F = None
        if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
            bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
            F = bbg_row.get('Underlying_Price', None)

        if F is None or pd.isna(F):
            F = row.get("Underlying_Price", row.get("Future_Price", None))
        if F is None or F == 0:
            print(f"⚠️ {ticker} : Pas de prix du future disponible, utilisation du Strike comme approximation")
            F = K

        print(f"\n🔍 {ticker} - Maturity = {row['Maturity']}, T = {T:.4f} ans")
        print(f"   F (Underlying Future) = {F}")
        print(f"   K (Strike) = {K}")
        print(f"   r_local (US curve) = {r_local:.4f}")

        if F == 0 or K == 0:
            print(f"❌ {ticker} : F={F} ou K={K} = 0 → Greeks = None")
            return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None}

        # CHECK 4 : IV
        iv = row.get('IV_Bloomberg', None)
        print(f"   IV Bloomberg (enrichie) = {iv}")
        if iv is None or pd.isna(iv):
            if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
                bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
                iv = bbg_row.get('IV', None)
                print(f"   IV Bloomberg (df) = {iv}")
        if iv is None or pd.isna(iv):
            iv = row.get("IV")
            print(f"   IV from row = {iv}")

        if iv in (None, '', ' ') or pd.isna(iv):
            sigma = 0.30
            print(f"   ⚠️ Pas d'IV → sigma par défaut = {sigma}")
        else:
            sigma = float(iv) / 100.0
            print(f"   ✅ σ (sigma) = {sigma}")

        # CHECK 5 : Type
        raw_type = str(row.get("Put/Call", row.get("Put_Call", row.get("Type", "C")))).strip().upper()
        option_type = "P" if raw_type.startswith("P") else "C"
        position_size = row.get("Size", row.get("Position_Size", 1))
        print(f"   Type = {option_type}, Size = {position_size}")

        print(f"   📞 Appel black76_greeks(F={F}, K={K}, T={T:.4f}, r={r_local}, σ={sigma}, type={option_type})")

        greeks = black76_greeks(
            F=float(F), K=float(K), T=float(T),
            r=float(r_local), sigma=float(sigma),
            option_type=option_type
        )

        print(f"   ✅ Greeks retournés : {greeks}")

        return {
            "Delta": greeks["delta"],
            "Gamma": greeks["gamma"],
            "Vega": greeks["vega"],
            "Theta": greeks["theta"],
            "Rho": greeks["rho"]
        }
    except Exception as e:
        print(f"❌ ERREUR {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None}

def compute_bachelier_greeks_for_position(row, bloomberg_df, valuation_date=None):
    """Calcule les Greeks Bachelier pour une position"""
    if valuation_date is None:
        valuation_date = datetime.today().date()

    try:
        if "Maturity" not in row or pd.isna(row["Maturity"]):
            return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None, "Price": None}

        T = safe_time_to_maturity(row["Maturity"], valuation_date)

        # r(T) avec fallback global
        if st.session_state.get("rates_data") is not None:
            r_local = get_rate_for_T(T)
        else:
            r_local = st.session_state.risk_free_rate

        K = row.get("Strike", row.get("Strike_Px", row.get("Strike Px", 0)))

        ticker = row.get('Ticker', '')
        F = None
        if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
            bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
            F = bbg_row.get('Underlying_Price', None)

        if F is None or pd.isna(F) or F == 0:
            F = K

        if K == 0:
            return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None, "Price": None}

        iv = None
        if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
            bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
            iv = bbg_row.get('IV', None)
        if iv is None or pd.isna(iv):
            iv = row.get("IV")

        if iv in (None, '', ' ') or pd.isna(iv):
            sigma = 0.30
        else:
            sigma = float(iv) / 100.0

        raw_type = str(row.get("Put/Call", row.get("Put_Call", row.get("Type", "C")))).strip().upper()
        option_type = "P" if raw_type.startswith("P") else "C"

        greeks = bachelier_greeks(
            F=float(F), K=float(K), T=float(T),
            r=float(r_local), sigma=float(sigma),
            option_type=option_type
        )

        return {
            "Delta": greeks["delta"],
            "Gamma": greeks["gamma"],
            "Vega": greeks["vega"],
            "Theta": greeks["theta"],
            "Rho": greeks["rho"],
            "Price": greeks["price"]
        }
    except Exception as e:
        print(f"Erreur calcul Bachelier: {e}")
        return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None, "Price": None}

def compute_heston_greeks_for_position(row, bloomberg_df, heston_params, valuation_date=None):
    """Calcule les Greeks Heston pour une position"""
    if valuation_date is None:
        valuation_date = datetime.today().date()

    try:
        if "Maturity" not in row or pd.isna(row["Maturity"]):
            return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None, "Price": None}

        T = safe_time_to_maturity(row["Maturity"], valuation_date)

        # r(T) avec fallback global
        if st.session_state.get("rates_data") is not None:
            r_local = get_rate_for_T(T)
        else:
            r_local = st.session_state.risk_free_rate

        K = row.get("Strike", row.get("Strike_Px", row.get("Strike Px", 0)))

        ticker = row.get('Ticker', '')
        F = None
        if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
            bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
            F = bbg_row.get('Underlying_Price', None)

        if F is None or pd.isna(F) or F == 0:
            F = K

        if K == 0:
            return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None, "Price": None}

        raw_type = str(row.get("Put/Call", row.get("Put_Call", row.get("Type", "C")))).strip().upper()
        option_type = "P" if raw_type.startswith("P") else "C"

        greeks = heston_greeks(
            F=float(F), K=float(K), T=float(T), r=float(r_local),
            v0=heston_params['v0'],
            kappa=heston_params['kappa'],
            theta=heston_params['theta'],
            sigma=heston_params['sigma'],
            rho=heston_params['rho'],
            lambd=heston_params['lambd'],
            option_type=option_type
        )

        return {
            "Delta": greeks["delta"],
            "Gamma": greeks["gamma"],
            "Vega": greeks["vega"],
            "Theta": greeks["theta"],
            "Rho": greeks["rho"],
            "Price": greeks["price"]
        }
    except Exception as e:
        print(f"Erreur calcul Heston: {e}")
        return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Rho": None, "Price": None}


def local_vol_greeks_simplified(F, K, T, r, sigma_atm, skew, option_type='C'):
    """
    Calcul simplifié avec ajustement local de la volatilité
    """
    # Ajustement de la volatilité en fonction du moneyness
    log_moneyness = np.log(K / F)

    # Volatilité locale = σ_ATM + skew * log(K/F)
    sigma_local = sigma_atm + skew * log_moneyness

    # Floor à 5% minimum, cap à 200% maximum
    sigma_local = np.clip(sigma_local, 0.05, 2.0)

    # Black-76 avec volatilité ajustée
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma_local ** 2 * T) / (sigma_local * sqrtT)
    d2 = d1 - sigma_local * sqrtT
    df = np.exp(-r * T)

    # Utilise tes fonctions norm_cdf et norm_pdf
    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    nd1 = norm_pdf(d1)

    option_type = option_type.upper()[0]

    if option_type == 'C':
        price = df * (F * Nd1 - K * Nd2)
        delta = df * Nd1
        theta = -(F * nd1 * sigma_local * df) / (2 * sqrtT) - r * df * (F * Nd1 - K * Nd2)
    else:  # PUT
        price = df * (K * norm_cdf(-d2) - F * norm_cdf(-d1))
        delta = df * (Nd1 - 1.0)
        theta = (-(F * nd1 * sigma_local * df) / (2 * sqrtT) + r * df * (K * norm_cdf(-d2) - F * norm_cdf(-d1)))

    gamma = df * nd1 / (F * sigma_local * sqrtT)
    vega = df * F * nd1 * sqrtT * 0.01

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta * 0.01,  # Theta par jour
        'sigma_local': sigma_local  # ✅ En décimal (pas en %)
    }


def fetch_complete_smile_from_bloomberg(session, underlying_code, maturity_str, n_strikes=15):
    """
    CORRIGÉ v3: Format exact Bloomberg pour options sur futures CL
    Format: CLG6C 58.50 Comdty (pas CLG26 C58.50 Comdty)
    """
    if session is None:
        return pd.DataFrame()

    try:
        # Convertir maturity_str en code de mois Bloomberg
        month_codes = {
            'JAN': 'F', 'FEB': 'G', 'MAR': 'H', 'APR': 'J',
            'MAY': 'K', 'JUN': 'M', 'JUL': 'N', 'AUG': 'Q',
            'SEP': 'U', 'OCT': 'V', 'NOV': 'X', 'DEC': 'Z'
        }

        parts = maturity_str.strip().upper().split()
        if len(parts) != 2:
            print(f"⚠️ Format maturity invalide: {maturity_str}")
            return pd.DataFrame()

        month_str = parts[0]
        year_str = parts[1]

        if month_str not in month_codes:
            print(f"⚠️ Mois invalide: {month_str}")
            return pd.DataFrame()

        month_code = month_codes[month_str]

        # ✅ CORRECTION MAJEURE : Année sur 1 SEUL CHIFFRE
        # 2026 → 6, 2025 → 5, 2024 → 4
        year_code = year_str[-1]  # Dernier chiffre uniquement

        # Future ticker (pour obtenir le prix spot)
        future_ticker = f"{underlying_code}{month_code}{year_code} Comdty"

        print(f"\n🔍 Récupération smile pour {future_ticker}")

        # Récupérer le prix du future
        service = session.getService("//blp/refdata")
        request = service.createRequest("ReferenceDataRequest")
        request.append("securities", future_ticker)
        request.append("fields", "PX_LAST")

        session.sendRequest(request)

        F = None
        while True:
            ev = session.nextEvent()
            for msg in ev:
                if msg.hasElement("securityData"):
                    sd = msg.getElement("securityData").getValueAsElement(0)
                    if sd.hasElement("fieldData"):
                        fd = sd.getElement("fieldData")
                        if fd.hasElement("PX_LAST"):
                            F = fd.getElementAsFloat("PX_LAST")
            if ev.eventType() == blpapi.Event.RESPONSE:
                break

        if F is None or F <= 0:
            print(f"❌ Impossible de récupérer le prix du future {future_ticker}")
            return pd.DataFrame()

        print(f"   ✅ Future price: {F:.2f}")

        # Créer une grille de strikes autour du forward
        # Strikes par incréments de 2.5 pour le pétrole
        strike_increment = 2.5
        n_below = 7
        n_above = 7

        strikes = []
        for i in range(-n_below, n_above + 1):
            K = round((F + i * strike_increment) * 2) / 2  # Arrondir à 0.5
            if K > 0:
                strikes.append(K)

        strikes = sorted(set(strikes))

        print(f"   🎯 Grille de {len(strikes)} strikes: {strikes}")

        # ✅ CORRECTION MAJEURE : Format des tickers d'options
        # Format: CLG6C 58.50 Comdty (C/P collé, espace avant strike)
        option_tickers_call = []
        option_tickers_put = []

        for K in strikes:
            # ✅ Format correct : [CODE][MOIS][ANNEE][C/P] [STRIKE] Comdty
            ticker_call = f"{underlying_code}{month_code}{year_code}C {K:.2f} Comdty"
            ticker_put = f"{underlying_code}{month_code}{year_code}P {K:.2f} Comdty"

            option_tickers_call.append(ticker_call)
            option_tickers_put.append(ticker_put)

        all_tickers = option_tickers_call + option_tickers_put

        print(f"   📋 Exemple de tickers générés:")
        print(f"      Call: {option_tickers_call[len(option_tickers_call) // 2]}")
        print(f"      Put:  {option_tickers_put[len(option_tickers_put) // 2]}")
        print(f"   ✅ Format de référence: CLG6C 58.50 Comdty")

        # Requête Bloomberg pour les IV
        fields = [
            'IVOL_MID',  # Volatilité implicite mid
            'IMPLIED_VOLATILITY',  # Volatilité implicite générique
            'IVOL_BID',  # IV Bid
            'IVOL_ASK',  # IV Ask
            'BID', 'ASK', 'PX_LAST'
        ]

        print(f"   📡 Requête Bloomberg pour {len(all_tickers)} options...")

        request2 = service.createRequest("ReferenceDataRequest")
        for ticker in all_tickers:
            request2.append("securities", ticker)
        for field in fields:
            request2.append("fields", field)

        session.sendRequest(request2)

        # Collecter les résultats
        data = {}
        errors = []

        while True:
            ev = session.nextEvent()
            for msg in ev:
                if msg.hasElement("securityData"):
                    sd_array = msg.getElement("securityData")
                    for i in range(sd_array.numValues()):
                        sd = sd_array.getValueAsElement(i)
                        ticker = sd.getElementAsString("security")

                        # Vérifier les erreurs Bloomberg
                        if sd.hasElement("securityError"):
                            error = sd.getElement("securityError")
                            error_msg = error.getElementAsString("message")
                            errors.append((ticker, error_msg))
                            continue

                        if sd.hasElement("fieldData"):
                            fd = sd.getElement("fieldData")

                            # Récupérer le premier champ IV disponible
                            iv = None
                            for iv_field in ['IVOL_MID', 'IMPLIED_VOLATILITY', 'IVOL_BID', 'IVOL_ASK']:
                                if fd.hasElement(iv_field):
                                    try:
                                        iv = fd.getElementAsFloat(iv_field)
                                        if iv and iv > 0:  # Valider que l'IV est positive
                                            break
                                    except:
                                        continue

                            data[ticker] = {
                                'IV': iv,
                                'Bid': fd.getElementAsFloat("BID") if fd.hasElement("BID") else None,
                                'Ask': fd.getElementAsFloat("ASK") if fd.hasElement("ASK") else None,
                                'Last': fd.getElementAsFloat("PX_LAST") if fd.hasElement("PX_LAST") else None
                            }

            if ev.eventType() == blpapi.Event.RESPONSE:
                break

        # Afficher les erreurs (si peu nombreuses)
        if errors and len(errors) <= 5:
            print(f"   ⚠️ {len(errors)} erreurs Bloomberg:")
            for ticker, error_msg in errors:
                print(f"      {ticker}: {error_msg}")
        elif errors:
            print(f"   ⚠️ {len(errors)} erreurs Bloomberg (strikes probablement non cotés)")

        # Construire le DataFrame de résultats
        smile_records = []
        valid_calls = 0
        valid_puts = 0

        for i, K in enumerate(strikes):
            ticker_call = option_tickers_call[i]
            ticker_put = option_tickers_put[i]

            call_data = data.get(ticker_call, {})
            put_data = data.get(ticker_put, {})

            iv_call = call_data.get('IV')
            iv_put = put_data.get('IV')

            if iv_call is not None and iv_call > 0:
                valid_calls += 1
            if iv_put is not None and iv_put > 0:
                valid_puts += 1

            record = {
                'Strike': K,
                'Moneyness': K / F,
                'IV_Call': iv_call,
                'IV_Put': iv_put,
                'Bid_Call': call_data.get('Bid'),
                'Ask_Call': call_data.get('Ask'),
                'Last_Call': call_data.get('Last'),
                'Bid_Put': put_data.get('Bid'),
                'Ask_Put': put_data.get('Ask'),
                'Last_Put': put_data.get('Last')
            }

            smile_records.append(record)

        smile_df = pd.DataFrame(smile_records)

        print(f"   📊 Résultats: {valid_calls} Calls valides, {valid_puts} Puts valides")

        # Ne garder que les strikes avec au moins une IV valide
        smile_df = smile_df.dropna(subset=['IV_Call', 'IV_Put'], how='all')

        print(f"   ✅ {len(smile_df)} strikes avec données IV récupérés")

        if len(smile_df) > 0:
            print(f"   📈 Range IV Call: {smile_df['IV_Call'].min():.2f}% - {smile_df['IV_Call'].max():.2f}%")
            print(f"   📈 Range IV Put: {smile_df['IV_Put'].min():.2f}% - {smile_df['IV_Put'].max():.2f}%")

        return smile_df

    except Exception as e:
        print(f"❌ Erreur récupération smile: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def compute_heston_volatility_smile_filtered(df_positions, bloomberg_df, heston_params, underlying_filter=None, fetch_market_data=True):
    """
    Calcule le smile de volatilité Heston avec possibilité de filtrer par sous-jacent
    et de récupérer des données de marché complètes depuis Bloomberg

    Args:
        df_positions: DataFrame des positions
        bloomberg_df: DataFrame des Greeks Bloomberg
        heston_params: Paramètres du modèle Heston
        underlying_filter: Code du sous-jacent à filtrer (ex: 'CL', 'NG') ou None pour tous
        fetch_market_data: Si True, récupère plus de strikes depuis Bloomberg

    Returns:
        dict: {maturity: DataFrame(Strike, Moneyness, IV_Market, IV_Heston)}
    """
    from datetime import date

    if df_positions.empty:
        print("❌ df_positions est vide")
        return {}

    valuation_date = date.today()

    # 1. Filtrer par sous-jacent si demandé
    df_pos = df_positions.copy()

    if underlying_filter:
        df_pos['Underlying_Code'] = df_pos['Ticker'].apply(extract_underlying_code)
        df_pos = df_pos[df_pos['Underlying_Code'] == underlying_filter]

        if df_pos.empty:
            print(f"❌ Aucune position pour le sous-jacent {underlying_filter}")
            return {}

        print(f"🎯 Filtrage sur sous-jacent: {underlying_filter} ({len(df_pos)} positions)")

    # 2. Calculer les maturités
    if 'Maturity' not in df_pos.columns:
        print("❌ Colonne 'Maturity' manquante")
        return {}

    df_pos['T_years'] = df_pos.apply(
        lambda row: safe_time_to_maturity(row['Maturity'], valuation_date),
        axis=1
    )

    df_pos = df_pos[df_pos['T_years'] > 0]

    if df_pos.empty:
        print("❌ Aucune maturité valide")
        return {}

    # 3. Grouper par maturité
    df_pos['T_rounded'] = df_pos['T_years'].round(1)
    unique_maturities = sorted(df_pos['T_rounded'].unique())

    print(f"📊 {len(unique_maturities)} maturité(s) unique(s): {unique_maturities}")

    # Limiter à 4 maturités max
    if len(unique_maturities) > 4:
        indices = [0, len(unique_maturities) // 4, 3 * len(unique_maturities) // 4, -1]
        maturities_to_plot = [unique_maturities[i] for i in indices]
    else:
        maturities_to_plot = unique_maturities

    smile_data = {}

    # 4. Pour chaque maturité
    for T in maturities_to_plot:
        df_maturity = df_pos[abs(df_pos['T_rounded'] - T) < 0.05]

        if df_maturity.empty:
            continue

        print(f"\n📍 Maturité T={T:.2f} ans")

        # 5. Déterminer le forward price moyen
        F_list = []
        for idx, row in df_maturity.iterrows():
            ticker = row.get('Ticker', '')
            if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
                bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
                F_tmp = bbg_row.get('Underlying_Price', None)
                if F_tmp and not pd.isna(F_tmp) and F_tmp > 0:
                    F_list.append(F_tmp)

        if not F_list:
            print(f"  ⚠️ Aucun forward valide")
            continue

        F_avg = np.mean(F_list)
        print(f"  Forward moyen: {F_avg:.2f}")

        # 6. Récupérer les données de marché
        market_smile_df = pd.DataFrame()

        if fetch_market_data and underlying_filter and BLOOMBERG_AVAILABLE:
            # Récupérer la maturité au format 'MMM YY'
            maturity_str = df_maturity.iloc[0]['Maturity']

            print(f"  📡 Récupération smile de marché complet...")

            session = start_bloomberg_session()
            if session:
                market_smile_df = fetch_complete_smile_from_bloomberg(
                    session,
                    underlying_filter,
                    maturity_str,
                    n_strikes=15
                )
                session.stop()

        # 7. Si pas de données Bloomberg complètes, utiliser les positions existantes
        if market_smile_df.empty:
            print(f"  ℹ️ Utilisation des positions existantes pour le marché")

            market_data_raw = []
            for idx, row in df_maturity.iterrows():
                ticker = row.get('Ticker', '')
                K_market = row.get('Strike', row.get('Strike_Px', row.get('Strike Px', 0)))

                if K_market == 0:
                    continue

                if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
                    bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
                    iv_mkt = bbg_row.get('IV', None)
                    if iv_mkt and not pd.isna(iv_mkt) and iv_mkt > 0:
                        market_data_raw.append((K_market, iv_mkt))

            # Dédupliquer
            from collections import defaultdict
            strikes_dict = defaultdict(list)
            for k, iv in market_data_raw:
                strikes_dict[k].append(iv)

            market_data = [(k, np.mean(ivs)) for k, ivs in strikes_dict.items()]
            market_data = sorted(market_data, key=lambda x: x[0])

            print(f"  📈 {len(market_data)} points de marché")
        else:
            # Utiliser les données Bloomberg complètes
            # Moyenner Call et Put IV
            market_smile_df['IV_Market'] = market_smile_df[['IV_Call', 'IV_Put']].mean(axis=1)
            market_data = list(zip(market_smile_df['Strike'], market_smile_df['IV_Market']))

            print(f"  📈 {len(market_data)} points de marché (Bloomberg complet)")

        # 8. Créer une grille de strikes pour le modèle Heston
        moneyness_grid = np.linspace(0.70, 1.30, 25)
        strikes_grid = moneyness_grid * F_avg

        # 9. Interpoler les IV de marché sur la grille
        iv_market_interpolated = {}

        if len(market_data) >= 2:
            strikes_market = [x[0] for x in market_data]
            ivs_market = [x[1] for x in market_data]

            from scipy.interpolate import interp1d

            interpolator = interp1d(
                strikes_market,
                ivs_market,
                kind='cubic',  # Cubic pour un smile plus lisse
                fill_value='extrapolate',
                bounds_error=False
            )

            for K in strikes_grid:
                try:
                    iv_interp = float(interpolator(K))
                    iv_interp = np.clip(iv_interp, 5, 150)
                    iv_market_interpolated[K] = iv_interp
                except:
                    pass

        # 10. Calculer IV Heston pour chaque strike
        if st.session_state.get("rates_data") is not None:
            r_local = get_rate_for_T(T)
        else:
            r_local = st.session_state.risk_free_rate

        smile_points = []

        for K in strikes_grid:
            try:
                price_heston = heston_price_rec(
                    F_avg, K,
                    heston_params['v0'],
                    heston_params['kappa'],
                    heston_params['theta'],
                    heston_params['sigma'],
                    heston_params['rho'],
                    heston_params['lambd'],
                    T, r_local
                )

                iv_heston = implied_vol_from_price_black76(price_heston, F_avg, K, T, r_local, 'C')

                if iv_heston is not None and iv_heston > 0:
                    iv_market = iv_market_interpolated.get(K, None)

                    smile_points.append({
                        'Strike': K,
                        'Moneyness': K / F_avg,
                        'IV_Heston': iv_heston * 100,
                        'IV_Market': iv_market
                    })

            except Exception as e:
                continue

        if smile_points:
            smile_df = pd.DataFrame(smile_points).sort_values('Strike')
            smile_data[T] = smile_df
            print(f"  ✅ {len(smile_points)} points calculés")

    return smile_data

def implied_vol_from_price_black76(price, F, K, T, r, option_type='C'):
    """
    Calcule la volatilité implicite depuis un prix d'option (Newton-Raphson)
    Simple approximation pour le graphique
    """
    from scipy.optimize import brentq

    def objective(sigma):
        if sigma <= 0:
            return 1e10

        sqrtT = np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        df = np.exp(-r * T)

        if option_type.upper() == 'C':
            theo_price = df * (F * norm_cdf(d1) - K * norm_cdf(d2))
        else:
            theo_price = df * (K * norm_cdf(-d2) - F * norm_cdf(-d1))

        return theo_price - price

    try:
        iv = brentq(objective, 0.001, 5.0, maxiter=100)
        return iv
    except:
        return None

def estimate_skew_from_smile(smile_df):
    """
    Estime la pente du skew depuis les données de marché

    Args:
        smile_df: DataFrame avec colonnes ['Strike', 'Moneyness', 'IV_Market']

    Returns:
        skew: Pente du skew (∂σ/∂ln(K))
    """
    df_valid = smile_df.dropna(subset=['IV_Market', 'Moneyness'])

    if len(df_valid) < 3:
        return 0.0

    # Régression linéaire: IV = a + b * ln(Moneyness)
    log_moneyness = np.log(df_valid['Moneyness'].values)
    iv_values = df_valid['IV_Market'].values / 100.0

    # Fit linéaire
    coeffs = np.polyfit(log_moneyness, iv_values, 1)
    skew = coeffs[0]  # Pente

    return skew


def compute_local_vol_greeks_for_position(row, bloomberg_df, valuation_date=None):
    """
    Calcule les Greeks avec volatilité locale pour une position
    """
    ticker = row.get('Ticker', 'Unknown')
    print(f"\n🔍 LOCAL VOL - Processing: {ticker}")

    if valuation_date is None:
        from datetime import date
        valuation_date = date.today()

    try:
        # Vérifications de base
        if "Maturity" not in row or pd.isna(row["Maturity"]):
            print(f"   ❌ Maturity manquante")
            return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Price": None, "Sigma_Local": None}

        # Calcul de T
        T = safe_time_to_maturity(row["Maturity"], valuation_date)
        print(f"   T = {T:.4f} ans")

        # Taux sans risque
        if st.session_state.get("rates_data") is not None:
            r_local = get_rate_for_T(T)
        else:
            r_local = st.session_state.risk_free_rate
        print(f"   r = {r_local:.4f}")

        # Strike et Forward
        K = row.get("Strike", row.get("Strike_Px", row.get("Strike Px", 0)))

        F = None
        if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
            bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
            F = bbg_row.get('Underlying_Price', None)

        if F is None or pd.isna(F) or F == 0:
            F = K

        print(f"   F = {F}, K = {K}")

        if K == 0 or F == 0:
            print(f"   ❌ K ou F = 0")
            return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Price": None, "Sigma_Local": None}

        # Volatilité ATM
        iv = None
        if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
            bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
            iv = bbg_row.get('IV', None)
        if iv is None or pd.isna(iv):
            iv = row.get("IV")

        if iv in (None, '', ' ') or pd.isna(iv):
            sigma_atm = 0.30
            print(f"   ⚠️ Pas d'IV → sigma_atm = {sigma_atm}")
        else:
            sigma_atm = float(iv) / 100.0
            print(f"   ✅ sigma_atm = {sigma_atm}")

        # Skew par défaut
        skew = -0.15  # Skew négatif typique pour commodities
        print(f"   skew = {skew}")

        # Type d'option
        raw_type = str(row.get("Put/Call", row.get("Put_Call", row.get("Type", "C")))).strip().upper()
        option_type = "P" if raw_type.startswith("P") else "C"
        print(f"   Type = {option_type}")

        # Calcul des Greeks
        print(f"   📞 Appel local_vol_greeks_simplified...")
        greeks = local_vol_greeks_simplified(
            F=float(F),
            K=float(K),
            T=float(T),
            r=float(r_local),
            sigma_atm=float(sigma_atm),
            skew=float(skew),
            option_type=option_type
        )

        print(f"   ✅ Greeks calculés: Delta={greeks['delta']:.4f}, Sigma_Local={greeks['sigma_local']*100:.2f}%")

        # ✅ CORRECTION : Retourner sigma_local en %
        return {
            "Delta": greeks["delta"],
            "Gamma": greeks["gamma"],
            "Vega": greeks["vega"],
            "Theta": greeks["theta"],
            "Price": greeks["price"],
            "Sigma_Local": greeks["sigma_local"] * 100  # ✅ Convertir en %
        }

    except Exception as e:
        print(f"   ❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return {"Delta": None, "Gamma": None, "Vega": None, "Theta": None, "Price": None, "Sigma_Local": None}

def compute_local_vol_smile(df_positions, df_local):
    """
    Construit un dict {T_bucket: DataFrame(Strike, Moneyness, Sigma_Local)}
    à partir des positions et des greeks local vol.
    """
    from datetime import date

    if df_positions.empty or df_local.empty:
        return {}

    valuation_date = date.today()
    df_pos = df_positions.copy()

    # 1) Calcul de T_years comme pour Heston
    if "Maturity" not in df_pos.columns:
        return {}

    df_pos["T_years"] = df_pos.apply(
        lambda row: safe_time_to_maturity(row["Maturity"], valuation_date),
        axis=1
    )
    df_pos = df_pos[df_pos["T_years"] > 0]
    if df_pos.empty:
        return {}

    # 2) Merge T_years avec les greeks locaux via le ticker
    df_pos_t = df_pos[["Ticker", "T_years"]].rename(columns={"Ticker": "Product"})
    df_lv = df_local.copy().merge(df_pos_t, on="Product", how="left")

    # Normaliser Strike
    if "Strike_px" in df_lv.columns:
        df_lv["Strike"] = df_lv["Strike_px"]

    df_lv = df_lv.dropna(subset=["T_years", "Sigma_Local", "Strike"])
    if df_lv.empty:
        return {}

    # 3) Grouper par maturité arrondie
    df_lv["T_bucket"] = df_lv["T_years"].round(1)
    smile_local = {}

    for T, df_T in df_lv.groupby("T_bucket"):
        df_T = df_T.sort_values("Strike").copy()

        # Proxy de spot pour la moneyness
        spot_proxy = df_T["Strike"].mean()
        df_T["Moneyness"] = df_T["Strike"] / spot_proxy

        smile_local[T] = df_T[["Strike", "Moneyness", "Sigma_Local"]]

    return smile_local


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

if 'heston_greeks' not in st.session_state:
    st.session_state.heston_greeks = pd.DataFrame()

if 'heston_params' not in st.session_state:
    # Paramètres par défaut (tu pourras les calibrer)
    st.session_state.heston_params = {
        'v0': 0.04,      # Initial variance
        'kappa': 2.0,    # Mean reversion speed
        'theta': 0.04,   # Long-term variance
        'sigma': 0.3,    # Vol of vol
        'rho': -0.7,     # Correlation
        'lambd': 0.0     # Market price of vol risk
    }

if 'local_vol_greeks' not in st.session_state:
    st.session_state.local_vol_greeks = pd.DataFrame()

# ============================================================================
# FONCTION DE CALCUL PRINCIPALE
# ============================================================================

def run_calculation():
    """Calcule les Greeks pour toutes les positions et récupère la courbe des taux"""
    if st.session_state.positions.empty:
        st.warning("Aucune position à calculer")
        return

    # 1. RÉCUPÉRATION DES DONNÉES BLOOMBERG (sans modifier positions)
    bloomberg_df = fetch_bloomberg_greeks(st.session_state.positions)
    st.session_state.bloomberg_greeks = bloomberg_df

    # 1.5 RÉCUPÉRATION DE LA COURBE DES TAUX US TREASURY
    st.markdown("###  Récupération de la courbe des taux")
    rates_df = fetch_bloomberg_rates_curve()
    if not rates_df.empty:
        st.session_state.rates_data = rates_df

        # Afficher un aperçu de la courbe
        col1, col2, col3, col4 = st.columns(4)
        try:
            with col1:
                rate_3m = rates_df[rates_df['Tenor'] == '3M']['Rate_%'].values[0]
                st.metric("3M", f"{rate_3m:.3f}%")
            with col2:
                rate_2y = rates_df[rates_df['Tenor'] == '2Y']['Rate_%'].values[0]
                st.metric("2Y", f"{rate_2y:.3f}%")
            with col3:
                rate_10y = rates_df[rates_df['Tenor'] == '10Y']['Rate_%'].values[0]
                st.metric("10Y", f"{rate_10y:.3f}%")
            with col4:
                rate_30y = rates_df[rates_df['Tenor'] == '30Y']['Rate_%'].values[0]
                st.metric("30Y", f"{rate_30y:.3f}%")
        except:
            pass
    else:
        st.session_state.rates_data = None

    st.markdown("---")

    # 2. CALCUL BLACK-76
    b76_results = []
    for idx, row in st.session_state.positions.iterrows():
        ticker = row.get('Ticker', '')

        # Récupérer l'IV depuis Bloomberg en PREMIER
        iv_bloomberg = None
        underlying_price_bloomberg = None
        if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
            bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
            iv_bloomberg = bbg_row.get('IV', None)
            underlying_price_bloomberg = bbg_row.get('Underlying_Price', None)

        # Créer une copie enrichie de row avec les données Bloomberg
        row_enriched = row.copy()
        if iv_bloomberg is not None and not pd.isna(iv_bloomberg):
            row_enriched['IV_Bloomberg'] = iv_bloomberg
        if underlying_price_bloomberg is not None:
            row_enriched['Underlying_Price_Bloomberg'] = underlying_price_bloomberg

        # Calculer les Greeks avec les données enrichies
        greeks = compute_b76_greeks_for_position(row_enriched, bloomberg_df, st.session_state.risk_free_rate)

        position_size = row.get('Size', row.get('Position_Size', 0))
        strike = row.get('Strike Px', row.get('Strike_Px', row.get('Strike', 0)))
        settlement_price = row.get('Settlement Price', row.get('Settlement_Price', 0))

        # Récupérer les données de prix depuis Bloomberg
        bid = None
        ask = None
        last_price = None
        iv = None
        underlying_price = None

        if not bloomberg_df.empty and ticker in bloomberg_df['Product'].values:
            bbg_row = bloomberg_df[bloomberg_df['Product'] == ticker].iloc[0]
            bid = bbg_row.get('Bid', None)
            ask = bbg_row.get('Ask', None)
            last_price = bbg_row.get('Last_Price', None)
            iv = bbg_row.get('IV', None)
            underlying_price = bbg_row.get('Underlying_Price', None)

        result = {
            'Product': ticker,
            'Position_Size': position_size,
            'Strike_px': strike,
            'Settlement_Price': settlement_price,
            'Bid': bid,
            'Ask': ask,
            'Last_Price': last_price,
            'Underlying_Price': underlying_price,  # ← AJOUTE CETTE LIGNE
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

    # 4. CALCUL HESTON
    heston_results = []
    for idx, row in st.session_state.positions.iterrows():
        greeks = compute_heston_greeks_for_position(
            row, bloomberg_df, st.session_state.heston_params, st.session_state.risk_free_rate
        )

        ticker = row.get('Ticker', '')
        position_size = row.get('Size', row.get('Position_Size', 0))
        strike = row.get('Strike Px', row.get('Strike_Px', row.get('Strike', 0)))
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
            'Price': greeks['Price'],
            'PnL': calculate_pnl(row, last_price)
        }
        heston_results.append(result)

    st.session_state.heston_greeks = pd.DataFrame(heston_results)

    # 5. CALCUL LOCAL VOLATILITY
    local_vol_results = []
    for idx, row in st.session_state.positions.iterrows():
        greeks = compute_local_vol_greeks_for_position(
            row, bloomberg_df, valuation_date=None
        )

        ticker = row.get('Ticker', '')
        position_size = row.get('Size', row.get('Position_Size', 0))
        strike = row.get('Strike Px', row.get('Strike_Px', row.get('Strike', 0)))
        settlement_price = row.get('Settlement Price', row.get('Settlement_Price', 0))

        # Récupérer les données Bloomberg
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
            'Sigma_Local': greeks['Sigma_Local'],  # Volatilité locale utilisée
            'Delta': greeks['Delta'],
            'Gamma': greeks['Gamma'],
            'Vega': greeks['Vega'],
            'Theta': greeks['Theta'],
            'Price': greeks['Price'],
            'PnL': calculate_pnl(row, last_price)
        }
        local_vol_results.append(result)

    st.session_state.local_vol_greeks = pd.DataFrame(local_vol_results)

    st.success("✅ Greeks calculés avec succès!")

def save_data():
    data = {
        'positions': st.session_state.positions.to_dict(),
        'b76_greeks': st.session_state.b76_greeks.to_dict(),
        'bachelier_greeks': st.session_state.bachelier_greeks.to_dict(),
        'bloomberg_greeks': st.session_state.bloomberg_greeks.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'heston_greeks': st.session_state.heston_greeks.to_dict(),
        'heston_params': st.session_state.heston_params
    }
    with open('risk_monitor_data.json', 'w') as f:
        json.dump(data, f)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style="position: fixed; top: 10px; left: 10px; z-index: 999; background-color: rgba(0, 119, 181, 0.9); 
                padding: 8px 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
        <a href="https://linkedin.com/in/lucaslankry" target="_blank" style="text-decoration: none; color: white; 
           display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 14px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="white" viewBox="0 0 24 24">
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
            </svg>
            Lucas Lankry
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.header("Configuration")

    st.markdown("### Bloomberg Status")
    if BLOOMBERG_AVAILABLE:
        st.success(" Bloomberg API installée")
    else:
        st.error("❌ Bloomberg API non disponible")
        st.code("pip install blpapi", language="bash")


    st.markdown("### Import de données")

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

    if st.button("Calculer les risques", use_container_width=True, type="primary"):
        if not st.session_state.positions.empty:
            with st.spinner("Calcul en cours..."):
                run_calculation()
                save_data()
                st.rerun()
        else:
            st.warning("Aucune position a calculer")

    st.markdown("---")

    st.subheader("Export")

    if not st.session_state.positions.empty:
        # Créer les boutons en 3 colonnes
        export_col1, export_col2, export_col3 = st.columns(3)

        with export_col1:
            if st.button(" Excel", use_container_width=True):
                output_file = f"risk_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    st.session_state.positions.to_excel(writer, sheet_name='Positions', index=False)

                    if not st.session_state.b76_greeks.empty:
                        export_b76 = st.session_state.b76_greeks.copy()
                        export_b76['Delta'] = export_b76['Delta'] * export_b76['Position_Size']
                        export_b76['Gamma'] = export_b76['Gamma'] * export_b76['Position_Size']
                        export_b76['Vega'] = export_b76['Vega'] * export_b76['Position_Size']
                        export_b76['Theta'] = export_b76['Theta'] * export_b76['Position_Size']
                        export_b76['Rho'] = export_b76['Rho'] * export_b76['Position_Size']
                        export_b76.to_excel(writer, sheet_name='B76_Greeks', index=False)

                    if not st.session_state.bachelier_greeks.empty:
                        export_bach = st.session_state.bachelier_greeks.copy()
                        export_bach['Delta'] = export_bach['Delta'] * export_bach['Position_Size']
                        export_bach['Gamma'] = export_bach['Gamma'] * export_bach['Position_Size']
                        export_bach['Vega'] = export_bach['Vega'] * export_bach['Position_Size']
                        export_bach['Theta'] = export_bach['Theta'] * export_bach['Position_Size']
                        export_bach['Rho'] = export_bach['Rho'] * export_bach['Position_Size']
                        export_bach.to_excel(writer, sheet_name='Bachelier_Greeks', index=False)

                    if not st.session_state.heston_greeks.empty:
                        export_heston = st.session_state.heston_greeks.copy()
                        export_heston['Delta'] = export_heston['Delta'] * export_heston['Position_Size']
                        export_heston['Gamma'] = export_heston['Gamma'] * export_heston['Position_Size']
                        export_heston['Vega'] = export_heston['Vega'] * export_heston['Position_Size']
                        export_heston['Theta'] = export_heston['Theta'] * export_heston['Position_Size']
                        export_heston['Rho'] = export_heston['Rho'] * export_heston['Position_Size']
                        export_heston.to_excel(writer, sheet_name='Heston_Greeks', index=False)

                    if not st.session_state.bloomberg_greeks.empty:
                        export_bbg = st.session_state.bloomberg_greeks.copy()
                        export_bbg['Delta'] = export_bbg['Delta'] * export_bbg['Position_Size']
                        export_bbg['Gamma'] = export_bbg['Gamma'] * export_bbg['Position_Size']
                        export_bbg['Vega'] = export_bbg['Vega'] * export_bbg['Position_Size']
                        export_bbg['Theta'] = export_bbg['Theta'] * export_bbg['Position_Size']
                        export_bbg.to_excel(writer, sheet_name='Bloomberg_Greeks', index=False)

                    if not st.session_state.local_vol_greeks.empty:
                        export_lv = st.session_state.local_vol_greeks.copy()
                        export_lv['Delta'] = export_lv['Delta'] * export_lv['Position_Size']
                        export_lv['Gamma'] = export_lv['Gamma'] * export_lv['Position_Size']
                        export_lv['Vega'] = export_lv['Vega'] * export_lv['Position_Size']
                        export_lv['Theta'] = export_lv['Theta'] * export_lv['Position_Size']
                        export_lv.to_excel(writer, sheet_name='LocalVol_Greeks', index=False)

                    if st.session_state.rates_data is not None:
                        st.session_state.rates_data.to_excel(writer, sheet_name='US_Rates_Curve', index=False)

                st.success(f"✅ Fichier exporté : {output_file}")

        with export_col2:
            csv = st.session_state.positions.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=" CSV",
                data=csv,
                file_name=f"positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with export_col3:
            if st.button(" Email", use_container_width=True):
                st.session_state.show_email_form = True

        # Formulaire d'envoi d'email (apparaît sous les boutons)
        if st.session_state.get('show_email_form', False):
            st.markdown("---")
            st.markdown("#####  Envoyer par Email")

            with st.form("email_form"):
                recipient = st.text_input(
                    "Email destinataire",
                    placeholder="exemple@domaine.com"
                )

                email_subject = st.text_input(
                    "Sujet",
                    value=f"Risk Monitor Report - {datetime.now().strftime('%Y-%m-%d')}"
                )

                email_body = st.text_area(
                    "Message",
                    value=f"""Bonjour,

Veuillez trouver ci-joint le rapport Risk Monitor.

Nombre de positions: {len(st.session_state.positions)}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Cordialement,
Lucas Lankry""",
                    height=150
                )

                submit_email = st.form_submit_button("✉️ Envoyer", use_container_width=True)

                if submit_email:
                    if not recipient or '@' not in recipient:
                        st.error("❌ Veuillez entrer une adresse email valide")
                    else:
                        # Créer le fichier Excel en mémoire
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            st.session_state.positions.to_excel(writer, sheet_name='Positions', index=False)

                            if not st.session_state.b76_greeks.empty:
                                export_b76 = st.session_state.b76_greeks.copy()
                                export_b76['Delta'] = export_b76['Delta'] * export_b76['Position_Size']
                                export_b76['Gamma'] = export_b76['Gamma'] * export_b76['Position_Size']
                                export_b76['Vega'] = export_b76['Vega'] * export_b76['Position_Size']
                                export_b76['Theta'] = export_b76['Theta'] * export_b76['Position_Size']
                                export_b76['Rho'] = export_b76['Rho'] * export_b76['Position_Size']
                                export_b76.to_excel(writer, sheet_name='B76_Greeks', index=False)

                            if not st.session_state.bachelier_greeks.empty:
                                export_bach = st.session_state.bachelier_greeks.copy()
                                export_bach['Delta'] = export_bach['Delta'] * export_bach['Position_Size']
                                export_bach['Gamma'] = export_bach['Gamma'] * export_bach['Position_Size']
                                export_bach['Vega'] = export_bach['Vega'] * export_bach['Position_Size']
                                export_bach['Theta'] = export_bach['Theta'] * export_bach['Position_Size']
                                export_bach['Rho'] = export_bach['Rho'] * export_bach['Position_Size']
                                export_bach.to_excel(writer, sheet_name='Bachelier_Greeks', index=False)

                            if not st.session_state.heston_greeks.empty:
                                export_heston = st.session_state.heston_greeks.copy()
                                export_heston['Delta'] = export_heston['Delta'] * export_heston['Position_Size']
                                export_heston['Gamma'] = export_heston['Gamma'] * export_heston['Position_Size']
                                export_heston['Vega'] = export_heston['Vega'] * export_heston['Position_Size']
                                export_heston['Theta'] = export_heston['Theta'] * export_heston['Position_Size']
                                export_heston['Rho'] = export_heston['Rho'] * export_heston['Position_Size']
                                export_heston.to_excel(writer, sheet_name='Heston_Greeks', index=False)

                            if not st.session_state.bloomberg_greeks.empty:
                                export_bbg = st.session_state.bloomberg_greeks.copy()
                                export_bbg['Delta'] = export_bbg['Delta'] * export_bbg['Position_Size']
                                export_bbg['Gamma'] = export_bbg['Gamma'] * export_bbg['Position_Size']
                                export_bbg['Vega'] = export_bbg['Vega'] * export_bbg['Position_Size']
                                export_bbg['Theta'] = export_bbg['Theta'] * export_bbg['Position_Size']
                                export_bbg['Rho'] = export_bbg['Rho'] * export_bbg['Position_Size']
                                export_bbg.to_excel(writer, sheet_name='Bloomberg_Greeks', index=False)

                            if st.session_state.rates_data is not None:
                                st.session_state.rates_data.to_excel(writer, sheet_name='US_Rates_Curve', index=False)

                        excel_data = output.getvalue()
                        filename = f"risk_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

                        with st.spinner(" Envoi en cours..."):
                            success, message = send_email_with_attachment(
                                recipient,
                                email_subject,
                                email_body,
                                excel_data,
                                filename
                            )

                        if success:
                            st.success(f"✅ {message}")
                            st.session_state.show_email_form = False
                            st.rerun()
                        else:
                            st.error(f"{message}")

            if st.button("❌ Annuler", use_container_width=True):
                st.session_state.show_email_form = False
                st.rerun()


# ============================================================================
# AFFICHAGE PRINCIPAL
# ============================================================================

df = st.session_state.positions

# Metrics en haut
if not df.empty:
    col1, col2, col3, col4, col5 = st.columns(5)

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
        if not st.session_state.b76_greeks.empty and {'Delta', 'Position_Size'}.issubset(st.session_state.b76_greeks.columns):
            df_b76 = st.session_state.b76_greeks.copy()
            df_b76['Delta'] = df_b76['Delta'] * df_b76['Position_Size']
            total_delta = df_b76['Delta'].dropna().sum()

            st.metric(
                label="Delta Total (B76)",
                value=f"{total_delta:.2f}",
                delta=f"{total_delta:.2f}" if total_delta != 0 else None
            )
        else:
            st.metric(label="Delta Total (B76)", value="0.00")

    with col4:
        if not st.session_state.b76_greeks.empty and {'Gamma', 'Position_Size'}.issubset(st.session_state.b76_greeks.columns):
            df_b76 = st.session_state.b76_greeks.copy()
            df_b76['Gamma'] = df_b76['Gamma'] * df_b76['Position_Size']
            total_gamma = df_b76['Gamma'].dropna().sum()

            st.metric(
                label="Gamma Total (B76)",
                value=f"{total_gamma:.4f}",
                delta=f"{total_gamma:.4f}" if total_gamma != 0 else None
            )
        else:
            st.metric(label="Gamma Total (B76)", value="0.0000")

    with col5:
        if not st.session_state.b76_greeks.empty and 'PnL' in st.session_state.b76_greeks.columns:
            total_pnl = st.session_state.b76_greeks['PnL'].dropna().sum()
            st.metric(
                label="PnL Total",
                value=f"${total_pnl:,.2f}",
                delta=f"${total_pnl:,.2f}" if total_pnl != 0 else None,
                delta_color="normal" if total_pnl >= 0 else "inverse"
            )
        else:
            st.metric(label="PnL Total", value="$0.00")

    st.markdown("---")

# Tabs pour différentes vues
if st.session_state.show_positions:
    selected_tab = st.radio("Navigation", ["Positions", "Greeks", "Courbe des Taux"],
                            index=1, horizontal=True, label_visibility="collapsed")
    st.session_state.show_positions = False
else:
    selected_tab = st.radio("Navigation", ["Positions", "Greeks", "Courbe des Taux"],
                            index=0, horizontal=True, label_visibility="collapsed")


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
        st.info(
            "Editez directement les valeurs dans le tableau ci-dessous, puis appuyer sur Enregistrer")

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
        display_df = st.session_state.bloomberg_greeks.copy()
        display_df['Delta'] = display_df['Delta'] * display_df['Position_Size']
        display_df['Gamma'] = display_df['Gamma'] * display_df['Position_Size']
        display_df['Vega'] = display_df['Vega'] * display_df['Position_Size']
        display_df['Theta'] = display_df['Theta'] * display_df['Position_Size']

        # ✅ AJOUT : Styling PnL
        styled_df = display_df.style

        if 'PnL' in display_df.columns:
            styled_df = styled_df.applymap(
                style_pnl_gradient,
                subset=['PnL']
            )

        # Formater les nombres
        styled_df = styled_df.format({
            'Position_Size': '{:.0f}',
            'Strike_px': '{:.2f}',
            'Settlement_Price': '{:.4f}',
            'Bid': '{:.4f}',
            'Ask': '{:.4f}',
            'Last_Price': '{:.4f}',
            'Underlying_Price': '{:.2f}',
            'IV': '{:.2f}',
            'Delta': '{:.4f}',
            'Gamma': '{:.6f}',
            'Vega': '{:.4f}',
            'Theta': '{:.4f}',
            'PnL': '{:.2f}'
        }, na_rep='N/A')

        st.dataframe(
            styled_df,
            use_container_width=True,
            height=250
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
        st.info("Aucune donnee Bloomberg. Cliquez sur 'Calculer les risques' pour calculer.")

    # Vue agrégée Bloomberg : bar chart à gauche, heatmap à droite
    with st.expander(" Voir les graphiques agrégés Bloomberg", expanded=False):
        st.markdown("### Vue agrégée des Greeks")

        col_left_bbg, col_right_bbg = st.columns(2)

        with col_left_bbg:
            st.markdown("##### Greeks par produit")

            if not st.session_state.bloomberg_greeks.empty:
                df_bbg = st.session_state.bloomberg_greeks.copy()

                # Greeks pondérés par la taille de position
                df_bbg["Delta_w"] = df_bbg["Delta"] * df_bbg["Position_Size"]
                df_bbg["Gamma_w"] = df_bbg["Gamma"] * df_bbg["Position_Size"]
                df_bbg["Vega_w"] = df_bbg["Vega"] * df_bbg["Position_Size"]

                agg_bbg = (
                    df_bbg
                    .groupby("Product")[["Delta_w", "Gamma_w", "Vega_w"]]
                    .sum()
                    .reset_index()
                )

                fig_bbg_bar = px.bar(
                    agg_bbg,
                    x="Product",
                    y=["Delta_w", "Gamma_w", "Vega_w"],
                    barmode="group",
                    title="Greeks agrégés par produit "
                )
                fig_bbg_bar.update_layout(
                    plot_bgcolor="#242833",
                    paper_bgcolor="#242833",
                    font=dict(color="white"),
                    xaxis_title="Produit",
                    yaxis_title="Valeur agrégée",
                    legend_title="Greek"
                )
                st.plotly_chart(fig_bbg_bar, use_container_width=True)
            else:
                st.info("Aucune donnée Bloomberg pour le graphique agrégé.")

        with col_right_bbg:
            st.markdown("##### Greeks agrégés par maturité (Bloomberg)")

            if not st.session_state.bloomberg_greeks.empty and not st.session_state.positions.empty:
                df_bbg = st.session_state.bloomberg_greeks.copy()
                df_pos = st.session_state.positions.copy()

                from datetime import date

                valuation_date = date.today()


                def _compute_T(row):
                    try:
                        return safe_time_to_maturity(row["Maturity"], valuation_date)
                    except Exception:
                        return None


                if "Maturity" in df_pos.columns:
                    # Calcul de T pour chaque ligne de positions
                    df_pos["T_years"] = df_pos.apply(_compute_T, axis=1)

                    # Merge T avec les greeks Bloomberg via le ticker
                    df_pos_t = df_pos[["Ticker", "T_years"]].rename(
                        columns={"Ticker": "Product"}
                    )
                    df_bbg = df_bbg.merge(df_pos_t, on="Product", how="left")

                    # Choix du Greek à afficher (logiquement ici)
                    greek_choice = st.selectbox(
                        "Greek à afficher par maturité",
                        ["Delta", "Gamma", "Vega"],
                        index=2,
                        key="bbg_greek_by_T"
                    )

                    # Greek pondéré par la taille de position
                    df_bbg["Greek_w"] = df_bbg[greek_choice] * df_bbg["Position_Size"]

                    # Bucket de maturité pour lisibilité
                    df_bbg["T_bucket"] = df_bbg["T_years"].round(2)

                    agg_T = (
                        df_bbg
                        .dropna(subset=["T_bucket"])
                        .groupby("T_bucket")["Greek_w"]
                        .sum()
                        .reset_index()
                        .sort_values("T_bucket")
                    )

                    COLOR_MAP = {
                        "Delta": "#4DB6AC",  # vert-bleu doux
                        "Gamma": "#FFB74D",  # orange doux
                        "Vega": "#42A5F5",  # bleu
                    }

                    fig_T = px.bar(
                        agg_T,
                        x="T_bucket",
                        y="Greek_w",
                        title=f"{greek_choice} agrégé par maturité ",
                        color_discrete_sequence=[COLOR_MAP.get(greek_choice, "#42A5F5")]
                    )

                    fig_T.update_layout(
                        plot_bgcolor="#242833",
                        paper_bgcolor="#242833",
                        font=dict(color="white"),
                        xaxis_title="Time to maturity (années)",
                        yaxis_title=f"{greek_choice} agrégé",
                    )

                    st.plotly_chart(fig_T, use_container_width=True)

                    # Petit texte sous le graphique pour rappeler le choix
                    st.caption("Greek sélectionné : " + greek_choice)

                else:
                    st.info("La colonne 'Maturity' n'est pas présente dans st.session_state.positions.")

            else:
                st.info("Aucune donnée Bloomberg / positions pour ce graphique.")


    st.markdown("---")

    st.markdown("### Black-76 Model")
    if not st.session_state.b76_greeks.empty:
        display_df_b76 = st.session_state.b76_greeks.copy()
        display_df_b76['Delta'] = display_df_b76['Delta'] * display_df_b76['Position_Size']
        display_df_b76['Gamma'] = display_df_b76['Gamma'] * display_df_b76['Position_Size']
        display_df_b76['Vega'] = display_df_b76['Vega'] * display_df_b76['Position_Size']


        styled_df_b76 = display_df_b76.style

        if 'PnL' in display_df_b76.columns:
            styled_df_b76 = styled_df_b76.applymap(
                style_pnl_gradient,
                subset=['PnL']
            )

        styled_df_b76 = styled_df_b76.format({
            'Position_Size': '{:.0f}',
            'Bid': '{:.4f}',
            'Ask': '{:.4f}',
            'Last_Price': '{:.4f}',
            'IV': '{:.2f}',
            'Delta': '{:.4f}',
            'Gamma': '{:.6f}',
            'Vega': '{:.4f}',
            'PnL': '{:.2f}'
        }, na_rep='N/A')

        st.dataframe(
            styled_df_b76,
            use_container_width=True,
            height=250
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

    with st.expander(" Voir les graphiques agrégés Black-76", expanded=False):
        st.markdown("### Vue agrégée des Greeks")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Greeks par produit (B76)")

            if not st.session_state.b76_greeks.empty:
                df_b76 = st.session_state.b76_greeks.copy()
                # Greeks pondérés par la taille de position
                df_b76["Delta_w"] = df_b76["Delta"] * df_b76["Position_Size"]
                df_b76["Gamma_w"] = df_b76["Gamma"] * df_b76["Position_Size"]
                df_b76["Vega_w"] = df_b76["Vega"] * df_b76["Position_Size"]

                agg_b76 = (
                    df_b76
                    .groupby("Product")[["Delta_w", "Gamma_w", "Vega_w"]]
                    .sum()
                    .reset_index()
                )

                fig_bars = px.bar(
                    agg_b76,
                    x="Product",
                    y=["Delta_w", "Gamma_w", "Vega_w"],
                    barmode="group",
                    title="Greeks agrégés par produit (B76)"
                )
                fig_bars.update_layout(
                    plot_bgcolor="#242833",
                    paper_bgcolor="#242833",
                    font=dict(color="white"),
                    xaxis_title="Produit",
                    yaxis_title="Valeur agrégée",
                    legend_title="Greek"
                )
                st.plotly_chart(fig_bars, use_container_width=True)
            else:
                st.info("Aucun Greek B76 disponible pour le graphique.")

        with col_right:
            st.markdown("#### Greeks agrégés par maturité (B76)")

            if not st.session_state.b76_greeks.empty and not st.session_state.positions.empty:
                df_b76 = st.session_state.b76_greeks.copy()
                df_pos = st.session_state.positions.copy()

                from datetime import date

                valuation_date = date.today()


                def _compute_T(row):
                    try:
                        return safe_time_to_maturity(row["Maturity"], valuation_date)
                    except Exception:
                        return None


                if "Maturity" in df_pos.columns:
                    # Calcul de T pour chaque ligne de positions
                    df_pos["T_years"] = df_pos.apply(_compute_T, axis=1)

                    # Merge T avec les greeks B76 via le ticker
                    df_pos_t = df_pos[["Ticker", "T_years"]].rename(
                        columns={"Ticker": "Product"}
                    )
                    df_b76 = df_b76.merge(df_pos_t, on="Product", how="left")

                    # Choix du Greek à afficher
                    greek_choice_b76 = st.selectbox(
                        "Greek à afficher par maturité (B76)",
                        ["Delta", "Gamma", "Vega"],
                        index=2,
                        key="b76_greek_by_T"
                    )

                    # Greek pondéré par la taille de position
                    df_b76["Greek_w"] = df_b76[greek_choice_b76] * df_b76["Position_Size"]

                    # Bucket de maturité pour lisibilité
                    df_b76["T_bucket"] = df_b76["T_years"].round(2)

                    agg_T_b76 = (
                        df_b76
                        .dropna(subset=["T_bucket"])
                        .groupby("T_bucket")["Greek_w"]
                        .sum()
                        .reset_index()
                        .sort_values("T_bucket")
                    )

                    COLOR_MAP = {
                        "Delta": "#4DB6AC",  # vert-bleu doux
                        "Gamma": "#FFB74D",  # orange doux
                        "Vega": "#42A5F5",  # bleu
                    }

                    fig_T_b76 = px.bar(
                        agg_T_b76,
                        x="T_bucket",
                        y="Greek_w",
                        title=f"{greek_choice_b76} agrégé par maturité (B76)",
                        color_discrete_sequence=[COLOR_MAP.get(greek_choice_b76, "#42A5F5")]
                    )

                    fig_T_b76.update_layout(
                        plot_bgcolor="#242833",
                        paper_bgcolor="#242833",
                        font=dict(color="white"),
                        xaxis_title="Time to maturity (années)",
                        yaxis_title=f"{greek_choice_b76} agrégé"
                    )

                    st.plotly_chart(fig_T_b76, use_container_width=True)

                    # Rappel sous le graphique
                    st.caption("Greek sélectionné : " + greek_choice_b76)
                else:
                    st.info("La colonne 'Maturity' n'est pas présente dans st.session_state.positions.")
            else:
                st.info("Aucune donnée B76 / positions pour ce graphique.")

    st.markdown("---")

    st.markdown("### Bachelier Model")
    if not st.session_state.bachelier_greeks.empty:
        display_df_bach = st.session_state.bachelier_greeks.copy()
        display_df_bach['Delta'] = display_df_bach['Delta'] * display_df_bach['Position_Size']
        display_df_bach['Gamma'] = display_df_bach['Gamma'] * display_df_bach['Position_Size']
        display_df_bach['Vega'] = display_df_bach['Vega'] * display_df_bach['Position_Size']


        styled_df_bach = display_df_bach.style

        if 'PnL' in display_df_bach.columns:
            styled_df_bach = styled_df_bach.applymap(
                style_pnl_gradient,
                subset=['PnL']
            )

        styled_df_bach = styled_df_bach.format({
            'Position_Size': '{:.0f}',
            'Bid': '{:.4f}',
            'Ask': '{:.4f}',
            'Last_Price': '{:.4f}',
            'IV': '{:.2f}',
            'Delta': '{:.4f}',
            'Gamma': '{:.6f}',
            'Vega': '{:.4f}',
            'PnL': '{:.2f}'
        }, na_rep='N/A')

        st.dataframe(
            styled_df_bach,
            use_container_width=True,
            height=250
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

    with st.expander(" Voir les graphiques agrégés Bachelier", expanded=False):
        st.markdown("### Vue agrégée des Greeks (Bachelier)")

        col_left_bach, col_right_bach = st.columns(2)


        with col_left_bach:
            st.markdown("#### Greeks par produit (Bachelier)")

            if not st.session_state.bachelier_greeks.empty:
                df_bach = st.session_state.bachelier_greeks.copy()
                df_bach["Delta_w"] = df_bach["Delta"] * df_bach["Position_Size"]
                df_bach["Gamma_w"] = df_bach["Gamma"] * df_bach["Position_Size"]
                df_bach["Vega_w"] = df_bach["Vega"] * df_bach["Position_Size"]

                agg_bach = (
                    df_bach
                    .groupby("Product")[["Delta_w", "Gamma_w", "Vega_w"]]
                    .sum()
                    .reset_index()
                )

                fig_bach_bar = px.bar(
                    agg_bach,
                    x="Product",
                    y=["Delta_w", "Gamma_w", "Vega_w"],
                    barmode="group",
                    title="Greeks agrégés par produit (Bachelier)"
                )
                fig_bach_bar.update_layout(
                    plot_bgcolor="#242833",
                    paper_bgcolor="#242833",
                    font=dict(color="white"),
                    xaxis_title="Produit",
                    yaxis_title="Valeur agrégée",
                    legend_title="Greek"
                )
                st.plotly_chart(fig_bach_bar, use_container_width=True)
            else:
                st.info("Aucun Greek Bachelier disponible pour le graphique.")


        with col_right_bach:
            st.markdown("#### Greeks agrégés par maturité (Bachelier)")

            if not st.session_state.bachelier_greeks.empty and not st.session_state.positions.empty:
                df_bach = st.session_state.bachelier_greeks.copy()
                df_pos = st.session_state.positions.copy()

                from datetime import date

                valuation_date = date.today()


                def _compute_T(row):
                    try:
                        return safe_time_to_maturity(row["Maturity"], valuation_date)
                    except Exception:
                        return None


                if "Maturity" in df_pos.columns:
                    df_pos["T_years"] = df_pos.apply(_compute_T, axis=1)

                    df_pos_t = df_pos[["Ticker", "T_years"]].rename(
                        columns={"Ticker": "Product"}
                    )
                    df_bach = df_bach.merge(df_pos_t, on="Product", how="left")

                    greek_choice_bach = st.selectbox(
                        "Greek à afficher par maturité (Bachelier)",
                        ["Delta", "Gamma", "Vega"],
                        index=2,
                        key="bach_greek_by_T"
                    )

                    df_bach["Greek_w"] = df_bach[greek_choice_bach] * df_bach["Position_Size"]
                    df_bach["T_bucket"] = df_bach["T_years"].round(2)

                    agg_T_bach = (
                        df_bach
                        .dropna(subset=["T_bucket"])
                        .groupby("T_bucket")["Greek_w"]
                        .sum()
                        .reset_index()
                        .sort_values("T_bucket")
                    )

                    COLOR_MAP = {
                        "Delta": "#4DB6AC",
                        "Gamma": "#FFB74D",
                        "Vega": "#42A5F5",
                    }

                    fig_T_bach = px.bar(
                        agg_T_bach,
                        x="T_bucket",
                        y="Greek_w",
                        title=f"{greek_choice_bach} agrégé par maturité (Bachelier)",
                        color_discrete_sequence=[COLOR_MAP.get(greek_choice_bach, "#42A5F5")]
                    )

                    fig_T_bach.update_layout(
                        plot_bgcolor="#242833",
                        paper_bgcolor="#242833",
                        font=dict(color="white"),
                        xaxis_title="Time to maturity (années)",
                        yaxis_title=f"{greek_choice_bach} agrégé"
                    )

                    st.plotly_chart(fig_T_bach, use_container_width=True)
                    st.caption("Greek sélectionné : " + greek_choice_bach)
                else:
                    st.info("La colonne 'Maturity' n'est pas présente dans st.session_state.positions.")
            else:
                st.info("Aucune donnée Bachelier / positions pour ce graphique.")

    st.markdown("---")

    st.markdown("### Heston Model")
    if not st.session_state.heston_greeks.empty:

        display_df_heston = st.session_state.heston_greeks.copy()
        display_df_heston['Delta'] = display_df_heston['Delta'] * display_df_heston['Position_Size']
        display_df_heston['Gamma'] = display_df_heston['Gamma'] * display_df_heston['Position_Size']
        display_df_heston['Vega'] = display_df_heston['Vega'] * display_df_heston['Position_Size']
        display_df_heston['Theta'] = display_df_heston['Theta'] * display_df_heston['Position_Size']
        display_df_heston['Rho'] = (display_df_heston['Rho'] * display_df_heston['Position_Size']).round(4)


        styled_df_heston = display_df_heston.style.apply(style_model_price, axis=1)


        if 'PnL' in display_df_heston.columns:
            styled_df_heston = styled_df_heston.applymap(
                style_pnl_gradient,
                subset=['PnL']
            )


        styled_df_heston = styled_df_heston.format({
            'Position_Size': '{:.0f}',
            'Strike_px': '{:.2f}',
            'Settlement_Price': '{:.4f}',
            'Bid': '{:.4f}',
            'Ask': '{:.4f}',
            'Last_Price': '{:.4f}',
            'IV': '{:.2f}',
            'Delta': '{:.4f}',
            'Gamma': '{:.6f}',
            'Vega': '{:.4f}',
            'Theta': '{:.4f}',
            'Rho': '{:.4f}',
            'Price': '{:.4f}',
            'PnL': '{:.2f}'
        }, na_rep='N/A')


        st.dataframe(
            styled_df_heston,
            use_container_width=True,
            height=250
        )


        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_delta_heston = display_df_heston['Delta'].dropna().sum()
            st.metric("Delta Total", f"{total_delta_heston:.4f}")
        with col2:
            total_gamma_heston = display_df_heston['Gamma'].dropna().sum()
            st.metric("Gamma Total", f"{total_gamma_heston:.6f}")
        with col3:
            total_vega_heston = display_df_heston['Vega'].dropna().sum()
            st.metric("Vega Total", f"{total_vega_heston:.4f}")
        with col4:
            total_theta_heston = display_df_heston['Theta'].dropna().sum()
            st.metric("Theta Total", f"{total_theta_heston:.4f}")

        # ============ VOLATILITY SMILE ============
        st.markdown("####  Volatility Smile par Maturité (Heston)")


        if not st.session_state.positions.empty:
            df_pos_temp = st.session_state.positions.copy()
            df_pos_temp['Underlying_Code'] = df_pos_temp['Ticker'].apply(extract_underlying_code)
            available_underlyings = sorted(df_pos_temp['Underlying_Code'].dropna().unique())

            if available_underlyings:
                col_filter, col_fetch = st.columns([3, 1])

                with col_filter:
                    selected_underlying = st.selectbox(
                        " Sélectionner le sous-jacent",
                        options=['Tous'] + available_underlyings,
                        help="Filtrer par sous-jacent pour un smile plus précis"
                    )

                with col_fetch:
                    fetch_complete_market = st.checkbox(
                        " Récupérer smile complet depuis Bloomberg",
                        value=True,
                        help="Récupère 15+ strikes depuis Bloomberg pour un smile de marché complet"
                    )

                underlying_filter = None if selected_underlying == 'Tous' else selected_underlying

                if st.button(" Calculer le Smile", type="primary"):
                    with st.spinner(" Calcul du smile de volatilité..."):
                        smile_data = compute_heston_volatility_smile_filtered(
                            st.session_state.positions,
                            st.session_state.bloomberg_greeks,
                            st.session_state.heston_params,
                            underlying_filter=underlying_filter,
                            fetch_market_data=fetch_complete_market
                        )

                        # Stocker dans session state
                        st.session_state['smile_data'] = smile_data
                        st.session_state['smile_underlying'] = selected_underlying


        if 'smile_data' in st.session_state and st.session_state.smile_data:
            smile_data = st.session_state.smile_data

            st.success(
                f" Smile calculé pour {len(smile_data)} maturité(s) - Sous-jacent: {st.session_state.get('smile_underlying', 'Tous')}")


            fig_smile = go.Figure()

            colors = px.colors.qualitative.Plotly

            for i, (T, df_smile) in enumerate(smile_data.items()):
                color = colors[i % len(colors)]

                # Courbe Heston
                fig_smile.add_trace(go.Scatter(
                    x=df_smile['Moneyness'],
                    y=df_smile['IV_Heston'],
                    mode='lines',
                    name=f'Heston {T:.2f}Y',
                    line=dict(color=color, width=3),
                    hovertemplate='<b>Strike:</b> %{customdata:.2f}<br>' +
                                  '<b>Moneyness:</b> %{x:.2f}<br>' +
                                  '<b>IV Heston:</b> %{y:.2f}%<br>' +
                                  '<extra></extra>',
                    customdata=df_smile['Strike']
                ))

                # Points marché
                df_market = df_smile.dropna(subset=['IV_Market'])
                if not df_market.empty:
                    fig_smile.add_trace(go.Scatter(
                        x=df_market['Moneyness'],
                        y=df_market['IV_Market'],
                        mode='markers',
                        name=f'Market {T:.2f}Y',
                        marker=dict(
                            size=10,
                            color=color,
                            symbol='circle-open',
                            line=dict(width=2)
                        ),
                        hovertemplate='<b>Strike:</b> %{customdata:.2f}<br>' +
                                      '<b>Moneyness:</b> %{x:.2f}<br>' +
                                      '<b>IV Market:</b> %{y:.2f}%<br>' +
                                      '<extra></extra>',
                        customdata=df_market['Strike']
                    ))

            fig_smile.update_layout(
                title={
                    'text': f"Implied Volatility Smile - {st.session_state.get('smile_underlying', 'All Underlyings')}",
                    'font': {'size': 18}
                },
                xaxis_title="Moneyness (Strike/Spot)",
                yaxis_title="Implied Volatility (%)",
                plot_bgcolor='#242833',
                paper_bgcolor='#242833',
                font=dict(color='white'),
                hovermode='closest',
                height=600,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor='rgba(36, 40, 51, 0.8)'
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    range=[0.65, 1.35]
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.1)'
                )
            )

            st.plotly_chart(fig_smile, use_container_width=True)

            # Afficher les données pour debug
            with st.expander(" Données détaillées"):
                for T, df_smile in smile_data.items():
                    st.markdown(f"**Maturité: {T:.2f} ans**")
                    st.dataframe(
                        df_smile,
                        column_config={
                            "Strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                            "Moneyness": st.column_config.NumberColumn("Moneyness", format="%.3f"),
                            "IV_Heston": st.column_config.NumberColumn("IV Heston (%)", format="%.2f"),
                            "IV_Market": st.column_config.NumberColumn("IV Market (%)", format="%.2f")
                        }
                    )

            st.info("""
             **Lecture du graphique:**
            - **Lignes continues** : Volatilité implicite du modèle Heston
            - **Cercles ouverts** : Volatilité implicite de marché (Bloomberg)
            """)

        else:
            st.info(" Sélectionnez un sous-jacent et cliquez sur 'Calculer le Smile'")

    else:
        st.info("Aucune donnee Heston. Cliquez sur 'Actualiser avec Bloomberg' pour calculer.")

    st.markdown("---")

    st.markdown("### Local Volatility Model")
    if not st.session_state.local_vol_greeks.empty:
        display_df_lv = st.session_state.local_vol_greeks.copy()

        # Multiplier par Position_Size
        display_df_lv['Delta'] = display_df_lv['Delta'] * display_df_lv['Position_Size']
        display_df_lv['Gamma'] = display_df_lv['Gamma'] * display_df_lv['Position_Size']
        display_df_lv['Vega'] = display_df_lv['Vega'] * display_df_lv['Position_Size']
        display_df_lv['Theta'] = display_df_lv['Theta'] * display_df_lv['Position_Size']


        styled_df_lv = display_df_lv.style.apply(style_model_price, axis=1)

        if 'PnL' in display_df_lv.columns:
            styled_df_lv = styled_df_lv.applymap(
                style_pnl_gradient,
                subset=['PnL']
            )

        styled_df_lv = styled_df_lv.format({
            'Position_Size': '{:.0f}',
            'Strike_px': '{:.2f}',
            'Settlement_Price': '{:.4f}',
            'Bid': '{:.4f}',
            'Ask': '{:.4f}',
            'Last_Price': '{:.4f}',
            'IV': '{:.2f}',
            'Sigma_Local': '{:.2f}',
            'Delta': '{:.4f}',
            'Gamma': '{:.6f}',
            'Vega': '{:.4f}',
            'Theta': '{:.4f}',
            'Price': '{:.4f}',
            'PnL': '{:.2f}'
        }, na_rep='N/A')

        st.dataframe(styled_df_lv, use_container_width=True, height=250)

        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_delta_lv = display_df_lv['Delta'].dropna().sum()
            st.metric("Delta Total", f"{total_delta_lv:.4f}")
        with col2:
            total_gamma_lv = display_df_lv['Gamma'].dropna().sum()
            st.metric("Gamma Total", f"{total_gamma_lv:.6f}")
        with col3:
            total_vega_lv = display_df_lv['Vega'].dropna().sum()
            st.metric("Vega Total", f"{total_vega_lv:.4f}")
        with col4:
            total_theta_lv = display_df_lv['Theta'].dropna().sum()
            st.metric("Theta Total", f"{total_theta_lv:.4f}")

        # Info sur le modèle
        st.info("""
         **Local Volatility Model:**
        - Ajuste la volatilité en fonction du strike (skew)
        - Plus réaliste pour les marchés avec smile prononcé
        - Colonne `Sigma_Local` = volatilité effective utilisée pour chaque strike
        """)
    else:
        st.info("Aucune donnée Local Vol. Cliquez sur 'Calculer les risques' pour calculer.")

    st.markdown("####  Volatility Smile par Maturité (Local Vol)")


    if not st.session_state.positions.empty:
        df_pos_temp = st.session_state.positions.copy()
        df_pos_temp['Underlying_Code'] = df_pos_temp['Ticker'].apply(extract_underlying_code)
        available_underlyings_lv = sorted(df_pos_temp['Underlying_Code'].dropna().unique())

        if available_underlyings_lv:
            col_filter_lv, col_fetch_lv = st.columns([3, 1])

            with col_filter_lv:
                selected_underlying_lv = st.selectbox(
                    " Sélectionner le sous-jacent (Local Vol)",
                    options=['Tous'] + available_underlyings_lv,
                    help="Filtrer par sous-jacent pour un smile plus précis",
                    key="lv_underlying_selector"
                )

            with col_fetch_lv:
                fetch_complete_market_lv = st.checkbox(
                    " Récupérer smile complet depuis Bloomberg",
                    value=True,
                    help="Récupère 15+ strikes depuis Bloomberg pour un smile de marché complet",
                    key="lv_fetch_complete"
                )

            underlying_filter_lv = None if selected_underlying_lv == 'Tous' else selected_underlying_lv

            if st.button(" Calculer le Smile (Local Vol)", type="primary", key="calc_smile_lv"):
                with st.spinner(" Calcul du smile de volatilité locale..."):


                    df_pos_filtered = st.session_state.positions.copy()
                    if underlying_filter_lv:
                        df_pos_filtered['Underlying_Code'] = df_pos_filtered['Ticker'].apply(extract_underlying_code)
                        df_pos_filtered = df_pos_filtered[df_pos_filtered['Underlying_Code'] == underlying_filter_lv]

                    if df_pos_filtered.empty:
                        st.warning(f"Aucune position pour {underlying_filter_lv}")
                    else:

                        smile_data_lv = {}

                        from datetime import date

                        valuation_date = date.today()

                        # Calculer T pour chaque position
                        df_pos_filtered["T_years"] = df_pos_filtered.apply(
                            lambda row: safe_time_to_maturity(row["Maturity"], valuation_date),
                            axis=1
                        )
                        df_pos_filtered = df_pos_filtered[df_pos_filtered["T_years"] > 0]

                        if not df_pos_filtered.empty:
                            # Grouper par maturité arrondie
                            df_pos_filtered["T_bucket"] = df_pos_filtered["T_years"].round(1)
                            unique_maturities_lv = sorted(df_pos_filtered["T_bucket"].unique())

                            # Limiter à 4 maturités max
                            if len(unique_maturities_lv) > 4:
                                indices = [0, len(unique_maturities_lv) // 4, 3 * len(unique_maturities_lv) // 4, -1]
                                maturities_to_plot_lv = [unique_maturities_lv[i] for i in indices]
                            else:
                                maturities_to_plot_lv = unique_maturities_lv

                            # Pour chaque maturité
                            for T in maturities_to_plot_lv:
                                df_maturity = df_pos_filtered[abs(df_pos_filtered["T_bucket"] - T) < 0.05]

                                if df_maturity.empty:
                                    continue


                                F_list = []
                                for idx, row in df_maturity.iterrows():
                                    ticker = row.get('Ticker', '')
                                    if not st.session_state.bloomberg_greeks.empty and ticker in \
                                            st.session_state.bloomberg_greeks['Product'].values:
                                        bbg_row = st.session_state.bloomberg_greeks[
                                            st.session_state.bloomberg_greeks['Product'] == ticker].iloc[0]
                                        F_tmp = bbg_row.get('Underlying_Price', None)
                                        if F_tmp and not pd.isna(F_tmp) and F_tmp > 0:
                                            F_list.append(F_tmp)

                                if not F_list:
                                    continue

                                F_avg = np.mean(F_list)


                                market_smile_df = pd.DataFrame()

                                if fetch_complete_market_lv and underlying_filter_lv and BLOOMBERG_AVAILABLE:
                                    maturity_str = df_maturity.iloc[0]['Maturity']
                                    session = start_bloomberg_session()
                                    if session:
                                        market_smile_df = fetch_complete_smile_from_bloomberg(
                                            session,
                                            underlying_filter_lv,
                                            maturity_str,
                                            n_strikes=15
                                        )
                                        session.stop()

                                # Construire les données du smile
                                if market_smile_df.empty:

                                    market_data_raw = []
                                    for idx, row in df_maturity.iterrows():
                                        ticker = row.get('Ticker', '')
                                        K_market = row.get('Strike', row.get('Strike_Px', row.get('Strike Px', 0)))

                                        if K_market == 0:
                                            continue

                                        if not st.session_state.bloomberg_greeks.empty and ticker in \
                                                st.session_state.bloomberg_greeks['Product'].values:
                                            bbg_row = st.session_state.bloomberg_greeks[
                                                st.session_state.bloomberg_greeks['Product'] == ticker].iloc[0]
                                            iv_mkt = bbg_row.get('IV', None)
                                            if iv_mkt and not pd.isna(iv_mkt) and iv_mkt > 0:
                                                market_data_raw.append((K_market, iv_mkt))

                                    from collections import defaultdict

                                    strikes_dict = defaultdict(list)
                                    for k, iv in market_data_raw:
                                        strikes_dict[k].append(iv)

                                    market_data = [(k, np.mean(ivs)) for k, ivs in strikes_dict.items()]
                                    market_data = sorted(market_data, key=lambda x: x[0])
                                else:
                                    market_smile_df['IV_Market'] = market_smile_df[['IV_Call', 'IV_Put']].mean(axis=1)
                                    market_data = list(zip(market_smile_df['Strike'], market_smile_df['IV_Market']))

                                # Créer une grille de strikes
                                moneyness_grid = np.linspace(0.70, 1.30, 25)
                                strikes_grid = moneyness_grid * F_avg

                                # Interpoler les IV de marché
                                iv_market_interpolated = {}
                                if len(market_data) >= 2:
                                    strikes_market = [x[0] for x in market_data]
                                    ivs_market = [x[1] for x in market_data]

                                    from scipy.interpolate import interp1d

                                    interpolator = interp1d(
                                        strikes_market,
                                        ivs_market,
                                        kind='cubic',
                                        fill_value='extrapolate',
                                        bounds_error=False
                                    )

                                    for K in strikes_grid:
                                        try:
                                            iv_interp = float(interpolator(K))
                                            iv_interp = np.clip(iv_interp, 5, 150)
                                            iv_market_interpolated[K] = iv_interp
                                        except:
                                            pass

                                # Calculer la volatilité locale pour chaque strike
                                smile_points = []


                                skew = -0.10  # Valeur par défaut

                                if len(market_data) >= 3:
                                    # Calculer le skew empirique depuis les IV de marché
                                    strikes_market = np.array([x[0] for x in market_data])
                                    ivs_market = np.array([x[1] / 100.0 for x in market_data])  # Convertir en décimal

                                    # Calculer log-moneyness
                                    log_moneyness_market = np.log(strikes_market / F_avg)

                                    # Régression linéaire : IV = a + b * log(K/F)
                                    # Le coefficient b est le skew
                                    try:
                                        coeffs = np.polyfit(log_moneyness_market, ivs_market, 1)
                                        skew = coeffs[0]  # Pente = skew
                                        sigma_atm_calibrated = coeffs[1]  # Intercept = σ_ATM

                                        # Utiliser le sigma_atm calibré si raisonnable
                                        if 0.05 <= sigma_atm_calibrated <= 1.5:
                                            sigma_atm = sigma_atm_calibrated

                                        st.write(
                                            f" **Calibration** - Skew: {skew:.4f} | σ_ATM: {sigma_atm * 100:.2f}%")
                                    except Exception as e:
                                        st.warning(f"Calibration skew échouée: {e}")
                                        pass

                                # Récupérer sigma_atm moyen si pas calibré
                                if 'sigma_atm' not in locals():
                                    if not st.session_state.local_vol_greeks.empty:
                                        df_lv_maturity = st.session_state.local_vol_greeks.merge(
                                            df_maturity[['Ticker']].rename(columns={'Ticker': 'Product'}),
                                            on='Product',
                                            how='inner'
                                        )
                                        if not df_lv_maturity.empty:
                                            sigma_atm_values = df_lv_maturity['Sigma_Local'].dropna()
                                            if len(sigma_atm_values) > 0:
                                                sigma_atm = sigma_atm_values.mean() / 100.0
                                            else:
                                                sigma_atm = 0.30
                                        else:
                                            sigma_atm = 0.30
                                    else:
                                        sigma_atm = 0.30

                                # Taux sans risque
                                if st.session_state.get("rates_data") is not None:
                                    r_local = get_rate_for_T(T)
                                else:
                                    r_local = st.session_state.risk_free_rate

                                for K in strikes_grid:
                                    try:
                                        # Calculer la volatilité locale pour ce strike
                                        greeks_lv = local_vol_greeks_simplified(
                                            F=float(F_avg),
                                            K=float(K),
                                            T=float(T),
                                            r=float(r_local),
                                            sigma_atm=float(sigma_atm),
                                            skew=float(skew),
                                            option_type='C'
                                        )

                                        sigma_local = greeks_lv['sigma_local']  # Déjà en décimal
                                        iv_market = iv_market_interpolated.get(K, None)

                                        smile_points.append({
                                            'Strike': K,
                                            'Moneyness': K / F_avg,
                                            'IV_LocalVol': sigma_local * 100,  # Convertir en %
                                            'IV_Market': iv_market
                                        })

                                    except Exception as e:
                                        continue

                                if smile_points:
                                    smile_df = pd.DataFrame(smile_points).sort_values('Strike')
                                    smile_data_lv[T] = smile_df

                            # Stocker dans session state
                            st.session_state['smile_data_lv'] = smile_data_lv
                            st.session_state['smile_underlying_lv'] = selected_underlying_lv

    # Afficher le graphique si disponible
    if 'smile_data_lv' in st.session_state and st.session_state.smile_data_lv:
        smile_data_lv = st.session_state.smile_data_lv

        st.success(
            f" Smile calculé pour {len(smile_data_lv)} maturité(s) - Sous-jacent: {st.session_state.get('smile_underlying_lv', 'Tous')}")

        # Créer le graphique
        fig_smile_lv = go.Figure()

        colors = px.colors.qualitative.Plotly

        for i, (T, df_smile) in enumerate(smile_data_lv.items()):
            color = colors[i % len(colors)]

            # Courbe Local Vol
            fig_smile_lv.add_trace(go.Scatter(
                x=df_smile['Moneyness'],
                y=df_smile['IV_LocalVol'],
                mode='lines',
                name=f'Local Vol {T:.2f}Y',
                line=dict(color=color, width=3),
                hovertemplate='<b>Strike:</b> %{customdata:.2f}<br>' +
                              '<b>Moneyness:</b> %{x:.2f}<br>' +
                              '<b>IV Local Vol:</b> %{y:.2f}%<br>' +
                              '<extra></extra>',
                customdata=df_smile['Strike']
            ))

            # Points marché
            df_market = df_smile.dropna(subset=['IV_Market'])
            if not df_market.empty:
                fig_smile_lv.add_trace(go.Scatter(
                    x=df_market['Moneyness'],
                    y=df_market['IV_Market'],
                    mode='markers',
                    name=f'Market {T:.2f}Y',
                    marker=dict(
                        size=10,
                        color=color,
                        symbol='circle-open',
                        line=dict(width=2)
                    ),
                    hovertemplate='<b>Strike:</b> %{customdata:.2f}<br>' +
                                  '<b>Moneyness:</b> %{x:.2f}<br>' +
                                  '<b>IV Market:</b> %{y:.2f}%<br>' +
                                  '<extra></extra>',
                    customdata=df_market['Strike']
                ))

        fig_smile_lv.update_layout(
            title={
                'text': f"Implied Volatility Smile (Local Vol) - {st.session_state.get('smile_underlying_lv', 'All Underlyings')}",
                'font': {'size': 18}
            },
            xaxis_title="Moneyness (Strike/Spot)",
            yaxis_title="Implied Volatility (%)",
            plot_bgcolor='#242833',
            paper_bgcolor='#242833',
            font=dict(color='white'),
            hovermode='closest',
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(36, 40, 51, 0.8)'
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                range=[0.65, 1.35]
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            )
        )

        st.plotly_chart(fig_smile_lv, use_container_width=True)

        # Afficher les données pour debug
        with st.expander(" Données détaillées"):
            for T, df_smile in smile_data_lv.items():
                st.markdown(f"**Maturité: {T:.2f} ans**")
                st.dataframe(
                    df_smile,
                    column_config={
                        "Strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                        "Moneyness": st.column_config.NumberColumn("Moneyness", format="%.3f"),
                        "IV_LocalVol": st.column_config.NumberColumn("IV Local Vol (%)", format="%.2f"),
                        "IV_Market": st.column_config.NumberColumn("IV Market (%)", format="%.2f")
                    }
                )

        st.info("""
         **Lecture du graphique:**
        - **Lignes continues** : Volatilité implicite du modèle Local Volatility
        - **Cercles ouverts** : Volatilité implicite de marché (Bloomberg)
        - Le modèle ajuste la volatilité en fonction du strike (skew)
        """)

    else:
        st.info(" Sélectionnez un sous-jacent et cliquez sur 'Calculer le Smile'")

if selected_tab == "Courbe des Taux":
    st.subheader("Courbe des Taux US Treasury")

    if st.session_state.rates_data is not None and not st.session_state.rates_data.empty:
        rates_df = st.session_state.rates_data

        # Vérifier que les colonnes nécessaires existent
        if 'Maturity_Years' in rates_df.columns and 'Rate_%' in rates_df.columns:

            # Calculer le minimum pour l'axe Y
            min_rate = rates_df['Rate_%'].min()
            y_axis_min = min_rate - 0.20

            # Créer le graphique Plotly (style Bloomberg)
            fig_rates = go.Figure()

            # Ajouter la courbe avec ligne lissée
            fig_rates.add_trace(go.Scatter(
                x=rates_df['Maturity_Years'],
                y=rates_df['Rate_%'],
                mode='lines+markers',
                name='US Treasury Yield',
                line=dict(
                    color='#FF4136',
                    width=3,
                    shape='spline'
                ),
                marker=dict(
                    size=8,
                    color='#FF4136',
                    symbol='circle'
                ),
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                              '<b>Maturity:</b> %{x:.1f} years<br>' +
                              '<b>Yield:</b> %{y:.3f}%<br>' +
                              '<extra></extra>',
                customdata=rates_df[['Tenor']].values
            ))

            fig_rates.update_layout(
                title={
                    'text': 'US Treasury Yield Curve - Last Mid YTM',
                    'font': {'size': 18, 'color': '#FFFFFF'}
                },
                xaxis=dict(
                    title=dict(
                        text='Maturity (Years)',
                        font=dict(size=16, color="white", family="Arial")
                    ),
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.15)',
                    gridwidth=1,
                    zeroline=True,
                    zerolinecolor='rgba(255, 255, 255, 0.3)',
                    zerolinewidth=2,
                    tickfont=dict(size=12, color="white"),
                    range=[0, 30],
                    dtick=5,
                    tickformat='.1f'
                ),
                yaxis=dict(
                    title=dict(
                        text='Yield (%)',
                        font=dict(size=16, color="white", family="Arial")
                    ),
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.15)',
                    gridwidth=1,
                    zeroline=True,
                    zerolinecolor='rgba(255, 255, 255, 0.3)',
                    zerolinewidth=2,
                    tickfont=dict(size=12, color="white"),
                    range=[y_axis_min, rates_df['Rate_%'].max() + 0.2],
                    tickformat='.3f'
                ),
                hovermode='x unified',
                plot_bgcolor='#242833',
                paper_bgcolor='#242833',
                height=600,
                showlegend=False,
                margin=dict(l=80, r=40, t=80, b=60)
            )

            st.plotly_chart(fig_rates, use_container_width=True)

            # Métriques clés
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                try:
                    rate_3m = rates_df[rates_df['Tenor'] == '3M']['Rate_%'].values[0]
                    st.metric("3 Mois", f"{rate_3m:.3f}%")
                except:
                    st.metric("3 Mois", "N/A")

            with col2:
                try:
                    rate_2y = rates_df[rates_df['Tenor'] == '2Y']['Rate_%'].values[0]
                    st.metric("2 Ans", f"{rate_2y:.3f}%")
                except:
                    st.metric("2 Ans", "N/A")

            with col3:
                try:
                    rate_10y = rates_df[rates_df['Tenor'] == '10Y']['Rate_%'].values[0]
                    st.metric("10 Ans", f"{rate_10y:.3f}%")
                except:
                    st.metric("10 Ans", "N/A")

            with col4:
                try:
                    rate_30y = rates_df[rates_df['Tenor'] == '30Y']['Rate_%'].values[0]
                    st.metric("30 Ans", f"{rate_30y:.3f}%")
                except:
                    st.metric("30 Ans", "N/A")

            st.markdown("---")

            # Tableau des données
            st.markdown("###  Données détaillées")

            st.dataframe(
                rates_df,
                use_container_width=True,
                column_config={
                    "Tenor": st.column_config.TextColumn("Tenor", width="small"),
                    "Maturity_Years": st.column_config.NumberColumn(
                        "Maturity (Years)",
                        format="%.2f"
                    ),
                    "Ticker": st.column_config.TextColumn("Bloomberg Ticker"),
                    "Rate_%": st.column_config.NumberColumn(
                        "Yield (%)",
                        format="%.3f"
                    ),
                    "Rate_Decimal": st.column_config.NumberColumn(
                        "Yield (Decimal)",
                        format="%.5f"
                    )
                },
                hide_index=True
            )

        else:
            st.error("❌ Les colonnes nécessaires sont manquantes")

    else:
        st.info(" Aucune courbe des taux disponible.")
        st.markdown("""
        **Pour récupérer la courbe des taux US Treasury:**

        1. Assurez-vous que Bloomberg Terminal est actif
        2. Cliquez sur **"Calculer les risques"** dans la sidebar
        3. La courbe sera automatiquement récupérée depuis Bloomberg

        **Tickers Bloomberg utilisés:**
        - 3M: USGG3M Index
        - 2Y: USGG2YR Index
        - 10Y: USGG10YR Index
        - 30Y: USGG30YR Index
        - ... et 7 autres tenors
        """)

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center'>
        <p>Risk Monitor v2.1 | Derniere mise a jour : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    """,
    unsafe_allow_html=True
)
