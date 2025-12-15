import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
                        'Underlying_Price': underlying_price,  # ← AJOUT
                        'IV': None,
                        'Delta': None,
                        'Gamma': None,
                        'Vega': None,
                        'Theta': None,
                        'Rho': None,
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
# FONCTIONS HESTON MODEL
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
                        export_bbg['Rho'] = export_bbg['Rho'] * export_bbg['Position_Size']
                        export_bbg.to_excel(writer, sheet_name='Bloomberg_Greeks', index=False)

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
Risk Monitor System""",
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
    selected_tab = st.radio("Navigation", ["Positions", "Greeks", "Courbe des Taux", "Analyse"],
                            index=1, horizontal=True, label_visibility="collapsed")
    st.session_state.show_positions = False
else:
    selected_tab = st.radio("Navigation", ["Positions", "Greeks", "Courbe des Taux", "Analyse"],
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
        st.info("Aucune donnee Bloomberg. Cliquez sur 'Calculer les risques' pour calculer.")

    # Vue agrégée Bloomberg : bar chart à gauche, heatmap à droite
    st.markdown("#### Vue agrégée Bloomberg Greeks")

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

                fig_T = px.bar(
                    agg_T,
                    x="T_bucket",
                    y="Greek_w",
                    title=f"{greek_choice} agrégé par maturité "
                )
                fig_T.update_layout(
                    plot_bgcolor="#242833",
                    paper_bgcolor="#242833",
                    font=dict(color="white"),
                    xaxis_title="Time to maturity (années)",
                    yaxis_title=f"{greek_choice} agrégé"
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

                # Choix du Greek à afficher (sélecteur sous le graphe mais logique ici)
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

                fig_T_b76 = px.bar(
                    agg_T_b76,
                    x="T_bucket",
                    y="Greek_w",
                    title=f"{greek_choice_b76} agrégé par maturité (B76)"
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

    st.markdown("### Vue agrégée des Greeks (Bachelier)")

    col_left_bach, col_right_bach = st.columns(2)

    # Colonne gauche : par produit
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

    # Colonne droite : par maturité
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

                fig_T_bach = px.bar(
                    agg_T_bach,
                    x="T_bucket",
                    y="Greek_w",
                    title=f"{greek_choice_bach} agrégé par maturité (Bachelier)"
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
        # Créer une copie du dataframe avec les Greeks pondérés
        display_df_heston = st.session_state.heston_greeks.copy()
        display_df_heston['Delta'] = display_df_heston['Delta'] * display_df_heston['Position_Size']
        display_df_heston['Gamma'] = display_df_heston['Gamma'] * display_df_heston['Position_Size']
        display_df_heston['Vega'] = display_df_heston['Vega'] * display_df_heston['Position_Size']
        display_df_heston['Theta'] = display_df_heston['Theta'] * display_df_heston['Position_Size']
        display_df_heston['Rho'] = (display_df_heston['Rho'] * display_df_heston['Position_Size']).round(4)

        st.dataframe(
            display_df_heston,
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
                "Price": st.column_config.NumberColumn("Model Price", format="%.4f"),
                "PnL": st.column_config.NumberColumn("P&L", format="%.2f")
            }
        )

        # Afficher les totaux Heston
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
    else:
        st.info("Aucune donnee Heston. Cliquez sur 'Actualiser avec Bloomberg' pour calculer.")

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