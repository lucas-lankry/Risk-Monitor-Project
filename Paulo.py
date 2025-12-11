import pandas as pd
import blpapi
import os
from pathlib import Path
from datetime import datetime, date
import platform
import subprocess
import math

# ============================================================================
# CONFIGURATION
# ============================================================================
desktop = Path.home() / "Desktop"
input_file = desktop / "Commodities Deals.xlsx"
output_file = desktop / "Cleaned_Commodities_Deals.xlsx"

risk_free_rate = 0.05  # 5 %
valuation_date = datetime.today().date()

# ============================================================================
# ÉTAPE 1: NETTOYAGE DES DONNÉES
# ============================================================================
print("=" * 70)
print("ÉTAPE 1: NETTOYAGE DES DONNÉES")
print("=" * 70)

# Lecture du fichier Excel d'origine
df = pd.read_excel(input_file)

print(f"✓ Colonnes originales: {df.columns.tolist()}")
print(f"✓ Nombre de colonnes: {len(df.columns)}")
print(f"✓ Total de lignes: {len(df)}\n")

# Renommer les colonnes selon le nombre (AJOUT DE LA COLONNE SIZE)
if len(df.columns) == 8:
    df.columns = [
        'Ticker', 'Contract_Size', 'Maturity', 'Settlement_Price',
        'Strike_Px', 'Put_Call', 'Size', 'Extra_Column'
    ]
    print("⚠️  8 colonnes détectées - Vérifiez 'Extra_Column'")
elif len(df.columns) == 7:
    df.columns = [
        'Ticker', 'Contract_Size', 'Maturity', 'Settlement_Price',
        'Strike_Px', 'Put_Call', 'Size'
    ]
    print("✓ 7 colonnes détectées (avec Size)")
elif len(df.columns) == 6:
    df.columns = [
        'Ticker', 'Contract_Size', 'Maturity', 'Settlement_Price',
        'Strike_Px', 'Put_Call'
    ]
    print("⚠️  Pas de colonne Size détectée - Utilisation de 1 par défaut")
    df['Size'] = 1
else:
    raise ValueError(f"Nombre de colonnes inattendu: {len(df.columns)}")

# Créer un identifiant unique
df['Unique_ID'] = (
        df['Ticker'].astype(str) + "_" +
        df['Maturity'].astype(str) + "_" +
        df['Strike_Px'].astype(str) + "_" +
        df['Put_Call'].astype(str)
)

# Grouper par identifiant unique et SOMMER les Size
grouped = df.groupby('Unique_ID').agg({
    'Ticker': 'first',
    'Maturity': 'first',
    'Settlement_Price': 'first',
    'Strike_Px': 'first',
    'Put_Call': 'first',
    'Contract_Size': 'first',
    'Size': 'sum'
}).reset_index(drop=True)

# Créer le DataFrame final
result_df = pd.DataFrame({
    'Ticker': grouped['Ticker'],
    'Maturity': grouped['Maturity'],
    'Settlement Price': grouped['Settlement_Price'],
    'Position': grouped['Size'],
    'Bid': '',
    'Ask': '',
    'Last Price': '',
    'IV': '',
    'Delta': '',
    'Gamma': '',
    'Vega': '',
    'Theta': '',
    'Rho': '',
    'Strike_Px': grouped['Strike_Px'],
    'Contract_Size': grouped['Contract_Size'],
    'Premium': grouped['Settlement_Price'],
    'Put_Call': grouped['Put_Call']
})

print(f"✓ {len(result_df)} positions uniques créées")
print(f"✓ Total des positions (somme): {result_df['Position'].sum()}\n")

# ============================================================================
# ÉTAPE 2: CONNEXION BLOOMBERG - VERSION VECTORISÉE
# ============================================================================
print("=" * 70)
print("ÉTAPE 2: RÉCUPÉRATION DES DONNÉES BLOOMBERG (VECTORISÉE)")
print("=" * 70)


def start_bloomberg_session():
    """Démarre une session Bloomberg API"""
    options = blpapi.SessionOptions()
    options.serverHost = "localhost"
    options.serverPort = 8194

    session = blpapi.Session(options)
    if not session.start():
        raise Exception("Impossible de démarrer la session Bloomberg")
    if not session.openService("//blp/refdata"):
        raise Exception("Impossible d'ouvrir le service refdata")

    return session


def get_bloomberg_data_batch(session, tickers, fields):
    """
    Récupère plusieurs champs Bloomberg pour plusieurs tickers en UNE SEULE requête
    """
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")

    # Ajouter TOUS les tickers d'un coup
    for ticker in tickers:
        request.append("securities", ticker)

    # Ajouter TOUS les champs d'un coup
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


# Démarrage de la session Bloomberg
try:
    session = start_bloomberg_session()
    print("✓ Session Bloomberg démarrée avec succès\n")
except Exception as e:
    print(f"❌ Erreur de connexion Bloomberg: {str(e)}")
    print("⚠️  Le fichier sera créé sans les données Bloomberg\n")
    session = None

# Mapping des champs Bloomberg
bloomberg_fields_mapping = {
    'Bid': 'BID',
    'Ask': 'ASK',
    'Last Price': 'PX_LAST',
    'IV': 'IVOL_MID',
    'Delta': 'OPT_DELTA',
    'Gamma': 'OPT_GAMMA',
    'Vega': 'OPT_VEGA',
    'Theta': 'OPT_THETA',
    'Rho': 'OPT_RHO'
}

# Récupération des données Bloomberg en mode BATCH
if session:
    print(f"🚀 Récupération BATCH des données pour {len(result_df)} positions...")
    print(f"📊 Champs à récupérer: {list(bloomberg_fields_mapping.keys())}\n")

    start_time = datetime.now()

    all_tickers = result_df['Ticker'].tolist()
    bbg_fields = list(bloomberg_fields_mapping.values())

    try:
        bloomberg_data = get_bloomberg_data_batch(session, all_tickers, bbg_fields)

        # Remplir le DataFrame avec les données récupérées
        missing_fields = {field: 0 for field in bloomberg_fields_mapping.keys()}

        for idx, row in result_df.iterrows():
            ticker = row['Ticker']

            if ticker in bloomberg_data:
                for col_name, bbg_field in bloomberg_fields_mapping.items():
                    value = bloomberg_data[ticker].get(bbg_field)
                    result_df.at[idx, col_name] = value

                    # Compter les valeurs manquantes
                    if value is None or pd.isna(value):
                        missing_fields[col_name] += 1

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print(f"✅ Données récupérées en {elapsed:.2f} secondes!")
        if elapsed > 0:
            print(f"⚡ Vitesse: {len(result_df) / elapsed:.1f} positions/seconde\n")

        print("🔍 Diagnostic des champs Bloomberg:")
        for field, count in missing_fields.items():
            status = "✓" if count == 0 else f"⚠️  {count}/{len(result_df)} manquants"
            print(f"  {field:12} : {status}")
        print()

        print("📋 Échantillon des données récupérées:")
        sample = result_df[['Ticker', 'Position', 'Bid', 'Ask', 'Last Price', 'Delta']].head(5)
        print(sample.to_string(index=False))
        print()

    except Exception as e:
        print(f"❌ Erreur lors de la récupération Bloomberg: {str(e)}")

# ============================================================================
# ÉTAPE 2.5: CONSTRUCTION DE LA COURBE DE TAUX (US TREASURY)
# ============================================================================
print("=" * 70)
print("ÉTAPE 2.5: RÉCUPÉRATION DE LA COURBE DE TAUX US")
print("=" * 70)

# Définition des tenors standard pour la courbe US Treasury
TREASURY_TICKERS = {
    '1M': 'USGG1M Index',  # 1 Month
    '3M': 'USGG3M Index',  # 3 Month
    '6M': 'USGG6M Index',  # 6 Month
    '1Y': 'USGG1YR Index',  # 1 Year
    '2Y': 'USGG2YR Index',  # 2 Year
    '3Y': 'USGG3YR Index',  # 3 Year
    '5Y': 'USGG5YR Index',  # 5 Year
    '7Y': 'USGG7YR Index',  # 7 Year
    '10Y': 'USGG10YR Index',  # 10 Year
    '20Y': 'USGG20YR Index',  # 20 Year
    '30Y': 'USGG30YR Index'  # 30 Year
}

# Conversion des tenors en années (pour le plotting et l'interpolation)
TENOR_TO_YEARS = {
    '1M': 1 / 12, '3M': 3 / 12, '6M': 6 / 12,
    '1Y': 1, '2Y': 2, '3Y': 3, '5Y': 5,
    '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30
}

rates_curve_df = None

if session:
    try:
        print(f"🔄 Récupération de la courbe US Treasury ({len(TREASURY_TICKERS)} tenors)...\n")

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
                print(f"  ⚠️  {tenor:4} ({ticker:20}) : N/A")

        if curve_records:
            rates_curve_df = pd.DataFrame(curve_records)
            print(f"\n✅ Courbe de taux récupérée: {len(curve_records)}/{len(TREASURY_TICKERS)} points")

            if missing_count > 0:
                print(f"⚠️  {missing_count} point(s) manquant(s)")
        else:
            print("❌ Aucune donnée de taux récupérée\n")

    except Exception as e:
        print(f"❌ Erreur lors de la récupération de la courbe: {str(e)}\n")
else:
    print("⚠️  Pas de session Bloomberg - courbe de taux non disponible\n")

# Fermeture de la session Bloomberg
if session:
    session.stop()
    print("✓ Session Bloomberg fermée\n")


# ============================================================================
# BLACK-76 : FONCTIONS D'AIDE
# ============================================================================
def norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def norm_cdf(x: float) -> float:
    # CDF via erf -> pas de dépendance SciPy
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black76_greeks(F: float,
                   K: float,
                   T: float,
                   r: float,
                   sigma: float,
                   option_type: str):
    """
    Black-76 greeks pour une option sur future.

    F : forward/futures price
    K : strike
    T : time to maturity in years
    r : risk-free rate (continuous)
    sigma : volatility (0.25 pour 25%)
    option_type : 'C' ou 'P'

    Retourne un dict: delta, gamma, vega
    """
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "vega": None}

    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    df = math.exp(-r * T)

    Nd1 = norm_cdf(d1)
    nd1 = norm_pdf(d1)

    option_type = option_type.upper()[0]

    if option_type == "C":
        delta = df * Nd1
    else:  # Put
        delta = df * (Nd1 - 1.0)

    gamma = df * nd1 / (F * sigma * sqrtT)
    vega = df * F * nd1 * sqrtT

    return {"delta": delta, "gamma": gamma, "vega": vega}


# ============================================================================
# BACHELIER MODEL : FONCTIONS D'AIDE
# ============================================================================
def bachelier_greeks(F: float,
                     K: float,
                     T: float,
                     r: float,
                     sigma: float,
                     option_type: str):
    """
    Bachelier model greeks pour une option sur future.
    Le modèle de Bachelier suppose une distribution normale des prix (pas lognormale).

    F : forward/futures price
    K : strike
    T : time to maturity in years
    r : risk-free rate (continuous)
    sigma : volatility normale (en unités de prix, pas en %)
    option_type : 'C' ou 'P'

    Retourne un dict: delta, gamma, vega, price
    """
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "vega": None, "price": None}

    sqrtT = math.sqrt(T)
    df = math.exp(-r * T)

    # Dans Bachelier, sigma est exprimé en unités de prix
    # Si on a une vol en %, on doit la convertir: sigma_normal = sigma_pct * F
    sigma_normal = sigma * F  # Conversion de % à unités de prix

    # d dans le modèle de Bachelier
    d = (F - K) / (sigma_normal * sqrtT)

    Nd = norm_cdf(d)
    nd = norm_pdf(d)

    option_type = option_type.upper()[0]

    # Prix de l'option
    if option_type == "C":
        price = df * ((F - K) * Nd + sigma_normal * sqrtT * nd)
        delta = df * Nd
    else:  # Put
        price = df * ((K - F) * norm_cdf(-d) + sigma_normal * sqrtT * nd)
        delta = df * (Nd - 1.0)

    # Gamma (identique pour call et put)
    gamma = df * nd / (sigma_normal * sqrtT)

    # Vega (identique pour call et put)
    vega = df * sqrtT * nd

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "price": price
    }


def safe_time_to_maturity(maturity_value, today: date) -> float:
    """
    Convertit la maturité (string, Timestamp, etc.) en T (années).
    Gère les formats bizarres et évite OutOfBoundsDatetime.
    """
    if isinstance(maturity_value, (datetime, pd.Timestamp)):
        maturity_date = maturity_value.date()
    else:
        try:
            m = pd.to_datetime(maturity_value, errors="coerce", dayfirst=True)
        except Exception:
            m = pd.NaT

        if pd.isna(m):
            print(f"⚠️ Maturité invalide: {maturity_value} -> T fixé à 0.00001")
            return 0.00001
        maturity_date = m.date()

    days = (maturity_date - today).days
    T = days / 365.0
    if T <= 0:
        T = 0.00001
    return T


def compute_b76_row(row):
    """
    Calcule Delta/Gamma/Vega Black-76 pour une ligne de result_df.
    """
    # Temps jusqu'à maturité
    T = safe_time_to_maturity(row["Maturity"], valuation_date)

    # Forward / futures price :
    # - si tu as un 'Forward_Price', tu peux l'utiliser
    # - sinon on prend le Settlement Price comme approximation
    F = row.get("Forward_Price")
    if F is None or F == '' or pd.isna(F):
        F = row["Settlement Price"]

    K = row["Strike_Px"]

    # Vol (en % dans IV) -> en décimal
    iv = row.get("IV")
    if iv in (None, '', ' ') or pd.isna(iv):
        sigma = 0.30  # fallback 30%
    else:
        sigma = float(iv) / 100.0

    r = risk_free_rate
    option_type = row["Put_Call"]

    try:
        greeks = black76_greeks(F=float(F),
                                K=float(K),
                                T=float(T),
                                r=float(r),
                                sigma=float(sigma),
                                option_type=option_type)
        return pd.Series([greeks["delta"], greeks["gamma"], greeks["vega"]])
    except Exception as e:
        print(f"Error computing greeks for {row['Ticker']}: {e}")
        return pd.Series([None, None, None])


def compute_b76_row(row):
    """
    Calcule Delta/Gamma/Vega Black-76 pour une ligne de result_df.
    """
    # Temps jusqu'à maturité
    T = safe_time_to_maturity(row["Maturity"], valuation_date)

    # Forward / futures price :
    # - si tu as un 'Forward_Price', tu peux l'utiliser
    # - sinon on prend le Settlement Price comme approximation
    F = row.get("Forward_Price")
    if F is None or F == '' or pd.isna(F):
        F = row["Settlement Price"]

    K = row["Strike_Px"]

    # Vol (en % dans IV) -> en décimal
    iv = row.get("IV")
    if iv in (None, '', ' ') or pd.isna(iv):
        sigma = 0.30  # fallback 30%
    else:
        sigma = float(iv) / 100.0

    r = risk_free_rate
    option_type = row["Put_Call"]

    try:
        greeks = black76_greeks(F=float(F),
                                K=float(K),
                                T=float(T),
                                r=float(r),
                                sigma=float(sigma),
                                option_type=option_type)
        return pd.Series([greeks["delta"], greeks["gamma"], greeks["vega"]])
    except Exception as e:
        print(f"Error computing greeks for {row['Ticker']}: {e}")
        return pd.Series([None, None, None])


def compute_bachelier_row(row):
    """
    Calcule Delta/Gamma/Vega/Price Bachelier pour une ligne de result_df.
    """
    # Temps jusqu'à maturité
    T = safe_time_to_maturity(row["Maturity"], valuation_date)

    # Forward / futures price
    F = row.get("Forward_Price")
    if F is None or F == '' or pd.isna(F):
        F = row["Settlement Price"]

    K = row["Strike_Px"]

    # Vol (en % dans IV) -> en décimal
    iv = row.get("IV")
    if iv in (None, '', ' ') or pd.isna(iv):
        sigma = 0.30  # fallback 30%
    else:
        sigma = float(iv) / 100.0

    r = risk_free_rate
    option_type = row["Put_Call"]

    try:
        greeks = bachelier_greeks(F=float(F),
                                  K=float(K),
                                  T=float(T),
                                  r=float(r),
                                  sigma=float(sigma),
                                  option_type=option_type)
        return pd.Series([greeks["delta"], greeks["gamma"], greeks["vega"], greeks["price"]])
    except Exception as e:
        print(f"Error computing Bachelier greeks for {row['Ticker']}: {e}")
        return pd.Series([None, None, None, None])


# Calcul des greeks B76 APRÈS Bloomberg (pour utiliser IV)
print("=" * 70)
print("CALCUL DES GREEKS BLACK-76")
print("=" * 70)

result_df[['B76_Delta', 'B76_Gamma', 'B76_Vega']] = result_df.apply(
    compute_b76_row, axis=1
)
print("✓ Greeks Black-76 calculés\n")

# Calcul des greeks Bachelier
print("=" * 70)
print("CALCUL DES GREEKS BACHELIER")
print("=" * 70)

result_df[['Bach_Delta', 'Bach_Gamma', 'Bach_Vega', 'Bach_Price']] = result_df.apply(
    compute_bachelier_row, axis=1
)
print("✓ Greeks Bachelier calculés\n")

# ============================================================================
# ÉTAPE 3: CRÉATION DU FICHIER EXCEL AVEC FORMULES
# ============================================================================
print("=" * 70)
print("ÉTAPE 3: CRÉATION DU FICHIER EXCEL")
print("=" * 70)


def create_pnl_formula(row_num):
    """
    Crée la formule Excel pour le calcul P&L

    Structure des colonnes (Risk Monitor):
    A = Ticker
    B = Maturity
    C = Settlement Price (Premium)
    D = Position
    E = Bid
    F = Ask
    G = Last Price
    N = Strike_Px
    O = Contract_Size
    P = Premium
    Q = Put_Call

    P&L = Intrinsic Value - Premium Paid (Long) ou Premium Received - Intrinsic Value (Short)
    """
    return (
        f'=IF(D{row_num}>0,'
        # ========== LONG POSITION ==========
        # P&L = Intrinsic Value - Premium Paid
        f'IF(Q{row_num}="C",'
        # Long Call: max(S-K, 0) * Size * Contract_Size - Premium * Size * Contract_Size
        f'(MAX(G{row_num}-N{row_num},0)-C{row_num})*O{row_num}*D{row_num},'
        # Long Put: max(K-S, 0) * Size * Contract_Size - Premium * Size * Contract_Size
        f'(MAX(N{row_num}-G{row_num},0)-C{row_num})*O{row_num}*D{row_num}),'
        # ========== SHORT POSITION ==========
        # P&L = Premium Received - Intrinsic Value
        f'IF(Q{row_num}="C",'
        # Short Call: Premium * Size * Contract_Size - max(S-K, 0) * Size * Contract_Size
        f'(C{row_num}-MAX(G{row_num}-N{row_num},0))*O{row_num}*ABS(D{row_num}),'
        # Short Put: Premium * Size * Contract_Size - max(K-S, 0) * Size * Contract_Size
        f'(C{row_num}-MAX(N{row_num}-G{row_num},0))*O{row_num}*ABS(D{row_num})))'
    )


# Ajout de la colonne P&L (utilisée pour stocker les formules)
result_df['P&L'] = [create_pnl_formula(i + 2) for i in range(len(result_df))]

# Vérifier si le fichier existe et est ouvert
if output_file.exists():
    print(f"⚠️  Le fichier {output_file.name} existe déjà.")
    try:
        output_file.unlink()
        print("✓ Ancien fichier supprimé")
    except PermissionError:
        print("\n" + "=" * 70)
        print("❌ ERREUR: Le fichier est actuellement ouvert dans Excel!")
        print("=" * 70)
        print(f"Veuillez fermer: {output_file}")
        print("\nAppuyez sur Entrée une fois le fichier fermé...")
        input()

# Export vers Excel avec gestion d'erreur
try:
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 1) Vue complète Risk Monitor
        result_df.to_excel(writer, sheet_name='Risk Monitor', index=False)

        workbook = writer.book
        worksheet = writer.sheets['Risk Monitor']

        # Application des formules P&L en colonne R
        # (R = 18e colonne -> correspond à 'P&L' dans ce layout)
        for idx, formula in enumerate(result_df['P&L'], start=2):
            worksheet[f'R{idx}'] = formula

        # 2) Onglet dédié aux B76
        b76_cols = [
            'Ticker', 'Maturity', 'Put_Call', 'Strike_Px',
            'Settlement Price',  # proxy forward
            'IV', 'Position',
            'B76_Delta', 'B76_Gamma', 'B76_Vega'
        ]
        b76_cols = [c for c in b76_cols if c in result_df.columns]

        result_df[b76_cols].to_excel(writer, sheet_name='B76_Greeks', index=False)

        # 2b) Onglet dédié aux Bachelier
        bach_cols = [
            'Ticker', 'Maturity', 'Put_Call', 'Strike_Px',
            'Settlement Price',  # proxy forward
            'IV', 'Position',
            'Bach_Delta', 'Bach_Gamma', 'Bach_Vega', 'Bach_Price'
        ]
        bach_cols = [c for c in bach_cols if c in result_df.columns]

        result_df[bach_cols].to_excel(writer, sheet_name='Bachelier_Greeks', index=False)
        print("✓ Greeks Bachelier ajoutés à l'Excel")

        # 3) Onglet de la courbe de taux (si disponible)
        if rates_curve_df is not None and not rates_curve_df.empty:
            rates_curve_df.to_excel(writer, sheet_name='US_Rates_Curve', index=False)
            print("✓ Courbe de taux ajoutée à l'Excel")

            # 4) Créer le graphique de la courbe de taux (style Bloomberg)
            from openpyxl.chart import ScatterChart, Reference, Series
            from openpyxl.chart.marker import Marker
            from openpyxl.drawing.line import LineProperties

            ws_curve = writer.sheets['US_Rates_Curve']

            # Calculer le minimum des taux pour l'axe Y
            min_rate = rates_curve_df['Rate_%'].min()
            y_axis_min = min_rate - 0.20

            # Créer un scatter chart (XY)
            chart = ScatterChart()
            chart.title = "US Treasury Yield Curve - Last Mid YTM"
            chart.style = 2
            chart.height = 15
            chart.width = 28

            # Configurer l'axe X (Maturity)
            chart.x_axis.title = 'Maturity (Years)'
            chart.x_axis.scaling.min = 0
            chart.x_axis.scaling.max = 30
            chart.x_axis.majorUnit = 5  # Intervalles de 5 ans
            chart.x_axis.tickLblPos = "low"  # Position des labels

            # Configurer l'axe Y (Yield)
            chart.y_axis.title = 'Yield (%)'
            chart.y_axis.scaling.min = y_axis_min
            chart.y_axis.tickLblPos = "low"  # Position des labels

            # Pas de légende
            chart.legend = None

            # X-axis: Maturity en années (colonne B)
            xvalues = Reference(ws_curve, min_col=2, min_row=2, max_row=len(rates_curve_df) + 1)

            # Y-axis: Taux en % (colonne D)
            yvalues = Reference(ws_curve, min_col=4, min_row=2, max_row=len(rates_curve_df) + 1)

            # Créer la série
            series = Series(yvalues, xvalues)
            chart.series.append(series)

            # Style de la ligne - bleu foncé
            line_props = LineProperties(solidFill="00008B")  # Bleu foncé (DarkBlue)
            line_props.width = 30000  # Ligne épaisse
            series.graphicalProperties.line = line_props
            series.smooth = True  # Ligne lissée

            # Marqueurs aux points de données - ronds bleu foncé
            marker = Marker('circle')
            marker.size = 8
            marker.graphicalProperties.solidFill = "00008B"  # Bleu foncé
            marker.graphicalProperties.line.solidFill = "00008B"
            series.marker = marker

            # Positionner le graphique
            ws_curve.add_chart(chart, "G2")

            print(f"✓ Graphique de la courbe de taux créé (Y-axis min: {y_axis_min:.2f}%)")
        else:
            print("⚠️  Courbe de taux non ajoutée (données manquantes)")

except PermissionError:
    print("\n" + "=" * 70)
    print("❌ ERREUR: Impossible d'écrire le fichier!")
    print("=" * 70)
    print(f"Le fichier est toujours ouvert: {output_file}")
    print("Fermez Excel et réessayez.")
    exit(1)

print(f"✓ Fichier créé: {output_file}")
print(f"✓ Positions traitées: {len(result_df)}")
print(f"✓ Heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("=" * 70)
print("RÉSUMÉ")
print("=" * 70)
print(f"📊 Fichier d'origine: {input_file.name}")
print(f"📁 Fichier de sortie: {output_file.name}")
print(f"📈 Positions uniques: {len(result_df)}")
print(
    f"📊 Positions longues: {(result_df['Position'] > 0).sum()} "
    f"({result_df[result_df['Position'] > 0]['Position'].sum()} contrats)"
)
print(
    f"📉 Positions short: {(result_df['Position'] < 0).sum()} "
    f"({result_df[result_df['Position'] < 0]['Position'].sum()} contrats)"
)
if rates_curve_df is not None and not rates_curve_df.empty:
    print(f"📈 Courbe de taux: {len(rates_curve_df)} points (US Treasury)")
    try:
        rate_3m = rates_curve_df[rates_curve_df['Tenor'] == '3M']['Rate_%'].values[0]
        rate_10y = rates_curve_df[rates_curve_df['Tenor'] == '10Y']['Rate_%'].values[0]
        print(f"   Taux 3M: {rate_3m:.3f}% | Taux 10Y: {rate_10y:.3f}%")
    except:
        pass
print(f"✅ Statut: Terminé avec succès!\n")

print("Aperçu des données:")
print(result_df[['Ticker', 'Maturity', 'Position', 'Last Price', 'Delta', 'B76_Delta', 'Bach_Delta']].head(10))

# Ouverture automatique du fichier
try:
    if platform.system() == 'Windows':
        os.startfile(output_file)
    elif platform.system() == 'Darwin':
        subprocess.run(['open', output_file])
    else:
        subprocess.run(['xdg-open', output_file])
    print("\n📂 Fichier ouvert automatiquement!")
except Exception:
    print("\n💡 Ouvrez le fichier manuellement depuis votre Desktop.")