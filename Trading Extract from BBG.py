import pandas as pd
import blpapi
from pathlib import Path
from datetime import datetime

# Get desktop path
desktop = Path.home() / "Desktop"
input_file = desktop / "Cleaned_Commodities_Deals.xlsx"

# Read the cleaned Excel file
df = pd.read_excel(input_file, sheet_name='Risk Monitor')

print(f"📊 Loading {len(df)} positions from Cleaned_Commodities_Deals.xlsx")
print(f"Tickers to update: {df['Ticker'].tolist()}\n")


def start_bloomberg_session():
    """
    Ouvre une session Bloomberg API et la retourne.
    """
    options = blpapi.SessionOptions()
    options.serverHost = "localhost"
    options.serverPort = 8194

    session = blpapi.Session(options)
    if not session.start():
        raise Exception("Impossible de démarrer la session Bloomberg.")
    if not session.openService("//blp/refdata"):
        raise Exception("Impossible d'ouvrir le service refdata")

    return session


def get_bid(session, ticker):
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.append("securities", ticker)
    request.append("fields", "BID")

    session.sendRequest(request)
    bid_price = None

    while True:
        event = session.nextEvent()
        for msg in event:
            if msg.hasElement("securityData"):
                security_data = msg.getElement("securityData").getValueAsElement(0)
                if security_data.hasElement("fieldData"):
                    field_data = security_data.getElement("fieldData")
                    if field_data.hasElement("BID"):
                        bid_price = field_data.getElementAsFloat("BID")
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    return bid_price


def get_ask(session, ticker):
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.append("securities", ticker)
    request.append("fields", "ASK")

    session.sendRequest(request)
    ask_price = None

    while True:
        event = session.nextEvent()
        for msg in event:
            if msg.hasElement("securityData"):
                security_data = msg.getElement("securityData").getValueAsElement(0)
                if security_data.hasElement("fieldData"):
                    field_data = security_data.getElement("fieldData")
                    if field_data.hasElement("ASK"):
                        ask_price = field_data.getElementAsFloat("ASK")
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    return ask_price


def get_last_price(session, ticker):
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.append("securities", ticker)
    request.append("fields", "PX_LAST")

    session.sendRequest(request)
    last_price = None

    while True:
        event = session.nextEvent()
        for msg in event:
            if msg.hasElement("securityData"):
                security_data = msg.getElement("securityData").getValueAsElement(0)
                if security_data.hasElement("fieldData"):
                    field_data = security_data.getElement("fieldData")
                    if field_data.hasElement("PX_LAST"):
                        last_price = field_data.getElementAsFloat("PX_LAST")
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    return last_price


def get_iv(session, ticker):
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.append("securities", ticker)
    request.append("fields", "IVOL_MID")

    session.sendRequest(request)
    iv = None

    while True:
        event = session.nextEvent()
        for msg in event:
            if msg.hasElement("securityData"):
                security_data = msg.getElement("securityData").getValueAsElement(0)
                if security_data.hasElement("fieldData"):
                    field_data = security_data.getElement("fieldData")
                    if field_data.hasElement("IVOL_MID"):
                        iv = field_data.getElementAsFloat("IVOL_MID")
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    return iv


def get_delta(session, ticker):
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.append("securities", ticker)
    request.append("fields", "OPT_DELTA")

    session.sendRequest(request)
    delta = None

    while True:
        event = session.nextEvent()
        for msg in event:
            if msg.hasElement("securityData"):
                security_data = msg.getElement("securityData").getValueAsElement(0)
                if security_data.hasElement("fieldData"):
                    field_data = security_data.getElement("fieldData")
                    if field_data.hasElement("OPT_DELTA"):
                        delta = field_data.getElementAsFloat("OPT_DELTA")
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    return delta


def get_gamma(session, ticker):
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.append("securities", ticker)
    request.append("fields", "OPT_GAMMA")

    session.sendRequest(request)
    gamma = None

    while True:
        event = session.nextEvent()
        for msg in event:
            if msg.hasElement("securityData"):
                security_data = msg.getElement("securityData").getValueAsElement(0)
                if security_data.hasElement("fieldData"):
                    field_data = security_data.getElement("fieldData")
                    if field_data.hasElement("OPT_GAMMA"):
                        gamma = field_data.getElementAsFloat("OPT_GAMMA")
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    return gamma


def get_vega(session, ticker):
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.append("securities", ticker)
    request.append("fields", "OPT_VEGA")

    session.sendRequest(request)
    vega = None

    while True:
        event = session.nextEvent()
        for msg in event:
            if msg.hasElement("securityData"):
                security_data = msg.getElement("securityData").getValueAsElement(0)
                if security_data.hasElement("fieldData"):
                    field_data = security_data.getElement("fieldData")
                    if field_data.hasElement("OPT_VEGA"):
                        vega = field_data.getElementAsFloat("OPT_VEGA")
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    return vega


def get_theta(session, ticker):
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.append("securities", ticker)
    request.append("fields", "OPT_THETA")

    session.sendRequest(request)
    theta = None

    while True:
        event = session.nextEvent()
        for msg in event:
            if msg.hasElement("securityData"):
                security_data = msg.getElement("securityData").getValueAsElement(0)
                if security_data.hasElement("fieldData"):
                    field_data = security_data.getElement("fieldData")
                    if field_data.hasElement("OPT_THETA"):
                        theta = field_data.getElementAsFloat("OPT_THETA")
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    return theta


def get_rho(session, ticker):
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.append("securities", ticker)
    request.append("fields", "OPT_RHO")

    session.sendRequest(request)
    rho = None

    while True:
        event = session.nextEvent()
        for msg in event:
            if msg.hasElement("securityData"):
                security_data = msg.getElement("securityData").getValueAsElement(0)
                if security_data.hasElement("fieldData"):
                    field_data = security_data.getElement("fieldData")
                    if field_data.hasElement("OPT_RHO"):
                        rho = field_data.getElementAsFloat("OPT_RHO")
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    return rho


# Start Bloomberg session
try:
    session = start_bloomberg_session()
    print("✅ Bloomberg session started successfully\n")
except Exception as e:
    print(f"❌ Error starting Bloomberg session: {str(e)}")
    exit()

# Update each row with Bloomberg data
print("🔄 Fetching Bloomberg data for each position...\n")

for idx, row in df.iterrows():
    ticker = row['Ticker']
    print(f"Processing {idx + 1}/{len(df)}: {ticker}")

    try:
        # Get each field individually
        bid = get_bid(session, ticker)
        ask = get_ask(session, ticker)
        last_price = get_last_price(session, ticker)
        iv = get_iv(session, ticker)
        delta = get_delta(session, ticker)
        gamma = get_gamma(session, ticker)
        vega = get_vega(session, ticker)
        theta = get_theta(session, ticker)
        rho = get_rho(session, ticker)

        # Update dataframe
        df.at[idx, 'Bid'] = bid
        df.at[idx, 'Ask'] = ask
        df.at[idx, 'Last Price'] = last_price
        df.at[idx, 'IV'] = iv
        df.at[idx, 'Delta'] = delta
        df.at[idx, 'Gamma'] = gamma
        df.at[idx, 'Vega'] = vega
        df.at[idx, 'Theta'] = theta
        df.at[idx, 'Rho'] = rho

        print(f"  ✓ Bid: {bid}, Ask: {ask}, Last: {last_price}")
        print(f"    Delta: {delta}, Gamma: {gamma}, Vega: {vega}")

    except Exception as e:
        print(f"  ❌ Error fetching data for {ticker}: {str(e)}")
        continue

# Close Bloomberg session
session.stop()
print("\n✅ Bloomberg session closed")

# Save updated Excel file with formulas preserved
with pd.ExcelWriter(input_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    # Write data without P&L column first
    columns_to_write = [col for col in df.columns if col != 'P&L']
    df_to_write = df[columns_to_write].copy()
    df_to_write.to_excel(writer, sheet_name='Risk Monitor', index=False)

    # Re-apply P&L formulas
    workbook = writer.book
    worksheet = writer.sheets['Risk Monitor']

    # P&L formula (column R)
    for i in range(len(df)):
        row_num = i + 2
        formula = (
            f'=IF(D{row_num}>0,'
            f'IF(Q{row_num}="C",(G{row_num}-N{row_num})*O{row_num}*D{row_num},'
            f'(N{row_num}-G{row_num})*O{row_num}*D{row_num}),'
            f'IF(Q{row_num}="C",'
            f'IF(G{row_num}>N{row_num},-(G{row_num}-N{row_num})*O{row_num}*ABS(D{row_num}),P{row_num}*O{row_num}*ABS(D{row_num})),'
            f'IF(G{row_num}<N{row_num},-(N{row_num}-G{row_num})*O{row_num}*ABS(D{row_num}),P{row_num}*O{row_num}*ABS(D{row_num}))))'
        )
        worksheet[f'R{row_num}'] = formula

print(f"\n✅ Success! Updated file saved at: {input_file}")
print(f"📊 Total positions updated: {len(df)}")
print(f"⏰ Update completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Show sample of updated data
print("\n📋 Sample of updated data:")
print(df[['Ticker', 'Bid', 'Ask', 'Last Price', 'Delta', 'Gamma']].head(10))