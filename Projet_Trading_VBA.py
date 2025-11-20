"""
Real-Time Options Risk Monitor with Bloomberg API
Monitors options positions with live market data and Greeks calculation
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import blpapi
from datetime import datetime, date
import time
from threading import Thread
import warnings
warnings.filterwarnings('ignore')


class BlackScholesCalculator:
    """Black-Scholes options pricing and Greeks calculation"""
    
    @staticmethod
    def calculate_d1(S, K, T, r, sigma):
        """Calculate d1 parameter"""
        if T <= 0 or sigma <= 0 or K <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def calculate_d2(d1, T, sigma):
        """Calculate d2 parameter"""
        if T <= 0:
            return 0
        return d1 - sigma * np.sqrt(T)
    
    @staticmethod
    def black_scholes_price(S, K, T, r, sigma, option_type):
        """Calculate option price using Black-Scholes"""
        if T <= 0:
            if option_type.upper() == 'CALL':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.calculate_d2(d1, T, sigma)
        
        if option_type.upper() == 'CALL':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.upper() == 'PUT':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError(f"Invalid option type: {option_type}")
        
        return price
    
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type):
        """Calculate all Greeks"""
        if T <= 0:
            return {
                'delta': 1.0 if (option_type.upper() == 'CALL' and S > K) else 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0
            }
        
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.calculate_d2(d1, T, sigma)
        
        # Delta
        if option_type.upper() == 'CALL':
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega (same for calls and puts, divided by 100 for 1% change)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Theta (per day)
        if option_type.upper() == 'CALL':
            theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Rho (per 1% change in rate)
        if option_type.upper() == 'CALL':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Vega': vega,
            'Theta': theta,
            'Rho': rho
        }


class BloombergDataFetcher:
    """Handles Bloomberg API connection and data retrieval"""
    
    def __init__(self):
        self.session = None
        self.refDataService = None
        
    def connect(self):
        """Establish Bloomberg API connection"""
        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost("localhost")
        sessionOptions.setServerPort(8194)
        
        self.session = blpapi.Session(sessionOptions)
        
        if not self.session.start():
            raise Exception("Failed to start Bloomberg session")
        
        if not self.session.openService("//blp/refdata"):
            raise Exception("Failed to open //blp/refdata")
        
        self.refDataService = self.session.getService("//blp/refdata")
        print("✓ Connected to Bloomberg API")
        
    def get_market_data(self, tickers, fields):
        """
        Fetch market data for given tickers and fields
        
        Args:
            tickers: list of Bloomberg tickers
            fields: list of Bloomberg fields (e.g., ['PX_LAST', 'PX_BID', 'PX_ASK'])
        
        Returns:
            dict: {ticker: {field: value}}
        """
        if not self.session:
            self.connect()
        
        request = self.refDataService.createRequest("ReferenceDataRequest")
        
        for ticker in tickers:
            request.append("securities", ticker)
        
        for field in fields:
            request.append("fields", field)
        
        self.session.sendRequest(request)
        
        data = {}
        
        while True:
            event = self.session.nextEvent(500)
            
            if event.eventType() == blpapi.Event.RESPONSE or \
               event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                
                for msg in event:
                    securityDataArray = msg.getElement("securityData")
                    
                    for i in range(securityDataArray.numValues()):
                        securityData = securityDataArray.getValueAsElement(i)
                        ticker = securityData.getElementAsString("security")
                        fieldData = securityData.getElement("fieldData")
                        
                        data[ticker] = {}
                        
                        for field in fields:
                            try:
                                data[ticker][field] = fieldData.getElementAsFloat(field)
                            except:
                                try:
                                    data[ticker][field] = fieldData.getElementAsString(field)
                                except:
                                    data[ticker][field] = None
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        return data
    
    def get_implied_volatility(self, ticker):
        """Fetch implied volatility for an option"""
        data = self.get_market_data([ticker], ['IVOL_MID'])
        return data.get(ticker, {}).get('IVOL_MID', None)
    
    def disconnect(self):
        """Close Bloomberg connection"""
        if self.session:
            self.session.stop()
            print("✓ Disconnected from Bloomberg API")


class RiskMonitor:
    """Main risk monitoring system"""
    
    def __init__(self, excel_path, risk_free_rate=0.05):
        self.excel_path = excel_path
        self.risk_free_rate = risk_free_rate
        self.bbg = BloombergDataFetcher()
        self.bs_calc = BlackScholesCalculator()
        self.positions_df = None
        self.running = False
        
    def load_positions(self):
        """Load positions from Excel file"""
        df = pd.read_excel(self.excel_path)
        
        required_cols = ['underlying', 'product', 'lot_size', 'maturity', 
                        'settlement_price', 'strike', 'type', 'bloomberg_ticker']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert maturity to datetime
        df['maturity'] = pd.to_datetime(df['maturity'])
        
        # Calculate time to maturity in years
        df['T'] = df['maturity'].apply(
            lambda x: max((x - pd.Timestamp.now()).days / 365.0, 0)
        )
        
        # Determine position direction: positive = long, negative = short
        df['position_sign'] = np.sign(df['lot_size'])
        df['abs_lot_size'] = np.abs(df['lot_size'])
        
        self.positions_df = df
        print(f"Loaded {len(df)} positions from {self.excel_path}")
        
    def calculate_position_risk(self, row, market_data):
        """Calculate risk metrics for a single position"""
        ticker = row['bloomberg_ticker']
        position_type = row['type'].upper()
        position_sign = row['position_sign']  # +1 for long, -1 for short
        lot_size = row['lot_size']  # Can be negative for short
        abs_lot_size = row['abs_lot_size']
        
        # Get market data
        if ticker not in market_data:
            return None
        
        md = market_data[ticker]
        
        result = {
            'Product': row['product'],
            'Underlying': row['underlying'],
            'Position': lot_size,  # Shows sign (negative = short)
            'Direction': 'LONG' if position_sign > 0 else 'SHORT',
            'Strike': row['strike'],
            'Maturity': row['maturity'].strftime('%Y-%m-%d'),
            'Type': position_type
        }
        
        if position_type in ['CALL', 'PUT']:
            # Options
            S = md.get('PX_LAST')
            bid = md.get('PX_BID')
            ask = md.get('PX_ASK')
            iv = md.get('IVOL_MID')  # Default to 30% if not available
            
            if iv:
                iv = iv / 100  # Convert from percentage
            
            result.update({
                'bid': bid,
                'ask': ask,
                'last': S,
                'IV': iv * 100 if iv else None
            })
            
            # Calculate Greeks
            if S > 0 and row['T'] > 0 and iv:
                greeks = self.bs_calc.calculate_greeks(
                    S, row['strike'], row['T'], self.risk_free_rate, iv, position_type
                )
                
                # IMPORTANT: Scale Greeks by position size (including sign)
                # Negative lot_size (short) will flip the Greek signs
                result.update({
                    'Delta': greeks['delta'] * lot_size,
                    'Gamma': greeks['gamma'] * lot_size,
                    'Vega': greeks['vega'] * lot_size,
                    'Theta': greeks['theta'] * lot_size,
                    'Rho': greeks['rho'] * lot_size
                })
                
                # Choose a market price (mid in this example)
                if bid is not None and ask is not None:
                    market_price = (bid + ask) / 2
                else:
                    market_price = md.get('PX_LAST')
                
                if position_sign > 0:  # long
                    result['PnL'] = (market_price - entry_price) * abs_lot_size
                else:  # short
                    result['PnL'] = (entry_price - market_price) * abs_lot_size

                # # Calculate theoretical price
                # theo_price = self.bs_calc.black_scholes_price(
                #     S, row['strike'], row['T'], self.risk_free_rate, iv, position_type
                # )
                
                # # P&L calculation taking into account position direction
                # entry_price = row['settlement_price']
                
                # if position_sign > 0:  # LONG position
                #     # Long: profit when price goes up
                #     result['PnL'] = (theo_price - entry_price) * abs_lot_size
                # else:  # SHORT position
                #     # Short: profit when price goes down
                #     result['PnL'] = (entry_price - theo_price) * abs_lot_size
            
        elif position_type == 'FUTURE':
            # Futures
            S = md.get('PX_LAST')
            bid = md.get('PX_BID')
            ask = md.get('PX_ASK')
            
            result.update({
                'bid': bid,
                'ask': ask,
                'last': S,
                'IV': None,
                'delta': lot_size,  # Futures delta = +1 per long contract, -1 per short
                'gamma': 0,
                'vega': 0,
                'theta': 0,
                'rho': 0
            })
            
            # P&L for futures taking into account position direction
            entry_price = row['settlement_price']
            
            if position_sign > 0:  # LONG future
                result['PnL'] = (S - entry_price) * abs_lot_size
            else:  # SHORT future
                result['PnL'] = (entry_price - S) * abs_lot_size
        
        return result
    
    def update_risk_table(self):
        """Update risk metrics for all positions"""
        if self.positions_df is None:
            self.load_positions()
        
        # Get unique tickers
        tickers = self.positions_df['bloomberg_ticker'].unique().tolist()
        
        # Fetch market data
        fields = ['PX_LAST', 'PX_BID', 'PX_ASK', 'IVOL_MID']
        
        try:
            market_data = self.bbg.get_market_data(tickers, fields)
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None
        
        # Calculate risk for each position
        risk_data = []
        for idx, row in self.positions_df.iterrows():
            risk = self.calculate_position_risk(row, market_data)
            if risk:
                risk_data.append(risk)
        
        risk_df = pd.DataFrame(risk_data)
        
        # Add portfolio totals
        if len(risk_df) > 0:
            totals = {
                'product': 'TOTAL',
                'underlying': '',
                'position': risk_df['position'].sum(),
                'direction': '',
                'strike': '',
                'maturity': '',
                'type': '',
                'bid': '',
                'ask': '',
                'last': '',
                'IV': '',
                'delta': risk_df['delta'].sum(),
                'gamma': risk_df['gamma'].sum(),
                'vega': risk_df['vega'].sum(),
                'theta': risk_df['theta'].sum(),
                'rho': risk_df['rho'].sum(),
                'PnL': risk_df['PnL'].sum()
            }
            risk_df = pd.concat([risk_df, pd.DataFrame([totals])], ignore_index=True)
        
        return risk_df
    
    def run_continuous_monitor(self, refresh_interval=5):
        """Run continuous monitoring with specified refresh interval (seconds)"""
        self.running = True
        self.bbg.connect()
        
        print(f"\n{'='*80}")
        print("REAL-TIME RISK MONITOR - Press Ctrl+C to stop")
        print(f"{'='*80}\n")
        
        try:
            while self.running:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Updating risk metrics...")
                
                risk_df = self.update_risk_table()
                
                if risk_df is not None:
                    # Format display
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', None)
                    pd.set_option('display.float_format', '{:.4f}'.format)
                    
                    print("\n" + "="*150)
                    print(risk_df.to_string(index=False))
                    print("="*150)
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping monitor...")
            self.running = False
        finally:
            self.bbg.disconnect()
    
    def export_snapshot(self, output_path='risk_snapshot.xlsx'):
        """Export current risk snapshot to Excel"""
        risk_df = self.update_risk_table()
        if risk_df is not None:
            risk_df.to_excel(output_path, index=False)
            print(f"✓ Risk snapshot exported to {output_path}")


# Example usage
if __name__ == "__main__":
    """
    Excel file format expected:
    
    | underlying | product | lot_size | maturity   | settlement_price | strike | type | bloomberg_ticker |
    |------------|---------|----------|------------|------------------|--------|------|------------------|
    | CLZ1       | CLZ1 C85| 10       | 2025-12-31 | 85.50           | 85     | Call | CLZ1C85 Comdty  |
    | CLZ1       | CLZ1 P80| -5       | 2025-12-31 | 80.20           | 80     | Put  | CLZ1P80 Comdty  |
    | CLZ1       | CLZ1    | 20       | 2025-12-31 | 82.30           | 0      | Future| CLZ1 Comdty     |
    """
    
    # Initialize monitor
    monitor = RiskMonitor(
        excel_path='positions.xlsx',
        risk_free_rate=0.05  # 5% risk-free rate
    )
    
    # Load positions
    monitor.load_positions()
    
    # Option 1: Run continuous monitoring (updates every 5 seconds)
    monitor.run_continuous_monitor(refresh_interval=5)
    
    # Option 2: Get single snapshot
    # risk_df = monitor.update_risk_table()
    # print(risk_df)
    
    # Option 3: Export to Excel
    # monitor.export_snapshot('risk_snapshot.xlsx')