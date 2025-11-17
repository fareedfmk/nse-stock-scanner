#!/usr/bin/env python3
"""
NSE Double Bottom + MACD + QQE Web Scanner
Deploy to cloud and access from iPhone Safari
"""

from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import os
from apscheduler.schedulers.background import BackgroundScheduler

try:
    import yfinance as yf
    from scipy.signal import find_peaks
except ImportError:
    print("Installing required packages...")
    os.system("pip install yfinance pandas scipy openpyxl flask apscheduler")
    import yfinance as yf
    from scipy.signal import find_peaks

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
MACD_THRESHOLD = -0.80
QQE_RSI_PERIOD = 14
QQE_RSI_SMOOTHING = 5
QQE_FACTOR = 4.236
QQE_SIGNAL_PERIOD = 5
BOTTOM_TOLERANCE = 0.03
MIN_PEAK_HEIGHT = 0.10
MIN_BOTTOM_DISTANCE = 15
MAX_BOTTOM_DISTANCE = 60
VOLUME_CONFIRMATION = True

# Sample symbols (replace with your full list)
NSE_SYMBOLS = [
"GVPIL","AARVI","IPCALAB","KRBL","MHLXMIRU","CSLFINANCE",
"MANINDS","NITIRAJ","UNIVCABLES","DCM","BHAGYANGR","TARIL",
"HILTON","WEL","MUTHOOTFIN","EXPLEOSOL","SRM","SPANDANA",
"PRIMO","AJAXENGG","VIDHIING","CUPID","AETHER","ASHAPURMIN",
"MIRZAINT","HBLENGINE","GROWW","RELCHEMQ","TIL","ALLTIME",
"JUBLFOOD","GRMOVER","PARAS","YATHARTH","FABTECH","ARMANFIN",
"BDL","AKSHARCHEM","FEDFINA","DYNAMATECH","SAGILITY","SPAL",
"GRSE","DOLLAR","POWERMECH","ENGINERSIN","GPIL","SKIPPER",
"SOLEX","AXISCADES","PIRAMALFIN","SABTNL","IZMO","SOMATEX",
"INDOTHAI","RISHABH","TRIGYN","HGM","MANAKSTEEL","RVHL",
"VLEGOV","HYBRIDFIN","TECILCHEM","GUJAPOLLO","TAKE","MODISONLTD",
"BAJAJHLDNG","THYROCARE","GCSL","ZENTEC","ADOR","GRWRHITECH",
"TEMBO","GEOJITFSL","KAPSTON","NRAIL","NBCC","KITEX",
"DOMS","IMFA","TRANSRAILL","AUTOIND","JAYBARMARU","NCLIND",
"STARHEALTH","HMVL","SURAJEST","NIRAJISPAT","RPEL","ARIES",
"RHETAN","CUB","HINDWAREAP","INTENTECH","TVSHLTD","ICDSLTD",
"RELTD","MIDWESTLTD","WIPL","ABDL","LINCOLN","SCHAND",
"EMMBI","ZOTA","THOMASCOOK","JAMNAAUTO","GODAVARIB","RNBDENIMS",
"XELPMOC","SESHAPAPER","DCBBANK","EIMCOELECO","GKENERGY","AMBER",
"SGMART","STARPAPER","SANSTAR","CPCAP","FINEORG","SKMEGGPROD",
"MCLOUD","VINCOFE","LGBBROSLTD","FIEMIND","ACEINTEG","PGEL",
"TEAMGTY","OSWALPUMPS","FCL","EPACKPEB","NIRAJ","SAGARDEEP",
"DATAPATTNS","INDIANHUME","FORCEMOT","TVSSRICHAK","CREATIVE","UDS",
"GEEKAYWIRE","BHARATSE","RACLGEAR","SPMLINFRA","BBTCL","GICHSGFIN",
"CANTABIL","SIGNPOST","PFOCUS","SHREEPUSHK","ADL","SUPREMEINF",
"DELPHIFX","BANCOINDIA","INNOVACAP","GSLSU","ISGEC","IDBI",
"SUTLEJTEX","MUTHOOTMF","JSWHL","PIGL","MENONBE","PAISALO",
"CONCORDBIO","ELIN","HOMEFIRST","TIMETECHNO","HPIL","DPABHUSHAN",
"MANAPPURAM","MOLDTECH","SWIGGY","KICL","BALUFORGE","USHAMART",
"STEL","ZENITHEXPO","EQUITASBNK","KRN","SILVERTUC","EUROPRATIK",
"DTIL","BSE","SMSLIFE","SUPERHOUSE","JUNIPER","KAYA",
"KMSUGAR","MARICO","DMCC","RADICO","PRSMJOHNSN","DONEAR",
"CANFINHOME","SANGAMIND","INDIASHLTR","TERASOFT","SEAMECLTD","LENSKART",
"APARINDS","PUNJABCHEM","MANORAMA","SWARAJENG","MARKSANS","RVNL",
"YASHO","ERIS","POKARNA","RPTECH","VINYLINDIA","GARUDA",
"PKTEA","DIGITIDE","ADVANCE","STUDDS","RUBYMILLS","KCPSUGIND",
"AEROFLEX","GFLLIMITED","ZAGGLE","ETERNAL","WELINV","PVP",
"ARKADE","NORBTEAEXP","MAHSCOOTER","CENTUM","ASHIANA","NRBBEARING",
"SUVEN","MIRCELECTR","ROSSTECH","SPORTKING","BLS","PRAENG",
"MTARTECH","TRF","BORANA","DCXINDIA","BMWVENTLTD","GANESHBE",
"PREMIERPOL","BIRLACABLE","ASKAUTOLTD","BORORENEW","AVTNPL","OMFREIGHT",
"NORTHARC","VIJAYA","RBZJEWEL","VIKRAN","MAHABANK","CANBK",
"BLSE","VLSFINANCE","IIFL","HPL","UNIMECH","MGL",
"CARRARO","MAYURUNIQ","WCIL","STLTECH","ACUTAAS","KREBSBIO",
"ASAHISONG","JTEKTINDIA","SHIVAMILLS","MAZDOCK","TEAMLEASE","DREDGECORP",
"EMSLIMITED","BEL","PANSARI","RATEGAIN","TALBROAUTO","UNIVASTU",
"V2RETAIL","ATHERENERG","REDTAPE","SURYALAXMI","DCMNVL","ASIANTILES",
"CONTROLPR","MAWANASUG","BALRAMCHIN","SJVN","SPECTRUM","EUREKAFORB",
"FINOPB","VISAKAIND","SHRINGARMS","LOTUSDEV","KALYANIFRG","MARINE",
"SOMICONVEY","SBGLP","HGINFRA","MAHASTEEL","RAILTEL","IFBIND",
"ASIANHOTNR","JNKINDIA","AVANTIFEED","SIRCA","MAHLIFE","TRENT",
"SANGHVIMOV","MAHESHWARI","HATSUN","ARISINFRA","SENCO","NURECA",
"INGERRAND","GOKEX","VENUSREM","MANOMAY","EKC","SBIN",
"APOLLO","WHIRLPOOL","OSWALAGRO","IRB","DIAMINESQ","MALUPAPER",
"JSFB","VBL","SHREERAMA","TIRUMALCHM","LEXUS","NEULANDLAB",
"GRAVITA","SVLL","PNC","IFGLEXPOR","STEELCAS","PYRAMID",
"ADANIPOWER","INDOTECH","JIOFIN","MODEFENCE","AXISBANK","BALAJEE",
"ORIENTCER","LATENTVIEW","DPWIRES","NATHBIOGEN","IRCON","DCAL",
"GLOBECIVIL","SANDUMA","BANKINDIA","BAJFINANCE","SHREYANIND","PSPPROJECT",
"HUBTOWN","FUSION","SUPRAJIT","WELSPUNLIV","CESC","AERONEU",
"PHOENIXLTD","LLOYDSME","SIYSIL","JARO","APOLLOPIPE","GPTINFRA",
"APTECHT","MUKTAARTS","NOVAAGRI","DIGIDRIVE","LUXIND","BEML",
"POCL","BANKBARODA","NYKAA","SOLARWORLD","UNIVPHOTO","DBEIL",
"CENTRALBK","RAJESHEXPO","VARROC","PSUBNK", "NAVA","KAKATCEM",
"MSUMI","MAGADSUGAR","NDTV"]

# Global variables
scan_results = []
scan_status = {"running": False, "message": "Ready", "progress": 0, "total": 0}
last_scan_time = None
scheduler = None

# ============================================================
# INDICATOR FUNCTIONS (same as desktop version)
# ============================================================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(window=period).mean()
    down = (-delta.clip(upper=0)).rolling(window=period).mean()
    down = down.replace(0, 0.0001)
    rs = up / down
    return 100 - 100 / (1 + rs)

def normalize_macd(raw_macd: pd.Series, window: int = 10) -> pd.Series:
    norm = pd.Series(index=raw_macd.index, dtype=float)
    for i in range(len(raw_macd)):
        if i < window:
            norm.iloc[i] = 0.0
        else:
            win = raw_macd.iloc[i-window+1:i+1]
            mn, mx = win.min(), win.max()
            if mx == mn:
                norm.iloc[i] = 0.0
            else:
                norm.iloc[i] = 2 * (raw_macd.iloc[i] - mn) / (mx - mn) - 1
    return norm

def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    ema5 = ema(close, 5)
    ema13 = ema(close, 13)
    raw_macd = ema5 - ema13
    norm_macd = normalize_macd(raw_macd)
    macd_line = ema(norm_macd, 2)
    signal_line = ema(macd_line, 3)
    df["MACD"] = macd_line
    df["Signal"] = signal_line
    return df

def calculate_qqe(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    rsi_values = rsi(close, QQE_RSI_PERIOD)
    qqe_line = ema(rsi_values, QQE_RSI_SMOOTHING)
    rsi_ma = qqe_line.rolling(window=QQE_RSI_PERIOD * 2).mean()
    
    atr_rsi = pd.Series(index=df.index, dtype=float)
    dar = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(qqe_line)):
        if i < QQE_RSI_PERIOD:
            atr_rsi.iloc[i] = 0
            dar.iloc[i] = 0
        else:
            if not pd.isna(qqe_line.iloc[i]) and not pd.isna(rsi_ma.iloc[i]):
                dar.iloc[i] = abs(qqe_line.iloc[i] - rsi_ma.iloc[i])
            else:
                dar.iloc[i] = 0
    
    atr_rsi = ema(dar, QQE_RSI_PERIOD * 2) * QQE_FACTOR
    
    trailing_long = pd.Series(index=df.index, dtype=float)
    trailing_short = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(qqe_line)):
        if i == 0 or pd.isna(qqe_line.iloc[i]) or pd.isna(atr_rsi.iloc[i]):
            trailing_long.iloc[i] = 0
            trailing_short.iloc[i] = 0
        else:
            new_long = qqe_line.iloc[i] - atr_rsi.iloc[i]
            if pd.isna(trailing_long.iloc[i-1]):
                trailing_long.iloc[i] = new_long
            else:
                trailing_long.iloc[i] = max(new_long, trailing_long.iloc[i-1]) if qqe_line.iloc[i] > trailing_long.iloc[i-1] else new_long
            
            new_short = qqe_line.iloc[i] + atr_rsi.iloc[i]
            if pd.isna(trailing_short.iloc[i-1]):
                trailing_short.iloc[i] = new_short
            else:
                trailing_short.iloc[i] = min(new_short, trailing_short.iloc[i-1]) if qqe_line.iloc[i] < trailing_short.iloc[i-1] else new_short
    
    qqe_signal = ema(qqe_line, QQE_SIGNAL_PERIOD)
    
    df["QQE"] = qqe_line
    df["QQE_Signal"] = qqe_signal
    df["QQE_Long"] = trailing_long
    df["QQE_Short"] = trailing_short
    
    return df

def check_macd_crossover(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    
    macd_today = df["MACD"].iloc[-1]
    signal_today = df["Signal"].iloc[-1]
    macd_yesterday = df["MACD"].iloc[-2]
    signal_yesterday = df["Signal"].iloc[-2]
    
    if pd.isna(macd_today) or pd.isna(signal_today) or pd.isna(macd_yesterday) or pd.isna(signal_yesterday):
        return False
    
    today_cross = (macd_yesterday <= signal_yesterday and macd_today > signal_today)
    
    if len(df) >= 3:
        macd_day_before = df["MACD"].iloc[-3]
        signal_day_before = df["Signal"].iloc[-3]
        
        if not (pd.isna(macd_day_before) or pd.isna(signal_day_before)):
            yesterday_cross = (macd_day_before <= signal_day_before and macd_yesterday > signal_yesterday)
            
            if (today_cross or yesterday_cross) and macd_today < MACD_THRESHOLD:
                return True
    
    if today_cross and macd_today < MACD_THRESHOLD:
        return True
    
    return False

def check_qqe_buy_signal(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return {"signal": False, "qqe": 0, "qqe_signal": 0, "trailing_long": 0, "cross_type": "none"}
    
    qqe_now = df["QQE"].iloc[-1]
    qqe_prev = df["QQE"].iloc[-2]
    signal_now = df["QQE_Signal"].iloc[-1]
    signal_prev = df["QQE_Signal"].iloc[-2]
    trailing_long = df["QQE_Long"].iloc[-1]
    
    if pd.isna(qqe_now) or pd.isna(qqe_prev) or pd.isna(signal_now) or pd.isna(signal_prev):
        return {"signal": False, "qqe": 0, "qqe_signal": 0, "trailing_long": 0, "cross_type": "none"}
    
    if qqe_now <= signal_now:
        return {"signal": False, "qqe": round(qqe_now, 2), "qqe_signal": round(signal_now, 2), 
                "trailing_long": round(trailing_long, 2) if not pd.isna(trailing_long) else 0, "cross_type": "none"}
    
    bullish_cross = (qqe_prev <= signal_prev and qqe_now > signal_now)
    above_trailing = qqe_now > trailing_long if not pd.isna(trailing_long) else True
    
    qqe_prev2 = df["QQE"].iloc[-3] if len(df) >= 3 else qqe_prev
    signal_prev2 = df["QQE_Signal"].iloc[-3] if len(df) >= 3 else signal_prev
    recent_cross = (qqe_prev2 <= signal_prev2 and qqe_now > signal_now)
    
    has_crossover = bullish_cross or recent_cross
    buy_signal = has_crossover and above_trailing
    
    return {
        "signal": buy_signal,
        "qqe": round(qqe_now, 2),
        "qqe_signal": round(signal_now, 2),
        "trailing_long": round(trailing_long, 2) if not pd.isna(trailing_long) else 0,
        "cross_type": "bullish_cross" if bullish_cross else ("recent_cross" if recent_cross else "none"),
        "has_crossover": has_crossover
    }

def detect_double_bottom(df: pd.DataFrame) -> dict:
    if len(df) < 30:
        return None
    
    close_prices = df["Close"].values
    volumes = df["Volume"].values
    
    valleys_idx, _ = find_peaks(-close_prices, distance=MIN_BOTTOM_DISTANCE)
    
    if len(valleys_idx) < 2:
        return None
    
    recent_valleys = valleys_idx[-10:]
    
    for i in range(len(recent_valleys) - 1):
        for j in range(i + 1, len(recent_valleys)):
            idx1 = recent_valleys[i]
            idx2 = recent_valleys[j]
            
            days_apart = idx2 - idx1
            if days_apart < MIN_BOTTOM_DISTANCE or days_apart > MAX_BOTTOM_DISTANCE:
                continue
            
            bottom1_price = close_prices[idx1]
            bottom2_price = close_prices[idx2]
            
            price_diff_pct = abs(bottom2_price - bottom1_price) / bottom1_price
            if price_diff_pct > BOTTOM_TOLERANCE:
                continue
            
            segment = close_prices[idx1:idx2+1]
            peak_idx_relative = np.argmax(segment)
            peak_idx = idx1 + peak_idx_relative
            peak_price = close_prices[peak_idx]
            neckline = peak_price
            
            avg_bottom = (bottom1_price + bottom2_price) / 2
            peak_height_pct = (neckline - avg_bottom) / avg_bottom
            
            if peak_height_pct < MIN_PEAK_HEIGHT:
                continue
            
            volume_confirmed = True
            if VOLUME_CONFIRMATION and idx2 > 0:
                vol1 = volumes[idx1]
                vol2 = volumes[idx2]
                if vol2 >= vol1:
                    volume_confirmed = False
            
            current_idx = len(close_prices) - 1
            days_since_bottom2 = current_idx - idx2
            
            if days_since_bottom2 > 5:
                continue
            
            current_price = close_prices[current_idx]
            entry_threshold = bottom2_price * 1.02
            
            if current_price > entry_threshold:
                continue
            
            if current_price < bottom2_price * 0.98:
                continue
            
            target_price = neckline + (neckline - avg_bottom)
            potential_gain_pct = ((target_price - current_price) / current_price) * 100
            
            pattern_strength = 100
            if not volume_confirmed:
                pattern_strength -= 20
            pattern_strength -= int(price_diff_pct * 1000)
            pattern_strength -= int(days_since_bottom2 * 2)
            
            return {
                "found": True,
                "bottom1_price": bottom1_price,
                "bottom2_price": bottom2_price,
                "peak_price": neckline,
                "current_price": current_price,
                "target_price": target_price,
                "days_apart": days_apart,
                "days_since_bottom2": days_since_bottom2,
                "volume_confirmed": volume_confirmed,
                "potential_gain": potential_gain_pct,
                "pattern_strength": max(0, min(100, pattern_strength)),
                "entry_zone": f"₹{bottom2_price:.2f} - ₹{entry_threshold:.2f}",
                "stop_loss": bottom2_price * 0.97
            }
    
    return None

def try_download(symbol):
    formats = [f"{symbol}.NS", f"{symbol}.BO"]
    for fmt in formats:
        try:
            stock = yf.Ticker(fmt)
            data = stock.history(period="3mo", timeout=10)
            if data is not None and not data.empty and len(data) >= 30:
                return data, fmt
        except:
            continue
    return None, None

# ============================================================
# SCANNING FUNCTION
# ============================================================
def run_scan():
    global scan_results, scan_status, last_scan_time
    
    scan_status["running"] = True
    scan_status["message"] = "Scanning..."
    scan_status["progress"] = 0
    scan_status["total"] = len(NSE_SYMBOLS)
    
    results = []
    
    for idx, symbol in enumerate(NSE_SYMBOLS, 1):
        scan_status["progress"] = idx
        
        try:
            data, ticker_format = try_download(symbol)
            
            if data is None:
                continue
            
            data = calculate_macd(data)
            data = calculate_qqe(data)
            
            macd_cross = check_macd_crossover(data)
            qqe_result = check_qqe_buy_signal(data)
            pattern = detect_double_bottom(data)
            
            # Validation
            if macd_cross:
                latest = data.iloc[-1]
                if latest["MACD"] <= latest["Signal"]:
                    macd_cross = False
            
            if qqe_result["signal"]:
                latest = data.iloc[-1]
                if latest["QQE"] <= latest["QQE_Signal"]:
                    qqe_result["signal"] = False
                if not qqe_result.get("has_crossover", False):
                    qqe_result["signal"] = False
            
            if pattern:
                current_price = data["Close"].iloc[-1]
                entry_threshold = pattern["bottom2_price"] * 1.02
                if current_price > entry_threshold or current_price < pattern["bottom2_price"] * 0.98:
                    pattern = None
            
            if macd_cross and qqe_result["signal"] and pattern:
                latest = data.iloc[-1]
                
                if latest["MACD"] > latest["Signal"]:
                    results.append({
                        "Symbol": symbol,
                        "Exchange": ticker_format,
                        "Price": round(pattern["current_price"], 2),
                        "Target": round(pattern["target_price"], 2),
                        "Gain%": round(pattern["potential_gain"], 2),
                        "StopLoss": round(pattern["stop_loss"], 2),
                        "Strength": pattern["pattern_strength"],
                        "Volume": "✓" if pattern["volume_confirmed"] else "✗",
                        "MACD": round(latest["MACD"], 4),
                        "QQE": qqe_result["qqe"],
                        "EntryZone": pattern["entry_zone"],
                        "Date": latest.name.strftime("%Y-%m-%d")
                    })
        except Exception as e:
            continue
        
        time.sleep(0.1)
    
    scan_results = sorted(results, key=lambda x: x['Gain%'], reverse=True)
    last_scan_time = datetime.now()
    scan_status["running"] = False
    scan_status["message"] = f"Scan complete - {len(scan_results)} stocks found"
    
    # Save results
    if scan_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(scan_results)
        df.to_excel(f"results/scan_{timestamp}.xlsx", index=False)

# ============================================================
# FLASK ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scan', methods=['POST'])
def start_scan():
    if not scan_status["running"]:
        thread = threading.Thread(target=run_scan, daemon=True)
        thread.start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})

@app.route('/api/status')
def get_status():
    return jsonify({
        "running": scan_status["running"],
        "message": scan_status["message"],
        "progress": scan_status["progress"],
        "total": scan_status["total"],
        "results_count": len(scan_results),
        "last_scan": last_scan_time.strftime("%Y-%m-%d %H:%M:%S") if last_scan_time else "Never"
    })

@app.route('/api/results')
def get_results():
    return jsonify(scan_results)

@app.route('/api/download')
def download_results():
    if scan_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scan_{timestamp}.xlsx"
        df = pd.DataFrame(scan_results)
        df.to_excel(filename, index=False)
        return send_file(filename, as_attachment=True)
    return jsonify({"error": "No results available"})

# ============================================================
# SCHEDULER
# ============================================================
def setup_scheduler():
    global scheduler
    scheduler = BackgroundScheduler()
    
    # Schedule scans every hour from 9:15 AM to 3:30 PM on weekdays
    scheduler.add_job(run_scan, 'cron', day_of_week='mon-fri', hour='9-15', minute='15')
    
    scheduler.start()

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    setup_scheduler()
    app.run(host='0.0.0.0', port=5000, debug=False)
