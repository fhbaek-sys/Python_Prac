#가상 자동매매 실시
#학습 파일은 ver3 에서 구현하여 저장 함

import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import time
import json
from datetime import datetime

# ==========================================
# [1. 사용자 설정 변수]
# ==========================================
# 선물 거래소 설정을 위해 기본 객체 생성 시 옵션 추가
exchange = ccxt.bybit({
    'options': {
        'defaultType': 'future', # 선물(Linear Perpetual) 마켓 고정
    }
})

INITIAL_BALANCE = 1000000  # 초기 투자금 100만원 (1,000,000원)
LEVERAGE = 2               # 레버리지 2배 설정
TARGET_PROFIT = 0.04       # 익절 4%
STOP_LOSS = -0.02          # 손절 -2%
TRADE_FEE = 0.0006         # 수수료 가정 (0.06%)
CONFIDENCE_THRESHOLD = 0.65 # 모델 확신도 (65% 이상 진입)

MAX_POSITIONS = 5          # 동시 진입 최대 종목 수
TRADE_INTERVAL = 1800      # 매매 의사결정 주기 (30분)
MONITOR_INTERVAL = 600     # 모니터링/수익체크 주기 (10분)

WHITELIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "AVAXUSDT",
    "LINKUSDT", "BCHUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT", "LTCUSDT", "ICPUSDT",
    "KASUSDT", "FETUSDT", "ETCUSDT", "XLMUSDT", "STXUSDT", "RENDERUSDT", "HBARUSDT", "ARBUSDT", "OPUSDT", "FILUSDT",
    "FLOWUSDT", "HYPEUSDT", "TIAUSDT", "SEIUSDT", "INJUSDT", "ORDIUSDT", "WLDUSDT", "RNDRUSDT", "TAOUSDT",
    "LDOUSDT", "MKRUSDT", "AAVEUSDT", "ALGOUSDT", "EGLDUSDT", "THETAUSDT", "VETUSDT", "EOSUSDT"]

MODEL_FILE = "xgb_trading_model.json"

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
trade_log_path = os.path.join(current_dir, "trade_history.csv")
monitor_log_path = os.path.join(current_dir, "monitor_log.txt")
portfolio_path = os.path.join(current_dir, "portfolio.json")

# ==========================================
# [2. 포트폴리오 관리 함수]
# ==========================================
def save_portfolio(portfolio):
    with open(portfolio_path, 'w') as f:
        json.dump(portfolio, f, indent=4)

def load_portfolio():
    if os.path.exists(portfolio_path):
        with open(portfolio_path, 'r') as f:
            return json.load(f)
    return {'cash': INITIAL_BALANCE, 'holdings': {}, 'total_pnl': 0}

portfolio = load_portfolio()

# ==========================================
# [3. 유동성 및 데이터 처리 함수]
# ==========================================
def get_top_volume_symbols(whitelist, n=10):
    """서버 심볼명에 화이트리스트 글자가 포함되면 무조건 매칭 (최종 병기)"""
    try:
        # ccxt 바이비트 객체에서 모든 티커 로드
        tickers = exchange.fetch_tickers()
        volume_list = []
        
        # 비교를 위해 내 리스트를 대문자로 통일
        clean_whitelist = [s.upper().replace('/', '').replace(':', '') for s in whitelist]

        for ticker_symbol, data in tickers.items():
            # 서버 심볼에서 특수문자 제거 (예: BTC/USDT:USDT -> BTCUSDTUSDT)
            target_name = ticker_symbol.upper().replace('/', '').replace(':', '')
            
            # 내 화이트리스트의 '알맹이'가 서버 이름에 포함되어 있는지 확인
            # 예: "BTCUSDT"가 "BTCUSDTUSDT" 안에 있는가? -> YES
            for original_w, clean_w in zip(whitelist, clean_whitelist):
                if clean_w in target_name:
                    vol = data['quoteVolume'] if data['quoteVolume'] else 0
                    volume_list.append((ticker_symbol, vol))
                    break # 하나 찾았으면 다음 티커로
        
        if not volume_list:
            print("!! [경고] 여전히 매칭 실패. 바이비트 응답 구조 확인 필요.")
            return whitelist[:n]
            
        # 거래대금 순 정렬 후 중복 제거 및 상위 n개 반환
        volume_list.sort(key=lambda x: x[1], reverse=True)
        
        # 중복된 종목이 들어가지 않도록 처리
        seen = set()
        final_symbols = []
        for sym, vol in volume_list:
            if sym not in seen:
                final_symbols.append(sym)
                seen.add(sym)
            if len(final_symbols) >= n: break
            
        print(f"-> 유동성 필터 성공! 현재 거래량 1위: {final_symbols[0]}")
        return final_symbols

    except Exception as e:
        print(f"❌ 필터 에러: {e}")
        return whitelist[:n]

def get_processed_data(symbol):
    """지정한 종목의 1시간봉 200개를 가져와 피처(f_) 생성"""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['f_rsi'] = 100 - (100 / (1 + (gain/loss)))
    df['f_rsi_slope'] = df['f_rsi'].diff(3)
    
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma40'] = df['close'].rolling(40).mean()
    df['f_ma_spread'] = (df['ma20'] - df['ma40']) / df['ma40']
    df['f_price_dist'] = (df['close'] - df['ma20']) / df['ma20']
    
    df['f_body_size'] = abs(df['open'] - df['close']) / (df['high'] - df['low'].replace(0, np.nan))
    df['f_upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'])
    df['f_vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['f_atr'] = (df['high'] - df['low']).rolling(14).mean() / df['close']
    df['f_hour'] = pd.to_datetime(df['timestamp'], unit='ms').dt.hour
    df['f_gap_7_25'] = (df['close'].rolling(7).mean() - df['close'].rolling(25).mean()) / df['close'].rolling(25).mean()
    
    return df.dropna()

# ==========================================
# [4. 메인 매매 엔진]
# ==========================================
def run_trade_decision():
    print(f"\n[{datetime.now()}] ★ 선물 매매(Lev:{LEVERAGE}) 사이클 시작")
    target_symbols = get_top_volume_symbols(WHITELIST, n=10)
    print(f"-> 유동성 필터 결과: {target_symbols}")
    
    try:
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(current_dir, MODEL_FILE))
        features = model.get_booster().feature_names
        print(f"-> 모델 로드 성공. (피처 개수: {len(features)})")
    except Exception as e:
        print(f"-> 모델 로드 에러: {e}")
        return

    for symbol in target_symbols:
        if symbol in portfolio['holdings']:
            print(f"   - {symbol}: 이미 포지션 보유 중 (패스)")
            continue
        
        # 동시 진입 제한 체크
        if len(portfolio['holdings']) >= MAX_POSITIONS:
            print(f"!! 최대 진입 한도({MAX_POSITIONS}개) 초과로 분석 중단.")
            break
        
        try:
            df = get_processed_data(symbol)
            if df.empty: continue
            
        # [추가 필터] 직전 10개 봉 내 10% 이상 변동성 있을 시 패스 (지뢰 피하기)            
            last_10 = df.tail(10)
            volatility = (last_10['high'] - last_10['low']) / last_10['low']
            if (volatility >= 0.10).any():
                print(f"   > {symbol}: 최근 10% 급등락 이력 감지. 패스.")
                continue

            # 최신 캔들 하나에 대한 예측 점수 계산
            X = df[features].tail(1)
            prob = model.predict_proba(X)[0][1]
            status_msg = f"   > {symbol}: 모델 확신도 {prob:.4f}"
            
            if prob >= CONFIDENCE_THRESHOLD:
                # 총 자산의 10%를 한 종목의 '증거금'으로 사용
                total_assets = portfolio['cash'] + sum(h['margin'] for h in portfolio['holdings'].values())
                invest_margin = total_assets * 0.10 
                
                # 가용 현금이 부족하면 가진 현금만큼만 투자
                if portfolio['cash'] < invest_margin:
                    invest_margin = portfolio['cash']
                
                if invest_margin > 1000: # 최소 주문 금액 1,000원 기준
                    entry_price = float(df['close'].iloc[-1])
                    # 레버리지를 곱하여 실제 구매 수량 결정 (invest_margin 사용)
                    buy_amount = (invest_margin * LEVERAGE) / entry_price
                    
                    portfolio['holdings'][symbol] = {
                        'entry_price': entry_price,
                        'amount': buy_amount,
                        'margin': invest_margin, # 증거금 저장
                        'reason_score': float(prob),
                        'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    portfolio['cash'] -= invest_margin
                    save_portfolio(portfolio)
                    # 메시지 출력 시에도 invest_margin 사용
                    print(f"   ★ [매수 완료] {symbol} | 가격: {entry_price} | 증거금: {round(invest_margin)}원 (2배 레버리지)")
                else:
                    print(f"    !! 잔액 부족으로 매수 불가 (가용현금: {round(portfolio['cash'])}원)")
            else:
                print(f"{status_msg} [점수 미달 - 관망]")
                    
        except Exception as e:
            print(f"   ! {symbol} 에러: {e}")
            
    print(f"[{datetime.now()}] 매매 사이클 종료.다름 사이클 대기 중")

# ==========================================
# [5. 모니터링 및 수익률 체크]
# ==========================================
def run_monitoring():
    print(f"[{datetime.now()}] 🔍 모니터링 및 수익률 체크")
    symbols_to_check = list(portfolio['holdings'].keys())
    
    for symbol in symbols_to_check:
        try:
            ticker = exchange.fetch_ticker(symbol)
            curr_p = ticker['last']
            hold = portfolio['holdings'][symbol]
            
            # 가격 변동률 계산
            price_change = (curr_p - hold['entry_price']) / hold['entry_price']
            # 실제 수익률 = 가격 변동률 * 레버리지
            pnl_rate = price_change * LEVERAGE
            
            exit_reason = ""
            if pnl_rate >= TARGET_PROFIT: exit_reason = "익절(+4%)"
            elif pnl_rate <= STOP_LOSS: exit_reason = "손절(-2%)"
            
            if exit_reason:
                # 최종 정산금: 투입 증거금 + (수익금 - 수수료)
                # 수수료는 전체 거래 규모(증거금 * 레버리지)에 대해 발생
                fee = (hold['entry_price'] * hold['amount'] + curr_p * hold['amount']) * TRADE_FEE
                pnl_cash = (curr_p - hold['entry_price']) * hold['amount'] - fee
                portfolio['cash'] += (hold['margin'] + pnl_cash)
                
                log = {
                    '시간': datetime.now(), '종목': symbol, '진입가': hold['entry_price'], 
                    '매도가': curr_p, '수익률': f"{pnl_rate*100:.2f}%", 
                    '수익금': round(pnl_cash, 2), '사유': exit_reason, 'Score': round(hold['reason_score'], 2)
                }
                pd.DataFrame([log]).to_csv(trade_log_path, mode='a', header=not os.path.exists(trade_log_path), index=False, encoding='utf-8-sig')
                
                del portfolio['holdings'][symbol]
                save_portfolio(portfolio)
                print(f"   -> [매도] {symbol} | {exit_reason} | 수익금: {round(pnl_cash)}원")
        except Exception as e:
            print(f"   ! {symbol} 감시 에러: {e}")

    with open(monitor_log_path, "a", encoding='utf-8-sig') as f:
        status = f"[{datetime.now()}] 현금: {round(portfolio['cash'], 2)} | 보유: {list(portfolio['holdings'].keys())}\n"
        f.write(status)
        print(f"-> 생존신고: {status.strip()}")

def start_bot():
    print("==========================================")
    print(f" 가상 매매 봇 v4.2 가동 (보유 종목: {len(portfolio['holdings'])}개)")
    print("==========================================")
    
    last_trade_time = 0; 
    last_monitor_time = 0
    
    while True:
        now = time.time()
        if now - last_monitor_time >= MONITOR_INTERVAL:
            run_monitoring(); last_monitor_time = now
        if now - last_trade_time >= TRADE_INTERVAL:
            run_trade_decision(); last_trade_time = now
        time.sleep(1)

if __name__ == "__main__":
    start_bot()
