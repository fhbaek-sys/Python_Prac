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
# [사용자 설정 변수]
# ==========================================
INITIAL_BALANCE = 1000000  # 초기 투자금 100만원
TARGET_PROFIT = 0.04      # 익절 4%
STOP_LOSS = -0.02         # 손절 -2%
TRADE_FEE = 0.0006        # 수수료 가정 (0.06%)
CONFIDENCE_THRESHOLD = 0.65 # 모델 확신도 (65% 이상 진입)

MAX_POSITIONS = 5          # 동시 진입 최대 종목 수
TRADE_INTERVAL = 1800      # 매매 의사결정 주기 (초단위, 30분)
MONITOR_INTERVAL = 600    # 모니터링/수익체크 주기 (초단위, 10분)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "AVAXUSDT",
    "LINKUSDT", "BCHUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT", "LTCUSDT", "ICPUSDT",
    "KASUSDT", "FETUSDT", "ETCUSDT", "XLMUSDT", "STXUSDT", "RENDERUSDT", "HBARUSDT", "ARBUSDT", "OPUSDT", "FILUSDT",
    "FLOWUSDT", "HYPEUSDT", "TIAUSDT", "SEIUSDT", "INJUSDT", "ORDIUSDT", "WLDUSDT", "RNDRUSDT", "TAOUSDT",
    "LDOUSDT", "MKRUSDT", "AAVEUSDT", "ALGOUSDT", "EGLDUSDT", "THETAUSDT", "VETUSDT", "EOSUSDT"]

MODEL_FILE = "xgb_trading_model.json"

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
trade_log_path = os.path.join(current_dir, "trade_history.csv")
monitor_log_path = os.path.join(current_dir, "monitor_log.txt")
portfolio_path = os.path.join(current_dir, "portfolio.json") # 상태 저장 파일

# ==========================================
# [상태 관리 함수: 저장 및 불러오기]
# ==========================================
def save_portfolio(portfolio):
    """현재 자산 및 포지션 상태를 JSON으로 저장"""
    with open(portfolio_path, 'w') as f:
        json.dump(portfolio, f, indent=4)

def load_portfolio():
    """저장된 포트폴리오 로드, 없으면 초기화"""
    if os.path.exists(portfolio_path):
        with open(portfolio_path, 'r') as f:
            return json.load(f)
    return {
        'cash': INITIAL_BALANCE,
        'holdings': {}, # { 'BTC/USDT': { ... } }
        'total_pnl': 0
    }

# 전역 포트폴리오 변수 로드
portfolio = load_portfolio()

# ==========================================
# [데이터 처리 및 피처 엔지니어링] - 기존 로직 유지
# ==========================================
def get_processed_data(symbol, model_features):
    exchange = ccxt.bybit()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # [피처 생성 로직]
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['f_rsi'] = 100 - (100 / (1 + (gain/loss)))
    df['f_rsi_slope'] = df['f_rsi'].diff(3)
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['f_ma_spread'] = (df['ma20'] - df['ma60']) / df['ma60']
    df['f_price_dist'] = (df['close'] - df['ma20']) / df['ma20']
    df['f_body_size'] = abs(df['open'] - df['close']) / (df['high'] - df['low'])
    df['f_upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'])
    df['f_vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['f_atr'] = (df['high'] - df['low']).rolling(14).mean() / df['close']
    df['f_hour'] = pd.to_datetime(df['timestamp'], unit='ms').dt.hour
    df['f_gap_7_25'] = (df['close'].rolling(7).mean() - df['close'].rolling(25).mean()) / df['close'].rolling(25).mean()
    
    # # [2] ★ 중요: 에러 방지 로직 ★
    # # 모델이 요구하는 이름(feat_...)과 내가 만든 이름(f_...)이 다를 경우를 대비해 이름을 강제로 매칭합니다.
    # # 에러 메시지에 뜬 이름들로 수동 변환해줍니다.
    # rename_map = {
    #     'f_rsi_slope': 'feat_rsi_slope',
    #     'f_ma_spread': 'feat_ma_spread',
    #     'f_price_dist': 'feat_price_dist',
    #     'f_vol_ratio': 'feat_vol_accel', # 이름이 다를 경우 매칭
    #     'f_atr': 'feat_range'            # 이름이 다를 경우 매칭
    # }
    # df = df.rename(columns=rename_map)
    
    return df.dropna().tail(1)

# ==========================================
# [매매 엔진: 진입 로직 (TRADE_INTERVAL 마다)]
# ==========================================
def run_trade_decision():
    print(f"\n[{datetime.now()}] ★ 매매 의사결정 사이클 시작...")
    
    # 모델 로드
    try:
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(current_dir, MODEL_FILE))
        # 모델에서 피처 이름 직접 추출 (안전장치)
        features = model.get_booster().feature_names
        print(f"-> 모델 로드 완료. (분석 피처: {len(features)}개)")
    except Exception as e:
        print(f"!! 모델 로드 실패: {e}")
        return

    for symbol in SYMBOLS:
        # 1. 포지션 제한 및 중복 체크 로그
        if symbol in portfolio['holdings']:
            print(f"   - {symbol}: 이미 보유 중 (패스)")
            continue
        
        if len(portfolio['holdings']) >= MAX_POSITIONS:
            print(f"!! 진입 제한: 최대 포지션({MAX_POSITIONS}개) 도달.")
            break
            
        print(f"   - {symbol} 분석 중...", end="\r") # 한 줄에서 표시
        
        try:
            # 데이터 수집 및 계산
            df_last = get_processed_data(symbol, features)
            
            # 모델 예측
            X = df_last[features]
            prob = model.predict_proba(X)[0][1]
            
            print(f"   > {symbol}: 모델 점수 {prob:.4f} " + (" [진입 시도!]" if prob >= CONFIDENCE_THRESHOLD else " [점수 미달]"))
            
            if prob >= CONFIDENCE_THRESHOLD:
                entry_price = float(df_last['close'].values[0])
                
                # 투자금 계산 (전체 자산의 1/5)
                total_asset = portfolio['cash'] + sum(h['entry_price']*h['amount'] for h in portfolio['holdings'].values())
                invest_amount = total_asset / MAX_POSITIONS
                
                if portfolio['cash'] < invest_amount:
                    invest_amount = portfolio['cash']
                
                if invest_amount > 1000: # 최소 1000원 이상일 때만 매수
                    buy_amount = invest_amount / entry_price
                    
                    portfolio['holdings'][symbol] = {
                        'entry_price': entry_price,
                        'amount': buy_amount,
                        'reason_score': float(prob),
                        'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    portfolio['cash'] -= invest_amount
                    save_portfolio(portfolio)
                    print(f"   ★★ [매수 완료] {symbol} | 가격: {entry_price} | 투자금: {round(invest_amount)}원")
                else:
                    print(f"   !! {symbol}: 잔액 부족으로 매수 실패 (가용현금: {round(portfolio['cash'])}원)")
                    
        except Exception as e:
            print(f"   ! {symbol} 분석 에러: {e}")

    print(f"[{datetime.now()}] 매매 사이클 종료.다음 사이클 대기중...")

# ==========================================
# [모니터링 엔진: 수익/손실 체크 (MONITOR_INTERVAL 마다)]
# ==========================================
def run_monitoring():
    print(f"\n[{datetime.now()}] 🔍 모니터링 및 자산 체크 중...")
    exchange = ccxt.bybit()
    
    # 딕셔너리 순회 중 삭제를 위해 리스트 복사본 사용
    symbols_to_check = list(portfolio['holdings'].keys())
    
    for symbol in symbols_to_check:
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            hold_info = portfolio['holdings'][symbol]
            
            profit_rate = (current_price - hold_info['entry_price']) / hold_info['entry_price']
            
            exit_reason = ""
            if profit_rate >= TARGET_PROFIT: exit_reason = "익절(Target)"
            elif profit_rate <= STOP_LOSS: exit_reason = "손절(Stop)"
            
            if exit_reason:
                final_value = (current_price * hold_info['amount']) * (1 - TRADE_FEE)
                pnl_amount = final_value - (hold_info['entry_price'] * hold_info['amount'])
                portfolio['cash'] += final_value
                
                log_data = {
                    '시간': datetime.now(), '종목': symbol, '진입가': hold_info['entry_price'],
                    '매도가': current_price, '수익률': f"{profit_rate*100:.2f}%",
                    '수익금': round(pnl_amount, 2), '이유': exit_reason, 'Score': round(hold_info['reason_score'], 2)
                }
                pd.DataFrame([log_data]).to_csv(trade_log_path, mode='a', header=not os.path.exists(trade_log_path), index=False,encoding='utf-8-sig')
                
                del portfolio['holdings'][symbol]
                save_portfolio(portfolio) # 상태 저장
                print(f"-> [매도] {symbol} | 사유: {exit_reason} | 수익: {log_data['수익금']}원")
        except Exception as e:
            print(f"{symbol} 모니터링 중 에러: {e}")

    # 생존 신고 로그 작성
    with open(monitor_log_path, "a",encoding='utf-8-sig') as f:
        status = f"[{datetime.now()}] 현금: {round(portfolio['cash'], 2)} | 보유: {list(portfolio['holdings'].keys())}\n"
        f.write(status)

# ==========================================
# [메인 루프]
# ==========================================
def start_bot():
    print(f"가상 투자 봇 가동... (보유 종목: {len(portfolio['holdings'])}개 로드됨)")
    last_trade_time = 0
    last_monitor_time = 0
    
    while True:
        now = time.time()
        
        # 1. 모니터링 사이클 (수익/손실 체크)
        if now - last_monitor_time >= MONITOR_INTERVAL:
            run_monitoring()
            last_monitor_time = now
            
        # 2. 매매 사이클 (신규 진입 체크)
        if now - last_trade_time >= TRADE_INTERVAL:
            run_trade_decision()
            last_trade_time = now
            
        time.sleep(1)

if __name__ == "__main__":
    start_bot()
