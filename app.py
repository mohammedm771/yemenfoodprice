# -*- coding: utf-8 -*-
# ================================================================
#  🌾 منصة عدن الذكية للتنبؤ بأسعار السلع الأساسية
#  🤖 تطوير: E.M-ALKAMEL | واتساب: 771257332 | بريد: mm11kk22mo@gmail.com
#  LSTM-Ensemble × 3 | البيانات: WFP + FRED | التحقق: Walk-Forward × 24
# ================================================================
import os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title='منصة عدن الذكية — التنبؤ بأسعار السلع',
                   page_icon='🌾', layout='wide')

# ==================== 🎨 الهوية البصرية — النسخة الذهبية ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800;900&display=swap');
html, body, #root, .stApp { font-family: 'Cairo', sans-serif !important; }
.stApp { background: radial-gradient(1200px 600px at 85% -10%, #12325e 0%, #0a1428 55%, #060c18 100%) !important; }

/* ===== ✨ خلفية الشفق القطبي ===== */
.stApp::before { content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(600px 320px at 15% 25%, rgba(63,127,224,.14), transparent 70%),
    radial-gradient(520px 300px at 85% 60%, rgba(127,224,167,.09), transparent 70%),
    radial-gradient(480px 280px at 45% 95%, rgba(147,112,219,.11), transparent 70%);
  animation: aurora 14s ease-in-out infinite alternate; }
@keyframes aurora {
  0%   { opacity:.55; transform: scale(1)   translateX(0); }
  50%  { opacity:.9;  transform: scale(1.06) translateX(2%); }
  100% { opacity:.7;  transform: scale(1.02) translateX(-2%); } }
.main .block-container { direction: rtl; text-align: right; padding-top: 1rem; max-width: 1300px; position: relative; z-index: 1; }
h1,h2,h3,h4, span, p, div, label, td, th { color: #e8eefc !important; }

/* ===== 🧑‍💻 شارة المطور (الزاوية المقابلة) ===== */
.devbadge { display: flex; align-items: center; gap: 12px; background: linear-gradient(135deg, rgba(127,224,167,.1), rgba(63,127,224,.14)); border: 1px solid rgba(127,224,167,.45); border-radius: 18px; padding: 12px 20px; animation: botGlow 3s ease-in-out infinite; }
@keyframes botGlow { 0%,100% { box-shadow: 0 0 16px rgba(127,224,167,.25); } 50% { box-shadow: 0 0 42px rgba(127,224,167,.55); } }
.devbadge .bot { font-size: 2.3rem; display: inline-block; animation: botFloat 2.6s ease-in-out infinite; filter: drop-shadow(0 0 8px rgba(127,224,167,.6)); }
@keyframes botFloat { 0%,100% { transform: translateY(0) rotate(0deg); } 50% { transform: translateY(-5px) rotate(4deg); } }
.devbadge .n { font-weight: 900; color: #7fe0a7 !important; font-size: 1.12rem; letter-spacing: 1px; text-shadow: 0 0 12px rgba(127,224,167,.5); }
.devbadge .r { font-size: .74rem; color: #9db8e8 !important; margin-top: -2px; }

/* ===== الترويسة: العنوان يمين — الشارة يسار ===== */
.hdr { position: relative; overflow: hidden; background: linear-gradient(135deg, rgba(20,50,100,.8), rgba(8,18,38,.92)); border: 1px solid rgba(100,160,255,.35); border-radius: 22px; padding: 22px 28px; display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 8px; box-shadow: 0 12px 45px rgba(0,0,0,.5), inset 0 0 40px rgba(63,127,224,.08); }
.hdr::before { content: ''; position: absolute; inset: 0; background: linear-gradient(100deg, transparent 30%, rgba(120,180,255,.14) 50%, transparent 70%); animation: sweep 5s ease-in-out infinite; pointer-events: none; }
@keyframes sweep { 0% { transform: translateX(-120%);} 60%,100% { transform: translateX(120%);} }
.hdr .titlewrap { display: flex; align-items: center; gap: 18px; }
.hdr .logo { font-size: 2.6rem; filter: drop-shadow(0 0 12px rgba(120,200,120,.5)); }
.hdr h1 { font-size: 1.6rem; font-weight: 900; margin: 0; background: linear-gradient(90deg, #ffffff 30%, #8fc2ff 70%, #7fe0a7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; filter: drop-shadow(0 2px 10px rgba(63,127,224,.45)); }
.hdr small { color: #9db8e8; font-size: .88rem; }

/* ===== عناوين الأقسام ===== */
h2 { position: relative; padding-inline-start: 16px; }
h2::before { content: ''; position: absolute; inset-inline-start: 0; top: 12%; height: 76%; width: 5px; border-radius: 5px; background: linear-gradient(180deg, #4d9dff, #7fe0a7); box-shadow: 0 0 12px rgba(77,157,255,.6); }

/* ===== ⭐ قوائم الاختيار والبحث: مضيئة وواضحة أخيراً ===== */
div[data-baseweb="popover"] { background: #10203f !important; border: 1px solid rgba(120,170,255,.5) !important; border-radius: 14px !important; box-shadow: 0 14px 45px rgba(0,0,0,.6) !important; }
div[data-baseweb="popover"] div, div[data-baseweb="popover"] span, div[data-baseweb="popover"] p,
div[data-baseweb="popover"] li, div[data-baseweb="popover"] input { color: #e8eefc !important; font-family: 'Cairo' !important; }
div[data-baseweb="popover"] input { background: rgba(63,127,224,.15) !important; font-weight: 700; }
li[role="option"] { background: transparent !important; color: #e8eefc !important; font-weight: 700; border-radius: 10px; margin: 2px 6px; transition: all .15s; }
li[role="option"]:hover { background: rgba(63,127,224,.35) !important; transform: translateX(-3px); }
button[data-baseweb="tag"] { background: linear-gradient(135deg, #1a5fd0, #0f3d8a) !important; color: #fff !important; border: 1px solid #3f7fe0 !important; border-radius: 12px !important; font-family: 'Cairo' !important; font-weight: 900 !important; padding: 6px 12px !important; box-shadow: 0 4px 14px rgba(63,127,224,.35) !important; }
button[data-baseweb="tag"] span { color: #fff !important; font-weight: 900 !important; }
button[data-baseweb="tag"] svg { fill: #ffd166 !important; stroke: #ffd166 !important; }
[data-testid="stMultiSelect"] label { color: #bcd2f5 !important; font-weight: 800; }
[data-testid="stMultiSelect"] > div { background: rgba(15,30,60,.7) !important; border: 1px solid rgba(120,170,255,.35) !important; border-radius: 12px !important; }

/* ===== بطاقات KPI ===== */
.kpi { position: relative; overflow: hidden; background: linear-gradient(145deg, rgba(30,60,110,.5), rgba(10,20,40,.7)); border: 1px solid rgba(120,170,255,.22); border-radius: 18px; padding: 16px 18px; box-shadow: 0 8px 30px rgba(0,0,0,.35); height: 100%; transition: all .3s cubic-bezier(.2,.8,.3,1); }
.kpi::before { content: ''; position: absolute; top: 0; right: 0; left: 0; height: 3px; background: linear-gradient(90deg, #4d9dff, #7fe0a7, #ffd166); opacity: .85; }
.kpi:hover { transform: translateY(-6px); border-color: rgba(127,184,255,.7) !important; box-shadow: 0 18px 50px rgba(63,127,224,.5), inset 0 0 25px rgba(63,127,224,.1) !important; }
.kpi h4 { margin: 0 0 6px; font-size: .9rem; color: #9db8e8 !important; }
.kpi .v { font-size: 1.8rem; font-weight: 900; color: #fff !important; line-height: 1.1; }
.kpi .s { font-size: .8rem; color: #8fd3a7 !important; margin-top: 4px; } .kpi .s.red { color: #ff9b9b !important; }

/* ===== الإنذار النابض (رادار الدولار) ===== */
.alertbox { border-radius: 18px; padding: 18px 22px; font-weight: 800; margin: 12px 0; border: 1px solid; animation: siren 1.6s ease-in-out infinite; }
@keyframes siren { 0%,100% { box-shadow: 0 0 12px rgba(255,80,80,.25); } 50% { box-shadow: 0 0 45px rgba(255,80,80,.6); } }
.alertbox.green { animation: greenPulse 2.5s ease-in-out infinite; }
@keyframes greenPulse { 0%,100% { box-shadow: 0 0 12px rgba(127,224,167,.2); } 50% { box-shadow: 0 0 32px rgba(127,224,167,.4); } }
.alertbox .big { font-size: 1.3rem; }
.alertbox .num { font-size: 1.6rem; font-weight: 900; color: #fff !important; }

/* ===== دليل الاستخدام ===== */
.guide { background: linear-gradient(145deg, rgba(30,60,110,.35), rgba(10,20,40,.55)); border: 1px dashed rgba(120,170,255,.4); border-radius: 18px; padding: 18px 22px; margin: 10px 0; }
.guide .step { display: flex; align-items: flex-start; gap: 12px; margin: 10px 0; }
.guide .num { min-width: 34px; height: 34px; border-radius: 50%; background: linear-gradient(135deg, #1a5fd0, #0f3d8a); display: flex; align-items: center; justify-content: center; font-weight: 900; color: #fff !important; box-shadow: 0 0 14px rgba(63,127,224,.5); }
.guide .t { color: #dce8ff !important; font-weight: 600; line-height: 1.7; padding-top: 4px; }

/* ===== بطاقة التوقع ===== */
.bigfc { background: linear-gradient(135deg, #0f2a55, #14418a); border: 1px solid #3f7fe0; border-radius: 24px; padding: 28px 22px; text-align: center; animation: pulse 3.5s ease-in-out infinite; }
@keyframes pulse { 0%,100% { box-shadow: 0 0 35px rgba(63,127,224,.3);} 50% { box-shadow: 0 0 75px rgba(63,127,224,.6);} }
.bigfc .price { font-size: 3rem; font-weight: 900; color: #fff !important; } .bigfc .cap { color: #bcd2f5 !important; font-size: 1rem; }

/* ===== التنبيهات ===== */
.alert { border-radius: 16px; padding: 14px 20px; font-weight: 800; font-size: 1.05rem; margin: 10px 0; border: 1px solid; background: rgba(20,40,80,.5); }

/* ===== الأزرار ===== */
div[data-testid="stButton"] > button { position: relative; overflow: hidden; background: linear-gradient(135deg, #1a5fd0 0%, #0f3d8a 55%, #0b2c66 100%) !important; color: #fff !important; border: 1px solid #3f7fe0 !important; border-radius: 16px !important; padding: 14px 30px !important; font-weight: 900 !important; font-family: 'Cairo' !important; font-size: 1.08rem !important; box-shadow: 0 6px 25px rgba(63,127,224,.4), inset 0 1px 0 rgba(255,255,255,.15) !important; transition: all .3s cubic-bezier(.2,.8,.3,1); }
div[data-testid="stButton"] > button::after { content: ''; position: absolute; top: 0; left: -80%; width: 50%; height: 100%; background: linear-gradient(100deg, transparent, rgba(255,255,255,.35), transparent); transform: skewX(-20deg); transition: left .6s ease; }
div[data-testid="stButton"] > button:hover { transform: translateY(-4px) scale(1.02); box-shadow: 0 14px 40px rgba(63,127,224,.65) !important; border-color: #7fb8ff !important; }
div[data-testid="stButton"] > button:hover::after { left: 130%; }
div[data-testid="stButton"] > button p { color: #fff !important; font-size: 1.08rem !important; font-weight: 900 !important; }
div[data-testid="stDownloadButton"] > button { background: linear-gradient(135deg, #1a7a42 0%, #0c4a27 100%) !important; color: #c9ffd9 !important; border: 1px solid #2e9b5f !important; border-radius: 16px !important; padding: 12px 28px !important; font-weight: 800 !important; font-family: 'Cairo' !important; box-shadow: 0 6px 22px rgba(46,155,95,.35) !important; transition: all .3s; }
div[data-testid="stDownloadButton"] > button:hover { transform: translateY(-3px); box-shadow: 0 12px 32px rgba(46,155,95,.55) !important; border-color: #7fe0a7 !important; }
div[data-testid="stDownloadButton"] > button p { color: #c9ffd9 !important; font-weight: 800 !important; }

/* ===== التبويبات ===== */
button[data-baseweb="tab"] { font-family: 'Cairo' !important; font-size: .98rem; font-weight: 800; color: #8fa8d8 !important; padding: 10px 18px !important; transition: all .25s; }
button[data-baseweb="tab"]:hover { color: #fff !important; background: rgba(63,127,224,.14); }
button[data-baseweb="tab"][aria-selected="true"] { color: #7fb8ff !important; text-shadow: 0 0 14px rgba(127,184,255,.7); }

/* ===== الحقول والمقاييس ===== */
div[data-baseweb="input"] input { background: rgba(15,30,60,.7) !important; border: 1px solid rgba(120,170,255,.3) !important; border-radius: 10px !important; color: #fff !important; font-weight: 700; font-family: 'Cairo' !important; }
div[data-baseweb="input"] label { color: #bcd2f5 !important; font-weight: 700; }
div[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(30,60,110,.4), rgba(10,20,40,.6)); border: 1px solid rgba(120,170,255,.22); border-radius: 16px; padding: 14px 18px; transition: all .3s; }
div[data-testid="stMetric"]:hover { transform: translateY(-4px); border-color: rgba(127,184,255,.6) !important; box-shadow: 0 12px 35px rgba(63,127,224,.4) !important; }

[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; border: 1px solid rgba(120,170,255,.25); box-shadow: 0 8px 28px rgba(0,0,0,.35); }

.expert { background: rgba(18,34,66,.7); border: 1px solid rgba(120,170,255,.2); border-radius: 16px; padding: 14px; text-align: center; transition: all .3s; }
.expert:hover { border-color: rgba(127,224,167,.5) !important; transform: translateY(-3px); box-shadow: 0 10px 30px rgba(127,224,167,.25) !important; }
.expert .n { font-weight: 800; color: #bcd2f5 !important; } .expert .p { font-size: 1.5rem; font-weight: 900; color: #7fe0a7 !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #2a5aa8, #14418a); border-radius: 10px; }
::-webkit-scrollbar-track { background: rgba(10,20,40,.5); }

.footer { color: #6c86b8 !important; font-size: .84rem; text-align: center; padding: 20px; border-top: 1px solid rgba(120,170,255,.15); margin-top: 30px; line-height: 2; }
.footer b { color: #7fb8ff !important; font-size: .95rem; }
.footer a { color: #7fe0a7 !important; text-decoration: none; font-weight: 800; }
.footer a:hover { text-shadow: 0 0 10px rgba(127,224,167,.8); }
</style>
""", unsafe_allow_html=True)

# ==================== 📂 البيانات والنماذج ====================
ROOT = next((r for r in [os.getcwd(), '/content/drive/MyDrive/YemenFoodPrice']
             if os.path.exists(r + '/data/processed/aden_features_ext.csv')), None)
if ROOT is None:
    st.error('لم أجد ملفات البيانات — شغّل التطبيق من مجلد المشروع'); st.stop()
PROC, MODELS = f'{ROOT}/data/processed', f'{ROOT}/models'

@st.cache_data
def load_data():
    t  = pd.read_csv(f'{PROC}/aden_features_ext.csv', index_col=0, parse_dates=True).sort_index()
    fc = pd.read_csv(f'{PROC}/next_month_forecast.csv')
    return t, fc

@st.cache_resource
def load_product():
    with open(f'{MODELS}/product.pkl','rb') as f: return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_tf_models():
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from tensorflow import keras
        return {s: keras.models.load_model(f'{MODELS}/product_seed{s}.keras') for s in [42, 7, 2026]}
    except Exception:
        return None

table, fcsv = load_data()
prod = load_product()
FEATS, LOOKBACK = prod['features'], prod['lookback']
TF_MODELS = load_tf_models()

table['flour_ret'] = table['flour'].pct_change() * 100
for c in ['exchange','brent','wheat','rice','sugar','oil','diesel']:
    table[c + '_pct'] = table[c].pct_change() * 100

LAST = table.iloc[-1]
last_flour, last_date = float(LAST['flour']), table.index.max()
pred_ret = float(fcsv['pred_ret_pct'].iloc[0])
forecast_flour = float(fcsv['forecast_flour'].iloc[0])
next_d = str(fcsv['forecast_date'].iloc[0])
BETA_D = 0.59

def signal(r):
    if r > 1.0:  return ('صاعد بقوة', 'شراء مبكر مفيد قبل موجة الغلاء', '#ff9b9b')
    if r > 0.5:  return ('صاعد ببطء', 'مراقبة، الشراء المبكر خيار معقول', '#ffc46b')
    if r < -1.0: return ('نازل بقوة', 'انتظار قليل قبل الشراء قد يوفر', '#7fe0a7')
    if r < -0.5: return ('نازل ببطء', 'استقرار قريب، لا استعجال', '#b7e0a7')
    return ('مستقر', 'الشراء على مهل — لا ضغط سعري', '#7fb8ff')
sig_name, sig_advice, sig_color = signal(pred_ret)

AR_M = {1:'يناير',2:'فبراير',3:'مارس',4:'أبريل',5:'مايو',6:'يونيو',
        7:'يوليو',8:'أغسطس',9:'سبتمبر',10:'أكتوبر',11:'نوفمبر',12:'ديسمبر'}
CNAMES = {'flour':'الدقيق','rice':'الأرز','sugar':'السكر','oil':'الزيت',
          'diesel':'الديزل','petrol':'البنزين','exchange':'الدولار',
          'brent':'البرنت','wheat':'القمح العالمي'}

# سلة المستخدم — محفوظة عبر كل التبويبات
if 'basket' not in st.session_state:
    st.session_state['basket'] = {'flour': 50, 'rice': 20, 'sugar': 10, 'oil': 8}

# ==================== 🏠 الترويسة: العنوان يمين — شارتك يسار ====================
st.markdown("""<div class="hdr">
<div class="titlewrap"><span class="logo">🌾</span>
<div><h1>منصة عدن الذكية للتنبؤ بأسعار السلع</h1>
<small>مدينة عدن • نموذج LSTM-Ensemble • مدرَّب على 13 سنة • مُثبَّت بمنهجية Walk-Forward</small></div></div>
<div class="devbadge"><span class="bot">🤖</span>
<div><div class="n">E.M-ALKAMEL</div><div class="r">مطور النظام الذكي</div></div></div>
</div>""", unsafe_allow_html=True)

tab1, tabR, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ['📊 لوحة القيادة','🚨 رادار الدولار الذكي','🔮 محرك التوقع','🧪 مختبر السيناريوهات',
     '📈 الاستكشاف التاريخي','🗓️ الأنماط الموسمية','🔗 العلاقات','⚖️ عن النموذج'])

# ============ 1: لوحة القيادة ============
with tab1:
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.markdown(f'<div class="kpi"><h4>💵 آخر سعر معروف</h4><div class="v">{last_flour:,.1f}</div><div class="s">ريال/كيلو • {last_date.date()}</div></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="kpi"><h4>🔮 توقع {AR_M[pd.Timestamp(next_d).month]}</h4><div class="v">{forecast_flour:,.1f}</div><div class="s">ريال/كيلو — LSTM-Ensemble</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi"><h4>📈 التغير المتوقع</h4><div class="v">{pred_ret:+.2f}%</div><div class="s {"red" if pred_ret>1 else ""}">{(forecast_flour-last_flour):+.1f} ريال على الكيلو</div></div>', unsafe_allow_html=True)
    with k4: st.markdown(f'<div class="kpi"><h4>🧭 إشارة السوق</h4><div class="v" style="font-size:1.3rem;color:{sig_color} !important">{sig_name}</div><div class="s">{sig_advice}</div></div>', unsafe_allow_html=True)
    with k5: st.markdown(f'<div class="kpi"><h4>✅ جودة النظام</h4><div class="v">91%</div><div class="s">مبنية على MAPE التقييمي (8.69%)</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="alert" style="border-color:{sig_color};color:{sig_color} !important">⚠️ توصية المنصة: <span style="color:#fff">{sig_name} — {sig_advice}</span></div>', unsafe_allow_html=True)

    st.subheader('حركة سعر الدقيق + نقطة التوقع القادمة')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table.index, y=table['flour'], name='🔵 سعر الدقيق (مسجل فعلياً)',
        line=dict(color='#4d9dff', width=2.4), fill='tozeroy', fillcolor='rgba(77,157,255,.08)'))
    fig.add_trace(go.Scatter(x=[last_date, pd.Timestamp(next_d)], y=[last_flour, forecast_flour],
        name='🔴 مسار التوقع (تنبؤ)', mode='lines', line=dict(color='#ff6b6b', width=2.5, dash='dot')))
    fig.add_trace(go.Scatter(x=[pd.Timestamp(next_d)], y=[forecast_flour], name='💎 نقطة التوقع',
        marker=dict(color='#ff6b6b', size=16, symbol='diamond', line=dict(color='#fff', width=1)),
        text=[f'{forecast_flour:.1f} ريال'], textposition='top center'))
    fig.update_layout(template='plotly_dark', height=430,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Cairo'), hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)),
        yaxis=dict(title='ريال/كيلو'),
        margin=dict(t=20, l=10, r=10, b=10),
        legend=dict(orientation='h', y=1.12,
                    font=dict(size=17, color='#ffffff', family='Cairo'),
                    bgcolor='rgba(10,20,40,.85)',
                    bordercolor='rgba(120,170,255,.45)', borderwidth=1.5))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.subheader('🛒 محاكي سلة المشتريات الشهرية')
    st.caption('أدخل كميات بيتك/محلك — تُحفظ سلتك تلقائياً وتُستخدم في رادار الدولار وجميع الحسابات')
    cA, cB, cC, cD = st.columns(4)
    q = {}
    for col, key, default in [(cA,'flour',50),(cB,'rice',20),(cC,'sugar',10),(cD,'oil',8)]:
        with col:
            q[key] = st.number_input(f'{CNAMES[key]} (كيلو/لتر)', 0, 500, default, 5,
                                     key=f'basket_{key}')
            st.session_state['basket'][key] = q[key]
    rows = []
    for k, qty in q.items():
        cur = float(LAST[k])
        if k == 'flour': nxt = forecast_flour
        else:
            m3 = table[k + '_pct'].iloc[-3:].mean()
            nxt = cur * (1 + (0 if pd.isna(m3) else m3)/100)
        rows.append([CNAMES[k], qty, cur, qty*cur, qty*nxt, qty*nxt - qty*cur])
    bt = pd.DataFrame(rows, columns=['السلعة','الكمية','السعر الآن','التكلفة الآن','التكلفة المتوقعة','الفرق المتوقع'])
    tot_now, tot_next = bt['التكلفة الآن'].sum(), bt['التكلفة المتوقعة'].sum()
    st.dataframe(bt, use_container_width=True, hide_index=True,
        column_config={
            'السلعة': st.column_config.TextColumn('📦 السلعة', width='medium'),
            'الكمية': st.column_config.NumberColumn('⚖️ الكمية', format='%d'),
            'السعر الآن': st.column_config.NumberColumn('💵 سعر الكيلو الآن (ريال)', format='%.0f'),
            'التكلفة الآن': st.column_config.NumberColumn('🧾 تكلفة السلة اليوم (ريال)', format='%.0f'),
            'التكلفة المتوقعة': st.column_config.NumberColumn('🔮 التكلفة المتوقعة للشهر القادم', format='%.0f'),
            'الفرق المتوقع': st.column_config.NumberColumn('📊 الفرق المتوقع (ريال)', format='%+.0f'),
        })
    delta_txt = (f'💰 متوقع توفير {abs(tot_next-tot_now):,.0f} ريال بالانتظار' if tot_next < tot_now
                 else f'⚠️ متوقع زيادة {tot_next-tot_now:,.0f} ريال — الشراء المبكر قد يوفر')
    st.markdown(f'<div class="alert" style="border-color:{sig_color}">🧾 إجمالي السلة الآن: <b>{tot_now:,.0f}</b> ريال ← متوقع: <b>{tot_next:,.0f}</b> ريال &nbsp;|&nbsp; {delta_txt if abs(tot_next-tot_now)>5 else "💵 التكلفة شبه ثابتة — اشترِ على مهل"}</div>', unsafe_allow_html=True)

# ============ ⭐ 1.5: رادار الدولار الذكي ============
with tabR:
    st.subheader('🚨 رادار الدولار الذكي — أخبار السوق تصير قرارات')
    st.caption('لا تحتاج أي حسابات — اكتب سعرين فقط والذكاء الاصطناعي يفهم الخبر ويعطيك القرار')

    st.markdown("""<div class="guide">
<div style="font-weight:900; font-size:1.05rem; margin-bottom:4px;">📖 كيف تستخدم هذا الرادار؟ (30 ثانية فقط)</div>
<div class="step"><div class="num">1</div><div class="t">افتح أي مصدر موثوق لسعر الدولار في عدن (موقع صرافة، قناة أخبار يومية، أو اسأل أقرب صرّاف)</div></div>
<div class="step"><div class="num">2</div><div class="t">سجّل <b>سعر اليوم</b> و <b>سعر أمس</b> فقط — مثل ما تسجل درجة الحرارة صباحاً</div></div>
<div class="step"><div class="num">3</div><div class="t">اكتبهما في الحقلين تحت — والمنصة تحسب كل شيء: النسبة، التصنيف، الأثر على الدقيق، والقرار</div></div>
<div class="step"><div class="num">4</div><div class="t">اتبع توصية القرار التي تظهر لك بالألوان — خضراء اطمئن، حمراء تحرك بسرعة!</div></div>
</div>""", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        usd_today = st.number_input('💵 سعر الدولار اليوم (ريال)', min_value=1.0, value=940.0, step=5.0,
                                    help='اكتب سعر الدولار الذي سمعته اليوم في عدن')
    with g2:
        usd_yest = st.number_input('💵 سعر الدولار أمس (ريال)', min_value=1.0, value=930.0, step=5.0,
                                   help='سعر الدولار بالأمس — أو آخر سعر تعرفه قبل اليوم')

    if usd_today > 0 and usd_yest > 0:
        chg_pct = (usd_today - usd_yest) / usd_yest * 100
        flour_impact = chg_pct * BETA_D
        impact_price   = last_flour * (1 + flour_impact/100)
        delta_vs_base  = impact_price - forecast_flour

        if abs(chg_pct) < 0.5:
            lvl, lvl_c, lvl_icon = ('هادئ 🟢', '#7fe0a7',
                'حركة طبيعية ضمن الضجيج اليومي — لا داعي لأي قلق أو تصرف متسرع')
        elif abs(chg_pct) < 2:
            lvl, lvl_c, lvl_icon = ('نشاط ملحوظ 🟡', '#ffc46b',
                'حركة واضحة تستحق المتابعة — راقب السوق يومين قبل قرارات كبيرة')
        elif abs(chg_pct) < 5:
            lvl, lvl_c, lvl_icon = ('تحذير 🟠', '#ff9b5b',
                'تقلب مؤثر — إذا كنت تخطط شراء كميات كبيرة فالتحرك الآن أذكى من الانتظار')
        else:
            lvl, lvl_c, lvl_icon = ('خطر — صدمة! 🔴', '#ff6b6b',
                'صدمة نادرة بمستوى صدمة 2025! إذا كانت الزيادة صاعدة: اشترِ احتياجك فوراً. إذا هابطة: انتظر يومين')

        sgn = '+' if chg_pct >= 0 else ''
        is_danger = abs(chg_pct) >= 2 and chg_pct != 0
        box_cls = 'alertbox' if is_danger else 'alertbox green'
        st.markdown(f"""
<div class="{box_cls}" style="border-color:{lvl_c}; background:rgba(20,30,60,.55)">
<div class="big" style="color:{lvl_c}">{lvl_icon} تصنيف الخبر: <span style="color:#fff">{lvl}</span></div>
<div style="margin-top:6px">📈 تغير الدولار: <span class="num">{sgn}{chg_pct:.2f}%</span>
&nbsp;→&nbsp; الأثر المتوقع على الدقيق: <span class="num" style="color:{lvl_c}">{flour_impact:+.2f}%</span>
&nbsp;(≈ {impact_price - last_flour:+.0f} ريال على الكيلو — يصير {impact_price:,.0f})</div>
<div style="margin-top:8px; color:#dce8ff">{lvl_icon}</div>
</div>""", unsafe_allow_html=True)

        st.subheader('🧭 جدول القرار الذكي — ماذا يحدث لسلتك؟')
        q0 = dict(st.session_state['basket'])
        st.caption('🛒 يستخدم سلتك الحالية: ' +
                   ' • '.join(f"{CNAMES[k]} {q0[k]}" for k in q0) +
                   ' — عدّلها من محاكي السلة في تبويب لوحة القيادة')
        tot_now_b = sum(q0[k] * float(LAST[k]) for k in q0)
        dec_rows = []
        for label, eff_ret in [('✅ إذا بقي الوضع هادئاً (المتوقع الرسمي)', pred_ret),
                               ('⚠️ إذا اكتمل أثر خبرك اليوم', flour_impact),
                               ('🚨 إذا اجتمع خبرك مع توقع الشهر', flour_impact + pred_ret)]:
            total = 0
            for k, qty in q0.items():
                cur = float(LAST[k])
                if k == 'flour': nxt = cur * (1 + eff_ret/100)
                else:
                    m3 = table[k + '_pct'].iloc[-3:].mean()
                    nxt = cur * (1 + (0 if pd.isna(m3) else m3)/100)
                total += qty * nxt
            dec_rows.append([label, total, total - tot_now_b])
        dt = pd.DataFrame(dec_rows, columns=['السيناريو','تكلفة سلتك المتوقعة (ريال)','الفرق عن شراء اليوم'])
        st.dataframe(dt, use_container_width=True, hide_index=True,
            column_config={
                'السيناريو': st.column_config.TextColumn('🎭 السيناريو', width='large'),
                'تكلفة سلتك المتوقعة (ريال)': st.column_config.NumberColumn('💰 تكلفة سلتك', format='%.0f'),
                'الفرق عن شراء اليوم': st.column_config.NumberColumn('📊 الفرق عن الشراء الآن', format='%+.0f'),
            })
        st.markdown(f'<div class="alert" style="border-color:{sig_color}">🧠 القرار المُوصى به: <b>اشترِ سلتك اليوم بـ {tot_now_b:,.0f} ريال</b> — وكلما ساء السيناريو زادت التكلفة تدريجياً كما يوضح الجدول</div>', unsafe_allow_html=True)

        hist_ex = table['exchange_pct'].dropna().abs()
        worse = (hist_ex > abs(chg_pct)).mean() * 100
        st.info(f'📏 على مقياس التاريخ: خبرك اليوم **أكبر من {worse:.0f}%** من تحركات الدولار المسجلة في عدن منذ 2013 '
                f'(أكبر حركة شهرية سجلتها: {hist_ex.max():.0f}%)')

# ============ 2: محرك التوقع ============
with tab2:
    l, r = st.columns([1.1, 1])
    with l:
        st.markdown(f'<div class="bigfc"><div class="cap">🔮 توقع سعر كيلو الدقيق — شهر {AR_M[pd.Timestamp(next_d).month]} {pd.Timestamp(next_d).year}</div><div class="price">{forecast_flour:,.1f} <span style="font-size:1.2rem">ريال</span></div><div class="cap">{pred_ret:+.2f}% مقارنة بآخر سعر معروف ({last_flour:,.1f})</div></div>', unsafe_allow_html=True)
        st.markdown('')
        if st.button('⚡ شغّل Ensemble الحي الآن (3 نماذج TensorFlow)', use_container_width=True):
            if TF_MODELS is None:
                st.info('نماذج TensorFlow غير مثبتة محلياً — يُعرض التوقع المحفوظ من آخر تشغيل إنتاجي (نفس النتيجة)')
            else:
                win = table[FEATS].iloc[-LOOKBACK:]
                X = prod['scaler'].transform(win.values.reshape(-1,len(FEATS))).reshape(1,LOOKBACK,-1)
                preds = {s: float(m.predict(X, verbose=0).flatten()[0]) for s, m in TF_MODELS.items()}
                mean_ret = float(np.mean(list(preds.values())))
                s1,s2,s3 = st.columns(3)
                for col,(s,v) in zip([s1,s2,s3], preds.items()):
                    col.markdown(f'<div class="expert"><div class="n">🌱 النموذج {s}</div><div class="p">{v:+.2f}%</div></div>', unsafe_allow_html=True)
                st.success(f'✅ متوسط الخبراء الثلاثة: {mean_ret:+.2f}% ← السعر: {last_flour*(1+mean_ret/100):,.1f} ريال')
                st.caption(f'اتفاق الخبراء: الانحراف المعياري {np.std(list(preds.values())):.3f} — كلما قل زادت الثقة')
    with r:
        st.markdown('**درجة ثقة النظام** (مبنية على MAPE التقييمي)')
        fig = go.Figure(go.Indicator(mode='gauge', value=91.3,
            gauge=dict(axis=dict(range=[0,100], nticks=5), bar=dict(color='#4d9dff', thickness=.35),
                steps=[dict(range=[0,60],color='rgba(255,80,80,.25)'),
                       dict(range=[60,85],color='rgba(255,200,80,.2)'),
                       dict(range=[85,100],color='rgba(80,220,140,.25)')],
                threshold=dict(line=dict(color='#fff',width=3), value=91.3))))
        fig.update_layout(template='plotly_dark', height=280, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Cairo'), margin=dict(t=30, l=20, r=20, b=10))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.caption('لماذا +0.34%؟ آخر 3 أشهر كلها ثابتة عند 933.3 والسوق هادئ — النموذج يرى لا سبب لحركة كبيرة')
    st.download_button('⬇️ حمّل بطاقة التوقع (CSV)', fcsv.to_csv(index=False).encode(), 'forecast_card.csv', 'text/csv', use_container_width=True)

# ============ 3: مختبر السيناريوهات ============
with tab3:
    st.subheader('🧪 ماذا لو تحركت المحركات العالمية؟')
    st.caption('المعاملات مُقدَّرة انحدارياً من تاريخ عدن نفسه — تحليل حساسية لا تنبؤ حدثي')
    ok = table[['flour_ret','exchange_pct','brent_pct']].dropna()
    beta = np.linalg.lstsq(ok[['exchange_pct','brent_pct']].values, ok['flour_ret'].values, rcond=None)[0]
    b_d, b_o = float(beta[0]), float(beta[1])
    s1, s2 = st.columns(2)
    with s1: d_chg = st.slider('💵 تغير سعر الدولار %', -10.0, 10.0, 0.0, 0.5, format='%+.1f%%')
    with s2: o_chg = st.slider('🛢️ تغير البرنت العالمي %', -15.0, 15.0, 0.0, 1.0, format='%+.1f%%')
    scen_ret = b_d*d_chg + b_o*o_chg
    scen_next = last_flour * (1 + (scen_ret + pred_ret)/100)
    m1, m2, m3 = st.columns(3)
    m1.metric('أثر السيناريو على العائد', f'{scen_ret:+.2f}%')
    m2.metric('السعر القادم تحت السيناريو', f'{scen_next:,.1f} ريال', f'{scen_next - forecast_flour:+.1f} عن الأساسي')
    m3.metric('معاملات الحساسية', f'دولار: {b_d:.2f} | برنت: {b_o:.2f}')
    scen = pd.DataFrame({'السيناريو': ['الأساسي','دولار +10%','دولار -10%','برنت +30%','برنت -30%'],
                         'العائد المتوقع %': [pred_ret, pred_ret+b_d*10, pred_ret-b_d*10, pred_ret+b_o*30, pred_ret-b_o*30]})
    scen['السعر المتوقع'] = last_flour * (1 + scen['العائد المتوقع %']/100)
    fig = px.bar(scen, x='السيناريو', y='السعر المتوقع', text_auto='.1f', color='السعر المتوقع', color_continuous_scale='RdYlGn_r')
    fig.update_layout(template='plotly_dark', height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Cairo'), title='مقارنة السيناريوهات على سعر الشهر القادم', margin=dict(t=50, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
    st.info(f'💡 قراءة: كل رفع 1% للدولار يسحب الدقيق نحو {b_d:.2f}% في الشهر التالي — وتاريخنا يثبتها (تبويب العلاقات)')

# ============ 4: الاستكشاف التاريخي ============
with tab4:
    st.subheader('📈 كل السلاسل في شاشة واحدة')
    cols = st.multiselect('اختر السلاسل', list(CNAMES.keys()), default=['flour','exchange'], format_func=lambda k: CNAMES[k])
    norm = st.toggle('تطبيع للبدء من 100 (مقارنة صحيحة بين وحدات مختلفة)', value=True)
    if cols:
        d = table[cols].dropna(how='all')
        if norm:
            d = d / d.iloc[0] * 100
            ytitle = 'مؤشر (البداية = 100)'
        else:
            ytitle = 'القيمة'
        fig = go.Figure()
        for c in cols: fig.add_trace(go.Scatter(x=d.index, y=d[c], name=CNAMES[c], mode='lines'))
        fig.update_layout(template='plotly_dark', height=480, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Cairo'), yaxis_title=ytitle, xaxis=dict(rangeslider=dict(visible=True)), hovermode='x unified', margin=dict(t=15, l=10, r=10, b=10), legend=dict(orientation='h', y=1.08, font=dict(size=15, color='#fff', family='Cairo'), bgcolor='rgba(10,20,40,.85)', bordercolor='rgba(120,170,255,.45)', borderwidth=1))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
    st.subheader('التغير الشهري (آخر 36 شهراً)')
    mom = table['flour_ret'].iloc[-36:]
    fig = go.Figure(go.Bar(x=mom.index, y=mom.values, marker_color=np.where(mom.values>=0, '#ff8080', '#7fe0a7'), text=[f'{v:+.1f}' for v in mom.values], textposition='outside'))
    fig.update_layout(template='plotly_dark', height=340, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Cairo'), yaxis_title='تغير %', margin=dict(t=15,l=10,r=10,b=10))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

# ============ 5: الأنماط الموسمية ============
with tab5:
    st.subheader('🗓️ خريطة حرارية: تغير الدقيق % حسب الشهر والسنة')
    hm = table.assign(m=table.index.month, y=table.index.year).pivot_table(index='y', columns='m', values='flour_ret')
    hm.columns = [AR_M[m] for m in hm.columns]
    fig = px.imshow(hm, text_auto='+.1f', aspect='auto', color_continuous_scale='RdYlGn_r', color_continuous_midpoint=0, labels=dict(color='%'))
    fig.update_layout(template='plotly_dark', height=520, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Cairo'), margin=dict(t=15,l=10,r=10,b=10))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
    monthly_avg = table.groupby(table.index.month)['flour_ret'].mean()
    c1, c2 = st.columns(2)
    c1.metric('🔥 أكثر الشهور تقلباً', AR_M[int(monthly_avg.abs().idxmax())])
    c2.metric('🧊 أكثر الشهور هدوءاً', AR_M[int(monthly_avg.abs().idxmin())])

# ============ 6: العلاقات ============
with tab6:
    st.subheader('🔗 مصفوفة الارتباط بين السلاسل')
    corr = table[['flour','exchange','rice','sugar','oil','diesel','brent','wheat']].corr()
    corr.index = [CNAMES[c] for c in corr.columns]
    fig = px.imshow(corr, text_auto='.2f', aspect='auto', color_continuous_scale='RdBu_r', color_continuous_midpoint=0, zmin=-1, zmax=1)
    fig.update_layout(template='plotly_dark', height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Cairo'), margin=dict(t=15,l=10,r=10,b=10))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
    st.subheader('قصة المشروع: الدقيق يتبع الدولار')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table.index, y=table['flour'], name='الدقيق (يسار)', line=dict(color='#4d9dff', width=2)))
    fig.add_trace(go.Scatter(x=table.index, y=table['exchange'], name='الدولار (يمين)', yaxis='y2', line=dict(color='#ff6b6b', width=2)))
    fig.update_layout(template='plotly_dark', height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Cairo'), yaxis=dict(title='ريال/كيلو'), yaxis2=dict(title='ريال/دولار', overlaying='y', side='right'), hovermode='x unified', margin=dict(t=15,l=10,r=10,b=10), legend=dict(orientation='h', y=1.08, font=dict(size=15, color='#fff', family='Cairo'), bgcolor='rgba(10,20,40,.85)', bordercolor='rgba(120,170,255,.45)', borderwidth=1))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
    st.caption('الخطان يتحركان كأنهما شخص واحد — الدليل البصري لإدخال سعر الصرف أهم ميزة (القمح هبط 2023 والدقيق صعد — الدولار أقوى من الخام)')

# ============ 7: عن النموذج ============
with tab7:
    st.subheader('⚖️ بطاقة النموذج (Model Card)')
    st.markdown("""
| البند | التفاصيل |
|---|---|
| **المعمارية** | LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout → Dense(16) → Dense(1) |
| **الجمع** | Ensemble × 3 بذور (42، 7، 2026) — متوسط التوقعات |
| **الهدف** | عائد الدقيق الشهري % (علاج انزياح التوزيع) |
| **النافذة** | 6 أشهر × 9 مؤشرات |
| **التدريب** | 145 نافذة — 13 سنة كاملة (2013→2026) |
| **التقييم** | Walk-Forward على 24 شهراً لم يرها — MAE 87.3 ريال |
| **القضاة** | تفوق الاتجاه: 54.2% مقابل 25% للـ Naive |
| **رادار الدولار** | يحول خبر الدولار اليومي لقرار شراء — بمعامل حساسية موثق (0.59) |
| **حدود** | الصدمات السياسية المفاجئة خارج قدرة أي نموذج تاريخي |""")
    st.subheader('رحلة التطوير (شفافية كاملة)')
    steps = pd.DataFrame([('v1 مستويات 2017','MAE 155.2','فشل — انزياح توزيع','❌'),('+ تاريخ ممتد 2013','اختبار','ذروة 2025 دخلت عالم التدريب','✅'),('+ هدف عوائد %','اختبار','ثبات عبر العصور','✅'),('+ Walk-Forward ×24','MAE 87.2','معيار دولي','✅'),('+ Ensemble ×3','MAE 87.3','النسخة المعتمدة','🏆'),('+ رادار الدولار','تجربة مستخدم','الخبر اليومي يصير قرار شراء','🚨')], columns=['المرحلة','النتيجة','المعنى','الحالة'])
    st.dataframe(steps, use_container_width=True, hide_index=True)

# ==================== 📇 التذييل ====================
st.markdown("""<div class="footer">
🤖 تطوير وتصميم: <b>E.M-ALKAMEL</b> &nbsp;|&nbsp;
📱 <a href="https://wa.me/967771257332" target="_blank">واتساب: 771257332</a> &nbsp;|&nbsp;
✉️ <a href="mailto:mm11kk22mo@gmail.com">mm11kk22mo@gmail.com</a><br>
المصادر: WFP (برنامج الأغذية العالمي) • FRED الفيدرالي الأمريكي • سعر صرف ضمني من ملف WFP • مقرر التعلم العميق والشبكات العصبية
</div>""", unsafe_allow_html=True)