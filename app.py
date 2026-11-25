# 선행지표는 정말 경기를 앞서갈까?  - Streamlit 대시보드
# 실행: streamlit run app.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="선행지표는 정말 경기를 앞서갈까?", layout="wide")

# ---------------------------------------------------------------
# 데이터 불러오기 (prepare_data.py가 만든 data.csv)
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", encoding="utf-8-sig", parse_dates=["월"]).set_index("월")
    return df

df = load_data()
TARGET = "동행지수"                      # '정답' = 지금 경기
LEADERS = ["선행지수", "소비자심리지수"]    # '예상' 두 개
COLORS = {"동행지수": "#1f2937", "선행지수": "#1e40af", "소비자심리지수": "#e11d48"}
MAX_LAG = 12

# ---------------------------------------------------------------
# 사이드바 위젯 (조건 바꾸기)
# ---------------------------------------------------------------
st.sidebar.header("분석 조건 설정")
leader = st.sidebar.selectbox("비교할 선행지표 선택", LEADERS)
years = st.sidebar.slider("분석 기간(연도) 지정", int(df.index.year.min()), int(df.index.year.max()),
                          (int(df.index.year.min()), int(df.index.year.max())))
lag = st.sidebar.slider("시차 : 선행지표를 몇 개월 앞당길까?", 0, MAX_LAG, 3)
exclude_covid = st.sidebar.checkbox("코로나 기간(2020.1~2021.12) 제외", value=False)


# ---------------------------------------------------------------
# 계산 함수
#   시차 k = 'k개월 전 선행지표'와 '이번 달 동행지수'를 짝지어 비교
# ---------------------------------------------------------------
def paired(df, col, k):
    """동행지수(t월)와 선행지표(t-k월)를 짝지은 표를 만들고, 위젯 조건으로 거른다."""
    out = pd.DataFrame({"선행": df[col].shift(k), "동행": df[TARGET]}).dropna()
    out = out[(out.index.year >= years[0]) & (out.index.year <= years[1])]
    if exclude_covid:
        out = out[~((out.index >= "2020-01-01") & (out.index <= "2021-12-31"))]
    return out


def corr_by_lag(df, col):
    """시차 0~12개월마다 상관계수 r 계산"""
    return pd.Series({k: paired(df, col, k).corr().iloc[0, 1] for k in range(MAX_LAG + 1)})


# 조건에 맞는 데이터가 너무 적으면 계산하지 않고 안내만 표시 (예: 2020~2021년 + 코로나 제외)
if len(paired(df, LEADERS[0], MAX_LAG)) < 12:
    st.warning("선택한 조건에 맞는 데이터가 12개월보다 적어 분석할 수 없습니다. "
               "기간을 넓히거나 '코로나 기간 제외'를 해제해 주세요.")
    st.stop()

corrs = {c: corr_by_lag(df, c) for c in LEADERS}
cur = corrs[leader]
best_lag = int(cur.idxmax())
pairs = paired(df, leader, lag)

# ---------------------------------------------------------------
# 제목과 질문
# ---------------------------------------------------------------
st.title("선행지표는 정말 경기를 앞서갈까?")
st.markdown(
    "**질문** : 여러 경제 지표로 앞으로의 경기를 예측한 **선행지수**와, 가구 설문으로 사람들이 느끼는 경기를 잰 **소비자심리지수**는 동행지수(실제 경기)보다 몇 개월 먼저 움직일까?"
)

# ---------------------------------------------------------------
# 요약 숫자
# ---------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("가장 잘 맞는 시차", f"{best_lag}개월")
m2.metric("그때의 상관계수 r", f"{cur.max():.2f}")
m3.metric(f"선택한 시차({lag}개월)의 상관계수 r", f"{cur[lag]:.2f}",
          delta=f"{cur[lag] - cur[0]:+.2f} (시차 0 대비)")
m4.metric("분석한 개월 수", f"{len(pairs)}개월")

# ---------------------------------------------------------------
# 차트 1: 추이 비교
# ---------------------------------------------------------------
st.subheader(f"① 추이 : {leader} vs 동행지수")
shift_view = st.checkbox(f"{leader}를 {lag}개월 뒤로 밀어서 겹쳐 보기", value=False)

view = df[(df.index.year >= years[0]) & (df.index.year <= years[1])]
lead_line = df[leader].shift(lag) if shift_view else df[leader]
lead_line = lead_line.loc[view.index]

fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Scatter(x=view.index, y=view[TARGET], name="동행지수(실제 경기)",
                          line=dict(color=COLORS[TARGET], width=2.5)), secondary_y=False)
fig1.add_trace(go.Scatter(x=view.index, y=lead_line,
                          name=f"{leader}" + (f" ({lag}개월 뒤로 이동)" if shift_view else ""),
                          line=dict(color=COLORS[leader], width=2)), secondary_y=(leader == '소비자심리지수'))
if exclude_covid:
    fig1.add_vrect(x0="2020-01-01", x1="2021-12-31", fillcolor="gray", opacity=0.15,
                   line_width=0, annotation_text="분석 제외", annotation_position="top left")
fig1.add_hline(y=100, line_dash="dot", line_color="gray", secondary_y=False)

# 주요 사건 표시 (분석 기간 안에 있을 때만)
for date, label in [("2008-10-01", "금융위기"), ("2020-03-01", "코로나")]:
    if years[0] <= int(date[:4]) <= years[1]:
        fig1.add_vline(x=date, line_dash="dot", line_color="lightgray")
        fig1.add_annotation(x=date, y=1, yref="paper", text=label, showarrow=False, yanchor="bottom")

# 축 이름: 선행지수는 동행지수와 단위가 같아 축 하나, 소비자심리지수만 오른쪽 축 추가
fig1.update_yaxes(title_text="순환변동치 (100 = 추세 수준)", secondary_y=False)
if leader == "소비자심리지수":
    fig1.update_yaxes(title_text="소비자심리지수 (100 = 장기평균)", secondary_y=True)
else:
    fig1.update_yaxes(visible=False, secondary_y=True)

fig1.update_layout(height=380, hovermode="x unified", legend=dict(orientation="h", y=1.12),
                   margin=dict(t=30, b=10))
st.plotly_chart(fig1, use_container_width=True)
st.caption("선행지수와 동행지수는 같은 단위(순환변동치)라 축 하나로, 단위가 다른 소비자심리지수만 오른쪽 축을 씁니다. "
           "점선(100)은 장기 추세 수준이며, 2026.7~9월 경기종합지수는 잠정치입니다.")

left, right = st.columns(2)

# ---------------------------------------------------------------
# 차트 2: 시차별 상관계수 (두 선행지표 비교)
# ---------------------------------------------------------------
with left:
    st.subheader("② 시차별 상관계수")
    fig2 = go.Figure()
    for c in LEADERS:
        fig2.add_trace(go.Bar(x=corrs[c].index, y=corrs[c].values, name=c,
                              marker_color=COLORS[c], opacity=1.0 if c == leader else 0.35))
    fig2.add_vline(x=lag, line_dash="dash", line_color="gray",
                   annotation_text=f"선택: {lag}개월", annotation_position="top")
    fig2.add_annotation(x=best_lag, y=cur.max(), text=f"최고: {best_lag}개월",
                    showarrow=True, arrowhead=2, yshift=10)
    fig2.update_layout(barmode="group", height=360, xaxis_title="시차(개월): 선행지표를 앞당긴 기간",
                       yaxis_title="상관계수 r", yaxis_range=[min(0, min(v.min() for v in corrs.values()) - 0.05), 1],
                       legend=dict(orientation="h", y=1.25), margin=dict(t=70, b=10))
    fig2.update_xaxes(dtick=1)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("막대가 가장 높은 시차가 '몇 개월 앞서는지'의 답입니다. r은 1에 가까울수록 같이 움직입니다.")

# ---------------------------------------------------------------
# 차트 3: 선택한 시차의 산점도
# ---------------------------------------------------------------
with right:
    st.subheader(f"③ {lag}개월 전 {leader} vs 이번 달 동행지수")
    slope, intercept = np.polyfit(pairs["선행"], pairs["동행"], 1)
    xs = np.linspace(pairs["선행"].min(), pairs["선행"].max(), 50)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=pairs["선행"], y=pairs["동행"], mode="markers", name="월별 값",
                              marker=dict(color=COLORS[leader], size=6, opacity=0.6),
                              text=pairs.index.strftime("%Y.%m"),
                              hovertemplate="%{text}<br>선행: %{x}<br>동행: %{y}<extra></extra>"))
    fig3.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="추세선",
                              line=dict(color="gray", dash="dash")))
    fig3.update_layout(height=360, xaxis_title=f"{lag}개월 전 {leader}", yaxis_title="동행지수 순환변동치",
                       showlegend=False, margin=dict(t=30, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(f"점 하나가 한 달입니다. 회색 점선은 점들의 전체 방향을 보여 주는 추세선입니다. 상관계수 r = {cur[lag]:.2f}")

# ---------------------------------------------------------------
# 해석 (조건에 따라 문장이 바뀜)
# ---------------------------------------------------------------
st.subheader("이 분석 조건에서의 결과")
other = [c for c in LEADERS if c != leader][0]
o_best = int(corrs[other].idxmax())
st.markdown(
    f"- **{leader}**는 동행지수보다 약 **{best_lag}개월** 앞설 때 가장 잘 맞았습니다 (r = {cur.max():.2f}).\n"
    f"- 비교 : **{other}**는 **{o_best}개월** 앞설 때 가장 잘 맞았습니다 (r = {corrs[other].max():.2f}).\n"
    f"- 조건 : {years[0]}~{years[1]}년" + (", 코로나 기간 제외" if exclude_covid else ", 전체 기간 포함")
)

with st.expander("한계와 주의할 점"):
    st.markdown(
        "- 상관관계는 **함께 움직인다**는 뜻일 뿐, 선행지표가 경기를 **일으킨다**는 뜻은 아닙니다.\n"
        "- 선행지수를 만드는 7개 항목에 **경제심리지수가 포함**되어 있어, 두 선행지표가 완전히 독립적이지 않습니다.\n"
        "- 코로나처럼 갑작스러운 충격이 있으면 결과가 크게 달라집니다. 기간과 '코로나 제외'를 바꿔 확인해 보세요.\n"
        "- 최근 3개월(2026.7~9) 경기종합지수는 **잠정치**라서 나중에 수정될 수 있습니다.\n"
        "- 순환변동치는 추세를 추정해서 빼는 방식이라, 새 자료가 나오면 과거 값도 조금씩 바뀝니다."
    )

with st.expander("데이터 출처와 집계 기준"):
    st.markdown(
        "- **동행지수·선행지수 순환변동치**: 국가데이터처, 「경기종합지수(2020=100)(10차)」, KOSIS (월, 1996.1~2026.9)\n"
        "- **소비자심리지수(CCSI), 전체**: 한국은행, 「소비자동향조사(전국, 월, 2008.9~)」, ECOS (월)\n"
        "- **분석 기간**: 세 지표가 모두 있는 2008년 9월 ~ 2026년 9월 (217개월)\n"
        "- **계산 방법**: 시차 k개월이면 'k개월 전 선행지표'와 '이번 달 동행지수'의 상관계수를 계산"
    )
