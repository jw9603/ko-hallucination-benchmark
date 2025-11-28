"""Comparative analysis page between global and Korean LLMs."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_utils import ensure_page_config, get_benchmark_info, render_sidebar


def render_comparative_report_page() -> None:
    st.title("📑 글로벌 vs 한국 LLM 비교 분석")

    # 상단 Executive Summary
    st.markdown("## 📊 핵심 요약")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("평가 모델 수", "14", delta="글로벌 6개 + 한국 8개")
    with col2:
        st.metric("총 샘플 수", "244", delta="6개 태스크")
    with col3:
        st.metric("총 평가 횟수", "3,416", delta="244 × 14 모델")

    st.markdown("---")

    # Top 5 Models
    st.markdown("### 🏆 전체 성능 Top 5")

    top_models_data = {
        "순위": ["🥇 1위", "🥈 2위", "🥉 3위", "4위", "5위"],
        "모델": [
            "K-intelligence Midm-2.0 🇰🇷",
            "Google Gemma-3-4B 🌍",
            "Qwen3-4B 🌍",
            "Upstage SOLAR-10.7B 🇰🇷",
            "Kakao Kanana-1.5-8B 🇰🇷"
        ],
        "성공률": ["38.93%", "36.89%", "33.61%", "33.20%", "27.46%"],
        "성공 개수": ["95/244", "90/244", "82/244", "81/244", "67/244"]
    }

    top_df = pd.DataFrame(top_models_data)
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Global vs Korean Performance Comparison
    st.markdown("### 🌍 vs 🇰🇷 그룹 성능 비교")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='글로벌 LLM',
            x=['전체 성능'],
            y=[25.07],
            marker_color='#3498db',
            text=['25.07%'],
            textposition='auto',
        ))
        fig.add_trace(go.Bar(
            name='한국 LLM',
            x=['전체 성능'],
            y=[25.92],
            marker_color='#e74c3c',
            text=['25.92%'],
            textposition='auto',
        ))

        fig.update_layout(
            title='평균 성공률: 글로벌 vs 한국',
            yaxis_title='성공률 (%)',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.info("""
        **핵심 발견:**

        두 그룹의 전체 성능은 거의 동일합니다:
        - **글로벌 LLM**: 25.07%
        - **한국 LLM**: 25.92%
        - **차이**: 단 0.85%p

        → 통계적으로 유의미하지 않은 차이
        """)

        st.success("""
        **태스크별 우승자:**
        - 🇰🇷 **QA (질의응답)**: +11.01%p
        - 🇰🇷 **General (일반)**: +2.82%p
        - 🌍 **Coding (코딩)**: +4.38%p
        - 🌍 **Dialogue (대화)**: +3.66%p
        """)

    st.markdown("---")

    # Task별 성능 비교
    st.markdown("### 📈 태스크별 성능 비교")

    task_comparison_data = {
        "태스크": ["코딩", "대화", "일반", "수학", "질의응답", "요약"],
        "글로벌 LLM": [10.00, 16.38, 33.87, 19.79, 34.52, 30.05],
        "한국 LLM": [5.62, 12.72, 36.69, 19.92, 45.54, 29.30],
        "차이": [-4.38, -3.66, 2.82, 0.13, 11.01, -0.75]
    }

    task_df = pd.DataFrame(task_comparison_data)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='글로벌 LLM',
        x=task_df['태스크'],
        y=task_df['글로벌 LLM'],
        marker_color='#3498db',
        text=[f"{v:.2f}%" for v in task_df['글로벌 LLM']],
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name='한국 LLM',
        x=task_df['태스크'],
        y=task_df['한국 LLM'],
        marker_color='#e74c3c',
        text=[f"{v:.2f}%" for v in task_df['한국 LLM']],
        textposition='auto',
    ))

    fig.update_layout(
        title='태스크 유형별 성공률',
        xaxis_title='태스크',
        yaxis_title='성공률 (%)',
        barmode='group',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # 차이 강조 테이블
    st.markdown("#### 태스크별 성능 차이")

    def color_code_difference(row):
        diff = row['차이']
        if diff > 5:
            return '🟢 한국 우세'
        elif diff < -5:
            return '🔴 글로벌 우세'
        else:
            return '➖ 비슷'

    task_df['우세 그룹'] = task_df.apply(color_code_difference, axis=1)

    display_df = task_df[['태스크', '글로벌 LLM', '한국 LLM', '차이', '우세 그룹']]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption("🟢 한국 LLM이 5%p 이상 우수 | 🔴 글로벌 LLM이 5%p 이상 우수 | ➖ 차이가 5%p 미만")

    st.markdown("---")

    # Hallucination Pattern Analysis
    st.markdown("### 🚨 환각 패턴 분석")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 글로벌 LLM")
        global_hal_data = {
            "Type": ["Factual Contradiction", "Factual Fabrication", "Instruction Inconsistency", "Logical Inconsistency", "No Hallucination"],
            "Percentage": [29.03, 18.85, 11.13, 10.86, 25.07]
        }

        fig = px.pie(
            global_hal_data,
            values='Percentage',
            names='Type',
            title='글로벌 LLM 환각 분포',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 한국 LLM")
        korean_hal_data = {
            "Type": ["Factual Contradiction", "Factual Fabrication", "Instruction Inconsistency", "Logical Inconsistency", "No Hallucination"],
            "Percentage": [19.01, 29.30, 10.04, 10.25, 25.92]
        }

        fig = px.pie(
            korean_hal_data,
            values='Percentage',
            names='Type',
            title='한국 LLM 환각 분포',
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    st.warning("""
    **🔍 환각 패턴의 핵심 차이:**

    - **글로벌 LLM**: **사실 모순** 생성 경향 (29.03% vs 19.01%)
      → 기존 사실과 충돌하는 정보를 생성하는 경향이 높음

    - **한국 LLM**: **사실 조작** 생성 경향 (29.30% vs 18.85%)
      → 존재하지 않는 정보를 만들어내는 경향이 높음
    """)

    st.markdown("---")

    # 중요한 제한사항 섹션
    st.markdown("### ⚠️ 중요한 제한사항")

    with st.expander("🔴 **핵심 가정: 분류기 정확도** - 클릭하여 읽기", expanded=True):
        st.error("""
        **⚠️ 중요한 제한사항:**

        이 분석은 **환각 분류기가 100% 정확하다**고 가정합니다.
        하지만 실제로 우리 분류기의 성능은 다음과 같습니다:

        - **LoRA 분류기**: 61.0% 정확도 (100개 샘플 테스트 세트)
        - **Full Fine-tuned 분류기**: 64.0% 정확도 (100개 샘플 테스트 세트)

        즉, **약 36-39%의 분류가 틀릴 수 있습니다**.
        """)

        st.warning("""
        **📊 이것이 결과에 미치는 영향:**

        1. **순위가 부정확할 수 있음**: 모델 A가 모델 B보다 높은 순위를 차지한 것이 분류기 오류 때문일 수 있음
        2. **성공률은 추정치임**: 38% 성공률을 보이는 모델이 실제로는 30% 또는 45%일 수 있음
        3. **태스크별 성능**: 10%p 미만의 차이는 오차 범위 내일 수 있음
        4. **환각 패턴**: 환각 타입 분포가 분류기 편향에 의해 왜곡될 수 있음

        **권장 다음 단계:**
        - ✅ 무작위 샘플 수동 검토 (예: 모델당 50-100개 샘플)
        - ✅ 여러 분류기 사용 및 일치도 비교 (앙상블 접근법)
        - ✅ 성공률에 대한 신뢰 구간 계산
        - ✅ 분류기 혼동 행렬 분석으로 체계적 오류 이해
        """)

        st.info("""
        **🎯 검증 체크리스트:**

        이 결과를 신뢰하기 위해 다음을 수행해야 합니다:
        - [ ] 모델당 최소 10%의 샘플 수동 검증
        - [ ] 분류기가 태스크별로 다르게 작동하는지 확인
        - [ ] 인간 전문가 주석과 결과 비교
        - [ ] 분류기와 인간 간 평가자 간 신뢰도 측정
        - [ ] False positive/negative 패턴 분석
        """)

        st.markdown("#### 분류기 성능 vs 주요 AI 모델")

        classifier_perf_data = {
            "모델": ["GPT-3.5", "GPT-4o-mini", "GPT-4o", "Grok 4.1", "Claude Opus 4.1",
                     "Grok 4", "GPT-5.1", "Claude Opus 4.5", "Gemini 2.5", "Gemini 2.0",
                     "우리 LoRA", "우리 Full FT"],
            "정확도": [34, 52, 52, 53, 55, 63, 69, 70, 70, 73, 61, 64],
            "유형": ["상용 모델"]*10 + ["우리 분류기"]*2
        }

        fig = px.bar(
            classifier_perf_data,
            x="모델",
            y="정확도",
            color="유형",
            title="환각 분류기 정확도 비교",
            labels={"정확도": "정확도 (%)"},
            color_discrete_map={"상용 모델": "#3498db", "우리 분류기": "#e74c3c"},
            text="정확도"
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        **출처**: ko-hallucination-sft-v3 데이터셋의 100개 샘플 테스트 세트로 평가.
        우리 분류기는 GPT-4o, Grok 4.1과 비슷한 성능을 보이지만, 여전히 약 36-39%의 오류율을 가지고 있습니다.
        """)

    with st.expander("❓ **Unknown 레이블 & 분류기 불확실성**", expanded=False):
        st.markdown("""
        **약 5%의 샘플이 "Unknown"으로 분류**되었으며, 이는 분류기가 확신하지 못함을 나타냅니다:

        - **글로벌 LLM**: 74/1,464 샘플 (5.05%) → Unknown
        - **한국 LLM**: 107/1,952 샘플 (5.48%) → Unknown

        **"Unknown" 레이블의 원인:**
        - 어떤 카테고리에도 명확히 맞지 않는 모호한 출력
        - 여러 환각 타입 사이의 경계 케이스
        - 분류기 신뢰도가 임계값 미만
        - 예상치 못한 형식의 출력

        **분석에 미치는 영향:**
        - 이러한 샘플은 성공률 계산에서 제외됨
        - 중요한 실패 패턴을 숨길 수 있음
        - 특정 모델이 더 모호한 출력을 생성하는 경우 결과에 편향이 생길 수 있음
        """)

        unknown_data = {
            "그룹": ["글로벌 LLM", "한국 LLM"],
            "Unknown": [5.05, 5.48],
            "분류됨": [94.95, 94.52]
        }

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Unknown',
            x=unknown_data['그룹'],
            y=unknown_data['Unknown'],
            marker_color='#95a5a6',
            text=[f"{v:.2f}%" for v in unknown_data['Unknown']],
            textposition='auto',
        ))
        fig.add_trace(go.Bar(
            name='분류됨',
            x=unknown_data['그룹'],
            y=unknown_data['분류됨'],
            marker_color='#2ecc71',
            text=[f"{v:.2f}%" for v in unknown_data['분류됨']],
            textposition='auto',
        ))

        fig.update_layout(
            title='분류기 불확실성: Unknown 레이블',
            yaxis_title='비율 (%)',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 상세 리포트 탭
    st.markdown("### 📄 상세 리포트")

    tab1, tab2, tab3 = st.tabs(["📊 핵심 인사이트", "📋 포맷된 분석 (MD)", "📄 원본 분석 (TXT)"])
    project_root = Path(__file__).resolve().parents[1]

    with tab1:
        st.markdown("#### 💡 핵심 인사이트 & 권장사항")

        st.markdown("""
        ##### 1️⃣ 한국 LLM의 QA 태스크 우수성
        한국 LLM은 QA(질의응답) 태스크에서 **11%p 더 높은 성능**을 보이며, 다음에 특화되어 있음을 시사합니다:
        - 사실 검증
        - 정확한 정보 검색
        - 한국어 이해

        **QA 최고 성능 모델:**
        - 🥇 Upstage SOLAR-10.7B: 69.05%
        - 🥈 Gemma-3-4B: 61.90%
        - 🥉 K-intelligence Midm-2.0: 57.14%
        """)

        st.markdown("""
        ##### 2️⃣ 글로벌 LLM의 코딩 우위
        글로벌 LLM은 코드 생성에서 **4.38%p 더 우수**하며, 다음 이유로 추정됩니다:
        - 학습 데이터에 더 많은 코드 데이터 포함
        - 다국어 코드 이해 능력 향상
        - 강력한 프로그래밍 언어 지원

        **코딩 최고 성능 모델:**
        - 🥇 Phi-4-mini: 15.00%
        - 🥇 K-intelligence Midm-2.0: 15.00%
        - 🥉 Gemma-3-4B / Qwen3-4B: 10.00%
        """)

        st.markdown("""
        ##### 3️⃣ 대화는 보편적인 난제
        모든 모델이 대화 태스크에서 어려움을 겪습니다 (평균 14.08%), 이는 다음을 나타냅니다:
        - 맥락 이해가 어려움
        - 대화 일관성 유지가 어려움
        - 다중 턴 추론 개선 필요

        **대화 최고 성능 모델:**
        - 🥇 Gemma-3-4B: 36.21%
        - 🥈 K-intelligence Midm-2.0: 29.31%
        - 🥉 Qwen3-4B: 25.86%
        """)

        st.markdown("""
        ##### 4️⃣ 모델 크기 ≠ 성능
        더 큰 모델이 항상 더 나은 성능을 보이지는 않습니다:
        - **최고**: K-intelligence Midm-2.0 (~2B 파라미터) - 38.93%
        - **10위**: Llama-3.1-8B (8B 파라미터) - 19.67%
        - **13위**: Yanolja NEXT-EEVE (10.8B 파라미터) - 18.85%

        → **아키텍처, 학습 데이터, 파인튜닝 품질**이 크기보다 중요합니다
        """)

        st.markdown("---")

        st.markdown("#### 🎯 모델 선택 가이드")

        guide_cols = st.columns(2)

        with guide_cols[0]:
            st.success("""
            **코딩 태스크:**
            1. Phi-4-mini (15.00%)
            2. K-intelligence Midm-2.0 (15.00%)
            3. Gemma-3-4B / Qwen3-4B (10.00%)
            """)

            st.info("""
            **QA 태스크:**
            1. Upstage SOLAR-10.7B (69.05%)
            2. Gemma-3-4B (61.90%)
            3. K-intelligence Midm-2.0 (57.14%)
            """)

            st.warning("""
            **수학 태스크:**
            1. Gemma-3-4B (40.62%)
            2. SKT A.X-4.0-Light (40.62%)
            3. K-intelligence Midm-2.0 (31.25%)
            """)

        with guide_cols[1]:
            st.success("""
            **대화 태스크:**
            1. Gemma-3-4B (36.21%)
            2. K-intelligence Midm-2.0 (29.31%)
            3. Qwen3-4B (25.86%)
            """)

            st.info("""
            **요약 태스크:**
            1. LG EXAONE-3.5-7.8B (47.54%)
            2. K-intelligence Midm-2.0 (44.26%)
            3. Qwen3-4B / Phi-4-mini (39.34%)
            """)

            st.warning("""
            **범용 최고 모델:**
            1. K-intelligence Midm-2.0 (38.93%)
            2. Gemma-3-4B (36.89%)
            3. Qwen3-4B (33.61%)
            """)

    with tab2:
        st.subheader("FINAL_COMPARATIVE_ANALYSIS.md")
        md_path = project_root / "FINAL_COMPARATIVE_ANALYSIS.md"
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            st.markdown(md_content)
        except FileNotFoundError:
            st.error(f"⚠️ {md_path} file not found")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    with tab3:
        st.subheader("COMPLETE_ANALYSIS_RAW.txt")
        txt_path = project_root / "COMPLETE_ANALYSIS_RAW.txt"
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                txt_content = f.read()
            st.text(txt_content)
        except FileNotFoundError:
            st.error(f"⚠️ {txt_path} file not found")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")


def main() -> None:
    ensure_page_config()
    benchmark_info = get_benchmark_info()
    render_sidebar(benchmark_info)
    render_comparative_report_page()


if __name__ == "__main__":
    main()
