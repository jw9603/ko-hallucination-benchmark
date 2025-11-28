"""Visualization page for task-based and comparative charts."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app_utils import (
    HALLUCINATION_TYPES,
    create_task_based_radar_charts,
    ensure_page_config,
    format_task_name,
    get_benchmark_info,
    parse_benchmark_results,
    render_sidebar,
)


def render_visualization_page() -> None:
    st.title("🕸️ Task-based Visualization & Model Comparison")

    if "all_results" not in st.session_state or not st.session_state["all_results"]:
        st.warning("⚠️ No results available. Please run a benchmark first.")
        st.info("👈 Go to 'Main' page to run a benchmark")
        return

    all_models = list(st.session_state["all_results"].keys())

    # 단일 모델 보기 vs 비교 모드 선택
    view_mode = st.radio("View Mode", ["Single Model", "Compare Models"], horizontal=True)

    st.markdown("---")

    if view_mode == "Single Model":
        # 단일 모델 선택
        selected_model = st.selectbox(
            "📊 Select Model",
            all_models,
            index=all_models.index(st.session_state.get("current_model", all_models[0]))
            if st.session_state.get("current_model") in all_models else 0,
        )

        results = st.session_state["all_results"][selected_model]
        df = parse_benchmark_results(results)

        # 태스크별 레이더 차트
        st.subheader(f"📐 Task-based Hallucination Distribution: {selected_model}")
        st.markdown("""
        각 육각형은 하나의 태스크를 나타내며, 각 꼭짓점은 환각 타입을 의미합니다.
        값은 해당 환각 타입의 비율(%)입니다.
        """)

        radar_fig = create_task_based_radar_charts(df, selected_model)
        st.plotly_chart(radar_fig, use_container_width=True)

        return

    # 모델 비교 모드
    st.subheader("📊 Model Comparison")

    # 한국어 모델과 글로벌 모델 분류
    korean_companies = ["LDCC", "NCSOFT", "upstage", "skt", "naver-hyperclovax",
                        "LGAI-EXAONE", "kakaocorp", "K-intelligence", "yanolja"]

    korean_models = [m for m in all_models if any(comp in m for comp in korean_companies)]
    global_models = [m for m in all_models if m not in korean_models]

    def render_charts(selected_models: list[str], tab_key: str) -> None:
        if len(selected_models) < 2:
            st.warning("⚠️ Please select at least 2 models to compare.")
            return

        # 모델별 환각 타입 분포 비교 차트
        st.markdown("### Hallucination Type Distribution Comparison")

        comparison_data = []
        for model in selected_models:
            results = st.session_state["all_results"][model]
            df = parse_benchmark_results(results)

            label_counts = df["label"].value_counts().to_dict()
            total = len(df)

            for hal_type in HALLUCINATION_TYPES:
                count = label_counts.get(hal_type, 0)
                percentage = (count / total * 100) if total > 0 else 0.0
                comparison_data.append({
                    "Model": model,
                    "Hallucination Type": hal_type,
                    "Percentage": percentage,
                    "Count": count,
                })

        comparison_df = pd.DataFrame(comparison_data)

        fig = px.bar(
            comparison_df,
            x="Hallucination Type",
            y="Percentage",
            color="Model",
            barmode="group",
            title="Hallucination Type Distribution by Model",
            labels={"Percentage": "Percentage (%)", "Hallucination Type": "Type"},
            text="Count",
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, key=f"bar_chart_{tab_key}")

        # 태스크별 레이더 차트 비교
        st.markdown("### Task-based Radar Chart Comparison")
        st.markdown("각 태스크별로 선택한 모델들의 환각 타입 분포를 비교합니다.")

        # 모든 태스크 수집
        all_tasks = set()
        for model in selected_models:
            results = st.session_state["all_results"][model]
            df = parse_benchmark_results(results)
            all_tasks.update(df["task"].unique())

        sorted_tasks = sorted(all_tasks)

        n_tasks = len(sorted_tasks)
        n_cols = 2
        n_rows = (n_tasks + n_cols - 1) // n_cols

        radar_comparison_fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[format_task_name(task) for task in sorted_tasks],
            specs=[[{"type": "polar"}] * n_cols for _ in range(n_rows)],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
        if len(selected_models) > len(colors):
            colors = (colors * ((len(selected_models) // len(colors)) + 1))[:len(selected_models)]
        model_colors = colors[:len(selected_models)]

        for idx, task in enumerate(sorted_tasks):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            task_max_val = 0

            for model_idx, model in enumerate(selected_models):
                results = st.session_state["all_results"][model]
                df = parse_benchmark_results(results)

                task_df = df[df["task"] == task]
                total_task = len(task_df)

                if total_task == 0:
                    continue

                label_counts = task_df["label"].value_counts().to_dict()
                percentages = [
                    (label_counts.get(hal_type, 0) / total_task * 100)
                    for hal_type in HALLUCINATION_TYPES
                ]

                if percentages:
                    task_max_val = max(task_max_val, max(percentages))

                model_short = model.split('/')[-1] if '/' in model else model

                radar_comparison_fig.add_trace(
                    go.Scatterpolar(
                        r=percentages,
                        theta=HALLUCINATION_TYPES,
                        fill='toself',
                        name=model_short,
                        line=dict(color=model_colors[model_idx], width=2),
                        marker=dict(size=6),
                        showlegend=(idx == 0),
                    ),
                    row=row,
                    col=col,
                )

            range_max = min(100, max(50, task_max_val * 1.2))
            radar_comparison_fig.update_polars(
                radialaxis=dict(visible=True, range=[0, range_max]),
                row=row,
                col=col,
            )

        if len(selected_models) <= 3:
            model_names = [m.split('/')[-1] for m in selected_models]
            title_text = f"Task-based Model Comparison: {', '.join(model_names)}"
        else:
            title_text = f"Task-based Model Comparison ({len(selected_models)} models)"

        radar_comparison_fig.update_layout(
            height=350 * n_rows,
            title_text=title_text,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=10),
            ),
        )

        st.plotly_chart(radar_comparison_fig, use_container_width=True, key=f"radar_chart_{tab_key}")

        # 태스크별 비교 테이블
        st.markdown("### Task-wise Comparison Table")

        task_comparison = []
        for model in selected_models:
            results = st.session_state["all_results"][model]
            df = parse_benchmark_results(results)

            for task in sorted(df["task"].unique()):
                task_df = df[df["task"] == task]
                total_task = len(task_df)
                label_counts = task_df["label"].value_counts().to_dict()

                no_hal = label_counts.get("No Hallucination", 0)
                no_hal_pct = (no_hal / total_task * 100) if total_task > 0 else 0.0

                task_comparison.append({
                    "Model": model,
                    "Task": format_task_name(task),
                    "Total": total_task,
                    "No Hallucination": f"{no_hal} ({no_hal_pct:.1f}%)",
                })

        task_comp_df = pd.DataFrame(task_comparison)
        st.dataframe(task_comp_df, use_container_width=True)

    tab_all, tab_korean, tab_global = st.tabs(["🌏 All Models", "🇰🇷 Korean LLMs", "🌍 Global LLMs"])

    with tab_all:
        selected_models = st.multiselect(
            "Select models to compare (max 15)",
            all_models,
            default=all_models[:min(2, len(all_models))],
            max_selections=15,
            key="compare_all"
        )
        render_charts(selected_models, "all")

    with tab_korean:
        if not korean_models:
            st.info("No Korean LLM results available yet.")
        else:
            selected_models = st.multiselect(
                "Select Korean models to compare (max 10)",
                korean_models,
                default=korean_models[:min(2, len(korean_models))],
                max_selections=10,
                key="compare_korean"
            )
            render_charts(selected_models, "korean")

    with tab_global:
        if not global_models:
            st.info("No Global LLM results available yet.")
        else:
            selected_models = st.multiselect(
                "Select global models to compare (max 10)",
                global_models,
                default=global_models[:min(2, len(global_models))],
                max_selections=10,
                key="compare_global"
            )
            render_charts(selected_models, "global")


def main() -> None:
    ensure_page_config()
    benchmark_info = get_benchmark_info()
    render_sidebar(benchmark_info)
    render_visualization_page()


if __name__ == "__main__":
    main()
