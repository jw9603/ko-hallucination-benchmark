"""Results page showing benchmark summaries and downloads."""

import json

import pandas as pd
import streamlit as st

from app_utils import (
    HALLUCINATION_TYPES,
    ensure_page_config,
    format_task_name,
    get_benchmark_info,
    parse_benchmark_results,
    render_sidebar,
)


def render_results_page() -> None:
    st.title("📈 Benchmark Results")

    if "all_results" not in st.session_state or not st.session_state["all_results"]:
        st.warning("⚠️ No results available. Please run a benchmark first.")
        st.info("👈 Go to 'Main' page to run a benchmark")
        return

    all_models = list(st.session_state["all_results"].keys())

    # 1. 전체 모델 요약 테이블 (모든 모델 비교)
    st.subheader("📋 Summary Table - All Models")
    st.caption("💡 Click column headers to sort by count (numeric sorting enabled)")

    all_summary_data = []
    for model in all_models:
        results = st.session_state["all_results"][model]
        df = parse_benchmark_results(results)

        total = len(df)
        label_counts = df["label"].value_counts().to_dict()

        # 모델명 단축
        model_short = model.split('/')[-1] if '/' in model else model

        row_data = {"Model": model_short, "Total": total}
        for hal_type in HALLUCINATION_TYPES:
            count = label_counts.get(hal_type, 0)
            # 개수만 저장하여 정렬 가능하도록 함
            row_data[hal_type] = count

        all_summary_data.append(row_data)

    all_summary_df = pd.DataFrame(all_summary_data)

    st.dataframe(
        all_summary_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # 2. 개별 모델 상세 결과
    st.subheader("📊 Detailed Results by Model")

    # 모델 선택 드롭다운
    selected_model = st.selectbox(
        "Select Model for Detailed View",
        all_models,
        index=all_models.index(st.session_state.get("current_model", all_models[0]))
        if st.session_state.get("current_model") in all_models else 0,
    )

    results = st.session_state["all_results"][selected_model]
    model_name = selected_model

    # 결과 파싱
    df = parse_benchmark_results(results)

    # 태스크별 통계
    st.markdown("### Task-wise Statistics for Selected Model")

    # 태스크별 환각 타입 분포 테이블
    task_stats = []
    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task]
        total_task = len(task_df)

        task_row = {"Task": format_task_name(task), "Samples": total_task}
        label_counts = task_df["label"].value_counts().to_dict()

        for hal_type in HALLUCINATION_TYPES:
            count = label_counts.get(hal_type, 0)
            percentage = (count / total_task * 100) if total_task > 0 else 0.0
            task_row[hal_type] = f"{count} ({percentage:.1f}%)"

        task_stats.append(task_row)

    task_stats_df = pd.DataFrame(task_stats)
    st.dataframe(task_stats_df, use_container_width=True)

    st.markdown("---")

    # 3. 상세 결과 다운로드
    st.subheader("💾 Download Results")
    result_json = json.dumps(results, ensure_ascii=False, indent=2)
    st.download_button(
        label="📥 Download Full Results (JSON)",
        data=result_json,
        file_name=f"{model_name.replace('/', '_')}_benchmark_results.json",
        mime="application/json",
    )


def main() -> None:
    ensure_page_config()
    benchmark_info = get_benchmark_info()
    render_sidebar(benchmark_info)
    render_results_page()


if __name__ == "__main__":
    main()
