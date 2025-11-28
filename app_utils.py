#!/usr/bin/env python3
"""
Shared utilities and layout helpers for the LDCC Hallucination Benchmark app.
"""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# 환각 타입 정의
HALLUCINATION_TYPES = [
    "No Hallucination",
    "Factual Contradiction",
    "Factual Fabrication",
    "Instruction Inconsistency",
    "Logical Inconsistency",
]

# 태스크 정의
TASKS = ["coding", "dialogue", "general", "math", "qa", "summarization"]


def ensure_page_config() -> None:
    """Set page config once per session."""
    if st.session_state.get("_page_configured"):
        return

    st.set_page_config(
        page_title="LDCC Hallucination Benchmark",
        page_icon="🔍",
        layout="wide",
    )
    st.session_state["_page_configured"] = True


def format_task_name(task: str) -> str:
    """태스크 이름 포맷팅 (qa -> QA)"""
    if task.lower() == "qa":
        return "QA"
    return task.capitalize()


def load_benchmark_data(
    use_hub: bool = True,
    benchmark_path: Path = Path("benchmark/benchmark.json")
) -> dict:
    """
    벤치마크 데이터 로드

    Args:
        use_hub: HuggingFace Hub에서 로드 (기본값: True)
        benchmark_path: 로컬 파일 경로 (use_hub=False일 때 사용)
    """
    if use_hub:
        from datasets import load_dataset
        try:
            dataset = load_dataset("jiwon9703/hallucination-benchmark-v1", split="test")
            data = [dict(sample) for sample in dataset]
        except Exception as e:
            st.warning(f"⚠️ Failed to load from HuggingFace Hub: {e}")
            st.info("📂 Falling back to local file...")
            with open(benchmark_path, encoding="utf-8") as f:
                data = json.load(f)
    else:
        with open(benchmark_path, encoding="utf-8") as f:
            data = json.load(f)

    # 태스크별 샘플 개수 계산
    task_counts = {}
    for sample in data:
        task = sample.get("task", "unknown")
        task = task[0].upper() + task[1:]
        if task == "Qa":
            task = "QA"
        task_counts[task] = task_counts.get(task, 0) + 1

    return {
        "total_samples": len(data),
        "task_counts": task_counts,
        "samples": data,
    }


def get_benchmark_info() -> dict:
    """Fetch benchmark info once and reuse across pages."""
    if "benchmark_info" not in st.session_state:
        st.session_state["benchmark_info"] = load_benchmark_data(use_hub=True)
    return st.session_state["benchmark_info"]


def render_sidebar(benchmark_info: dict) -> None:
    """Render shared sidebar content."""
    st.sidebar.title("📊 Benchmark Overview")
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Benchmark Info")
    st.sidebar.metric("**Total Samples**", benchmark_info["total_samples"])
    st.sidebar.markdown("**📈 Task Distribution:**")
    for task, count in sorted(benchmark_info["task_counts"].items()):
        st.sidebar.text(f"  • {task}: {count}개")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**😵‍💫 Hallucination Types:**")
    for hal_type in HALLUCINATION_TYPES:
        st.sidebar.text(f"  • {hal_type}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        LDCC Hallucination Benchmark v1.0<br>
        Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_results(result_path: Path) -> Optional[dict[str, Any]]:
    """벤치마크 결과 로드"""
    if not result_path.exists():
        return None

    with open(result_path, encoding="utf-8") as f:
        return json.load(f)


def parse_benchmark_results(results) -> pd.DataFrame:
    """
    벤치마크 결과를 파싱하여 DataFrame으로 변환

    결과는 리스트 또는 딕셔너리 형태일 수 있습니다:
    - 리스트: [{"task": "...", "hallucination_label": "...", ...}, ...]
    - 딕셔너리: {"samples": [...], "summary": {...}}
    """
    # 결과가 리스트인 경우 (2.benchmark_classify.py의 새로운 출력 형식)
    if isinstance(results, list):
        samples = results
    # 결과가 딕셔너리인 경우 (이전 형식)
    elif isinstance(results, dict):
        samples = results.get("samples", [])
    else:
        samples = []

    # 각 샘플에서 task와 hallucination_label 추출
    rows = []
    for sample in samples:
        rows.append({
            "task": sample.get("task", "unknown"),
            "label": sample.get("hallucination_label", sample.get("predicted_label", "unknown")),
        })

    return pd.DataFrame(rows)


def extract_samples(results_data: Any) -> list[dict[str, Any]]:
    """벤치마크 결과에서 샘플 리스트만 추출"""
    if isinstance(results_data, list):
        return results_data
    if isinstance(results_data, dict):
        return results_data.get("samples", [])
    return []


def load_outputs_samples(outputs_dir: Path = Path("outputs")) -> dict[str, list[dict[str, Any]]]:
    """outputs 폴더 내 benchmark_results.json 파일에서 샘플 로드"""
    samples_by_model: dict[str, list[dict[str, Any]]] = {}

    if not outputs_dir.exists():
        return samples_by_model

    for model_dir in sorted(outputs_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        result_path = model_dir / "benchmark_results.json"
        if not result_path.exists():
            continue

        try:
            results = load_results(result_path)
        except json.JSONDecodeError:
            continue

        if not results:
            continue

        samples = extract_samples(results)
        if samples:
            samples_by_model[model_dir.name] = samples

    return samples_by_model


def create_task_based_radar_charts(df: pd.DataFrame, model_name: str) -> go.Figure:
    """
    태스크별 육각형(레이더) 차트 생성

    각 태스크마다 하나의 레이더 차트
    각 꼭지점: 환각 타입
    값: 해당 환각 타입의 비율(%)
    """
    # 태스크 목록 (실제 데이터에 있는 태스크만)
    tasks_in_data = sorted(df["task"].unique())

    # 서브플롯 레이아웃 계산 (2행 3열)
    n_tasks = len(tasks_in_data)
    n_cols = 3
    n_rows = (n_tasks + n_cols - 1) // n_cols

    # 서브플롯 생성
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[format_task_name(task) for task in tasks_in_data],
        specs=[[{"type": "polar"}] * n_cols for _ in range(n_rows)],
    )

    # 각 태스크에 대해 레이더 차트 생성
    for idx, task in enumerate(tasks_in_data):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # 해당 태스크의 데이터만 필터링
        task_df = df[df["task"] == task]
        total_task = len(task_df)

        if total_task == 0:
            continue

        # 환각 타입별 비율 계산
        label_counts = task_df["label"].value_counts().to_dict()
        percentages = [
            (label_counts.get(hal_type, 0) / total_task * 100)
            for hal_type in HALLUCINATION_TYPES
        ]

        # 레이더 차트 추가
        fig.add_trace(
            go.Scatterpolar(
                r=percentages,
                theta=HALLUCINATION_TYPES,
                fill="toself",
                name=model_name,
                line=dict(color="royalblue", width=2),
                marker=dict(size=8),
            ),
            row=row,
            col=col,
        )

        # 각 서브플롯의 레이아웃 설정 (동적 범위 조정)
        max_val = max(percentages) if percentages else 0
        range_max = min(100, max(50, max_val * 1.2))  # 최소 50, 최대값의 120%, 상한 100
        fig.update_polars(
            radialaxis=dict(visible=True, range=[0, range_max]),
            row=row,
            col=col,
        )

    # 전체 레이아웃 설정
    fig.update_layout(
        height=300 * n_rows,
        showlegend=False,
        title_text=f"Task-based Hallucination Distribution: {model_name}",
    )

    return fig


def render_sample_viewer(
    model_samples: dict[str, list[dict[str, Any]]],
    title: Optional[str],
    empty_message: str,
    state_prefix: str,
    default_model: Optional[str] = None,
) -> None:
    """공통 샘플 분석 UI"""
    if title:
        st.title(title)

    if not model_samples:
        st.warning(empty_message)
        return

    model_names = sorted(model_samples.keys())

    select_key = f"{state_prefix}_model_select"
    if select_key in st.session_state and st.session_state[select_key] not in model_names:
        del st.session_state[select_key]

    default_index = 0
    if default_model and default_model in model_names:
        default_index = model_names.index(default_model)

    selected_model = st.selectbox(
        "📊 Select Model to Analyze",
        model_names,
        index=default_index,
        key=select_key,
    )

    samples = model_samples.get(selected_model, [])
    if not samples:
        st.warning("⚠️ No samples found in results")
        return

    st.markdown(f"**Total Samples:** {len(samples)}")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    # 태스크 필터
    all_tasks = sorted(set(s.get("task", "unknown") for s in samples))
    task_options = ["All"] + all_tasks
    task_key = f"{state_prefix}_task_filter"
    if task_key in st.session_state and st.session_state[task_key] not in task_options:
        del st.session_state[task_key]

    with col1:
        selected_task = st.selectbox(
            "Filter by Task",
            task_options,
            key=task_key,
        )

    # 환각 타입 필터
    with col2:
        selected_hal_type = st.selectbox(
            "Filter by Hallucination Type",
            ["All"] + HALLUCINATION_TYPES,
            key=f"{state_prefix}_hal_filter",
        )

    # 필터링 적용
    filtered_samples = samples
    if selected_task != "All":
        filtered_samples = [s for s in filtered_samples if s.get("task") == selected_task]
    if selected_hal_type != "All":
        filtered_samples = [s for s in filtered_samples if s.get("hallucination_label") == selected_hal_type]

    if not filtered_samples:
        st.warning("⚠️ No samples match the selected filters")
        return

    st.markdown(f"**Filtered Samples:** {len(filtered_samples)} / {len(samples)}")
    st.markdown("---")

    filter_key = f"{selected_model}_{selected_task}_{selected_hal_type}"
    filter_state_key = f"{state_prefix}_current_filter_key"
    sample_state_key = f"{state_prefix}_current_sample_idx"
    goto_state_key = f"{state_prefix}_goto_idx"
    pending_goto_key = f"{state_prefix}_pending_goto_idx"

    if filter_state_key not in st.session_state or st.session_state[filter_state_key] != filter_key:
        st.session_state[filter_state_key] = filter_key
        st.session_state[sample_state_key] = 0
        st.session_state[goto_state_key] = 0
        st.session_state.pop(pending_goto_key, None)

    if sample_state_key not in st.session_state:
        st.session_state[sample_state_key] = 0
    if goto_state_key not in st.session_state:
        st.session_state[goto_state_key] = st.session_state[sample_state_key]

    max_idx = len(filtered_samples) - 1

    if pending_goto_key in st.session_state:
        st.session_state[goto_state_key] = st.session_state.pop(pending_goto_key)

    st.session_state[sample_state_key] = min(st.session_state[sample_state_key], max_idx)
    st.session_state[goto_state_key] = min(st.session_state[goto_state_key], max_idx)

    # 필터링 여부에 따라 라벨 변경
    is_filtered = selected_task != "All" or selected_hal_type != "All"
    goto_label = "Go to Position #" if is_filtered else "Go to Sample #"

    with col3:
        st.number_input(
            goto_label,
            min_value=0,
            max_value=max_idx,
            step=1,
            key=goto_state_key,
            help="Filtered view: position in filtered list (0-based)" if is_filtered else "Original sample number (0-based)"
        )

    if st.session_state[sample_state_key] != st.session_state[goto_state_key]:
        st.session_state[sample_state_key] = st.session_state[goto_state_key]

    current_idx = st.session_state[sample_state_key]

    sample = filtered_samples[current_idx]
    hal_label = sample.get("hallucination_label", "Unknown")

    st.subheader(f"📄 Sample #{sample.get('index', current_idx)}")

    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        st.metric("Task", sample.get("task", "unknown").capitalize())
    with meta_col2:
        if hal_label == "No Hallucination":
            st.markdown(f"**Hallucination Type:** :green[{hal_label}]")
        else:
            st.markdown(f"**Hallucination Type:** :red[{hal_label}]")

    st.markdown("---")

    with st.expander("📝 Instruction & System Prompt", expanded=True):
        st.markdown("**Instruction:**")
        st.info(sample.get("instruction", "N/A"))

        system_input = sample.get("input", "")
        if system_input:
            st.markdown("**System Prompt:**")
            st.info(system_input)

    st.markdown("### 📊 Response Comparison")

    resp_col1, resp_col2 = st.columns(2)
    with resp_col1:
        st.markdown("#### ✅ Ground Truth (Reference Answer)")
        ground_truth = sample.get("gold_answer", sample.get("answer", "N/A"))
        st.success(ground_truth)
        st.caption(f"Length: {len(ground_truth)} characters")

    with resp_col2:
        st.markdown("#### 🤖 Model Output")
        model_output = sample.get("model_output", "N/A")
        if hal_label == "No Hallucination":
            st.success(model_output)
        else:
            st.error(model_output)
        st.caption(f"Length: {len(model_output)} characters")

    st.markdown("---")
    st.markdown("### 🎯 Classifier Analysis")

    analysis_col1, analysis_col2 = st.columns([2, 1])
    with analysis_col1:
        st.markdown("**Classifier Raw Output:**")
        classifier_raw = sample.get("classifier_raw_output", "N/A")
        st.code(classifier_raw, language=None)

    with analysis_col2:
        st.markdown("**Classified Label:**")
        if hal_label == "No Hallucination":
            st.success(hal_label)
        else:
            st.error(hal_label)

    if sample.get("reference"):
        st.markdown("---")
        with st.expander("📚 Additional Reference"):
            st.markdown(sample.get("reference"))

    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])

    previous_key = f"{state_prefix}_prev_btn"
    next_key = f"{state_prefix}_next_btn"

    with nav_col1:
        if st.button("⬅️ Previous", disabled=(current_idx == 0), key=previous_key):
            new_idx = max(0, current_idx - 1)
            st.session_state[sample_state_key] = new_idx
            st.session_state[pending_goto_key] = new_idx
            st.rerun()

    with nav_col2:
        st.markdown(
            f"<div style='text-align: center;'>{current_idx + 1} / {len(filtered_samples)}</div>",
            unsafe_allow_html=True,
        )

    with nav_col3:
        if st.button("Next ➡️", disabled=(current_idx == len(filtered_samples) - 1), key=next_key):
            new_idx = min(len(filtered_samples) - 1, current_idx + 1)
            st.session_state[sample_state_key] = new_idx
            st.session_state[pending_goto_key] = new_idx
            st.rerun()
