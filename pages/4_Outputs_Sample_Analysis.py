"""Sample analysis that reads from the outputs directory."""

from pathlib import Path

import streamlit as st

from app_utils import (
    ensure_page_config,
    get_benchmark_info,
    load_outputs_samples,
    render_sample_viewer,
    render_sidebar,
)


def render_outputs_sample_page() -> None:
    st.title("🗂️ Sample Analysis (Outputs Folder)")

    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"

    if not outputs_dir.exists():
        st.warning(f"⚠️ Outputs directory not found at {outputs_dir}.")
        return

    model_samples = load_outputs_samples(outputs_dir)
    if not model_samples:
        st.warning(f"⚠️ No benchmark_results.json files found in {outputs_dir}.")
        st.info("새롭게 실행한 결과는 `outputs_live/`에 저장됩니다. 이 페이지에서 확인하려면 해당 폴더의 결과를 `outputs/`로 복사하세요.")
        return

    render_sample_viewer(
        model_samples=model_samples,
        title=None,
        empty_message="⚠️ No samples available in outputs directory.",
        state_prefix="outputs",
    )


def main() -> None:
    ensure_page_config()
    benchmark_info = get_benchmark_info()
    render_sidebar(benchmark_info)
    render_outputs_sample_page()


if __name__ == "__main__":
    main()
