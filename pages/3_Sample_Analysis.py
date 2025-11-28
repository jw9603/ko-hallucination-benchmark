"""Sample-by-sample analysis using in-memory benchmark results."""

import streamlit as st

from app_utils import (
    ensure_page_config,
    extract_samples,
    get_benchmark_info,
    render_sample_viewer,
    render_sidebar,
)


def render_sample_analysis_page() -> None:
    st.title("🔍 Sample-by-Sample Analysis")

    if "all_results" not in st.session_state or not st.session_state["all_results"]:
        st.warning("⚠️ No results available. Please run a benchmark first.")
        st.info("👈 Go to 'Main' page to run a benchmark")
        return

    model_samples: dict[str, list[dict]] = {}
    for model, results in st.session_state["all_results"].items():
        samples = extract_samples(results)
        if samples:
            model_samples[model] = samples

    if not model_samples:
        st.warning("⚠️ No samples found in the loaded results.")
        return

    render_sample_viewer(
        model_samples=model_samples,
        title=None,
        empty_message="⚠️ No samples available.",
        state_prefix="sample",
        default_model=st.session_state.get("current_model"),
    )


def main() -> None:
    ensure_page_config()
    benchmark_info = get_benchmark_info()
    render_sidebar(benchmark_info)
    render_sample_analysis_page()


if __name__ == "__main__":
    main()
