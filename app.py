#!/usr/bin/env python3
"""
LDCC Hallucination Benchmark - Streamlit Web Interface (Main Page)
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from app_utils import ensure_page_config, get_benchmark_info, load_results, render_sidebar


def run_benchmark_step(
    cmd: list[str],
    step_name: str,
    progress_bar,
    status_text,
) -> bool:
    """
    벤치마크 스텝 실행 (실시간 출력)
    """
    status_text.text(f"🔄 {step_name}...")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # 실시간 출력 표시
    output_container = st.expander(f"📜 {step_name} Logs", expanded=False)
    output_lines = []

    for line in iter(process.stdout.readline, ''):
        if line:
            output_lines.append(line)
            # 마지막 10줄만 표시
            output_container.code('\n'.join(output_lines[-10:]), language='bash')

    process.wait()

    if process.returncode == 0:
        status_text.text(f"✅ {step_name} completed!")
        return True
    else:
        status_text.error(f"❌ {step_name} failed!")
        return False


def run_benchmark(
    model_name: str,
    use_lora: bool = False,
    lora_path: str = "",
    use_quantization: bool = False,
    quantization_type: str = "awq",
    classifier_option: str = "LoRA Fine-tuned (chahyunmook/kanana-1.5-8b-instruct-2505-r64-lora)",
    max_model_len: Optional[int] = None,
) -> tuple[Optional[dict[str, Any]], Optional[Path]]:
    """
    벤치마크 실행

    1. 1.benchmark_generate.py - 타겟 모델 생성
    2. 2.benchmark_classify.py - 환각 분류
    """
    st.info(f"🚀 Starting benchmark for model: {model_name}")

    # 모델 이름을 파일 시스템 안전한 이름으로 변환
    safe_model_name = model_name.replace("/", "_").replace(":", "_")

    # 현재 스크립트의 디렉토리를 기준으로 경로 설정
    script_dir = Path(__file__).parent
    benchmark_path = script_dir / "benchmark" / "benchmark.json"

    # 모델별 디렉토리 생성 (live runs are stored under outputs_live)
    outputs_live_dir = script_dir / "outputs_live"
    model_output_dir = outputs_live_dir / safe_model_name
    target_output_path = model_output_dir / "target_outputs.json"
    result_output_path = model_output_dir / "benchmark_results.json"

    # HuggingFace에서 벤치마크 데이터셋 다운로드
    st.info("📥 Loading benchmark dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("jiwon9703/ko-hallucination-benchmark", split="test")
        benchmark_data = [dict(sample) for sample in dataset]

        # benchmark 디렉토리 생성 및 저장
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, ensure_ascii=False, indent=2)

        st.success(f"✅ Loaded {len(benchmark_data)} samples from HuggingFace")
    except Exception as e:
        st.error(f"❌ Failed to load benchmark from HuggingFace: {e}")
        return None, None

    # outputs 디렉토리 생성
    target_output_path.parent.mkdir(parents=True, exist_ok=True)

    # 프로그레스 바
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 모델 캐싱 (vLLM이 접근하기 전에 미리 다운로드)
    status_text.text("📦 Pre-caching model from HuggingFace...")
    try:
        from transformers import AutoConfig
        _ = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        st.success(f"✅ Model {model_name} cached successfully")
    except Exception as e:
        st.error(f"❌ Failed to cache model: {e}")
        return None, None

    # Step 1: 타겟 모델 생성
    generate_cmd = [
        "python", "1.benchmark_generate.py",
        "--benchmark", str(benchmark_path),
        "--target-model", model_name,
        "--output-target", str(target_output_path),
        "--dtype", "auto",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.7",
        "--trust-remote-code",
    ]

    # Max Model Length 옵션 추가
    if max_model_len is not None:
        generate_cmd.extend(["--max-model-len", str(max_model_len)])
        st.info(f"🔧 Using max_model_len: {max_model_len}")

    # LoRA 옵션 추가
    if use_lora and lora_path:
        generate_cmd.extend(["--target-lora", lora_path])

    # Quantization 옵션 추가
    if use_quantization:
        generate_cmd.extend(["--quantization", quantization_type])

    progress_bar.progress(10)
    if not run_benchmark_step(
        generate_cmd,
        "Step 1/2: Generating target model outputs",
        progress_bar,
        status_text,
    ):
        return None, None

    progress_bar.progress(50)

    # Step 2: 환각 분류
    # 분류기 모델 설정
    if "LoRA" in classifier_option:
        # LoRA Fine-tuned 모델
        classifier_base = "kakaocorp/kanana-1.5-8b-instruct-2505"
        classifier_lora = "chahyunmook/kanana-1.5-8b-instruct-2505-r64-lora"
    else:
        # Full Fine-tuned 모델
        classifier_base = "jiwon9703/Phi-4-mini-instruct-ko-hallucination-sft-v3"
        classifier_lora = None

    classify_cmd = [
        "python", "2.benchmark_classify.py",
        "--benchmark", str(benchmark_path),
        "--target-outputs", str(target_output_path),
        "--classifier-base", classifier_base,
        "--output", str(result_output_path),
        "--dtype", "auto",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.9",
    ]

    # LoRA 분류기인 경우 LoRA 경로 추가 및 trust-remote-code 플래그
    if classifier_lora:
        classify_cmd.extend(["--classifier-lora", classifier_lora])
        classify_cmd.append("--trust-remote-code")

    if not run_benchmark_step(
        classify_cmd,
        "Step 2/2: Classifying hallucinations",
        progress_bar,
        status_text,
    ):
        return None, None

    progress_bar.progress(100)
    status_text.success("🎉 All steps completed!")

    # 결과 로드
    return load_results(result_output_path), model_output_dir


def render_main_page() -> None:
    """메인 페이지: 모델 설정 및 벤치마크 실행"""
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://i.namu.wiki/i/pLUNbB1ntuLgPB2xzhK9u2qgGue6Dnbcx8oKgGVCAnbonNgcBJuCBloVzx5MxqSgpGrwPXkQI-dwoX3dfIGY3675K96XxdYffAHq3E-CC3thGy8oE-Rh6s1_dXI0Gzqfg29jYK1-PnhUXBYIj9YHOw.svg" width="280"/>
            <h1 style="font-size: 2.8rem; font-weight:700; margin-top:10px;">
                LDCC Hallucination Benchmark
            </h1>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("""
    **👥 Project Team**

    Developed by **Language AI Team**, LOTTE INNOVATE AI Tech LAB
    - Jiwon Jeong PRO
    - Hyunmook Cha PRO
    """)
    st.markdown("---")

    with st.expander("📊 Classifier Model Performance", expanded=False):
        st.markdown("""
        **Hallucination Classifier Accuracy on Test Set**

        Evaluated on [ko-hallucination-sft-v3](jiwon9703/Phi-4-mini-instruct-ko-hallucination-sft-v3) (100 samples)
        """)

        # HTML 테이블로 렌더링 (Our Classifier 행 위에 구분선 추가)
        html_table = """
        <style>
        .perf-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        .perf-table th {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            text-align: left;
            border-bottom: 2px solid #333;
            font-weight: 600;
        }
        .perf-table td {
            padding: 8px 10px;
            border-bottom: 1px solid #555;
        }
        .perf-table tr.separator {
            border-top: 3px double #FFA500;
        }
        </style>
        <table class="perf-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Samples</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>GPT-3.5-Turbo</td><td>34.0%</td><td>100</td></tr>
            <tr><td>GPT-4o-mini</td><td>52.0%</td><td>100</td></tr>
            <tr><td>GPT-4o</td><td>52.0%</td><td>100</td></tr>
            <tr><td>Grok 4.1</td><td>53.0%</td><td>100</td></tr>
            <tr><td>Claude Opus 4.1</td><td>55.0%</td><td>100</td></tr>
            <tr><td>Grok 4</td><td>63.0%</td><td>100</td></tr>
            <tr><td>GPT-5.1</td><td>69.0%</td><td>100</td></tr>
            <tr><td>Claude Opus 4.5</td><td><b>70.0%</b></td><td>100</td></tr>
            <tr><td>Gemini 2.5 Flash</td><td><b>70.0%</b></td><td>100</td></tr>
            <tr><td>Gemini 2.0 Flash</td><td><b>73.0%</b></td><td>100</td></tr>
            <tr class="separator"><td><b>Our Classifier (LoRA)</b></td><td><b>61.0%</b></td><td>100</td></tr>
            <tr><td><b>Our Classifier (Full Fine-Tuned)</b></td><td><b>64.0%</b></td><td>100</td></tr>
        </tbody>
        </table>
        """
        st.markdown(html_table, unsafe_allow_html=True)

        st.info("💡 Our classifier (chahyunmook/kanana-1.5-8b-instruct-2505-r64-lora and jiwon9703/Phi-4-mini-instruct-ko-hallucination-sft-v3) achieves competitive performance compared to GPT-4o and GPT-5.1.")

        st.markdown("---")

    # 메인 컨텐츠
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🤖 Model Configuration")

        # 빠른 모델 선택
        with st.expander("🔍 Quick Model Selector", expanded=False):
            st.markdown("**Popular Korean LLMs by Company:**")

            # 기업별 모델 정보
            companies = [
                {
                    "logo": "https://i.namu.wiki/i/pLUNbB1ntuLgPB2xzhK9u2qgGue6Dnbcx8oKgGVCAnbonNgcBJuCBloVzx5MxqSgpGrwPXkQI-dwoX3dfIGY3675K96XxdYffAHq3E-CC3thGy8oE-Rh6s1_dXI0Gzqfg29jYK1-PnhUXBYIj9YHOw.svg",
                    "name": "LOTTE INNOVATE",
                    "url": "https://huggingface.co/LDCC",
                    "models": [],
                },
                {
                    "logo": "https://i.namu.wiki/i/eUPJizKzvkNGRSCvI7CmaWF4AstwgmbTwU9Prho69T--hj20XBLuJz-2xDFLWaO1rmc7Q_mAeJPXIoOLvN5lE-J-lF-6pSGqrajdB4qK26ZZp8ftPJcGAja5AzmQbfnt1fR5mfMGjV7rNxgr9hFtTA.webp",
                    "name": "NC AI",
                    "url": "https://huggingface.co/NCSOFT",
                    "models": ["NCSOFT/Llama-VARCO-8B-Instruct"]
                },
                {
                    "logo": "https://i.namu.wiki/i/alncWPsUkVb-nNhTAwJTaKdjHT2sZHhuiTmanflD26DBYcBtKb9aU25CxH16bda6PH6-hLJTRMi5Rocncy3VrgHxTXZrYcmDBrIha4aJCXRdod8EhUaxTePglIeMtzgpKxvyVDwGJ1pnO2DWizA31Q.svg",
                    "name": "Upstage",
                    "url": "https://huggingface.co/upstage",
                    "models": ["upstage/SOLAR-10.7B-Instruct-v1.0", "upstage/SOLAR-10.7B-v1.0"]
                },
                {
                    "logo": "https://i.namu.wiki/i/0GT0O7NKUAHXD_6CI69nW3KQ5XIkj2y10_7lwYoVkbgFEXb7qxE5aDCImwpJ3eObD-hvCSAtUV7b4JDgKn6jpnn4wj_z6cECh0NTdyhrDf0gHK2RjenmfGAlRxkRlSdtvs_DTfSGPVjkhWjnAjUM6Q.svg",
                    "name": "SKT",
                    "url": "https://huggingface.co/skt",
                    "models": ["skt/A.X-4.0-Light", "skt/A.X-3.1-Light"]
                },
                {
                    "logo": "https://i.namu.wiki/i/fjo3Z9ZYz2QX__OCYguIQNYfWlB3FGKrHR83_OI2vmsIUN0oIUHpuVQ-1nYe-6dozVbA92rKyVrvQNmwEV2NxDGECPDNS1KGbAN72Qyt4w_5BNLmk2t3VD3DSm0HZLdAjevEtmqXPqsxFMa1wlHedg.svg",
                    "name": "Naver",
                    "url": "https://huggingface.co/naver-hyperclovax",
                    "models": ["naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"]
                },
                {
                    "logo": "https://i.namu.wiki/i/ph4BYEj7RzbUHsza3RJGOjGH0SmoXieUdWvPz5FSVGr625YwCnoHRKAZQEoNW73JQSO3-fXYltks0OGPJ_VqW9FsciyD-MB0yDcfUtoN80fdzc63WyGcrSGq5c_0BuxQV-t6KK4E6ZUehWzEihba0Q.webp",
                    "name": "LG AI Research",
                    "url": "https://huggingface.co/LGAI-EXAONE",
                    "models": ["LGAI-EXAONE/EXAONE-4.0-1.2B", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"]
                },
                {
                    "logo": "https://i.namu.wiki/i/6M4f7652U3PKkSHbzDeU-bdgW3Z9reeyeCyXpKmsmRrtvBd5fHgJoF_SRNMw3lcLMqZ7Z-ZOtbyWu_9GvNuJ77rw3RJI_R6-cra6N31CzOESg-zP4fE_UtoQjiTRxzh_reS9mDU7L5l83oxEaAxJsQ.svg",
                    "name": "Kakao",
                    "url": "https://huggingface.co/kakaocorp",
                    "models": ["kakaocorp/kanana-1.5-8b-base", "kakaocorp/kanana-1.5-8b-instruct-2505", "kakaocorp/kanana-1.5-2.1b-base", "kakaocorp/kanana-1.5-2.1b-instruct-2505", "kakaocorp/kanana-nano-2.1b-base", "kakaocorp/kanana-nano-2.1b-instruct"]
                },
                {
                    "logo": "https://i.namu.wiki/i/aueNAyNdYEFEDmV6uznKaKz9DYgDNfVAUPp3jqUoagRptHOVkpv75kY8tjLVUihEuKLvHkE7p3Al2l4qMsDdVWxmZOgBHmRaH6Fh1RoOLLE6rqVTTAgjVuw7QEpchRtWiwgsK0svgBzye9tpXyxO5g.svg",
                    "name": "KT",
                    "url": "https://huggingface.co/K-intelligence",
                    "models": ["K-intelligence/Midm-2.0-Base-Instruct", "K-intelligence/Midm-2.0-Mini-Instruct"]
                },
                {
                    "logo": "https://i.namu.wiki/i/K6hf9HNPwRTeNGtOFxJXVQwkF-xOJfoRoFgPl64xb8knveN1ofMNYs_R2pIPwQkD5Rt4WlpAQkK6XTq-DWI22A.svg",
                    "name": "야놀자",
                    "url": "https://huggingface.co/yanolja",
                    "models": ["yanolja/YanoljaNEXT-EEVE-Instruct-10.8B", "yanolja/YanoljaNEXT-EEVE-Instruct-2.8B"]
                }
            ]

            for company in companies:
                with st.expander(f"**{company['name']}**", expanded=False):
                    spacer_col, logo_col, info_col = st.columns([0.3, 1, 4])
                    with spacer_col:
                        st.write("")
                    with logo_col:
                        if company['name'] == "LOTTE INNOVATE":
                            logo_width = 160
                        elif company['name'] in ["NC AI", "SKT", "Upstage", "Naver", "LG AI Research", "야놀자", "Kakao"]:
                            logo_width = 120
                        else:
                            logo_width = 60
                        st.image(company["logo"], width=logo_width)
                    with info_col:
                        st.markdown(f"**[{company['name']}]({company['url']})** 🔗")
                        st.markdown(f"**{len(company['models'])} models available**")

                    st.markdown("---")

                    for model in company["models"]:
                        model_col, btn_col = st.columns([5, 1])
                        with model_col:
                            st.code(model, language=None)
                        with btn_col:
                            if st.button("📋", key=f"select_{model}", help="Click to use this model"):
                                st.session_state["selected_model"] = model
                                st.rerun()

            st.markdown("---")
            st.markdown("**Global Open-Source LLMs:**")

            global_companies = [
                {
                    "logo": "https://i.namu.wiki/i/PvqeGxPkldKis1fUOoavKKK0j9IRcJO6h2d8iPwuZxFbcXcC0jTD05GSy7bE2tyCgOuF4pXU6w01MNNfrlrKOetvA_NtQWtefvLb1xcsICG2JJrVmKEQyOeZQdOsfKlXyiAwEJEdqH08IEZLvzffgA.svg",
                    "name": "Google",
                    "url": "https://huggingface.co/google",
                    "models": [
                        "google/gemma-3-4b-it",
                        "google/gemma-2-9b-it",
                    ]
                },
                {
                    "logo": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Microsoft_logo_%282012%29.svg/512px-Microsoft_logo_%282012%29.svg.png",
                    "name": "Microsoft",
                    "url": "https://huggingface.co/microsoft",
                    "models": [
                        "microsoft/Phi-4-mini-instruct",
                        "microsoft/Phi-3.5-mini-instruct",
                    ]
                },
                {
                    "logo": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Meta_Platforms_Inc._logo.svg/200px-Meta_Platforms_Inc._logo.svg.png",
                    "name": "Meta",
                    "url": "https://huggingface.co/meta-llama",
                    "models": [
                        "meta-llama/Llama-3.2-3B-Instruct",
                        "meta-llama/Llama-3.2-1B-Instruct",
                        "meta-llama/Llama-3.1-8B-Instruct"
                    ]
                },
                {
                    "logo": "https://i.namu.wiki/i/iifhXt8GRre1RBkl-ZnWt8UCsnME7QEJa8CSMfD5pQyKvu6ckVPNimm-wuQwmLiWXfpyeSwVga7RVCVlIgb4Y0d1OZCWownWDDoI-g2THloZu2bkgGa_d2UsHcjLo0eTHGvNFIVmTKZWPy1GcZ-8zg.svg",
                    "name": "Mistral AI",
                    "url": "https://huggingface.co/mistralai",
                    "models": [
                        "mistralai/Mistral-7B-Instruct-v0.3",
                        "mistralai/Mistral-7B-v0.3",
                    ]
                },
                {
                    "logo": "https://i.namu.wiki/i/ULAdve4TEDTxQa9HnXsYlbKWApM9dtCPYLlw59-AoPUlq0MYDA8wpivH-jV-JN3Un5kJZscjc2XUSs8nVpW7lqvoQsKkIIhQUqu8nBXuTSG4D_RSZj_7SdgUsNfW3JjrdVZ3p0a1pKk4M82MnqbdVw.webp",
                    "name": "Qwen (Alibaba)",
                    "url": "https://huggingface.co/Qwen",
                    "models": [
                        "Qwen/Qwen3-4B-Instruct-2507",
                        "Qwen/Qwen2.5-7B-Instruct",
                    ]
                },
                {
                    "logo": "https://i.namu.wiki/i/PKpjKr5w63az9bePhlrpTf3F-sOM4e4nPcQNEdoStiOUqVfZ5s5wN1ZDj2cgGIQH7bzeB-G2wHap6da-y6nf3yr4AudrQ0HYFNF1-FrCBMsqrA2BIInv8AynRR41usWsi2OL6SsQAWTjTrNyyg87Fw.svg",
                    "name": "DeepSeek AI",
                    "url": "https://huggingface.co/deepseek-ai",
                    "models": [
                        "deepseek-ai/deepseek-llm-7b-chat",
                        "deepseek-ai/deepseek-llm-7b-base",
                    ]
                }
            ]

            for company in global_companies:
                with st.expander(f"**{company['name']}**", expanded=False):
                    spacer_col, logo_col, info_col = st.columns([0.3, 1, 4])
                    with spacer_col:
                        st.write("")
                    with logo_col:
                        logo_width = 120
                        st.image(company["logo"], width=logo_width)
                    with info_col:
                        st.markdown(f"**[{company['name']}]({company['url']})** 🔗")
                        st.markdown(f"**{len(company['models'])} models available**")

                    st.markdown("---")

                    for model in company["models"]:
                        model_col, btn_col = st.columns([5, 1])
                        with model_col:
                            st.code(model, language=None)
                        with btn_col:
                            if st.button("📋", key=f"select_global_{model}", help="Click to use this model"):
                                st.session_state["selected_model"] = model
                                st.rerun()

        default_model = st.session_state.get("selected_model", "")
        model_name = st.text_input(
            "**HuggingFace Model ID**",
            value=default_model,
            placeholder="e.g., NCSOFT/Llama-VARCO-8B-Instruct",
            help="Enter HuggingFace model ID (NOT full URL). Format: username/model-name",
        )

    with col2:
        st.subheader("⚙️ Target Model Settings")

        use_lora = st.checkbox("Use LoRA", value=False, key="target_lora")
        lora_path = ""
        if use_lora:
            lora_path = st.text_input(
                "LoRA Path",
                placeholder="e.g., daebakgazua/250526_OhLoRA_LLM_kanana",
                key="target_lora_path",
                help="HuggingFace LoRA adapter path (username/repo-name) or local directory path"
            )
            with st.expander("ℹ️ How to use LoRA?"):
                st.markdown("""
                **Example:**
                - Base Model: `kakaocorp/kanana-1.5-8b-instruct-2505`
                - LoRA Path: `daebakgazua/250526_OhLoRA_LLM_kanana`

                vLLM will load the base model and apply the LoRA adapter on top.
                """)

        use_quantization = st.checkbox("Use Quantization", value=False, key="target_quant")
        quantization_type = "awq"
        if use_quantization:
            quantization_type = st.selectbox(
                "Quantization Type",
                ["awq", "gptq", "squeezellm"],
                help="vLLM supported quantization methods",
                key="target_quant_type",
            )
            with st.expander("ℹ️ What is Quantization?"):
                st.markdown("""
                **Quantization** reduces model size and speeds up inference by using lower precision (e.g., INT4 instead of FP16).

                **Supported formats (vLLM):**
                - **AWQ**: Activation-aware Weight Quantization
                - **GPTQ**: GPT Quantization (widely used)
                - **SqueezeLLM**: Optimized for memory efficiency

                ⚠️ Your model must be **pre-quantized** with the selected method.

                ❌ **GGUF format is NOT supported** by vLLM (use llama.cpp/Ollama instead).
                """)

        use_max_model_len = st.checkbox(
            "Limit Max Context Length (Input + Output)",
            value=False,
            key="use_max_model_len",
            help="Enable this to prevent KV cache memory errors on long-context models"
        )
        max_model_len = None
        if use_max_model_len:
            max_model_len = st.number_input(
                "Max Model Length",
                min_value=512,
                max_value=32768,
                value=8192,
                step=512,
                key="max_model_len_value",
                help="Maximum context length. Recommended: 4096-8192 for most tasks"
            )
            with st.expander("ℹ️ When to use this?"):
                st.markdown("""
                **Use this option when:**
                - You get KV cache memory errors (e.g., "16.00 GiB KV cache is needed...")
                - Running long-context models (Llama-3.1 with 131K context)

                **Recommended values:**
                - **4096**: Safe for most GPUs, sufficient for most benchmark tasks
                - **8192**: Good balance between memory and capability
                - **16384**: For tasks requiring longer context

                ⚠️ Most benchmark tasks don't need more than 8192 tokens.
                """)

    st.markdown("---")

    col3, col4 = st.columns([2, 1])

    with col3:
        st.subheader("🔍 Classifier Model")
        classifier_option = st.radio(
            "Select Classifier:",
            [
                "LoRA Fine-tuned (chahyunmook/kanana-1.5-8b-instruct-2505-r64-lora)",
                "Full Fine-tuned (jiwon9703/Phi-4-mini-instruct-ko-hallucination-sft-v3)",
            ],
            index=0,
            key="classifier_option",
        )

    with col4:
        st.markdown("##### Classifier Info")
        if "LoRA" in classifier_option:
            st.info("🎯 LoRA 방식\n- Base: kanana-1.5-8b\n- Adapter: r64")
        else:
            st.info("🔧 Full Fine-tuning\n- Phi-4-mini 기반\n- SFT v3")

    st.markdown("---")
    run_button = st.button("🚀 Run Benchmark", type="primary", use_container_width=True)

    if run_button:
        if not model_name:
            st.error("⚠️ Please enter a model name!")
        else:
            results, artifacts_dir = run_benchmark(
                model_name=model_name,
                use_lora=use_lora,
                lora_path=lora_path,
                use_quantization=use_quantization,
                quantization_type=quantization_type,
                classifier_option=classifier_option,
                max_model_len=max_model_len,
            )

            if results:
                if "all_results" not in st.session_state:
                    st.session_state["all_results"] = {}

                st.session_state["all_results"][model_name] = results
                st.session_state["current_model"] = model_name
                st.success("🎉 Benchmark completed successfully!")
                if artifacts_dir:
                    st.caption(f"📁 Artifacts saved to {artifacts_dir}")
                st.info("👈 Go to the other pages to view detailed results")


def main() -> None:
    """Entrypoint for the Streamlit app."""
    ensure_page_config()
    benchmark_info = get_benchmark_info()
    render_sidebar(benchmark_info)
    render_main_page()


if __name__ == "__main__":
    main()
