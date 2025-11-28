# #!/usr/bin/env python3
# """Run target model on benchmark and save its outputs only."""

# from __future__ import annotations

# import argparse
# import json
# import pathlib
# from typing import Any

# from vllm import LLM, SamplingParams


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Run target model on benchmark and save outputs."
#     )
#     parser.add_argument(
#         "--benchmark",
#         type=pathlib.Path,
#         default=pathlib.Path("final_benchmark.json"),
#         help="Benchmark JSON file (list of samples).",
#     )
#     parser.add_argument(
#         "--target-model",
#         required=True,
#         help="Model ID/path for the model being evaluated.",
#     )
#     parser.add_argument(
#         "--target-max-new-tokens",
#         type=int,
#         default=512,
#         help="Max tokens for the target model outputs.",
#     )
#     parser.add_argument(
#         "--target-temperature",
#         type=float,
#         default=0.7,
#         help="Sampling temperature for the target model.",
#     )
#     parser.add_argument(
#         "--target-top-p",
#         type=float,
#         default=0.9,
#         help="Top-p setting for the target model.",
#     )
#     parser.add_argument(
#         "--max-samples",
#         type=int,
#         default=None,
#         help="Limit number of benchmark samples processed.",
#     )
#     parser.add_argument(
#         "--output-target",
#         type=pathlib.Path,
#         default=pathlib.Path("outputs/target_outputs.json"),
#         help="Where to save target model outputs.",
#     )
#     parser.add_argument(
#         "--dtype",
#         default="auto",
#         help="vLLM dtype (auto, half, float16, bfloat16, float32).",
#     )
#     parser.add_argument(
#         "--tensor-parallel-size",
#         type=int,
#         default=1,
#         help="Tensor parallelism for vLLM.",
#     )
#     parser.add_argument(
#         "--gpu-memory-utilization",
#         type=float,
#         default=0.9,
#         help="GPU memory utilization hint for vLLM.",
#     )
#     parser.add_argument(
#         "--trust-remote-code",
#         action="store_true",
#         help="Allow custom model code when loading via vLLM.",
#     )
#     parser.add_argument(
#         "--target-lora",
#         type=str,
#         default=None,
#         help="LoRA adapter path for the target model.",
#     )
#     parser.add_argument(
#         "--quantization",
#         type=str,
#         default=None,
#         choices=["awq", "squeezellm", "gptq"],
#         help="Quantization method (awq, squeezellm, gptq).",
#     )
#     return parser.parse_args()


# def load_benchmark(path: pathlib.Path, limit: int | None) -> list[dict[str, Any]]:
#     data = json.loads(path.read_text(encoding="utf-8"))
#     return data[:limit] if limit is not None else data


# def build_generation_prompt(instruction: str, input_text: str | None) -> str:
#     input_block = f"\n\n### Input:\n{input_text.strip()}" if input_text else ""
#     return f"### Instruction:\n{instruction.strip()}{input_block}\n\n### Response:\n"


# def run_vllm_generation(
#     llm: LLM,
#     prompts: list[str],
#     sampling_params: SamplingParams,
# ) -> list[str]:
#     outputs = llm.generate(prompts, sampling_params=sampling_params)
#     results: list[str] = []
#     for out in outputs:
#         if not out.outputs:
#             results.append("")
#             continue
#         results.append(out.outputs[0].text.strip())
#     return results


# def main() -> None:
#     args = parse_args()
#     samples = load_benchmark(args.benchmark, args.max_samples)
#     if not samples:
#         raise SystemExit("Benchmark set is empty.")
#     print(f"Loaded {len(samples)} sample(s) from {args.benchmark}")

#     print(f"Loading target model with vLLM: {args.target_model}")

#     # LLM 초기화 파라미터 구성
#     llm_kwargs = {
#         "model": args.target_model,
#         "dtype": args.dtype,
#         "tensor_parallel_size": args.tensor_parallel_size,
#         "gpu_memory_utilization": args.gpu_memory_utilization,
#         "trust_remote_code": args.trust_remote_code,
#         "max_num_seqs": 256,  # 배치 크기 증가로 처리량 향상
#     }

#     # LoRA 어댑터 추가
#     if args.target_lora:
#         print(f"  └─ Loading LoRA adapter: {args.target_lora}")
#         llm_kwargs["enable_lora"] = True
#         llm_kwargs["max_lora_rank"] = 64

#     # Quantization 추가
#     if args.quantization:
#         print(f"  └─ Using quantization: {args.quantization}")
#         llm_kwargs["quantization"] = args.quantization

#     llm = LLM(**llm_kwargs)

#     sampling = SamplingParams(
#         max_tokens=args.target_max_new_tokens,
#         temperature=args.target_temperature,
#         top_p=args.target_top_p,
#         n=1,
#     )

#     prompts = [
#         build_generation_prompt(sample["instruction"], sample.get("input"))
#         for sample in samples
#     ]

#     # LoRA를 사용하는 경우, LoRARequest 생성
#     if args.target_lora:
#         from vllm.lora.request import LoRARequest
#         lora_request = LoRARequest("target_lora", 1, args.target_lora)
#         print(f"Generating with LoRA adapter...")
#         outputs_objs = llm.generate(prompts, sampling, lora_request=lora_request)
#         outputs = []
#         for out in outputs_objs:
#             if not out.outputs:
#                 outputs.append("")
#                 continue
#             outputs.append(out.outputs[0].text.strip())
#     else:
#         outputs = run_vllm_generation(llm, prompts, sampling)

#     print("Target generation complete.")

#     # 먼저 /tmp에 저장 (디스크 쿼터 회피)
#     temp_output = pathlib.Path(f"/tmp/{args.output_target.name}")
#     temp_output.write_text(
#         json.dumps(outputs, ensure_ascii=False, indent=2) + "\n",
#         encoding="utf-8",
#     )
#     print(f"Saved to temp: {temp_output}")

#     # 최종 위치 디렉토리 생성 및 심볼릭 링크
#     args.output_target.parent.mkdir(parents=True, exist_ok=True)

#     # 기존 파일/링크 제거 후 심볼릭 링크 생성
#     if args.output_target.exists() or args.output_target.is_symlink():
#         args.output_target.unlink()
#     args.output_target.symlink_to(temp_output)

#     print(f"Saved target outputs to {args.output_target} -> {temp_output}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""Run target model on benchmark and save its outputs only."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run target model on benchmark and save outputs."
    )
    parser.add_argument(
        "--benchmark",
        type=pathlib.Path,
        default=pathlib.Path("final_benchmark.json"),
        help="Benchmark JSON file (list of samples).",
    )
    parser.add_argument(
        "--target-model",
        required=True,
        help="Model ID/path for the model being evaluated.",
    )
    parser.add_argument(
        "--target-max-new-tokens",
        type=int,
        default=512,
        help="Max tokens for the target model outputs.",
    )
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the target model.",
    )
    parser.add_argument(
        "--target-top-p",
        type=float,
        default=0.9,
        help="Top-p setting for the target model.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of benchmark samples processed.",
    )
    parser.add_argument(
        "--output-target",
        type=pathlib.Path,
        default=pathlib.Path("outputs/target_outputs.json"),
        help="Where to save target model outputs.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="vLLM dtype (auto, half, float16, bfloat16, float32).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism for vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization hint for vLLM.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length. If not set, vLLM auto-detects from model config. Use this to limit KV cache memory usage.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code when loading via vLLM.",
    )
    parser.add_argument(
        "--target-lora",
        type=str,
        default=None,
        help="LoRA adapter path for the target model.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "squeezellm", "gptq"],
        help="Quantization method (awq, squeezellm, gptq).",
    )
    return parser.parse_args()


def load_benchmark(path: pathlib.Path, limit: int | None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[:limit] if limit is not None else data


def build_generation_prompt(instruction: str, input_text: str | None) -> str:
    input_block = f"\n\n### Input:\n{input_text.strip()}" if input_text else ""
    return f"### Instruction:\n{instruction.strip()}{input_block}\n\n### Response:\n"


def run_vllm_generation(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> list[str]:
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    results: list[str] = []
    for out in outputs:
        if not out.outputs:
            results.append("")
            continue
        results.append(out.outputs[0].text.strip())
    return results


def truncate_to_max_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """토큰 길이가 max_tokens 이하가 될 때까지 안전하게 잘라내는 함수."""
    while True:
        enc = tokenizer(text, add_special_tokens=False)
        if len(enc["input_ids"]) <= max_tokens:
            return text
        truncated_ids = enc["input_ids"][:max_tokens]
        text = tokenizer.decode(truncated_ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    samples = load_benchmark(args.benchmark, args.max_samples)
    if not samples:
        raise SystemExit("Benchmark set is empty.")
    print(f"Loaded {len(samples)} sample(s) from {args.benchmark}")

    print(f"Loading target model with vLLM: {args.target_model}")

    # LLM 초기화 파라미터 구성
    llm_kwargs = {
        "model": args.target_model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "max_num_seqs": 256,
        "disable_custom_all_reduce": True,  # speculative decoding 관련 에러 방지
    }

    # max_model_len 설정 (지정된 경우에만)
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
        print(f"  └─ max_model_len set to: {args.max_model_len}")
    else:
        # vLLM이 모델의 max_position_embeddings를 자동 감지하도록 함
        # 매우 긴 context 모델(Llama-3.1의 131K 등)에서 KV cache 메모리 부족이 발생할 수 있음
        print(f"  └─ max_model_len: auto-detect from model config")

    # LoRA 어댑터 추가
    if args.target_lora:
        print(f"  └─ Loading LoRA adapter: {args.target_lora}")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    # Quantization 추가
    if args.quantization:
        print(f"  └─ Using quantization: {args.quantization}")
        llm_kwargs["quantization"] = args.quantization

    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        max_tokens=args.target_max_new_tokens,
        temperature=args.target_temperature,
        top_p=args.target_top_p,
        n=1,
    )

    # 기본 프롬프트 구성
    prompts = [
        build_generation_prompt(sample["instruction"], sample.get("input"))
        for sample in samples
    ]

    # --- 여기서부터 프롬프트 길이 제한 로직 추가 ---
    tokenizer = llm.get_tokenizer()
    max_context_len = llm.llm_engine.model_config.max_model_len

    # 템플릿/스페셜 토큰/여유분을 위해 약간의 margin을 둔다 (예: 512 토큰)
    SAFETY_MARGIN = 512
    max_prompt_tokens = max_context_len - sampling.max_tokens - SAFETY_MARGIN
    max_prompt_tokens = max(128, max_prompt_tokens)  # 혹시 음수/너무 작은 값 방지

    print(f"[Target] max_model_len = {max_context_len}")
    print(f"[Target] allowed prompt tokens (with safety margin) = {max_prompt_tokens}")

    truncated_count = 0
    filtered_prompts: list[str] = []

    for p in prompts:
        new_p = truncate_to_max_tokens(p, tokenizer, max_prompt_tokens)
        if new_p != p:
            truncated_count += 1
        filtered_prompts.append(new_p)

    prompts = filtered_prompts
    print(f"[Target] Truncated {truncated_count} prompt(s) longer than {max_prompt_tokens} tokens.")
    # --- 프롬프트 길이 제한 끝 ---

    # LoRA를 사용하는 경우, LoRARequest 생성
    if args.target_lora:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(
            "target_lora",
            1,
            args.target_lora,
        )
        print("Generating with LoRA adapter...")
        outputs_objs = llm.generate(prompts, sampling, lora_request=lora_request)
        outputs = []
        for out in outputs_objs:
            if not out.outputs:
                outputs.append("")
                continue
            outputs.append(out.outputs[0].text.strip())
    else:
        outputs = run_vllm_generation(llm, prompts, sampling)

    print("Target generation complete.")

    # 먼저 /tmp에 저장 (디스크 쿼터 회피)
    # 모델명을 포함하여 고유한 임시 파일 생성
    model_safe_name = args.target_model.replace('/', '_').replace('\\', '_')
    temp_output = pathlib.Path(f"/tmp/{model_safe_name}_{args.output_target.name}")
    temp_output.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved to temp: {temp_output}")

    # 최종 위치 디렉토리 생성 및 심볼릭 링크
    args.output_target.parent.mkdir(parents=True, exist_ok=True)

    # 기존 파일/링크 제거 후 심볼릭 링크 생성
    if args.output_target.exists() or args.output_target.is_symlink():
        args.output_target.unlink()
    args.output_target.symlink_to(temp_output)

    print(f"Saved target outputs to {args.output_target} -> {temp_output}")


if __name__ == "__main__":
    main()
