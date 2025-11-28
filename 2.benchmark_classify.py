#!/usr/bin/env python3
"""Classification-only hallucination benchmarking with vLLM + LoRA."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict
from typing import Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


LABEL_CANON = {
    "nohallucination": "No Hallucination",
    "factualcontradiction": "Factual Contradiction",
    "factualfabrication": "Factual Fabrication",
    "instructioninconsistency": "Instruction Inconsistency",
    "logicalinconsistency": "Logical Inconsistency",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hallucination classifier on target outputs."
    )
    parser.add_argument(
        "--benchmark",
        type=pathlib.Path,
        default=pathlib.Path("final_benchmark.json"),
        help="Benchmark JSON file (list of samples).",
    )
    parser.add_argument(
        "--target-outputs",
        type=pathlib.Path,
        default=pathlib.Path("outputs/target_outputs.json"),
        help="JSON file containing target model outputs (list[str]).",
    )
    parser.add_argument(
        "--classifier-base",
        default="kakaocorp/kanana-1.5-8b-instruct-2505",
        help="Base model used for the hallucination classifier.",
    )
    parser.add_argument(
        "--classifier-lora",
        type=pathlib.Path,
        default=None,
        help="Path to the classifier LoRA weights. If not provided, uses base model only.",
    )
    parser.add_argument(
        "--classifier-max-new-tokens",
        type=int,
        default=64,
        help="Max tokens for classifier outputs.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("outputs/benchmark_results.json"),
        help="Where benchmark details are saved.",
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
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code when loading via vLLM.",
    )
    return parser.parse_args()


def load_benchmark(path: pathlib.Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_target_outputs(path: pathlib.Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Target outputs file {path} must be a JSON list.")
    return [str(x) if x is not None else "" for x in data]


def classifier_prompt_template() -> str:
    return """### Instruction:
답변1을 기준으로 답변2를 비교하여, 답변2에 어떤 할루시네이션이 있는지 분류하세요.
반드시 아래 레이블 중 하나만 그대로 출력하세요.

가능한 레이블:
- No Hallucination
- Factual Contradiction
- Factual Fabrication
- Instruction Inconsistency
- Logical Inconsistency

### 질문:
{instruction}

### 시스템 프롬프트:
{input}

### 답변1 (정답):
{answer}

### 답변2 (모델 응답):
{response}

### 할루시네이션 타입:
"""


def normalize_label(text: str) -> str:
    if not text:
        return "Unknown"
    first_line = text.strip().splitlines()[0]
    normalized = "".join(ch for ch in first_line.lower() if ch.isalpha())
    for key, value in LABEL_CANON.items():
        if key in normalized:
            return value
    return "Unknown"


def run_vllm_generation(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None = None,
) -> list[str]:
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )
    results: list[str] = []
    for out in outputs:
        if not out.outputs:
            results.append("")
            continue
        results.append(out.outputs[0].text.strip())
    return results


def main() -> None:
    args = parse_args()

    samples = load_benchmark(args.benchmark)
    target_outputs = load_target_outputs(args.target_outputs)
    if len(samples) != len(target_outputs):
        raise SystemExit(
            f"Benchmark samples ({len(samples)}) and target outputs ({len(target_outputs)}) length mismatch."
        )

    print(
        f"Loaded {len(samples)} benchmark sample(s) and target outputs from {args.target_outputs}"
    )

    # LoRA 사용 여부 확인
    use_lora = args.classifier_lora is not None and args.classifier_lora.exists()

    if use_lora:
        print(
            f"Loading classifier base model via vLLM: {args.classifier_base} "
            f"+ LoRA {args.classifier_lora}"
        )
    else:
        print(f"Loading classifier model via vLLM: {args.classifier_base} (no LoRA)")

    # LLM 설정
    llm_kwargs = {
        "model": args.classifier_base,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": 4096,
        "enforce_eager": True,
        "max_num_seqs": 256,  # 배치 크기 증가로 처리량 향상
    }

    # LoRA 사용 시에만 LoRA 관련 파라미터 추가
    if use_lora:
        llm_kwargs.update({
            "enable_lora": True,
            "max_lora_rank": 64,
        })

    classifier_llm = LLM(**llm_kwargs)

    # LoRA 사용 시에만 LoRARequest 생성
    lora_request = None
    if use_lora:
        lora_request = LoRARequest(
            lora_name="hallucination_classifier",
            lora_int_id=1,
            lora_local_path=str(args.classifier_lora.resolve()),
            base_model_name=args.classifier_base,
        )

    sampling = SamplingParams(
        max_tokens=args.classifier_max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        n=1,
    )

    template = classifier_prompt_template()
    prompts: list[str] = []
    for sample, response in zip(samples, target_outputs):
        prompts.append(
            template.format(
                instruction=sample["instruction"],
                input=sample.get("input", "") or "<empty>",
                answer=sample["answer"],
                response=response or "<empty>",
            )
        )
        
    tokenizer = classifier_llm.get_tokenizer()
    max_context_len = classifier_llm.llm_engine.model_config.max_model_len
    max_prompt_tokens = max_context_len - sampling.max_tokens

    print(f"Classifier max_model_len = {max_context_len}")
    print(f"Max prompt tokens for prompts = {max_prompt_tokens}")

    truncated_count = 0
    filtered_prompts: list[str] = []

    for p in prompts:
        enc = tokenizer(p, add_special_tokens=False)
        token_len = len(enc["input_ids"])
        if token_len > max_prompt_tokens:
            # 앞부분만 남기고 잘라서 사용
            truncated_ids = enc["input_ids"][:max_prompt_tokens]
            new_p = tokenizer.decode(truncated_ids, skip_special_tokens=True)
            filtered_prompts.append(new_p)
            truncated_count += 1
        else:
            filtered_prompts.append(p)

    print(f"Truncated {truncated_count} prompt(s) longer than {max_prompt_tokens} tokens.")
    prompts = filtered_prompts

    classifier_outputs = run_vllm_generation(
        classifier_llm,
        prompts,
        sampling,
        lora_request=lora_request,
    )
    print("Classification complete.")

    task_distributions: dict[str, Counter[str]] = defaultdict(Counter)
    records: list[dict[str, Any]] = []

    for idx, (sample, model_out, clf_out) in enumerate(
        zip(samples, target_outputs, classifier_outputs)
    ):
        label = normalize_label(clf_out)
        task = sample.get("task", "unknown")
        task_distributions[task][label] += 1
        records.append(
            {
                "index": idx,
                "task": task,
                "instruction": sample["instruction"],
                "input": sample.get("input"),
                "reference": sample.get("reference"),
                "model_output": model_out,
                "gold_answer": sample["answer"],
                "hallucination_label": label,
                "classifier_raw_output": clf_out,
            }
        )

    # 먼저 /tmp에 저장 (디스크 쿼터 회피)
    # 모델명을 포함하여 고유한 임시 파일 생성
    import tempfile
    # args.output 경로에서 모델명 추출 (outputs/MODEL_NAME/benchmark_results.json)
    model_dir_name = args.output.parent.name
    temp_output = pathlib.Path(f"/tmp/{model_dir_name}_{args.output.name}")
    temp_output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved to temp: {temp_output}")

    # 최종 위치 디렉토리 생성 및 심볼릭 링크
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 기존 파일/링크 제거 후 심볼릭 링크 생성
    if args.output.exists() or args.output.is_symlink():
        args.output.unlink()
    args.output.symlink_to(temp_output)

    print(f"Saved detailed benchmark results to {args.output} -> {temp_output}")

    print("\nHallucination distribution by task:")
    for task, counter in sorted(task_distributions.items()):
        total = sum(counter.values())
        breakdown = ", ".join(
            f"{label}: {count} ({count/total:.1%})"
            for label, count in counter.most_common()
        )
        print(f"- {task}: {total} sample(s) -> {breakdown}")


if __name__ == "__main__":
    main()
