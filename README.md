# ko-hallucination-benchmark

Korean Hallucination Benchmark - Streamlit Application

이 디렉토리는 app.py를 독립적으로 실행할 수 있도록 필요한 파일들만 모아둔 곳입니다.

## 디렉토리 구조

```
streamlit-app/
├── app.py                  # Streamlit 웹 애플리케이션
├── requirements.txt        # Python 패키지 의존성
├── .env                    # API 키 설정 (비공개)
├── benchmark/              # 벤치마크 데이터
│   ├── benchmark.json     # 벤치마크 데이터 (HuggingFace에서 자동 로드 또는 로컬 파일)
│   └── judge_llm_result.txt
├── outputs/                # 기존 제공 결과
└── outputs_live/           # Streamlit 앱에서 새로 실행한 결과
```

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
streamlit run app.py
```

브라우저가 자동으로 열리며, 일반적으로 `http://localhost:8501`에서 접속할 수 있습니다.

## 주요 기능

- 벤치마크 데이터 시각화
- 태스크별 환각 유형 분석
- 모델 성능 비교
- 결과 리포트 생성

## 참고사항

- `.env` 파일에는 API 키가 포함되어 있음.
- `benchmark.json`은 HuggingFace Hub에서 자동으로 로드됩니다
- `outputs/` 디렉토리는 기존 결과를 보관하고, Streamlit 앱에서 새로 실행하는 결과는 `outputs_live/`에 저장됩니다
