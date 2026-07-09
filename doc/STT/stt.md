# Transformers STT 학습 순서

## 1. 오디오 기본 개념 배우기

먼저 Hugging Face Audio Course로 오디오 데이터의 기본 개념을 익힌다.

- Waveform
- Sampling rate
- Spectrogram
- 오디오 데이터셋 로딩
- STT, ASR의 기본 흐름

링크:

https://huggingface.co/learn/audio-course/chapter1/introduction

## 2. Transformers ASR 공식 문서 보기

Hugging Face Transformers의 Automatic Speech Recognition 공식 문서를 본다.

여기서 배울 내용:

- `pipeline("automatic-speech-recognition")`
- `datasets`로 음성 데이터셋 불러오기
- `processor`, `tokenizer`, `feature_extractor`
- 모델 학습용 데이터 전처리
- `Trainer`를 이용한 학습
- WER 평가 지표

링크:

https://huggingface.co/docs/transformers/tasks/asr

## 3. Whisper Fine-tuning 튜토리얼 따라하기

Whisper를 실제 STT 데이터로 fine-tuning하는 튜토리얼을 따라한다.

여기서 배울 내용:

- Whisper 모델 구조와 입력 형식
- 다국어 STT 학습
- 한국어 음성 데이터 학습
- 학습 데이터 전처리
- 평가와 추론

링크:

https://huggingface.co/blog/fine-tune-whisper

## 4. Whisper 모델 페이지 확인하기

처음에는 `openai/whisper-small`이나 `openai/whisper-base`부터 확인한다.

모델 페이지에서 확인할 내용:

- 모델 설명
- 사용 예제
- 지원 언어
- 라이선스
- 입력/출력 형식

링크:

https://huggingface.co/openai/whisper-small

## 5. OpenAI Whisper 원본 자료 보기

Whisper의 원본 소개와 GitHub 저장소도 같이 본다.

링크:

https://openai.com/index/whisper/

https://github.com/openai/whisper

## 추천 학습 순서

```text
Hugging Face Audio Course
→ Transformers ASR 공식 문서
→ Whisper fine-tuning 튜토리얼
→ Whisper 모델 페이지 확인
→ 내 한국어 음성 데이터로 base 또는 small 모델 학습
```

## 처음 실습할 때 추천 모델

| 모델 | 특징 |
|---|---|
| `openai/whisper-tiny` | 가장 빠르지만 정확도는 낮음 |
| `openai/whisper-base` | 가벼운 실습용 |
| `openai/whisper-small` | 속도와 정확도의 균형 |
| `openai/whisper-medium` | 더 정확하지만 GPU 메모리를 더 사용 |
| `openai/whisper-large-v3` | 정확도 높음, GPU 권장 |

## 핵심 정리

`transformers` STT는 Hugging Face Transformers 라이브러리로 Whisper 같은 음성 인식 모델을 실행하거나 학습하는 방식이다.

```text
음성 파일
→ 전처리
→ Whisper 같은 STT 모델
→ 텍스트 결과
```

PyTorch 기반으로 STT를 공부하거나 fine-tuning하려면 `transformers` 방식이 적합하다.
