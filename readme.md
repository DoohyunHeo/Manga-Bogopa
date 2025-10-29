# Manga-Bogopa: 만화 자동 번역 및 식자 프로젝트

Manga-Bogopa는 만화 이미지 속의 텍스트를 자동으로 감지하고, 번역하며, 원본 텍스트를 지운 자리에 번역된 텍스트를 자연스럽게 다시 그려 넣어주는(식자) 자동화 파이프라인입니다.

## 주요 기능

- **텍스트 영역 감지**: Manga Dialogue Extractor(YOLO 기반) 모델을 사용하여 이미지에서 말풍선과 텍스트 영역을 찾습니다.
- **텍스트 속성 분석**: OCR로 텍스트를 추출하고, AI 모델로 폰트 스타일, 크기, 각도 등을 분석합니다.
- **번역**: Google Gemini API를 이용해 추출된 텍스트를 한국어로 번역합니다.
- **텍스트 제거**: LaMa 인페인팅 모델로 원본 텍스트 영역을 지웁니다.
- **식자**: 분석된 폰트 스타일을 바탕으로 번역된 텍스트를 이미지에 다시 그립니다.

## 프로젝트 구조

```
Manga-Bogopa/
├── data/
│   ├── inputs/         # 번역할 원본 이미지 폴더
│   ├── outputs/        # 번역 및 식자 완료된 이미지 폴더
│   ├── models/         # AI 모델 파일 폴더
│   └── fonts/          # 식자에 사용할 폰트 파일 폴더
├── src/
│   ├── pipeline.py     # 전체 번역 및 식자 워크플로 관리
│   ├── extractor.py    # 텍스트 영역 및 속성 추출
│   ├── translator.py   # 텍스트 번역
│   ├── inpainter.py    # 원본 텍스트 제거 (인페인팅)
│   ├── drawer.py       # 번역된 텍스트 식자
│   ├── config.py       # 프로젝트 주요 설정
│   └── ...
├── main.py             # 프로젝트 실행 스크립트
├── requirements.txt    # 필요한 Python 라이브러리 목록
└── readme.md           # 프로젝트 설명 파일
```

## 사용법

### 1. 환경 설정

**가. API 키 설정**

프로젝트 루트 디렉토리에 `api_key.txt` 파일을 생성하고, 파일 안에 Google Gemini API 키를 입력하세요.

**나. 필요 라이브러리 설치**

아래 명령어를 실행하여 필요한 Python 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

**다. 모델 및 폰트 다운로드**

`data/models`와 `data/fonts` 폴더에 필요한 AI 모델 파일과 폰트 파일을 위치시켜야 합니다. (모델 및 폰트 파일은 별도로 제공되어야 합니다.)

### 2. 실행

**가. 입력 이미지 준비**

`data/inputs` 폴더에 번역하고자 하는 만화 이미지 파일(jpg, png 등)들을 넣습니다.

**나. 파이프라인 실행**

아래 명령어를 실행하여 번역 및 식자 파이프라인을 시작합니다.

```bash
python main.py
```

### 3. 결과 확인

프로세스가 완료되면 `data/outputs` 폴더에서 결과물을 확인할 수 있습니다.

## 설정

프로젝트의 세부 동작은 `src/config.py` 파일에서 수정할 수 있습니다.

- `INPUT_DIR`, `OUTPUT_DIR`: 입/출력 폴더 경로
- `YOLO_CONF_THRESHOLD`: 객체 탐지 최소 신뢰도
- `GEMINI_MODEL`: 사용할 Gemini 모델 버전
- `FONT_MAP`: 폰트 스타일과 실제 폰트 파일 경로 매핑
- `DRAW_DEBUG_BOXES`: 결과 이미지에 탐지 영역 박스를 표시할지 여부 (디버깅용)

