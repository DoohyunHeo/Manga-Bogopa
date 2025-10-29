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

**다. 폰트 준비**

번역된 텍스트를 그리기 위해 폰트 파일이 반드시 필요합니다.

1.  `data/fonts` 폴더에 사용할 폰트 파일(.ttf, .otf)을 위치시킵니다.
2.  `src/config.py` 파일의 `FONT_MAP` 딕셔너리를 열어, 폰트 스타일 이름과 실제 폰트 파일 경로를 연결해줍니다.

    ```python
    # src/config.py 예시
    FONT_MAP = {
        "narration": "data/fonts/NanumMyeongjo-Bold.ttf",
        "shouting": "data/fonts/Pretendard-ExtraBold.otf",
        # 다른 폰트 스타일...
    }
    ```

**라. 모델 다운로드**

`data/models` 폴더에 필요한 AI 모델 파일을 위치시켜야 합니다. 모델 파일은 프로젝트의 GitHub Releases 페이지에서 다운로드할 수 있습니다.

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
- `FONT_MAP`: `src/config.py`의 `FONT_MAP` 딕셔너리에서 특정 폰트 스타일(예: 'narration', 'shouting')에 사용할 폰트 파일(.ttf, .otf)의 경로를 지정하거나 수정할 수 있습니다.
- `DRAW_DEBUG_BOXES`: 결과 이미지에 탐지 영역 박스를 표시할지 여부 (디버깅용)

## 향후 계획

- **GUI 프론트엔드 개발**: 사용자가 더 쉽게 파이프라인을 이용할 수 있도록 그래픽 사용자 인터페이스(GUI)를 개발할 예정입니다.
- **식자 기능 고도화**: 텍스트 배치, 폰트 크기 자동 조절, 스타일 적용 등을 개선하여 더 자연스러운 식자 결과를 만들 계획입니다.