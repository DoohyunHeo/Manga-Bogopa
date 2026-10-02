"""번역 CLI(Antigravity CLI·Claude Code·Codex CLI)로 대사 번역을 요청한다.

- 설정(TRANSLATION_BACKEND)에서 고른 CLI를 쓰고, auto면 설치된 것 중 agy → Claude Code → Codex 순으로 고른다
- 창 없이 하위 프로세스로 실행하고, 응답은 JSON 스키마로 받는다
- 요청마다 새 대화로 보내고 여러 요청을 동시에 돌린다 — 대화를 이어 쓰면 앞 내용이 입력으로 다시 들어가
  요청마다 무거워진다. 인명·용어는 응답마다 받아 책 하나의 사전(glossary)에 모아 모든 요청에 붙인다
- 응답에서 빠진 대사는 한 번 더 요청하고, 그래도 없으면 '실패'로 남겨 다음 실행에서
  그 대사만 다시 요청한다 (대사를 지우지 않는다)
- 한 화를 다 번역하면 프로그램이 찾을 수 있는 틀림 의심(한두 글자 다르게 읽힌 이름, 칸에 잘린 말, 대명사 오독,
  되묻는 말이 된 외침, 번역문에 남은 일본 글자)이 있는 대사만 한 번 더 묻는다 (의심 대사 확인) — 걸린 대사가 없으면
  묻지 않고, 실패하면 받은 번역을 그대로 쓴다
- 받은 번역문은 원문 부호에 맞춘다 — 원문에 없는 감싼 따옴표·쌍점, 원문 끝 뜻 없는 로마자 조각,
  한자 후리가나를 옮긴 괄호 병기를 빼고, 第가 빠진 회차 제목은 '제○화'로 적는다. 끝 마침표·쉼표는
  모델이 찍은 대로 두되 물결표 뒤 마침표만 뗀다 (한국어판은 원문에 。가 없어도 문장을
  마침표로 끝내고, 물결표로 끝낸 말에는 찍지 않는다)
- 확인이 원문에 없는 물음표를 붙이거나 대명사를 소리대로(와시·오레) 바꾼 고침은 받지 않는다
"""
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src import config
from src.data_models import PageData, TextElement, TranslationStatus
from src.glossary import PRONOUN_TARGETS as _PRONOUN_TARGETS
from src.progress import EventLevel, PipelinePhase, ProgressEvent
from src.reading_order import reading_order
from src.text_filters import dialogue_size, exclusion_reason

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
# 사용량 제한·일시 오류는 수십 초 단위로 풀리는 경우가 많아 점증 대기
RETRY_DELAYS = [10, 30]

# Ellipsis normalization: 모델이 ..., …, ．．．, 。。。, ・・・, 점 사이 공백 등 다양한 형태로 낸다.
# 모두 U+22EF(⋯)로 통일.
_ELLIPSIS_SPACED = re.compile(r'(?<=[.。．・])\s+(?=[.。．・])')
_ELLIPSIS_RUNS = re.compile(
    r'[…‥]+'                      # …  ‥
    r'|[.。．・]{2,}'           # .. .., 。。。, ．．．, ・・・
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "skip": {"type": "boolean"},
                },
                "required": ["id", "text"],
            },
        },
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "alias_of": {"type": "string"},
                },
                "required": ["source", "target"],
            },
        },
    },
    "required": ["translations", "terms"],
}

_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "fixes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "skip": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "text"],
            },
        },
        "terms": _RESPONSE_SCHEMA["properties"]["terms"],
    },
    "required": ["fixes", "terms"],
}

_SHORTEN_SCHEMA = {
    "type": "object",
    "properties": {
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "texts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "texts"],
            },
        },
    },
    "required": ["alternatives"],
}

_CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "verdict": {"type": "string"}},
                "required": ["id", "verdict"],
            },
        },
    },
    "required": ["verdicts"],
}

# 전각 영숫자(ＡＢＣ, １２)는 요청 전에 반각으로 — 번역문에 전각 글자가 그대로 옮겨 오지 않게
_FULLWIDTH_ALNUM = re.compile("[０-９Ａ-Ｚａ-ｚ]")

_FORMAT_INSTRUCTION = """## 입력과 출력 형식
- 입력은 JSON 배열이고, 항목마다 id(페이지.번호)와 원문 text가 있습니다. 원문은 글자 인식으로 읽은 것이라
  틀린 글자가 섞일 수 있습니다 — 뜻이 안 통하면 모양이 비슷하고 문맥에 맞는 글자로 짐작해 번역합니다.
- free가 true인 항목은 말풍선 밖 글자(해설·제목·간판·화면 글자·손글씨)입니다. 효과음·장면 소품 글자가 섞이기 쉽습니다.
- ruby는 원문 한자·로마자 약어에 붙은 후리가나를 '본문(읽기)'로 적은 것입니다(글자 인식이라 가끔 틀림). 인명·지명의
  읽기는 짐작하지 말고 후리가나를 따릅니다: 小鳥遊(たかなし) → 타카나시. 특별한 읽기(本気(マジ))도 따릅니다.
  로마자 약어 위의 읽기는 약어의 뜻풀이입니다 — 약어는 두고 괄호로 읽기를 옮겨 적어 한국 독자도 뜻을 알게 합니다
  (SP(スペシャル) → SP(스페셜)). 같은 약어가 여러 번 나오면 처음 나온 곳에만 괄호를 답니다.
  한자 본문 위의 특별한 읽기는 괄호로 둘 다 적지 않고 한 가지 표기만 씁니다 — 괄호 병기는 로마자 약어에만 합니다.
  같은 한자 말이 책의 다른 곳에 읽기 없이 나오면 그곳과 같은 표기(대개 한자 뜻)를 씁니다.
- 출력은 지정된 JSON 형식으로만 합니다. translations 배열에 받은 id를 하나도 빠짐없이 넣고
  text에 한국어 번역을 적습니다. 위 지침에 다른 출력 형식이 적혀 있어도 이 형식을 따릅니다.
- 번역문 안에서 줄을 바꾸지 않습니다.
- 문장 끝 부호는 한국어 정식판처럼 찍습니다: 평서문·명령문은 원문 끝에 。가 없어도 마침표로 끝내고(갈게요.),
  말줄임표·늘임표로 끝난 문장도 '….'·'—.'로 끝내며(그렇구나…. 좋아—.), 묻는 말·권하는 말은 원문에 ?가 없어도
  물음표로 끝냅니다(갈까?). 문장이 끝나지 않은 말(다음 말풍선으로 이어지는 말, ~면·~은·~고로 끊긴 조각)과
  물결표로 끝난 말, 길게 늘인 환호(와아아아)·늘여 외친 구호와 맞장구(만세—·그렇지—), 간판·이름표의 이름에는
  마침표를 찍지 않습니다. 늘임표는 하나(—)만 쓰고 늘임표·말줄임표 뒤에는 쉼표를 찍지 않습니다. 한 번역문의 여러
  문장은 문장마다 끝 부호를 찍고 쉼표로 잇지 않습니다. 부르는 말·감탄사·대답 뒤에는 쉼표를 씁니다(아니, 그게 아니야.).
  원문에 없는 따옴표로 속마음·혼잣말을 감싸지 않습니다.
  원문이 !로 끝나는 단정·외침은 되묻는 말(~다고?!)로 바꾸지 않습니다. 원문에 없는 쌍점(:)도 넣지 않습니다.
- 번역할 수 없는 항목(지침의 '번역 불가'에 해당)은 text를 빈 문자열로 두고 skip을 true로 합니다.
- terms 배열에는 이번 묶음에 나온 고유명사(인명·별명·지명·단체·기술·물건 이름)를 원문 표기(source)와
  번역에 쓴 한국어 표기(target)로 적습니다. 한자나 가타카나로 쓴 이름은 빠짐없이 넣고,
  히라가나로만 쓴 말과 일반 명사(숫자가 붙은 말 포함 — 조회수·구독자 수 같은 수치)는 넣지 않습니다. source는 원문에
  쓰인 그대로 적되 호칭(さん·くん·ちゃん·様·先輩 등)은 빼고, target도 호칭과 조사 없이 적습니다.
- 글자 인식이 잘못 읽은 것으로 보이는 이름(일본어에 없는 한자 조합, 후리가나나 책의 다른 곳 표기와 안 맞는 이름)은
  terms에 넣지 않습니다 — 오독이 사전에 굳으면 책 전체에 퍼집니다.
- 작품·곡·프로그램·앱·기술 이름(히라가나가 섞여도, 예: キミのせい), 이야기에서 되풀이되는 용어(影武者 등)와
  사람을 가리키는 호칭(奥様 등)도 terms에 넣습니다 — 뒤 묶음이 같은 표기를 쓰게 됩니다. 작품 속 종족·요괴·설정
  용어는 한 글자 한자여도 넣습니다(鬼 → 오니, 天狗 → 텐구 — 일본 요괴 이름은 한국 요괴 이름으로 바꾸지 않고 음역).
- 1인칭·2인칭 대명사(ワシ·オレ·ボク·アタシ·ウチ·オヌシ·アンタ 등)는 이름이 아니므로 terms에 넣지 않습니다.
- 한 인물의 다른 이름(애칭·활동명·줄인 이름)은 따로 한 줄로 넣고 alias_of에 본 이름의 원문을 적습니다
  (ミナミ의 애칭 ミナ → source ミナ, target 미나, alias_of ミナミ). 비슷한 두 이름을 한 표기로 섞지 않습니다.
- target 표기: 인명·지명·가게·브랜드 같은 고유명만 음을 옮깁니다. 곡·작품 제목, 일반 용어·호칭은 뜻을 옮긴
  한국어로 적습니다(한국 정식 발매명이 있으면 그것). 影武者 → 대역, キミのせい → 너 때문이야. 한자로 된 기술·유파·단체
  이름은 한국 한자음으로 읽습니다(竜巻斬 → 용권참 — 히라가나 후리가나가 있어도 일본어 소리로 적지 않고, 후리가나가 가타카나
  외래어이면 그 소리). 제목의 괄호(『』「」)는
  번역문에 그대로 두고, 뜻을 옮기는 말은 한자를 한국 한자음으로 읽지 말고 한국 독자가 실제로 쓰는 말로 적습니다
  (영상의 再生 → 조회수, 登録者 → 구독자). 일·역할을 뜻하는 말이 사람을 부르는 별명이면 사람을 가리키는 말로 적습니다
  (使いっ走り → 심부름꾼).
- 파일을 읽거나 만들지 말고 명령도 실행하지 마세요. 번역 결과만 답합니다."""

_GLOSSARY_HEADER = """## 인명·용어 사전
이 책에서 이미 정한 표기입니다. 아래 원문이 나오면 반드시 이 표기를 그대로 씁니다 (조사는 문장에 맞게).
다른 표기가 더 나아 보여도 바꾸지 않고, terms에도 이 표기로 적습니다."""

_CHECK_INSTRUCTION = """## 의심 대사 확인
이 책의 번역이 끝났습니다. 아래 JSON 배열의 대사는 프로그램이 틀렸을 수 있다고 찾은 것입니다. 항목마다 원문(text),
지금 번역(ko), 앞뒤 대사(before·after, 읽는 순서의 원문과 번역, 문맥용), 의심한 까닭(why)이 있습니다.
id는 '쪽.번호'이고, free가 true면 말풍선 밖 글자, ruby는 원문 한자·로마자 약어에 붙은 후리가나 '본문(읽기)'입니다
(글자 인식이라 가끔 틀림). 원문도 글자 인식으로 읽은 것이라 틀린 글자가 섞일 수 있습니다.

항목마다 why를 보고, 정말 틀렸을 때만 고친 번역을 fixes에 넣습니다. 확실하지 않거나 지금 번역이 맞으면 넣지 않습니다.
- 글자 인식 오독: 책의 다른 곳과 한두 글자만 다르게 읽힌 이름·앱 이름은 모양이 닮은 글자(シ↔ツ, ソ↔ン, ワ↔フ,
  バ↔ラ, ロ↔口처럼)로 읽은 것인지 문맥으로 보고, 맞으면 그 이름으로 옮깁니다. 엉뚱한 뜻을 그럴듯하게 꾸며 옮기지 않습니다.
- 칸·말풍선에 잘려 첫 한두 글자만 남은 말(ご·あり·すみ)은 문맥상 무슨 말인지 확실하면 되살려 옮깁니다
  (ご → ごめん '미안해', すみ → すみません '죄송해요').
- 1인칭 대명사로 보이는 오독(ワシ가 フシ로 읽힘)은 대명사로 옮깁니다 — 소리대로('와시') 적지 않습니다(나 / 이 몸).
- 번역문에 남은 일본 글자는 한국어 표기로 고칩니다(셰ン론 → 셴론, 보ンド → 본드). 한자가 섞였으면 한국어로 옮깁니다
  (攻撃했다 → 공격했다).
- 원문이 !로 끝나는 단정·외침을 되묻는 말('~했다고?!')로 옮겼으면 원문대로 단정으로 고치고, 물음표를 새로 붙이지 않습니다.
  원문이 !로 끝나도 뜻이 물음(~のか!)이면 그대로 둡니다.

고칠 때 지킬 것
- 원문에 없는 내용을 더하지 않고, 원래 번역보다 길게 늘이지 않습니다. 번역문 안에서 줄을 바꾸지 않습니다.
- 문장 끝 마침표·쉼표는 지금 번역을 따릅니다. 원문에 없는 따옴표를 붙이지 않습니다.

돌려주는 법
- fixes에는 고칠 항목만 넣습니다: id, 고친 번역문 전체(text), 이유(reason, 짧게). 고칠 것이 없으면 빈 배열입니다.
- 인명·용어 표기를 바꿨으면 terms에 원문 표기(source, 호칭 빼고)와 새 표기(target)를 적습니다. 바꾼 것이 없으면 빈 배열입니다.
- 파일을 읽거나 만들지 말고 명령도 실행하지 마세요. 결과만 답합니다."""

_SHORTEN_INSTRUCTION = """## 말풍선에 맞춘 짧은 번역
아래 JSON 배열의 대사는 번역은 맞지만 말풍선이 좁아 글자가 작아지거나 낱말이 줄 끝에서 쪼개지는 것들입니다.
항목마다 원문(text), 지금 번역(ko), 앞뒤 대사(before·after, 읽는 순서, 문맥용), 그리고 이 말풍선에 원하는 크기로
넣을 수 있는 길이(max_chars_per_line: 한 줄 최대 글자 수, max_lines: 최대 줄 수, about_chars: 공백 포함 전체 글자 수 안팎)가
있습니다. ruby는 원문 한자·로마자 약어의 후리가나입니다.

항목마다 지금 번역보다 짧은 대안을 2~3개, 짧은 것부터 적어 주세요.
- 뜻·말투(존댓말·반말·사투리·어미)·인명·용어 표기는 지금 번역 그대로 지킵니다. 원문에 없는 말을 더하지 않습니다.
- 원문에 없는 뜻·표현·관용구·감정을 더하지 않습니다(『부탁이야』가 아니잖아 → 『부탁』은 얼어 죽을 ❌). 군더더기 조사·어미·
  겹말을 덜거나, 같은 뜻의 더 짧은 말(준말·동의어·자연스러운 의역)로 바꾸는 건 괜찮습니다(신경 쓰지 마 → 신경 꺼).
- 뜻을 빼지 않습니다 — 부르는 말(야·이 자식아), 맞장구(응), 되풀이(새로고침 새로고침), 대상(다들)도 뜻입니다. 외래어로 바꾸거나
  (새로고침 → 리로드 ❌) 한국어에서 안 쓰는 꼴로 줄이지 않습니다(사과하지 마 → 사과 마 ❌). 줄여도 자연스러운 대사여야 합니다.
- 한 낱말이 max_chars_per_line보다 길면 줄 끝에서 쪼개지므로, 같은 뜻의 더 짧은 말로 쓰거나 앞뒤를 덜어 한 줄에 오게 합니다.
- 문장부호는 지금 번역을 따릅니다 — 끝 마침표·쉼표도 지금 번역대로 두고, 새 물음표·따옴표는 붙이지 않습니다.
  번역문 안에서 줄을 바꾸지 않습니다.
- 더 줄일 수 없으면 texts를 빈 배열로 둡니다.
- 파일을 읽거나 만들지 말고 명령도 실행하지 마세요. 결과만 답합니다."""

_CONFIRM_INSTRUCTION = """## 줄인 번역 확인
일본 만화 대사의 한국어 번역을 말풍선에 맞추려고 줄였습니다. 아래 JSON 배열은 그 줄인 번역입니다. 항목마다 일본어 원문(text), 지금 번역(ko), 줄인 번역(short)이 있습니다.
줄인 번역을 그대로 써도 되는지 항목마다 verdict에 OK 또는 NG로만 답하세요.
- NG: 원문에 없는 뜻·관용구·감정·비꼼을 더했다 / 말투(반말·존댓말)·호칭·말하는 사람의 성격이 바뀌었다 / 원문의 핵심 정보가 빠졌다.
- OK: 같은 뜻을 다른 말(동의어·준말·자연스러운 의역)로 짧게 했다.
- 받은 id를 빠짐없이 답합니다. 파일을 읽거나 만들지 말고 명령도 실행하지 마세요. 결과만 답합니다."""

# 곁가지 요청(짧은 번역·짧은 번역 확인·의심 대사 확인)은 번역 품질 설정과 상관없이 생각을 짧게 — 설정대로 생각하면 확인
# 몇 개에도 몇 분이 걸린다. 시간 한도를 넘거나 실패하면 다시 묻지 않는다 (짧은 번역은 지금 번역, 확인은
# 대안을 버림, 의심 대사는 지금 번역)
SIDE_EFFORT = "low"
SHORTEN_LIMIT_SEC = 120
CONFIRM_LIMIT_SEC = 60
CHECK_LIMIT_SEC = 90
# 의심 대사 앞뒤로 붙이는 대사 수 (문맥용)
_CHECK_CONTEXT = 2

_LEGACY_SKIP_TEXTS = {"번역 불가", "(번역 불가)"}
_AUTH_HINTS = ("authentication required", "sign in", "log in", "login")
_MODEL_LEVEL_SUFFIX = re.compile(r"-(low|medium|high|max)$", re.I)

# 번역 CLI 이름 (로그·안내 문구)과 로그인 안내 — 다시 시도해도 소용없어 실행을 멈추고 알린다
BACKEND_LABELS = {"antigravity": "Antigravity CLI", "claude": "Claude Code", "codex": "Codex CLI"}
_LOGIN_HELP = {
    "antigravity": "Antigravity CLI 로그인이 필요합니다. 터미널에서 agy를 한 번 실행해 로그인해 주세요.",
    "claude": "Claude Code 로그인이 필요합니다. 터미널에서 claude를 한 번 실행해 로그인해 주세요.",
    "codex": "Codex CLI 로그인이 필요합니다. 터미널에서 codex login을 한 번 실행해 로그인해 주세요.",
}


class TranslationUnavailable(RuntimeError):
    """다시 시도해도 소용없는 실패 (CLI 없음·로그인 필요) — 실행을 멈추고 알린다."""


class TranslationTimeout(RuntimeError):
    """시간 한도 안에 답이 오지 않았다 — 곁가지 요청(짧은 번역·확인)은 다시 묻지 않고 건너뛴다."""


def find_agy_executable() -> Optional[str]:
    """설정의 경로, PATH, 기본 설치 위치 순으로 agy 실행 파일을 찾는다."""
    configured = str(config.AGY_PATH or "").strip()
    if configured:
        return configured if os.path.isfile(configured) else None
    found = shutil.which("agy")
    if found:
        return found
    default = os.path.join(os.environ.get("LOCALAPPDATA", ""), "agy", "bin", "agy.exe")
    return default if os.path.isfile(default) else None


def check_agy_login(executable: Optional[str] = None, timeout_sec: int = 60):
    """(로그인 여부, 안내 문구). 모델 목록 조회로 확인한다 — 번역 요청은 보내지 않는다."""
    executable = executable or find_agy_executable()
    if not executable:
        return False, "Antigravity CLI(agy)를 찾지 못했습니다."
    try:
        proc = subprocess.run(
            [executable, "models"], capture_output=True, timeout=timeout_sec,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"확인 실패: {e}"
    out = proc.stdout.decode("utf-8", errors="replace").strip()
    err = proc.stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode == 0:
        return True, out or "로그인되어 있습니다."
    lines = [line for line in err.splitlines() if line.strip() and "logging before google.Init" not in line]
    return False, _agy_error(err) or (lines[-1] if lines else f"exit {proc.returncode}")


def _find_cli(name: str, defaults: Sequence[str]) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    return next((path for path in defaults if path and os.path.isfile(path)), None)


def find_claude_executable() -> Optional[str]:
    """PATH, 기본 설치 위치(홈 폴더의 .local/bin, npm 전역) 순으로 Claude Code 실행 파일을 찾는다."""
    local_bin = os.path.join(os.path.expanduser("~"), ".local", "bin")
    npm = os.environ.get("APPDATA")
    return _find_cli("claude", [os.path.join(local_bin, "claude.exe"), os.path.join(local_bin, "claude"),
                                os.path.join(npm, "npm", "claude.cmd") if npm else ""])


def find_codex_executable() -> Optional[str]:
    """PATH, npm 전역 설치 위치 순으로 Codex CLI 실행 파일을 찾는다."""
    npm = os.environ.get("APPDATA")
    return _find_cli("codex", [os.path.join(npm, "npm", "codex.cmd") if npm else ""])


# npm 전역 설치의 껍데기(.cmd·.ps1)가 node로 부르는 스크립트
_NPM_SCRIPTS = {"claude": ("@anthropic-ai", "claude-code", "cli.js"), "codex": ("@openai", "codex", "bin", "codex.js")}


def _launch_prefix(executable: str, backend: str) -> List[str]:
    """실행 명령 앞부분. npm 껍데기는 껍데기가 하는 대로 node로 스크립트를 바로 부른다 — .cmd를 거치면
    cmd.exe가 인자(스키마 JSON 등)를 다시 해석한다. 스크립트를 못 찾으면 껍데기를 그대로 부른다."""
    if executable.lower().endswith(".exe") or backend not in _NPM_SCRIPTS:
        return [executable]
    folder = os.path.dirname(executable)
    script = os.path.join(folder, "node_modules", *_NPM_SCRIPTS[backend])
    node = os.path.join(folder, "node.exe")
    if not os.path.isfile(node):
        node = shutil.which("node")
    return [node, script] if node and os.path.isfile(script) else [executable]


# 이 프로그램을 Claude Code 안에서 띄웠을 때 물려받는 바깥 대화 표시 — 번역 요청은 따로 선 대화여야 한다
_CLAUDE_PARENT_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")


def _child_env(backend: str) -> Optional[dict]:
    if backend != "claude":
        return None  # 그대로 물려준다
    return {name: value for name, value in os.environ.items() if name not in _CLAUDE_PARENT_ENV}


def _kill_tree(proc: subprocess.Popen):
    """시간을 넘긴 CLI를 끈다 — npm 껍데기(node)가 띄운 실제 CLI까지 (Windows는 taskkill /T)."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.communicate(timeout=10)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass


def _run_quick(cmd: List[str], timeout_sec: int, env: Optional[dict] = None) -> Tuple[int, str, str]:
    """짧은 CLI 명령(로그인 상태·모델 목록) — (종료 코드, 표준 출력, 오류 출력). 시간을 넘기면 자식까지 끄고 TimeoutExpired."""
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        out, err = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        raise
    return proc.returncode, out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")


def check_login(backend: str, executable: Optional[str] = None, timeout_sec: int = 60) -> Tuple[bool, str]:
    """(로그인 여부, 안내 문구) — 번역 요청은 보내지 않는다 (agy는 모델 목록, Claude Code·Codex는 로그인 상태 명령)."""
    if backend == "antigravity":
        ok, message = check_agy_login(executable, timeout_sec)
        return (True, "로그인되어 있습니다.") if ok else (False, message)
    executable = executable or find_executable(backend)
    if not executable:
        return False, f"{BACKEND_LABELS[backend]}를 찾지 못했습니다."
    args = ["auth", "status", "--json"] if backend == "claude" else ["login", "status"]
    try:
        code, out, _ = _run_quick(_launch_prefix(executable, backend) + args, timeout_sec, env=_child_env(backend))
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"확인 실패: {e}"
    ok = code == 0
    if ok and backend == "claude":
        # 상태 JSON에는 계정 정보가 들어 있어 로그인 여부만 본다
        ok = bool((_parse_json_text(out.strip()) or {}).get("loggedIn"))
    return (True, "로그인되어 있습니다.") if ok else (False, _LOGIN_HELP[backend])


def _normalize_ellipsis(text: str) -> str:
    if not text:
        return text
    collapsed = _ELLIPSIS_SPACED.sub('', text)
    return _ELLIPSIS_RUNS.sub('⋯', collapsed)


# 번역문 전체를 감싼 따옴표 — 원문에 따옴표·낫표가 없으면 모델이 속마음 표시로 붙인 것이다
_WRAPPING_QUOTES = (("'", "'"), ("‘", "’"), ('"', '"'), ("“", "”"))
_SOURCE_QUOTES = re.compile("[「」『』\"'“”‘’〝〟]")
# 원문 끝에 붙은 뜻 없는 로마자 조각 — 잘린 글자를 글자 인식이 로마자로 읽은 것 (제목/g, 일본어 바로 뒤 소문자 한 자)
# (w는 웃음 표시라 뺀다)
_SOURCE_LATIN_TAIL = re.compile(r"(?:[/／|｜]\s*[A-Za-z]{1,2}|(?<=[ぁ-ゖァ-ヺ一-鿿])[a-vx-z])\s*$")


# 한자 본문의 후리가나를 번역문에 괄호로 덧붙인 것 — 괄호 병기는 로마자 약어 뒤에만 둔다
_HANGUL_GLOSS = re.compile(r"(?<=[가-힣])\s?[(（][가-힣 ]{1,12}[)）]")
# 회차 제목 — 글자 인식이 第를 놓쳐도 '제○화'로 (○話 → 제○화)
_SOURCE_EPISODE = re.compile(r"^[^\w第]*(\d+)\s*話")
_COLON = re.compile(r"\s*[:：]\s*")
# 물결표로 끝낸 말 뒤의 마침표 — 한국어판은 물결표 뒤에 마침표를 찍지 않는다
_WAVE_PERIOD = re.compile(r"([~〜～])\.+(?=\s|$)")
# 늘임표·물결표·말줄임표 바로 뒤 쉼표 — 한국어판은 늘인 소리 뒤에 쉼표를 찍지 않는다
_MARK_COMMA = re.compile(r"([—―─~〜～…⋯])[,，](?=\s|$)")
# 묻는 꼴인데 마침표로 끝낸 문장 — 'ㄹ까(요).'와 의문사+지('왜지.'). 한국어판은 원문에 ?가 없어도 묻는 말·권하는 말에
# 물음표를 찍는데, 번역은 지침을 두어도 마침표를 남기곤 한다
_KKA_PERIOD = re.compile(r"([가-힣])까(요?)\.(?=\s|$)")
_WH_JI_PERIOD = re.compile(r"(?<![가-힣])(왜|뭐|누구|어디|언제|어째서)지\.(?=\s|$)")


def _question_marks(text: str) -> str:
    """'갈까.'·'왜일까.'·'왜지.'의 마침표를 물음표로 — 'ㄹ까'는 앞 글자 받침이 ㄹ일 때만 ('~니까.'는 그대로)."""
    def kka(match):
        final = (ord(match.group(1)) - 0xAC00) % 28
        return f"{match.group(1)}까{match.group(2)}?" if final == 8 else match.group(0)
    return _WH_JI_PERIOD.sub(r"\1지?", _KKA_PERIOD.sub(kka, text))


# 느낌표 뒤에 붙은 물음표 — 한국어판은 원문의 '!?'를 '?!'로 쓴다
_EXCLAIM_QUESTION = re.compile("([!！]+)([?？]+)")
# 촉음(っ)으로 끊긴 짧은 비명·헉 소리 — 한국어판은 원문에 느낌표가 없어도 느낌표로 끝내는데, 번역은 마침표로
# 끝내곤 한다. 촉음으로 끝나지 않는 차분한 짧은 소리는 한국어판도 마침표라 그대로 둔다
_CLIPPED_CRY = re.compile(r"^[ぁ-ゖァ-ヺー〜~]{1,5}[っッ]$")
_END_PERIOD = re.compile(r"(?<![.…⋯])\.$")


def _question_first(text: str) -> str:
    """'!?'·'!!?'처럼 느낌표 뒤에 붙은 물음표를 앞으로 ('?!'·'?!!')."""
    return _EXCLAIM_QUESTION.sub(lambda m: "?" * len(m.group(2)) + "!" * len(m.group(1)), text)
# 번역문에 남은 일본 글자 — 한국어판에는 없어야 한다. 의심 대사 확인에 먼저 싣고(check_notes),
# 그래도 남으면 소리대로 한글로 바꾼다 (hangul_for_kana). 장음 부호 ー와 가운뎃점 ・도 여기 든다
_KANA = re.compile("[ぁ-ゖァ-ーｦ-ﾟ]")
# 번역문에 섞인 한자 — 소리대로 바꿀 수 없어 의심 대사 확인에만 싣는다 (check_notes)
_HANJA = re.compile("[㐀-䶿一-鿿々〆]")
_KANA_SOUNDS = {}  # 가타카나 → (첫소리, 가운뎃소리)의 한글 자모 번호
for _row, _initial in (("アイウエオ", 11), ("カキクケコ", 15), ("ガギグゲゴ", 0), ("サシスセソ", 9), ("ザジズゼゾ", 12),
                       ("タチツテト", 16), ("ダヂヅデド", 3), ("ナニヌネノ", 2), ("ハヒフヘホ", 18), ("バビブベボ", 7),
                       ("パピプペポ", 17), ("マミムメモ", 6), ("ラリルレロ", 5)):
    for _kana, _medial in zip(_row, (0, 20, 13, 5, 8)):
        _KANA_SOUNDS[_kana] = (_initial, _medial)
_KANA_SOUNDS.update({"ス": (9, 18), "ズ": (12, 18), "チ": (14, 20), "ツ": (14, 18), "ヂ": (12, 20), "ヅ": (12, 18),
                     "ヤ": (11, 2), "ユ": (11, 17), "ヨ": (11, 12), "ワ": (11, 9), "ヰ": (11, 20), "ヱ": (11, 5),
                     "ヲ": (11, 8), "ヴ": (7, 13)})
# 작은 글자는 앞 글자의 가운뎃소리를 바꾼다 (キャ→캬, ファ→파, ウィ→위, シェ→셰) — 홀로 있으면 제 소리
_SMALL_KANA = {"ァ": 0, "ィ": 20, "ゥ": 13, "ェ": 5, "ォ": 8, "ャ": 2, "ュ": 17, "ョ": 12, "ヮ": 9}
_W_GLIDES = {20: 16, 5: 15, 8: 14, 0: 9}  # ウ + 작은 모음 → ㅟ·ㅞ·ㅝ·ㅘ


def clean_translation(original: str, text: str, readings: Sequence[str] = ()) -> str:
    """번역문을 원문 부호에 맞춘다 — 공백·말줄임표 정리, 원문에 없는 감싼 따옴표·끝 로마자 조각·쌍점,
    한자 후리가나를 옮긴 괄호 병기, 물결표 뒤 마침표, 늘인 소리 뒤 쉼표를 빼고, 묻는 꼴 끝의 마침표를 물음표로,
    '!?'를 '?!'로, 촉음으로 끊긴 짧은 비명 끝의 마침표를 느낌표로, 第가 빠진 회차 제목에 '제'를 붙인다."""
    text = _WAVE_PERIOD.sub(r"\1", _normalize_ellipsis(" ".join(str(text or "").split())))
    text = _question_first(_question_marks(_MARK_COMMA.sub(r"\1", text)))
    source = request_text(original or "").strip()
    if not text or not source:
        return text
    if _CLIPPED_CRY.match(source):
        text = _END_PERIOD.sub("!", text)
    if not re.search("[:：]", source):
        # 채팅 이름 뒤에 모델이 붙인 쌍점 — 원문은 띄어쓰기로만 가른다
        text = _COLON.sub(" ", text).strip()
    if not re.search("[(（]", source) and any(not request_text(str(pair)).split("(")[0].isascii() for pair in readings or ()):
        text = _HANGUL_GLOSS.sub("", text)
    episode = _SOURCE_EPISODE.match(source)
    if episode:
        text = re.sub(rf"(?<![제\d]){episode.group(1)}\s*화", f"제{episode.group(1)}화", text, count=1)
    if not _SOURCE_QUOTES.search(source):
        for left, right in _WRAPPING_QUOTES:
            inner = text[len(left):-len(right)] if len(text) > 2 else ""
            if text.startswith(left) and text.endswith(right) and inner.strip() and left not in inner and right not in inner:
                text = inner.strip()
                break
    tail = _SOURCE_LATIN_TAIL.search(source)
    if tail:
        piece = tail.group().strip()
        if text.endswith(piece) and len(text) > len(piece):
            text = text[:-len(piece)].rstrip()
        else:
            letters = piece.lstrip("/／|｜ ")
            previous = text[-len(letters) - 1] if len(text) > len(letters) else "A"
            if text.endswith(letters) and not (previous.isascii() and previous.isalnum()):
                text = text[:-len(letters)].rstrip()
    return text


def _syllable(initial: int, medial: int, final: int = 0) -> str:
    return chr(0xAC00 + (initial * 21 + medial) * 28 + final)


def hangul_for_kana(text: str) -> str:
    """번역문에 남은 일본 글자를 소리대로 한글로 — ン은 앞 글자의 받침 ㄴ, ッ은 뒤에 글자가 이어지면 받침 ㅅ(끝이나
    부호 앞이면 뺌), ー는 늘임표(—), ・는 가운뎃점(·). 의심 대사 확인도 못 고친 것을 거르는 마지막 장치다."""
    if not text or not _KANA.search(text):
        return text
    text = re.sub("[ｦ-ﾟ]+", lambda m: unicodedata.normalize("NFKC", m.group()), text)  # 반각 가타카나
    chars = [chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text]  # 히라가나 → 가타카나
    out = []
    for i, ch in enumerate(chars):
        prev = out[-1] if out else ""
        open_syllable = bool(prev) and "가" <= prev <= "힣" and (ord(prev) - 0xAC00) % 28 == 0  # 받침 없는 앞 글자
        if ch == "ー":
            out.append("—")
        elif ch == "・":
            out.append("·")
        elif ch == "ン":
            if open_syllable:
                out[-1] = chr(ord(prev) + 4)
            else:
                out.append("응")
        elif ch == "ッ":
            following = chars[i + 1] if i + 1 < len(chars) else ""
            if open_syllable and (following in _KANA_SOUNDS or "가" <= following <= "힣"):
                out[-1] = chr(ord(prev) + 19)
        elif ch in _SMALL_KANA:
            medial = _SMALL_KANA[ch]
            if open_syllable and i > 0 and chars[i - 1] in _KANA_SOUNDS:
                initial = (ord(prev) - 0xAC00) // 28 // 21
                if chars[i - 1] == "フ":
                    initial = 17  # ファ → 파
                elif chars[i - 1] == "ウ":
                    medial = _W_GLIDES.get(medial, medial)
                elif chars[i - 1] == "シ" and medial == 5:
                    medial = 7  # シェ → 셰
                out[-1] = _syllable(initial, medial)
            else:
                out.append(_syllable(11, medial))
        elif ch in _KANA_SOUNDS:
            out.append(_syllable(*_KANA_SOUNDS[ch]))
        else:
            out.append(ch)
    return "".join(out)


def _agy_error(stderr_text: str) -> str:
    """stderr의 'AGY_ERROR: {...}' 줄에서 오류 문구를 꺼낸다 (없으면 빈 문자열)."""
    for line in reversed(stderr_text.splitlines()):
        if "AGY_ERROR:" in line:
            payload = line.split("AGY_ERROR:", 1)[1].strip()
            try:
                data = json.loads(payload)
                return str(data.get("message") or data.get("error") or payload)
            except json.JSONDecodeError:
                return payload
    return ""


def _parse_stream(stdout_text: str):
    """stream-json 출력에서 결과 이벤트를 꺼낸다. 결과 이벤트가 없으면 None."""
    result = None
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "result":
            result = event.get("result") or {}
    return result


def _stream_events(stdout_text: str) -> List[str]:
    """stream-json 출력에 온 이벤트 종류와 개수 (온 차례대로, 예: ['init', 'message×12', 'result'])."""
    counts: Dict[str, int] = {}
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            name = str(json.loads(line).get("event") or "?")
        except (json.JSONDecodeError, AttributeError):
            continue
        counts[name] = counts.get(name, 0) + 1
    return [name if count == 1 else f"{name}×{count}" for name, count in counts.items()]


def _parse_json_text(text: str) -> Optional[dict]:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


class AntigravitySession:
    """Antigravity CLI(agy) 번역 요청 — 요청마다 새 대화로 보낸다 (지침이 요청마다 실린다)."""

    backend = "antigravity"
    name = BACKEND_LABELS["antigravity"]

    def __init__(self, executable: str, system_prompt: str, model: str = "", effort: str = "",
                 timeout_sec: int = 600):
        self.executable = executable
        self.system_prompt = (system_prompt or "").strip()
        self.model = (model or "").strip()
        self.effort = (effort or "").strip().lower()
        self.timeout_sec = max(30, int(timeout_sec or 600))
        self.key = ""  # 번역 창구가 붙이는 요청 이름 — 사전에서 같은 대화의 표를 한 번만 센다
        self.last_events: List[str] = []  # 마지막 응답에서 받은 이벤트 종류 (쓸 번역이 없을 때 원인 확인용)
        self.last_response = ""  # 마지막 응답 본문 앞부분
        # 에이전트가 프로젝트 파일을 건드리지 않도록 빈 작업 폴더에서 실행한다
        self._workdir = tempfile.mkdtemp(prefix="bogopa_agy_")
        self._schema_path = self._write_schema("response_schema.json", _RESPONSE_SCHEMA)
        self._review_schema_path = self._write_schema("review_schema.json", _REVIEW_SCHEMA)
        self._shorten_schema_path = self._write_schema("shorten_schema.json", _SHORTEN_SCHEMA)
        self._confirm_schema_path = self._write_schema("confirm_schema.json", _CONFIRM_SCHEMA)

    def _write_schema(self, name: str, schema: dict) -> str:
        path = os.path.join(self._workdir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False)
        return path

    def _command(self, schema_path: str, timeout_sec: Optional[int] = None) -> List[str]:
        # 대사 목록은 명령줄 길이 제한(Windows 약 32K자)을 넘기 쉬워 표준 입력(stream-json)으로 보낸다.
        # --sandbox: 에이전트의 터미널 명령 실행을 막는다 (번역에는 도구가 필요 없다).
        cmd = [
            self.executable,
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--json-schema", schema_path,
            "--disable-slash-commands",
            "--sandbox",
            "--print-timeout", f"{int(timeout_sec or self.timeout_sec)}s",
        ]
        if self.model:
            cmd += ["--model", self.model]
        # 이름에 수준이 붙은 모델(gemini-3.8-flash-high 등)에 --effort를 같이 주면 CLI가 충돌로 거절한다.
        if self.effort and not _MODEL_LEVEL_SUFFIX.search(self.model):
            cmd += ["--effort", self.effort]
        # -p는 바로 뒤 값을 프롬프트로 삼으므로 빈 값으로 맨 끝에 둔다 (실제 프롬프트는 표준 입력)
        return cmd + ["-p="]

    def close(self):
        """빈 작업 폴더를 치운다 (대화 자체는 CLI 쪽에 남는다)."""
        shutil.rmtree(self._workdir, ignore_errors=True)

    def request(self, prompt: str, review: bool = False, shorten: bool = False, confirm: bool = False,
                limit_sec: Optional[int] = None) -> dict:
        """프롬프트 한 번을 보내고 스키마대로 받은 결과(dict)를 돌려준다. 실패하면 예외.

        review=True면 고칠 항목만 받는 형식(의심 대사 확인)으로 받는다. limit_sec를 주면 그 시간 안에 끝내고 넘기면
        TranslationTimeout을 올린다 (없으면 설정의 번역 시간 한도).
        기다리는 동안의 알림은 번역 창구를 기다리는 쪽(파이프라인)이 낸다 — 대화 여러 개가 동시에 돈다.
        """
        limit = int(limit_sec or self.timeout_sec)
        started = time.time()
        proc = subprocess.Popen(
            self._command(self._confirm_schema_path if confirm else self._shorten_schema_path if shorten else
                          self._review_schema_path if review else self._schema_path, limit), cwd=self._workdir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        payload = (json.dumps({"event": "user", "message": {"content": prompt}}, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            out, err = proc.communicate(input=payload, timeout=limit + (5 if limit_sec else 60))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise TranslationTimeout(f"번역 응답이 {limit}초 안에 오지 않았습니다")

        stdout_text = out.decode("utf-8", errors="replace")
        stderr_text = err.decode("utf-8", errors="replace")
        result = _parse_stream(stdout_text)
        self.last_events = _stream_events(stdout_text)
        self.last_response = str((result or {}).get("response") or "")[:300]
        status = str((result or {}).get("status", "")).upper()
        if proc.returncode != 0 or status != "SUCCESS":
            stderr_lines = [line for line in stderr_text.splitlines()
                            if line.strip() and "logging before google.Init" not in line]
            message = ((result or {}).get("error") or _agy_error(stderr_text)
                       or "\n".join(stderr_lines)[-500:] or f"exit {proc.returncode}")
            if any(hint in str(message).lower() for hint in _AUTH_HINTS):
                raise TranslationUnavailable(_LOGIN_HELP["antigravity"])
            if limit_sec and time.time() - started >= limit - 1:
                raise TranslationTimeout(f"번역 응답이 {limit}초 안에 오지 않았습니다")
            raise RuntimeError(f"번역 요청 실패 ({status or proc.returncode}): {message}")

        usage = result.get("usage") or {}
        logger.info("번역 응답 %.1f초 (%s), 입력 토큰 %s / 출력 토큰 %s", time.time() - started, self.key or "대화",
                    usage.get("input_tokens", "?"), usage.get("output_tokens", "?"))
        structured = result.get("structured_output")
        if not isinstance(structured, dict):
            # 스키마가 적용되지 않은 응답 대비: 본문 전체를 JSON으로 읽어 본다
            structured = _parse_json_text(str(result.get("response") or "")) or {}
        return structured


_SCHEMAS = {"response": _RESPONSE_SCHEMA, "review": _REVIEW_SCHEMA, "shorten": _SHORTEN_SCHEMA, "confirm": _CONFIRM_SCHEMA}


def _schema_kind(review: bool, shorten: bool, confirm: bool) -> str:
    return "confirm" if confirm else "shorten" if shorten else "review" if review else "response"


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _json_from_text(text: str) -> Optional[dict]:
    """응답 본문을 JSON으로 — 코드 울타리(```json)나 앞뒤 군말이 붙어 있으면 첫 { 부터 마지막 } 까지."""
    text = (text or "").strip()
    value = _parse_json_text(text) or _parse_json_text(_JSON_FENCE.sub("", text))
    if value is None and "{" in text:
        value = _parse_json_text(text[text.find("{"):text.rfind("}") + 1])
    return value


def _last_json_object(text: str) -> Optional[dict]:
    """출력 전체, 안 되면 마지막 JSON 줄 (앞에 경고 줄이 섞여 있어도)."""
    value = _parse_json_text((text or "").strip())
    if value is not None:
        return value
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            value = _parse_json_text(line)
            if value is not None:
                return value
    return None


def _strict_schema(schema):
    """엄격 스키마 — 객체마다 모든 속성을 required에 넣고 추가 속성을 막는다. 원래 선택이던 속성은 null도 받는다
    (Codex CLI가 요구한다. 받은 null은 _drop_nulls로 빼 다른 CLI와 같은 모양으로 만든다)."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if isinstance(out.get("properties"), dict):
        required = set(out.get("required") or ())
        properties = {}
        for name, sub in out["properties"].items():
            sub = _strict_schema(sub)
            if name not in required:
                kind = sub.get("type")
                sub = ({**sub, "type": [kind, "null"]} if isinstance(kind, str)
                       else {"anyOf": [sub, {"type": "null"}]})
            properties[name] = sub
        out.update(properties=properties, required=list(properties), additionalProperties=False)
    if isinstance(out.get("items"), dict):
        out["items"] = _strict_schema(out["items"])
    return out


def _drop_nulls(value):
    if isinstance(value, dict):
        return {key: _drop_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_nulls(item) for item in value]
    return value


def _tail(text: str, lines: int = 5) -> str:
    return "\n".join([line for line in (text or "").splitlines() if line.strip()][-lines:])


class _SingleShotSession:
    """요청마다 새 대화를 여는 번역 CLI(Claude Code·Codex)의 공통 틀 — AntigravitySession과 같은 모양이다.

    요청마다 지침이 실린다. 프로젝트 파일을 건드리지 않도록 빈 임시 폴더에서 도구 없이 실행하고, 프롬프트는 표준 입력으로
    보낸다 (명령줄 길이 제한).
    """

    backend = ""
    name = ""

    def __init__(self, executable: str, system_prompt: str, model: str = "", effort: str = "",
                 timeout_sec: int = 600):
        self.executable = executable
        self.system_prompt = (system_prompt or "").strip()
        self.model = (model or "").strip()
        self.effort = (effort or "").strip().lower()
        self.timeout_sec = max(30, int(timeout_sec or 600))
        self.key = ""  # 번역 창구가 붙이는 요청 이름
        self.last_events: List[str] = []  # 마지막 응답에서 받은 이벤트 종류 (쓸 번역이 없을 때 원인 확인용)
        self.last_response = ""  # 마지막 응답 본문 앞부분
        self._prefix = _launch_prefix(executable, self.backend)
        self._workdir = tempfile.mkdtemp(prefix=f"bogopa_{self.backend}_")

    def close(self):
        """빈 작업 폴더를 치운다."""
        shutil.rmtree(self._workdir, ignore_errors=True)

    def _run(self, cmd: List[str], prompt: str, limit: int) -> Tuple[int, str, str]:
        proc = subprocess.Popen(cmd, cwd=self._workdir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=_child_env(self.backend),
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            out, err = proc.communicate(input=prompt.encode("utf-8"), timeout=limit)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            raise TranslationTimeout(f"번역 응답이 {limit}초 안에 오지 않았습니다")
        return proc.returncode, out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")

    def _failure(self, message: str, auth_hints: Sequence[str], status) -> Exception:
        if any(hint in message.lower() for hint in auth_hints):
            return TranslationUnavailable(_LOGIN_HELP[self.backend])
        return RuntimeError(f"번역 요청 실패 ({status}): {message[-500:]}")

    def _log_usage(self, started: float, input_tokens, output_tokens):
        logger.info("번역 응답 %.1f초 (%s), 입력 토큰 %s / 출력 토큰 %s", time.time() - started, self.key or "대화",
                    input_tokens, output_tokens)


# Claude Code의 기본 시스템 프롬프트(코딩 도우미 설명)를 대신하는 한 줄 — 지침·형식은 요청 본문에 실린다
_CLAUDE_SYSTEM_PROMPT = "사용자 메시지에 적힌 지침대로 답합니다. 답은 지정된 JSON 형식으로만 합니다."
_CLAUDE_AUTH_HINTS = ("not logged in", "/login", "invalid api key", "oauth", "authentication_error", "failed to authenticate")


class ClaudeCodeSession(_SingleShotSession):
    """Claude Code(claude -p) 번역 요청 — 도구·슬래시 명령·MCP·세션 저장을 끄고, 전역·프로젝트 설정(CLAUDE.md·훅)을
    읽지 않는다. --bare는 쓰지 않는다 (API 키만 받아 구독 로그인으로는 못 쓴다)."""

    backend = "claude"
    name = BACKEND_LABELS["claude"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._system_path = os.path.join(self._workdir, "system_prompt.txt")
        with open(self._system_path, "w", encoding="utf-8") as f:
            f.write(_CLAUDE_SYSTEM_PROMPT)

    def _command(self, schema: dict) -> List[str]:
        cmd = self._prefix + [
            "-p", "--output-format", "json",
            "--json-schema", json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
            "--system-prompt-file", self._system_path,
            "--tools", "", "--disable-slash-commands", "--no-session-persistence", "--strict-mcp-config",
            "--setting-sources", "",
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def request(self, prompt: str, review: bool = False, shorten: bool = False, confirm: bool = False,
                limit_sec: Optional[int] = None) -> dict:
        """AntigravitySession.request와 같다 — 스키마대로 받은 결과(dict). 실패하면 예외."""
        limit = int(limit_sec or self.timeout_sec)
        started = time.time()
        code, out, err = self._run(self._command(_SCHEMAS[_schema_kind(review, shorten, confirm)]), prompt, limit)
        data = _last_json_object(out) or {}
        self.last_events = [str(data.get("subtype") or "결과 없음")] + (
            [f"turn×{data['num_turns']}"] if data.get("num_turns") else [])
        self.last_response = str(data.get("result") or "")[:300]
        if code != 0 or not data or data.get("is_error"):
            message = str(data.get("result") or "") or _tail(err) or _tail(out) or f"exit {code}"
            raise self._failure(message, _CLAUDE_AUTH_HINTS, data.get("subtype") or code)
        usage = data.get("usage") or {}
        # 입력 토큰 = 새로 읽은 것 + 캐시에 쓴 것 + 캐시에서 읽은 것
        inputs = sum(int(usage.get(name) or 0) for name in
                     ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")) if usage else "?"
        self._log_usage(started, inputs, usage.get("output_tokens", "?"))
        structured = data.get("structured_output")
        if not isinstance(structured, dict):
            # 스키마가 적용되지 않은 응답 대비: 본문을 JSON으로 읽어 본다
            structured = _json_from_text(str(data.get("result") or "")) or {}
        return structured


# 명령 실행·그림·브라우저·앱·여러 에이전트 같은 도구와 기술(skill) 사용 안내를 끈다 — 번역에는 도구가 필요 없다.
# -c로 주면 모르는 이름은 CLI가 조용히 넘기므로 버전이 바뀌어도 실행이 막히지 않는다 (--disable은 모르는 이름이면 멈춘다)
_CODEX_OFF = (
    "features.shell_tool=false", "features.unified_exec=false", "features.view_image=false",
    "features.image_generation=false", "features.browser_use=false", "features.computer_use=false",
    "features.apps=false", "features.plugins=false", "features.multi_agent=false", "features.goals=false",
    "features.hooks=false", "agents.enabled=false", "skills.include_instructions=false",
    # 웹 검색은 기본으로 켜져 있어 따로 끈다
    "web_search=disabled",
)
_CODEX_AUTH_HINTS = ("not logged in", "codex login", "please log in", "unauthorized", "authentication required",
                     "sign in")


def _codex_error(event: dict) -> str:
    if event.get("type") == "error":
        return str(event.get("message") or "")
    error = event.get("error")
    return str(error.get("message") or "") if isinstance(error, dict) else str(error or "")


def _codex_event_names(events: List[dict]) -> List[str]:
    counts: Dict[str, int] = {}
    for event in events:
        name = str(event.get("type") or "?")
        item = event.get("item")
        if isinstance(item, dict) and item.get("type"):
            name += f":{item['type']}"
        counts[name] = counts.get(name, 0) + 1
    return [name if count == 1 else f"{name}×{count}" for name, count in counts.items()]


class CodexSession(_SingleShotSession):
    """Codex CLI(codex exec) 번역 요청 — 전역 설정(config.toml)·규칙 파일을 읽지 않고 세션을 저장하지 않으며,
    빈 임시 폴더에서 읽기 전용으로 실행한다. 스키마는 엄격 규칙(모든 속성 required·추가 속성 금지)으로 바꿔 보낸다."""

    backend = "codex"
    name = BACKEND_LABELS["codex"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._schema_paths = {}
        for kind, schema in _SCHEMAS.items():
            path = os.path.join(self._workdir, f"{kind}_schema.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_strict_schema(schema), f, ensure_ascii=False)
            self._schema_paths[kind] = path
        self._answer_path = os.path.join(self._workdir, "answer.json")

    def _command(self, kind: str) -> List[str]:
        cmd = self._prefix + [
            "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
            "-s", "read-only", "-C", self._workdir, "--color", "never", "--json",
            "--output-schema", self._schema_paths[kind], "--output-last-message", self._answer_path,
        ]
        for option in _CODEX_OFF:
            cmd += ["-c", option]
        if self.effort:
            cmd += ["-c", f"model_reasoning_effort={self.effort}"]
        if self.model:
            cmd += ["-m", self.model]
        return cmd + ["-"]  # 프롬프트는 표준 입력

    def request(self, prompt: str, review: bool = False, shorten: bool = False, confirm: bool = False,
                limit_sec: Optional[int] = None) -> dict:
        """AntigravitySession.request와 같다 — 스키마대로 받은 결과(dict). 실패하면 예외."""
        limit = int(limit_sec or self.timeout_sec)
        if os.path.exists(self._answer_path):
            os.remove(self._answer_path)
        started = time.time()
        code, out, err = self._run(self._command(_schema_kind(review, shorten, confirm)), prompt, limit)
        events = [event for event in (_parse_json_text(line.strip()) for line in out.splitlines()
                                      if line.strip().startswith("{")) if event]
        self.last_events = _codex_event_names(events)
        text = ""
        if os.path.exists(self._answer_path):
            with open(self._answer_path, encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
        if not text:
            text = next((str(event["item"].get("text") or "") for event in reversed(events)
                         if event.get("type") == "item.completed" and isinstance(event.get("item"), dict)
                         and event["item"].get("type") == "agent_message"), "")
        self.last_response = text[:300]
        errors = [_codex_error(event) for event in events if event.get("type") in ("error", "turn.failed")]
        failed = any(event.get("type") == "turn.failed" for event in events)
        if code != 0 or failed or not text:
            message = "; ".join(m for m in errors if m) or _tail(err) or f"exit {code}"
            raise self._failure(message, _CODEX_AUTH_HINTS, code)
        usage = next((event.get("usage") for event in reversed(events)
                      if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict)), {})
        self._log_usage(started, usage.get("input_tokens", "?"), usage.get("output_tokens", "?"))
        return _drop_nulls(_json_from_text(text) or {})


# ── 번역 CLI 고르기 ──

_SESSION_TYPES = {"antigravity": AntigravitySession, "claude": ClaudeCodeSession, "codex": CodexSession}
_FINDERS = {"antigravity": find_agy_executable, "claude": find_claude_executable, "codex": find_codex_executable}


def find_executable(backend: str) -> Optional[str]:
    return _FINDERS[backend]()


def detect_backends() -> Dict[str, Optional[str]]:
    """설치된 번역 CLI — {이름: 실행 파일 경로(없으면 None)}, auto가 고르는 순서대로."""
    return {backend: find_executable(backend) for backend in config.TRANSLATION_BACKENDS}


def resolve_backend(choice: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """설정에서 고른 번역 CLI의 (이름, 실행 파일). auto면 설치된 것 중 agy → Claude Code → Codex 순. 못 찾으면 None."""
    choice = choice or config.TRANSLATION_BACKEND
    for backend in (config.TRANSLATION_BACKENDS if choice == "auto" else (choice,)):
        executable = find_executable(backend) if backend in _FINDERS else None
        if executable:
            return backend, executable
    return None


def missing_backend_message(choice: Optional[str] = None) -> str:
    choice = choice or config.TRANSLATION_BACKEND
    if choice == "antigravity":
        return ("Antigravity CLI(agy)를 찾지 못했습니다. 설치한 뒤 터미널에서 agy를 한 번 실행해 로그인하세요. "
                "다른 곳에 설치했다면 설정의 'CLI 위치'에 경로를 적어 주세요.")
    if choice in BACKEND_LABELS:
        return (f"설정에서 고른 {BACKEND_LABELS[choice]}를 찾지 못했습니다. 설치한 뒤 터미널에서 한 번 실행해 "
                "로그인하거나, 설정에서 다른 번역 CLI를 골라 주세요.")
    return ("번역을 맡을 CLI(Antigravity CLI·Claude Code·Codex CLI)를 찾지 못했습니다. 하나를 설치한 뒤 "
            "터미널에서 한 번 실행해 로그인해 주세요.")


def create_session(backend: str, executable: str, system_prompt: str, model: str = "", effort: str = "",
                   timeout_sec: int = 600):
    """고른 번역 CLI의 요청 세션 — 셋 다 request(prompt, review/shorten/confirm, limit_sec)·close()가 같다."""
    return _SESSION_TYPES[backend](executable, system_prompt, model=model, effort=effort, timeout_sec=timeout_sec)


# Claude Code에는 모델 목록 명령이 없어 도움말(--model)에 적힌 별칭만 둔다 — 별칭은 그 계열의 최신 모델이다
_CLAUDE_MODEL_ALIASES = (("fable", "Fable"), ("opus", "Opus"), ("sonnet", "Sonnet"))
_OPTIONS_TTL_SEC = 1800
_options_cache: Dict[Tuple[str, str], Tuple[float, dict]] = {}
_options_lock = threading.Lock()


def _agy_models(executable: str) -> List[dict]:
    code, out, err = _run_quick([executable, "models"], 60)
    if code != 0:
        raise RuntimeError(_agy_error(err) or f"exit {code}")
    models = []
    for line in out.splitlines():
        model, _, label = line.strip().partition("\t")
        if label.strip() and config.valid_model_name(model):
            entry = {"id": model, "label": label.strip()}
            if _MODEL_LEVEL_SUFFIX.search(model):
                entry["efforts"] = []  # 이름에 수준이 붙은 모델 — --effort를 같이 주면 CLI가 거절한다
            models.append(entry)
    return models


def _codex_models(executable: str) -> List[dict]:
    code, out, _ = _run_quick(_launch_prefix(executable, "codex") + ["debug", "models"], 60)
    catalog = _parse_json_text(out) if code == 0 else None
    if not isinstance(catalog, dict):
        raise RuntimeError(f"모델 목록을 읽지 못했습니다 (exit {code})")
    listed = [m for m in catalog.get("models") or [] if isinstance(m, dict) and m.get("visibility") == "list"
              and config.valid_model_name(str(m.get("slug") or ""))]
    models = []
    for model in sorted(listed, key=lambda m: m.get("priority") or 0):
        levels = [str(level.get("effort") if isinstance(level, dict) else level)
                  for level in model.get("supported_reasoning_levels") or []]
        models.append({"id": model["slug"], "label": str(model.get("display_name") or model["slug"]),
                       "efforts": [level for level in config.TRANSLATION_EFFORT_LEVELS["codex"] if level in levels]})
    return models


def backend_options(backend: str, executable: Optional[str] = None) -> dict:
    """고를 수 있는 모델과 생각 깊이 — {"models": [{"id", "label"[, "efforts"]}], "efforts": [...]}.

    models의 첫 항목(id "")은 CLI 기본 모델이다. 모델 목록은 CLI가 알려 주는 것(agy models, codex debug models)이고,
    목록 명령이 없는 Claude Code는 도움말에 적힌 별칭만 둔다. 모델에 efforts가 있으면 그 모델은 그 값만 받는다
    (빈 목록 = 이름에 수준이 붙어 생각 깊이를 따로 주지 않는 모델). efforts는 기본 모델이 받는 값 — Codex는 목록의
    모든 모델이 받는 값이다. 목록은 잠시 기억해 둔다 (명령마다 몇 초 걸린다).
    """
    executable = executable or find_executable(backend)
    key = (backend, executable or "")
    with _options_lock:
        hit = _options_cache.get(key)
        if hit and time.time() < hit[0]:
            return hit[1]
    options = {"models": [{"id": "", "label": "기본"}], "efforts": list(config.TRANSLATION_EFFORT_LEVELS[backend])}
    keep = _OPTIONS_TTL_SEC
    try:
        if backend == "claude":
            options["models"] += [{"id": alias, "label": label} for alias, label in _CLAUDE_MODEL_ALIASES]
        elif executable and backend == "antigravity":
            options["models"] += _agy_models(executable)
        elif executable and backend == "codex":
            models = _codex_models(executable)
            options["models"] += models
            common = [level for level in options["efforts"] if all(level in m["efforts"] for m in models)]
            options["efforts"] = common or options["efforts"]
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as e:
        logger.info("%s 모델 목록을 읽지 못했습니다: %s", BACKEND_LABELS[backend], e)
        keep = 60  # 로그인 전일 수 있어 곧 다시 본다
    with _options_lock:
        _options_cache[key] = (time.time() + keep, options)
    return options


def _request_with_retry(session: AntigravitySession, prompt_builder: Callable[[], str],
                        callback: Optional[Callable] = None) -> dict:
    """일시 오류는 점증 대기 후 다시 요청한다. 로그인 필요 등은 바로 올려보낸다.

    프롬프트는 시도마다 다시 만든다 — 그사이 동시에 도는 다른 요청이 사전에 정한 표기도 붙는다.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            return session.request(prompt_builder())
        except TranslationUnavailable:
            raise
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
            logger.info(f"번역 요청 오류 (시도 {attempt + 1}/{MAX_ATTEMPTS}): {e}. {delay}초 후 재시도...")
            if callback:
                callback(ProgressEvent(
                    PipelinePhase.TRANSLATION, 0, 0,
                    f"번역 요청 오류, {delay}초 후 재시도 ({attempt + 1}/{MAX_ATTEMPTS})...",
                ))
            time.sleep(delay)


class KeyedItems(dict):
    """'쪽.번호' → 대사. free에는 말풍선 밖 글자의 번호를 둔다 (요청에 말풍선 안/밖을 싣는다)."""

    def __init__(self, *args, free=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.free = set(free)

    def subset(self, keys) -> "KeyedItems":
        keys = set(keys)
        return KeyedItems(((k, v) for k, v in self.items() if k in keys), free=self.free & keys)


def request_text(text: str) -> str:
    """요청에 싣는 원문 — 전각 영숫자를 반각으로 (ＡＢＣ１２ → ABC12)."""
    return _FULLWIDTH_ALNUM.sub(lambda m: chr(ord(m.group()) - 0xFEE0), text or "")


_RUBY_PAIR = re.compile(r"^(.+)\((.+)\)$")


def _ruby_pairs(text: str, pairs: Sequence[str]) -> List[str]:
    """요청에 싣는 후리가나 쌍 — 로마자 약어 본문은 원문 표기(대소문자·반각)로 맞춘다 (ab(…) → AB(…))."""
    lowered = text.lower()
    out = []
    for pair in pairs:
        match = _RUBY_PAIR.match(request_text(pair))
        if match and match.group(1).isascii():
            base = match.group(1)
            at = lowered.find(base.lower())
            if at >= 0:
                pair = f"{text[at:at + len(base)]}({match.group(2)})"
        out.append(pair)
    return out


def _item(key: str, element: TextElement, free: bool) -> dict:
    text = request_text(element.original_text)
    item = {"id": key, "text": text}
    if free:
        item["free"] = True
    if element.furigana:
        item["ruby"] = _ruby_pairs(text, element.furigana)
    return item


def _build_prompt(session: AntigravitySession, keyed_items: Dict[str, TextElement], note: str,
                  glossary_lines: Optional[List[str]] = None) -> str:
    free = getattr(keyed_items, "free", set())
    items = [_item(key, element, key in free) for key, element in keyed_items.items()]
    # 요청마다 새 대화라 지침과 형식을 늘 싣는다
    parts = [session.system_prompt, _FORMAT_INSTRUCTION]
    if glossary_lines:
        # 다른 대화가 정한 표기까지 — 이 요청에 나오는 말만 붙인다
        parts.append(_GLOSSARY_HEADER + "\n" + "\n".join(glossary_lines))
    parts.append(f"{note}\n{json.dumps(items, ensure_ascii=False)}")
    return "\n\n".join(parts)


_JAPANESE_LETTER = re.compile("[ぁ-ゖァ-ヺー㐀-䶿一-鿿々]")


def _echoes_original(original: str, text: str) -> bool:
    """가나·한자 없는 원문(영문 로고·표어·약어)을 번역이 그대로 옮겨 적었는지 — 지우고 같은 글을 다시 쓰면 원본
    글씨만 잃고, 글자 인식이 잘못 읽은 글자까지 그대로 옮겨 쓰게 된다.
    일본어 줄을 그대로 돌려준 실패는 의심 대사 확인이 다시 묻도록 남긴다."""
    if not original or _JAPANESE_LETTER.search(original):
        return False
    squash = lambda s: re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or "")).casefold()  # noqa: E731
    return bool(squash(original)) and squash(original) == squash(text)


def _apply_translations(structured: dict, keyed_items: Dict[str, TextElement]) -> List[str]:
    """응답을 대사에 채우고 답이 온 id 목록을 돌려준다. 원문을 그대로 옮겨 적은 답은 번역 제외(원본 그대로)로 둔다."""
    answered = []
    unknown = []
    for item in (structured or {}).get("translations", []) or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id", "")).strip()
        element = keyed_items.get(key)
        if element is None:
            unknown.append(key)
            continue
        text = clean_translation(element.original_text, item.get("text"), element.furigana or ())
        if item.get("skip") or not text or text in _LEGACY_SKIP_TEXTS or _echoes_original(element.original_text, text):
            element.translated_text = None
            element.translation_status = TranslationStatus.EXCLUDED
        else:
            element.translated_text = text
            element.translation_status = TranslationStatus.TRANSLATED
        answered.append(key)
    if unknown:
        logger.warning(f"번역 응답에 알 수 없는 번호 {len(unknown)}개: {unknown[:5]}")
    return answered


def pages_label(page_numbers: Sequence[int]) -> str:
    """책에서 몇 쪽인지 — 'a~b쪽', 띄엄띄엄이면 'a~b쪽 중 n쪽'."""
    numbers = sorted(set(page_numbers))
    if not numbers:
        return ""
    first, last = numbers[0], numbers[-1]
    span = f"{first}쪽" if first == last else f"{first}~{last}쪽"
    return span if len(numbers) == last - first + 1 else f"{span} 중 {len(numbers)}쪽"


def prepare_items(pages: List[PageData], page_numbers: Optional[Sequence[int]] = None) -> Dict[str, TextElement]:
    """번역할 대사(대기·실패)를 읽는 순서대로 '쪽.번호'로 모은다. 효과음·부속물·빈 판독은 여기서 번역 제외로 둔다.

    읽는 순서는 컷을 나눠 정하므로 쪽 그림(image_rgb)이 있을 때 부른다 (그림이 없으면 줄 순서).
    """
    numbers = page_numbers or range(1, len(pages) + 1)
    keyed_items = KeyedItems()
    skipped = Counter()
    # 말풍선이 적은 쪽은 이번 묶음 전체의 말풍선 대사 크기로 효과음 크기를 잰다
    fallback_size = dialogue_size([b for page in pages for b in page.speech_bubbles])
    for page_number, page_data in zip(numbers, pages):
        reference = dialogue_size(page_data.speech_bubbles, fallback_size)
        freeform = {id(element) for element in page_data.freeform_texts}
        # 탐지 순서가 아니라 읽는 순서로 보낸다 — 앞뒤 대사가 이어져야 문맥이 산다
        for text_idx, element in enumerate(reading_order(page_data), start=1):
            if not element.needs_translation:
                continue
            if not (element.original_text or "").strip():
                element.translation_status = TranslationStatus.EXCLUDED  # 읽힌 글자가 없음
                continue
            ratio = element.font_size / reference if reference and element.font_size else None
            reason = exclusion_reason(element.original_text, in_bubble=id(element) not in freeform, size_ratio=ratio)
            if reason:
                element.translation_status = TranslationStatus.EXCLUDED  # 효과음·부속물은 원본 그대로 둔다
                skipped[reason] += 1
                continue
            key = f"{page_number}.{text_idx}"
            keyed_items[key] = element
            if id(element) in freeform:
                keyed_items.free.add(key)
    if skipped:
        logger.info("번역하지 않는 글자: " + ", ".join(f"{reason} {count}개" for reason, count in skipped.items()))
    return keyed_items


def translate_items(session: AntigravitySession, keyed_items: Dict[str, TextElement], label: str = "",
                    glossary=None, callback: Optional[Callable] = None):
    """대사 묶음 하나를 번역해 채운다. 빠진 대사는 한 번 더 요청하고, 끝내 못 받은 대사는 '실패'로 둔다.

    실패가 남은 페이지는 식자하지 않고 다음 실행에서 그 대사만 다시 요청한다. 받은 인명·용어는 사전에 센다.
    로그인 필요 같은 실패는 그대로 올려 실행을 멈춘다.
    """
    if not keyed_items:
        return
    logger.info(f"{label or '묶음'} 대사 {len(keyed_items)}개 번역 요청 ({session.key or '대화'})...")
    originals = [element.original_text or "" for element in keyed_items.values()]
    heading = f"## 이번 묶음 — 책의 {label}" if label else "## 이번 묶음"

    def batch_note():
        return f"{heading}\n책을 여러 번에 나눠 보냅니다. 인명·용어 사전이 붙어 있으면 그 표기를 그대로 따르세요."

    request_originals = [request_text(text) for text in originals]
    given: Dict[str, str] = {}  # 이번 요청에 붙인 사전 표기 — 다르게 적어 온 표기는 세지 않는다

    def build(items, note):
        lines = None
        if glossary is not None:
            # 요청을 만드는 순간의 사전 — 동시에 도는 다른 요청이 그사이 정한 표기도 붙는다
            given.clear()
            given.update(glossary.given_map(request_originals))
            lines = glossary.prompt_lines(request_originals)
        return _build_prompt(session, items, note(), glossary_lines=lines)

    def ask(items, note):
        structured = _request_with_retry(session, lambda: build(items, note), callback)
        if glossary is not None:
            glossary.record(session.key, structured.get("terms"), originals, given=given)
        got = set(_apply_translations(structured, items))
        if items and not got:
            # 원인 확인용 — 이벤트 종류와 응답 본문 앞부분만 (대화 ID·오류 출력은 적지 않는다)
            logger.warning(f"번역 응답에 쓸 수 있는 번역이 없습니다 ({session.key or '대화'}) — 받은 이벤트: "
                           f"{', '.join(session.last_events) or '없음'} · 응답 키: {sorted((structured or {}).keys())}"
                           f" · 응답 앞부분: {session.last_response!r}")
        return got

    if not isinstance(keyed_items, KeyedItems):
        keyed_items = KeyedItems(keyed_items)
    answered = set()
    try:
        answered |= ask(keyed_items, batch_note)
        missing = keyed_items.subset(key for key in keyed_items if key not in answered)
        if missing:
            logger.info(f"누락된 번역 {len(missing)}개를 새 대화로 재요청: {list(missing)[:10]}")
            if callback:
                callback(ProgressEvent(
                    PipelinePhase.TRANSLATION, 0, len(missing),
                    f"누락된 번역 {len(missing)}개 재요청 중...", level=EventLevel.WARNING,
                ))
            # 빠진 대사는 새 대화로 다시 보낸다 (지침·사전·빠진 대사만)
            answered |= ask(missing, batch_note)
    except TranslationUnavailable:
        raise
    except Exception as e:
        # 첫 응답을 받은 뒤 빠진 항목 요청만 실패했으면 받은 번역은 그대로 둔다
        logger.error(f"번역 요청 실패 ({MAX_ATTEMPTS}회 시도): {e}")

    failed = [key for key in keyed_items if key not in answered]
    for key in failed:
        keyed_items[key].translation_status = TranslationStatus.FAILED
    if failed and callback:
        callback(ProgressEvent(
            PipelinePhase.TRANSLATION, len(keyed_items) - len(failed), len(keyed_items),
            f"번역 실패 {len(failed)}개 — 다시 실행하면 이 대사만 다시 요청합니다",
            level=EventLevel.WARNING, extras={"failed_keys": failed[:20]},
        ))


def _key_order(key: str):
    page, _, index = key.partition(".")
    try:
        return int(page), int(index)
    except ValueError:
        return 0, 0


_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
# 말 사이에 홀로 낀 인사·사과의 첫 글자 — 칸에 잘린 ごめん·ありがとう·すみません일 수 있다
_CUT_FRAGMENT = re.compile(r"(?<=[てで、!！?？…⋯])(?:ご|あり|すみ)(?=[一-鿿])")
# ワ↔フ 오독으로 대명사 ワシ가 フシ로 읽힌 것 (문장 첫머리·조사 앞)
_PRONOUN_MISREAD = re.compile(r"(?:^|(?<=[!！?？。、…⋯\s]))フシ(?=[はがもの])")


def _near(a: str, b: str) -> bool:
    """두 로마자 낱말이 한두 글자만 다른지 (빠짐·바뀜)."""
    a, b = a.lower(), b.lower()
    if a == b or abs(len(a) - len(b)) > 2:
        return False
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1] <= 2 and previous[-1] < min(len(a), len(b)) - 1


_KATAKANA_WORD = re.compile(r"[ァ-ヺー]{2,}")
# 글자 인식이 가타카나를 모양이 닮은 한자로 읽는 것 (ロ → 口, カ → 力) — 이것을 가나로 되돌려 책의 이름과 맞춰 본다
_KANJI_LOOKALIKE = str.maketrans({"口": "ロ", "回": "ロ", "二": "ニ", "力": "カ", "夕": "タ", "工": "エ", "卜": "ト",
                                  "八": "ハ", "一": "ー"})
_KATAKANA_LOOKALIKE_WORD = re.compile(r"[ァ-ヺー口回二力夕工卜八一]{2,}")
_EXCLAIM = re.compile("[!！]")
_REASK = re.compile("[다라냐]고[?？]")  # 단정을 되묻는 말로 바꾼 꼴 (~했다고?!)
_NAME_MIN_COUNT = 3  # 이 번 이상 나온 가타카나 낱말만 '책의 이름'으로 보고 한 글자 다르게 읽힌 것을 의심한다


def _one_kana_off(a: str, b: str) -> bool:
    """길이가 같고 한 글자만 다른 가타카나 낱말인지."""
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def check_notes(keyed_items: KeyedItems) -> Dict[str, List[str]]:
    """번역을 마친 대사 가운데 프로그램이 찾을 수 있는 틀림 의심 — {id: [까닭, ...]}.

    책에 더 자주 나온 로마자·가타카나 이름과 한두 글자만 다르게 읽힌 낱말, 가타카나를 닮은 한자로 읽힌 이름,
    칸에 잘린 말의 첫 글자(ご·あり·すみ), 대명사 오독(フシ), 원문은 느낌표뿐인데 되묻는 말이 된 번역, 번역문에 남은
    일본 글자·한자. 번역한 대사만 본다.
    """
    notes: Dict[str, List[str]] = {}
    done = {key: element for key, element in keyed_items.items()
            if element.translation_status == TranslationStatus.TRANSLATED and element.translated_text}
    where: Dict[str, List[str]] = {}
    for key, element in keyed_items.items():
        text = request_text(element.original_text)
        for word in _LATIN_WORD.findall(text) + _KATAKANA_WORD.findall(text):
            where.setdefault(word, []).append(key)
    for word, keys in where.items():
        if word[0].isascii():
            better = [other for other, other_keys in where.items() if len(other_keys) > len(keys) and _near(word, other)]
        else:
            better = [other for other, other_keys in where.items() if not other[0].isascii()
                      and len(other_keys) >= max(_NAME_MIN_COUNT, 2 * len(keys)) and _one_kana_off(word, other)]
        if better:
            best = max(better, key=lambda other: len(where[other]))
            for key in keys:
                notes.setdefault(key, []).append(f"'{word}'는 책에 {len(where[best])}번 나온 '{best}'를 잘못 읽은 것일 수 있음")
    names = {word for word, keys in where.items() if not word[0].isascii() and len(keys) >= 2}
    for key, element in done.items():
        text = request_text(element.original_text)
        for word in _KATAKANA_LOOKALIKE_WORD.findall(text):
            fixed = word.translate(_KANJI_LOOKALIKE)
            if fixed != word and fixed in names and _KATAKANA_WORD.fullmatch(fixed):
                notes.setdefault(key, []).append(f"'{word}'는 책에 나온 이름 '{fixed}'를 닮은 한자로 잘못 읽은 것일 수 있음")
        fragment = _CUT_FRAGMENT.search(text)
        if fragment:
            notes.setdefault(key, []).append(f"'{fragment.group()}'는 칸에 잘린 말의 첫 글자일 수 있음 — 확실하면 되살려 옮기기")
        misread = _PRONOUN_MISREAD.search(text)
        if misread:
            notes.setdefault(key, []).append(f"'{misread.group()}'는 1인칭 대명사 ワシ를 잘못 읽은 것일 수 있음")
        if (_REASK.search(element.translated_text) and not _QUESTION.search(element.original_text or "")
                and _EXCLAIM.search(element.original_text or "")):
            notes.setdefault(key, []).append("원문은 느낌표뿐인데 번역이 되묻는 말('~다고?')이 됨 — 원문이 단정이면 단정으로")
        if _KANA.search(element.translated_text):
            notes.setdefault(key, []).append("번역에 일본 글자가 남음 — 한국어 표기로")
        if _HANJA.search(element.translated_text):
            notes.setdefault(key, []).append("번역에 한자가 남음 — 한국어로 옮기기")
    return {key: reasons for key, reasons in notes.items() if key in done}


def _check_prompt(session: AntigravitySession, keyed_items: KeyedItems, flagged: Dict[str, List[str]],
                  glossary=None) -> str:
    """의심 대사만 앞뒤 대사와 함께 싣는다 (앞뒤는 같은 쪽에서 읽는 순서로 _CHECK_CONTEXT개씩)."""
    order = sorted(keyed_items, key=_key_order)
    position = {key: i for i, key in enumerate(order)}

    def neighbours(key, step):
        page, found, i = _key_order(key)[0], [], position[key] + step
        while 0 <= i < len(order) and len(found) < _CHECK_CONTEXT and _key_order(order[i])[0] == page:
            element = keyed_items[order[i]]
            found.append({"text": request_text(element.original_text), "ko": element.translated_text or ""})
            i += step
        return found[::-1] if step < 0 else found

    items = []
    for key in sorted(flagged, key=_key_order):
        element = keyed_items[key]
        item = _item(key, element, key in keyed_items.free)
        item["ko"] = element.translated_text or ""
        for side, step in (("before", -1), ("after", 1)):
            context = neighbours(key, step)
            if context:
                item[side] = context
        item["why"] = flagged[key]
        items.append(item)
    parts = [session.system_prompt, _CHECK_INSTRUCTION]
    if glossary is not None:
        lines = glossary.prompt_lines(request_text(keyed_items[key].original_text) for key in flagged)
        if lines:
            parts.append(_GLOSSARY_HEADER + "\n" + "\n".join(lines))
    parts.append("## 확인할 대사\n" + json.dumps(items, ensure_ascii=False))
    return "\n\n".join(parts)


def check_items(session: AntigravitySession, keyed_items: KeyedItems, prompt: str,
                limit_sec: int = CHECK_LIMIT_SEC) -> Optional[dict]:
    """의심 대사를 한 번 보내 고칠 항목만 받는다 — 곁가지 요청이라 생각을 짧게(SIDE_EFFORT) 하고, limit_sec 안에
    못 받거나 실패하면 다시 묻지 않고 None (지금 번역을 그대로 쓴다)."""
    logger.info(f"의심 대사 {len(keyed_items)}개 확인 요청 ({session.key or '대화'})...")
    session.effort = SIDE_EFFORT
    try:
        return session.request(prompt, review=True, limit_sec=limit_sec)
    except TranslationTimeout:
        logger.warning(f"의심 대사 확인이 {limit_sec}초 안에 오지 않아 지금 번역을 그대로 씁니다")
    except Exception as e:  # noqa: BLE001 — 확인은 다듬기일 뿐, 실패해도 번역은 멈추지 않는다
        logger.warning(f"의심 대사 확인을 받지 못해 지금 번역을 그대로 씁니다: {e}")
    return None


_SENTENCE = re.compile(r"[^.!?。！？⋯…]*[^\s.!?。！？⋯…][^.!?。！？⋯…]*(?:[.!?。！？⋯…]+|$)")
_PUNCT_OR_SPACE = re.compile(r"[\s.,!?。、！？⋯…~〜ー—\-・「」『』()（）'\"“”‘’☆★♡♪]")


def _sentences(text: str) -> int:
    return len(_SENTENCE.findall(text or ""))


def _letters(text: str) -> int:
    return len(_PUNCT_OR_SPACE.sub("", text or ""))


def _adds_content(original: str, before: str, after: str) -> bool:
    """고침이 원문에 없는 말을 덧붙였는지 — 문장이 원문·원래 번역보다 늘었거나, 원래 번역이 원문만큼 긴데
    고친 번역이 훨씬 길어졌으면 뜻을 더한 것으로 본다 (빠진 뜻을 채운 고침은 원래 번역이 짧다)."""
    grown = _letters(after) - _letters(before)
    # 잘려 읽힌 말(ご… → ごめん)을 되살리면 짧은 문장이 하나 는다 — 글자가 거의 늘지 않았으면 받는다
    if _sentences(after) > max(_sentences(before), _sentences(original)) and grown > 2:
        return True
    full = _letters(before) >= 0.6 * _letters(original)
    return full and grown > max(4, 0.35 * _letters(before))


_QUESTION = re.compile("[?？]")
# 대명사를 소리대로 적은 말 ('와시는', '오레가') — 말투 대신 소리로 바꾼 고침은 받지 않는다
_PRONOUN_SOUND = re.compile("(?<![가-힣])(?:" + "|".join(sorted(_PRONOUN_TARGETS, key=len, reverse=True))
                            + ")(?=[는가를도의야이랑한]|[^가-힣]|$)")


def _adds_question(original: str, before: Optional[str], after: str) -> bool:
    """원문·원래 번역에 없던 물음표를 고침이 붙였는지 — 단정·외침이 되묻는 말로 바뀐다."""
    return bool(_QUESTION.search(after)) and not _QUESTION.search(original or "") and not _QUESTION.search(before or "")


def apply_review(structured: dict, keyed_items: KeyedItems, glossary=None) -> List[tuple]:
    """확인 응답(고칠 항목만)을 대사에 채운다. 고친 (id, 전, 후, 이유) 목록을 돌려준다.

    보낸 대사(번역함·번역 안 함)만 고친다. 원래 번역보다 훨씬 길어진 답은 말풍선에 안 들어가므로 받지 않고,
    문장이 늘었거나 원문만큼 길던 번역이 크게 불어난 고침은 원문에 없는 말을 더한 것으로 보아 받지 않는다.
    원문에 없는 물음표를 붙이거나, 원문에 있는 사전 이름을 빼거나, 대명사를 소리대로 적은 고침도 받지 않는다.
    """
    if glossary is not None:
        # 사전 고침을 먼저 굳힌다 — 아래에서 원문의 이름을 뺀 고침을 가릴 때 새 표기로 본다
        haystack = "\n".join(request_text(element.original_text) for element in keyed_items.values())
        # 대명사(ワシ·オレ)와 그 소리 표기(와시·오레)는 clean_term이 걸러 사전에 굳히지 않는다
        for term in (structured or {}).get("terms", []) or []:
            if isinstance(term, dict) and term.get("source") and str(term["source"]).strip() in haystack:
                if glossary.fix(term.get("source"), term.get("target")):
                    logger.info(f"의심 대사 확인이 사전 표기를 고쳤습니다: {term.get('source')} → {term.get('target')}")
    changes = []
    for item in (structured or {}).get("fixes", []) or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id", "")).strip()
        element = keyed_items.get(key)
        if element is None or element.translation_status not in (TranslationStatus.TRANSLATED,
                                                                  TranslationStatus.EXCLUDED):
            continue
        before = element.translated_text if element.translation_status == TranslationStatus.TRANSLATED else None
        reason = " ".join(str(item.get("reason") or "").split())
        text = clean_translation(element.original_text, item.get("text"), element.furigana or ())
        # 새로 번역하는 글자는 원문 길이로, 고치는 번역은 원래 번역 길이로 잰다
        limit = (len(before) * 1.6 + 6) if before is not None else (len(element.original_text or "") * 2 + 6)
        if item.get("skip") or text in _LEGACY_SKIP_TEXTS or _echoes_original(element.original_text, text):
            if before is None:
                continue
            element.translated_text = None
            element.translation_status = TranslationStatus.EXCLUDED
            changes.append((key, before, None, reason))
        elif text and text != before and len(text) <= limit:
            if before is not None and _adds_content(element.original_text or "", before, text):
                logger.info(f"의심 대사 확인 {key}:원문에 없는 말을 더한 고침은 받지 않습니다 — {before} → {text}")
                continue
            if _adds_question(element.original_text, before, text):
                logger.info(f"의심 대사 확인 {key}:원문에 없는 물음표를 붙인 고침은 받지 않습니다 — {before} → {text}")
                continue
            dropped = [target for _, target in (glossary.terms_for([request_text(element.original_text)])
                                                if glossary is not None and before else [])
                       if target in before and target not in text]
            if dropped:
                logger.info(f"의심 대사 확인 {key}:원문에 있는 이름을 뺀 고침은 받지 않습니다({', '.join(dropped)}) — {before} → {text}")
                continue
            if _PRONOUN_SOUND.search(text) and not _PRONOUN_SOUND.search(before or ""):
                logger.info(f"의심 대사 확인 {key}:대명사를 소리대로 옮긴 고침은 받지 않습니다 — {before} → {text}")
                continue
            element.translated_text = text
            element.translation_status = TranslationStatus.TRANSLATED
            changes.append((key, before, text, reason))
    return changes


def request_shorter(session: AntigravitySession, items: List[dict], glossary=None,
                    limit_sec: int = SHORTEN_LIMIT_SEC) -> Tuple[Dict[str, List[str]], dict]:
    """배치가 어려운 말풍선 대사들의 짧은 대안을 한 번에 받는다 — ({id: [대안, ...]}, {status, seconds}).

    곁가지 요청이라 생각을 짧게(SIDE_EFFORT) 하고, limit_sec 안에 못 받거나 실패하면 다시 묻지 않고 빈 dict
    (status: timeout·failed) — 지금 번역을 그대로 쓴다.
    items: {id, element(TextElement), before(앞 대사 목록), after(뒤 대사 목록), max_chars_per_line, max_lines, about_chars}
    """
    if not items:
        return {}, {"status": "ok", "seconds": 0.0}
    payload = []
    for item in items:
        element = item["element"]
        entry = _item(item["id"], element, False)
        entry["ko"] = element.translated_text or ""
        for side in ("before", "after"):
            if item.get(side):
                entry[side] = item[side]
        for field in ("max_chars_per_line", "max_lines", "about_chars"):
            entry[field] = item[field]
        payload.append(entry)
    parts = [session.system_prompt, _SHORTEN_INSTRUCTION]
    if glossary is not None:
        lines = glossary.prompt_lines(request_text(item["element"].original_text) for item in items)
        if lines:
            parts.append(_GLOSSARY_HEADER + "\n" + "\n".join(lines))
    parts.append("## 줄일 대사\n" + json.dumps(payload, ensure_ascii=False))
    prompt = "\n\n".join(parts)
    logger.info(f"말풍선에 맞춘 짧은 번역 {len(items)}개 요청 ({session.key or '대화'})...")
    session.effort = SIDE_EFFORT
    started = time.time()
    try:
        structured = session.request(prompt, shorten=True, limit_sec=limit_sec)
    except TranslationTimeout:
        logger.warning(f"짧은 번역이 {limit_sec}초 안에 오지 않아 지금 번역을 그대로 씁니다")
        return {}, {"status": "timeout", "seconds": round(time.time() - started, 1)}
    except Exception as e:  # noqa: BLE001 — 다듬기일 뿐, 실패하면 지금 번역을 그대로 쓴다
        logger.warning(f"짧은 번역을 받지 못해 지금 번역을 그대로 씁니다: {e}")
        return {}, {"status": "failed", "seconds": round(time.time() - started, 1)}
    known = {item["id"]: item["element"] for item in items}
    out: Dict[str, List[str]] = {}
    for entry in (structured or {}).get("alternatives", []) or []:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("id", "")).strip()
        element = known.get(key)
        if element is None:
            continue
        texts = [clean_translation(element.original_text, text, element.furigana or ())
                 for text in (entry.get("texts") or []) if isinstance(text, str)]
        out[key] = [text for text in texts if text]
    return out, {"status": "ok", "seconds": round(time.time() - started, 1)}


_POLITE_END = re.compile(r"(?:요|니다|니까|세요|죠)[\s!?！？.。⋯…~〜—\-]*$")
_SHORTER_MIN_SHARE = 0.6  # 짧은 대안은 지금 번역 글자 수의 이 비율 이상을 남긴다


def confirm_shorter(session: AntigravitySession, items: List[dict],
                    limit_sec: int = CONFIRM_LIMIT_SEC) -> Tuple[Optional[Dict[str, bool]], dict]:
    """새 낱말을 들여온 짧은 대안들이 원문 뜻을 지키는지 한 번에 묻는다 — ({id: OK인지}, {status, seconds}).

    번역 지침 없이 확인 지침과 항목만 보내고 생각을 짧게(SIDE_EFFORT) 한다. limit_sec 안에 못 받거나 실패하면
    다시 묻지 않고 None (status: timeout·failed) — 확인이 필요한 대안은 쓰지 않는다.
    items: {id, element(TextElement — 원문·지금 번역), short(줄인 번역)}. 답이 없거나 OK가 아닌 항목은 NG로 본다.
    """
    if not items:
        return {}, {"status": "ok", "seconds": 0.0}
    payload = [{"id": item["id"], "text": request_text(item["element"].original_text),
                "ko": item["element"].translated_text or "", "short": item["short"]} for item in items]
    prompt = "\n\n".join([_CONFIRM_INSTRUCTION, "## 확인할 번역\n" + json.dumps(payload, ensure_ascii=False)])
    logger.info(f"줄인 번역 {len(items)}개 뜻 확인 요청 ({session.key or '대화'})...")
    session.effort = SIDE_EFFORT
    started = time.time()
    try:
        structured = session.request(prompt, confirm=True, limit_sec=limit_sec)
    except TranslationTimeout:
        logger.warning(f"줄인 번역 확인이 {limit_sec}초 안에 오지 않아 새 낱말이 든 대안은 쓰지 않습니다")
        return None, {"status": "timeout", "seconds": round(time.time() - started, 1)}
    except Exception as e:  # noqa: BLE001 — 확인을 못 받으면 확인이 필요한 대안은 쓰지 않는다
        logger.warning(f"줄인 번역 확인을 받지 못해 새 낱말이 든 대안은 쓰지 않습니다: {e}")
        return None, {"status": "failed", "seconds": round(time.time() - started, 1)}
    verdicts = {str(v.get("id", "")).strip(): str(v.get("verdict", "")).strip().upper().startswith("OK")
                for v in (structured or {}).get("verdicts", []) or [] if isinstance(v, dict)}
    return ({item["id"]: verdicts.get(item["id"], False) for item in items},
            {"status": "ok", "seconds": round(time.time() - started, 1)})


def acceptable_shorter(element: TextElement, text: str, glossary=None) -> bool:
    """짧은 대안이 지금 번역의 뜻·표기를 지키는지 — 더 길거나, 물음표·문장·대명사 소리를 더했거나, 존댓말을 반말로 줄였거나,
    사전 이름을 뺐으면 아님. 지금 번역에 없는 낱말을 들여온 대안은 여기서 거르지 않고 new_words로 확인 요청에 올린다."""
    before = element.translated_text or ""
    if not text or text == before or _letters(text) > _letters(before):
        return False
    if _letters(text) < _SHORTER_MIN_SHARE * _letters(before):
        return False  # 이만큼 넘게 줄인 대안은 뜻(되풀이·부르는 말·맞장구)을 뺀 것으로 본다
    if _adds_content(element.original_text or "", before, text) or _adds_question(element.original_text, before, text):
        return False
    if _PRONOUN_SOUND.search(text) and not _PRONOUN_SOUND.search(before):
        return False
    if _POLITE_END.search(before) and not _POLITE_END.search(text):
        return False  # 존댓말을 반말로 줄였다
    if glossary is not None:
        for _, target in glossary.terms_for([request_text(element.original_text)]):
            if target and target in before and target not in text:
                return False
    return True


# 짧은 대안의 낱말이 지금 번역에서 왔는지(새 낱말이 있으면 번역기에 뜻 확인을 한 번 더 받는다) — 형태소 분석기 없이 어절 머리(두 글자 어절은 첫 글자, 세 글자 이상은 첫
# 두 글자 = 대개 체언·용언 어간)가 지금 번역(띄어쓰기·부호 뺀 글)에 있는지 본다. 흔한 준말은 본말로도 찾는다
_CONTRACTIONS = {
    "얘": ("이야",), "뭐": ("무엇", "무슨"), "뭔": ("무슨", "무엇", "뭐"), "뭘": ("무엇", "무얼", "뭐"),
    "거": ("것", "걸", "건", "게"), "건": ("것", "거"), "걸": ("것", "거"), "게": ("것", "거"),
    "넌": ("너",), "난": ("나",), "전": ("저",), "맘": ("마음",), "젤": ("제일",), "좀": ("조금",),
    "둘": ("두",), "셋": ("세",), "넷": ("네",), "어찌": ("어떻", "어떡"), "그리": ("그렇",), "이리": ("이렇",),
    "저리": ("저렇",), "근데": ("그런데",),
}
_CONNECTIVES = {"근데", "그런데", "하지만", "그래서", "그리고", "그럼", "그러니까"}
_HANGUL_RUN = re.compile("[가-힣]+")


def _head_in(head: str, base: str) -> bool:
    """어절 머리가 base에 있는지 — 받침 없는 글자는 받침이 붙은 꼴과도 맞는다 (소리 ↔ 소릴, 되나 ↔ 될)."""
    for i in range(len(base) - len(head) + 1):
        for j, ch in enumerate(head):
            other = base[i + j]
            if ch == other:
                continue
            no_final = "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 == 0
            if no_final and "가" <= other <= "힣" and ord(other) - (ord(other) - 0xAC00) % 28 == ord(ch):
                continue
            break
        else:
            return True
    return False


def new_words(before: str, text: str) -> List[str]:
    """짧은 대안(text)에서 지금 번역(before)에 없는 낱말 목록."""
    base = re.sub(r"[^0-9A-Za-z가-힣]", "", before or "").lower()
    found = []
    for token in (text or "").split():
        core = re.sub(r"[^0-9A-Za-z가-힣]", "", token).lower()
        if not core:
            continue
        run = _HANGUL_RUN.match(core)
        if not run:  # 숫자·로마자로 시작하는 낱말은 그대로 있어야 한다
            head = re.match(r"[0-9a-z]+", core).group()
            if head not in base:
                found.append(token)
            continue
        word = run.group()
        if word in _CONNECTIVES:
            continue
        head = word if len(word) == 1 else word[:1] if len(word) == 2 else word[:2]
        if word in base or _head_in(head, base):
            continue
        if any(alt in base for key in (word, head) for alt in _CONTRACTIONS.get(key, ())):
            continue
        found.append(token)
    return found


class TranslationPool:
    """번역 요청을 여러 개 동시에 돌리는 창구 — 요청은 뒤에서 돌고, 받은 인명·용어는 한 사전에 모인다.

    요청마다 새 대화를 연다. 대화를 이어 쓰면 앞 내용이 입력으로 다시 들어가 번역 토큰이
    약 2배가 되고, 대화 여럿이 번갈아 받으면 어차피 앞 내용이 띄엄띄엄이다 — 이름 표기는 사전으로 맞춘다.
    """

    def __init__(self, session_factory: Callable[[], AntigravitySession], size: int, glossary=None,
                 callback: Optional[Callable] = None, should_stop: Optional[Callable[[], bool]] = None):
        self.size = max(1, int(size))  # 동시에 도는 요청 수
        self.glossary = glossary
        self.reading = False  # 그래픽카드가 아직 쪽을 읽는 중인지 — 알림에 실어 화면이 단계를 헷갈리지 않게 한다
        self.submitted = 0
        self.finished = 0
        self.busy_seconds = 0.0  # 요청들이 답을 기다린 시간의 합 (동시에 돌아 실제 걸린 시간보다 길다)
        self._factory = session_factory
        self._callback = callback or (lambda event: None)
        self._should_stop = should_stop or (lambda: False)
        self._lock = threading.Lock()
        self._started = 0
        self._executor = ThreadPoolExecutor(max_workers=self.size, thread_name_prefix="agy")

    def check(self):
        """번역 CLI와 지침을 미리 확인한다 — 못 찾으면 쪽을 읽기 전에 알린다."""
        session = self._factory()
        logger.info("번역 CLI 준비: %s — %s (model=%s, effort=%s) · 요청은 동시에 %d개까지",
                    getattr(session, "name", "번역 CLI"), session.executable, session.model or "기본",
                    session.effort or "기본", self.size)
        session.close()

    def progress(self) -> dict:
        """화면 알림에 싣는 번역 진행 — 보낸 요청 수, 받은 요청 수, 쪽을 읽는 중인지."""
        with self._lock:
            return {"requests_done": self.finished, "requests_total": self.submitted, "reading": self.reading}

    def _notify(self, event: ProgressEvent):
        event.extras = {**self.progress(), **(event.extras or {})}
        self._callback(event)

    def submit(self, keyed_items: Dict[str, TextElement], label: str) -> Future:
        """대사 묶음을 뒤에서 번역한다. 결과는 대사에 바로 채워지고, 돌려준 Future로 끝났는지 본다."""
        with self._lock:
            self.submitted += 1
        return self._executor.submit(self._run, keyed_items, label)

    def _run(self, keyed_items, label):
        if self._should_stop():
            # 멈춰 달라고 했으면 보내지 않는다 — 대사는 대기로 남아 다음 실행에서 요청한다
            with self._lock:
                self.finished += 1
            return False
        with self._lock:
            self._started += 1
            key = f"r{self._started}"
        started = time.perf_counter()
        session = None
        try:
            session = self._factory()
            session.key = key
            translate_items(session, keyed_items, label=label, glossary=self.glossary, callback=self._notify)
        finally:
            if session is not None:
                session.close()
            with self._lock:
                self.finished += 1
                self.busy_seconds += time.perf_counter() - started
        progress = self.progress()
        self._notify(ProgressEvent(
            PipelinePhase.TRANSLATION, progress["requests_done"], progress["requests_total"],
            f"{label} 번역을 받았습니다 ({progress['requests_done']}/{progress['requests_total']})",
        ))
        return True

    def start_check(self, keyed_items: KeyedItems) -> List[tuple]:
        """의심 대사(check_notes)만 뒤에서 다시 묻는다 — 걸린 대사가 없으면 요청하지 않고 빈 목록.

        받은 답은 finish_check가 채우므로 그동안 다른 단계(짧은 번역)를 돌린다. 보낼 글과 그때의 번역은 지금 떠 둔다.
        책 전체를 다시 읽히는 검토는 오래 걸리는 데 비해 바로잡는 곳이 드물고 맞는 번역을 틀리게 고치기도 해서
        의심 대사만 묻는다.
        """
        flagged = check_notes(keyed_items)
        if not flagged:
            return []
        keys = sorted(flagged, key=_key_order)
        subset = KeyedItems(((key, keyed_items[key]) for key in keys), free=keyed_items.free & set(keys))
        session = self._factory()
        session.key = "의심 대사 확인"
        prompt = _check_prompt(session, keyed_items, flagged, self.glossary)
        sent = {key: (element.translated_text, element.translation_status) for key, element in subset.items()}
        return [(subset, sent, self._executor.submit(self._request_check, session, subset, prompt))]

    def _request_check(self, session, keyed_items: KeyedItems, prompt: str) -> Optional[dict]:
        started = time.perf_counter()
        try:
            if self._should_stop():
                return None
            return check_items(session, keyed_items, prompt)
        finally:
            session.close()
            with self._lock:
                self.busy_seconds += time.perf_counter() - started

    @staticmethod
    def check_waiting(pending: List[tuple]) -> bool:
        """start_check로 보낸 요청의 답이 아직 오지 않았는지."""
        return any(not future.done() for _, _, future in pending)

    def finish_check(self, pending: List[tuple]) -> List[tuple]:
        """start_check로 보낸 확인 답을 기다려 대사에 채운다 — 고친 (id, 전, 후, 이유) 목록.

        확인은 보낸 번역을 보고 답했다. 보낸 뒤 바뀐 대사(짧은 번역)는 보낸 번역에 대고 답을 채운 뒤, 확인이 고치지
        않은 것만 바뀐 번역으로 되돌린다 — 둘 다 고친 대사는 뜻을 바로잡는 확인 쪽을 쓴다.
        """
        changes = []
        for subset, sent, future in pending:
            try:
                structured = future.result()
            except Exception as e:  # noqa: BLE001 — 확인 실패는 번역을 멈추지 않는다
                logger.warning(f"의심 대사 확인 실패 — 지금 번역을 그대로 씁니다: {e}")
                continue
            if not structured:
                continue
            later = {key: (element.translated_text, element.translation_status) for key, element in subset.items()
                     if (element.translated_text, element.translation_status) != sent[key]}
            for key in later:
                subset[key].translated_text, subset[key].translation_status = sent[key]
            applied = apply_review(structured, subset, self.glossary)
            fixed = {change[0] for change in applied}
            for key, (text, status) in later.items():
                if key in fixed:
                    logger.info(f"의심 대사 확인 {key}: 짧은 번역 '{text}' 대신 확인이 고친 번역을 씁니다")
                else:
                    subset[key].translated_text, subset[key].translation_status = text, status
            changes += applied
        return changes

    def close(self):
        """시작하지 않은 요청은 거두고 창구를 닫는다 — 돌던 요청은 끝날 때까지 기다린다."""
        self._executor.shutdown(wait=True, cancel_futures=True)
