"""배치가 어려운 말풍선만 번역으로 되돌려 짧은 대안을 받는다 (한 권에 요청 한 번, 설정 ENABLE_FIT_TRANSLATION을 켰을 때만).

식자를 한 번 맞춰 본 뒤, 말풍선 안 대사 가운데
  (a) 한국어가 원문 크기(쪽 대사 중앙보다 크면 중앙)의 0.8배 미만으로 작아졌거나
  (b) 어절 중간에서 줄이 끊겼거나
  (c) 강조 대사인데 쪽 대사 중앙의 1.25배에 못 미친
것만 모아 번역기에 한 번 보낸다. 한 줄에 서너 자씩 짧게 끊긴 줄은 고칠 거리로 보지 않는다 — 좁은 말풍선에서는
흔한 모양이고, 이걸 고치려고 줄인 번역은 크기는 그대로인 채 주어·감탄사·호칭을 뺀다. 실제 글꼴로 잰 '한 줄 최대 N자·최대 M줄'을 같이 보내 짧은 대안 2~3개를 받고,
대안마다 다시 맞춰 봐서 크기·줄 모양이 확실히 나아진 것만 번역으로 바꾼다. 나아진 대안 가운데 지금 번역에 없는 낱말을
들여온 것은 책마다 한 번 번역기에 원문 뜻을 지키는지(OK/NG) 묻고, OK인 것만 쓴다 (확인을 못 받으면 쓰지 않는다).
바꾼 원래 번역과 확인 결과는 출력 폴더의
translation_fit.json에 남기고, 한 번 물어본 말풍선은 다시 요청하지 않는다 (식자만 다시 해도 흔들리지 않게).
기준이 바뀌어 새로 걸린 말풍선이 있을 때만 그 줄들을 한 번 더 요청한다.
"""
import json
import logging
import math
import os
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from src import page_drawer, translator
from src.data_models import PageData
from src.reading_order import reading_order
from src.serialization import load_page_data_json, save_page_data_json
from src.text_fitting import count_determiner_breaks
from src.text_renderer import measure_text
from src.text_wrapping import count_midword_breaks
from src.utils import read_image_bgr

logger = logging.getLogger(__name__)

FIT_FILE = "translation_fit.json"
_SMALL_RATIO = 0.8        # (a) 쪽 대사 중앙의 이 배 미만
_CONTEXT_LINES = 2        # 앞뒤로 붙이는 대사 수
_ADOPT_MARGIN = 0.1       # 대안 점수가 이만큼(쪽 대사 중앙 크기의 10%에 해당) 넘게 좋아야 바꾼다
_ADOPT_GROWTH = 1.1       # 끊김을 줄이지 않는 대안은 글자가 이만큼 커져야 바꾼다 (작아지는 대안은 받지 않는다)


def _erased_look(page: PageData, input_dir: str) -> np.ndarray:
    """지운 뒤 모습 흉내 — 말풍선 글자 상자를 흰색으로 칠한 원본 (배치만 맞춰 볼 때 쓴다)."""
    image = cv2.cvtColor(read_image_bgr(os.path.join(input_dir, page.source_page)), cv2.COLOR_BGR2RGB)
    for bubble in page.speech_bubbles:
        x1, y1, x2, y2 = (int(v) for v in bubble.text_element.text_box[:4])
        image[max(0, y1):y2, max(0, x1):x2] = 255
    return image


def _layouts(page: PageData, image: np.ndarray) -> Dict[int, object]:
    """말풍선 번호 → 식자 검수까지 마친 계획."""
    _, report = page_drawer.render_page_text(image.copy(), page)
    return {er.index: er.plan_after for er in report.elements if er.kind == "bubble"}


def _desired_size(element, plan, dialogue_px) -> float:
    if page_drawer._is_emphasis(element, dialogue_px):
        return page_drawer._EMPHASIS_MIN * dialogue_px
    base = dialogue_px if element.font_size >= dialogue_px else element.font_size
    return max(float(plan.font_size), float(base))


def _hard_reasons(element, plan, dialogue_px) -> List[str]:
    reasons = []
    text = element.translated_text or ""
    # 원문이 쪽 대사보다 작은 대사도 원문 크기에서 한참 작아졌으면
    if plan.font_size < _SMALL_RATIO * min(float(element.font_size), dialogue_px):
        reasons.append("작아짐")
    if not plan.vertical and count_midword_breaks(text, plan.text):
        reasons.append("어절 끊김")
    if page_drawer._is_emphasis(element, dialogue_px) and plan.font_size < page_drawer._EMPHASIS_MIN * dialogue_px:
        reasons.append("강조 부족")
    return reasons


def _score(element, plan, desired) -> float:
    """크기(원하는 크기 대비, 1까지)에서 어절 끊김·관형사 끊김을 뺀 점수."""
    text = element.translated_text or ""
    score = min(float(plan.font_size) / desired, 1.0)
    if not plan.vertical:
        score -= 0.3 * count_midword_breaks(text, plan.text)
        score -= 0.15 * count_determiner_breaks(text, plan.text)
    return score


def _breaks(element, plan) -> int:
    """어절 끊김·관형사 끊김의 수."""
    if plan.vertical:
        return 0
    text = element.translated_text or ""
    return count_midword_breaks(text, plan.text) + count_determiner_breaks(text, plan.text)


def _room(job, size) -> Dict[str, int]:
    """그 크기로 이 말풍선에 들어가는 한 줄 최대 글자 수·최대 줄 수 (실제 글꼴로 잼)."""
    size = max(1, int(round(size)))
    advance = measure_text("가나다라마바사아자차", job.font_path, size, job.style)[0] / 10.0
    one = measure_text("가", job.font_path, size, job.style)[1]
    two = measure_text("가\n가", job.font_path, size, job.style)[1]
    per_line = max(2, int(job.target_width // max(advance, 1.0)))
    lines = max(1, 1 + int(max(0.0, job.target_height - one) // max(two - one, 1.0)))
    return {"max_chars_per_line": per_line, "max_lines": lines, "about_chars": per_line * lines}


def _context(page: PageData) -> Dict[int, tuple]:
    """말풍선 대사 id → (앞 대사들, 뒤 대사들) — 읽는 순서."""
    order = [e for e in reading_order(page) if e.translated_text]
    where = {id(b.text_element): i for i, b in enumerate(page.speech_bubbles)}
    out = {}
    for pos, element in enumerate(order):
        if id(element) in where:
            before = [e.translated_text for e in order[max(0, pos - _CONTEXT_LINES):pos]]
            after = [e.translated_text for e in order[pos + 1:pos + 1 + _CONTEXT_LINES]]
            out[where[id(element)]] = (before, after)
    return out


def shorten_hard_bubbles(book_pages: List[PageData], typeset_pages: List[PageData], output_dir: str, input_dir: str,
                         json_path: str, session_factory: Callable, glossary=None) -> Optional[dict]:
    """배치가 어려운 말풍선의 짧은 대안을 받아 나아진 것만 번역에 넣는다. 이미 한 책이면 None.

    typeset_pages는 글씨체까지 정한 식자용 쪽(대사는 book_pages와 같은 객체). 바꾼 번역은 json_path(대사집)에 저장한다.
    """
    record_path = os.path.join(output_dir, FIT_FILE)
    record = {"requested": 0, "adopted": {}, "kept": {}}
    if os.path.exists(record_path):
        with open(record_path, encoding="utf-8") as f:
            record = json.load(f)
    seen = {(v["page"], v["bubble"]) for part in ("adopted", "kept") for v in record.get(part, {}).values()}
    candidates = []  # (쪽 번호, 말풍선 번호, 요소, 이유, 원하는 크기, 원래 점수)
    images, layouts = {}, {}
    for page_number, page in enumerate(typeset_pages, start=1):
        dialogue_px = page_drawer.page_dialogue_px(page)
        if not dialogue_px or not page.speech_bubbles:
            continue
        image = _erased_look(page, input_dir)
        plans = _layouts(page, image)
        gray = np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        jobs = {job.index: job for job in page_drawer._plan_speech_bubbles(page, gray)}
        page.image_rgb = cv2.cvtColor(read_image_bgr(os.path.join(input_dir, page.source_page)), cv2.COLOR_BGR2RGB)
        context = _context(page)
        page.image_rgb = None
        for index, plan in plans.items():
            element = page.speech_bubbles[index].text_element
            if not element.translated_text or index not in jobs or (page.source_page, index) in seen:
                continue
            reasons = _hard_reasons(element, plan, dialogue_px)
            if not reasons:
                continue
            desired = _desired_size(element, plan, dialogue_px)
            before, after = context.get(index, ([], []))
            candidates.append(dict(page=page_number, index=index, element=element, reasons=reasons, desired=desired,
                                   score=_score(element, plan, desired), size=int(plan.font_size),
                                   breaks=_breaks(element, plan),
                                   before=before, after=after, room=_room(jobs[index], desired)))
        images[page_number], layouts[page_number] = image, plans
    if not candidates:
        if not os.path.exists(record_path):
            with open(record_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=1)
        return None
    if candidates:
        items = [dict(id=f"{c['page']}.{c['index']}", element=c["element"], before=c["before"], after=c["after"],
                      **c["room"]) for c in candidates]
        session = session_factory()
        session.key = "짧은 번역"
        try:
            answers, info = translator.request_shorter(session, items, glossary)
        finally:
            session.close()
        record["requested"] = record.get("requested", 0) + len(items)
        record.setdefault("answers", {}).update(answers)
        record.setdefault("requests", []).append(dict(kind="짧은 번역", items=len(items), **info))
        _evaluate(typeset_pages, candidates, answers, images, glossary)
        verdicts, info = _confirm(candidates, session_factory)
        if info is not None:
            record["requests"].append(dict(kind="짧은 번역 확인", items=len(verdicts), **info))
        _decide(typeset_pages, candidates, verdicts, record)
    if record["adopted"]:
        _save_adopted(book_pages, typeset_pages, candidates, json_path)
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    adopted = sum(1 for c in candidates if c.get("best"))
    logger.info("말풍선에 맞춘 짧은 번역: 어려운 말풍선 %d개 중 %d개를 바꿨습니다", len(candidates), adopted)
    return record


def _save_adopted(book_pages, typeset_pages, candidates, json_path):
    """고른 짧은 번역만 대사집에 쓴다."""
    save_translations(book_pages, [c["element"] for c in candidates if c.get("best")], json_path)


def save_translations(book_pages: List[PageData], elements, json_path: str):
    """고친 대사의 번역과 번역할지만 대사집에 쓴다 (짧은 번역·의심 대사 확인). 메모리의 대사에는 책 단위 글씨체·크기
    보정(font_relative.apply_book_styles)이 들어가 있어 통째로 저장하면 식자만 다시 할 때 보정이 한 번 더 들어간다."""
    wanted = {id(element) for element in elements}
    if not wanted:
        return
    saved = load_page_data_json(json_path)
    saved_pages = {page.source_page: page for page in saved}
    for page in book_pages:
        target = saved_pages.get(page.source_page)
        if target is None:
            continue
        for mine, theirs in zip(page.text_elements(), target.text_elements()):
            if id(mine) in wanted and mine.original_text == theirs.original_text:
                theirs.translated_text = mine.translated_text
                theirs.translation_status = mine.translation_status
    save_page_data_json(saved, json_path)


def _evaluate(pages, candidates, answers, images, glossary):
    """대안마다 실제로 맞춰 보고 점수가 확실히 오른 대안을 c["qualified"]에 (점수, 번역, 크기, 새 낱말)로 모은다."""
    by_page: Dict[int, list] = {}
    for c in candidates:
        key = f"{c['page']}.{c['index']}"
        options = [t for t in answers.get(key, []) if translator.acceptable_shorter(c["element"], t, glossary)]
        c["options"], c["qualified"], c["original"] = options, [], c["element"].translated_text
        if options:
            by_page.setdefault(c["page"], []).append(c)
    for page_number, group in by_page.items():
        page = pages[page_number - 1]
        rounds = max(len(c["options"]) for c in group)
        for k in range(rounds):
            trial = [c for c in group if k < len(c["options"])]
            for c in trial:
                c["element"].translated_text = c["options"][k]
            try:
                plans = _layouts(page, images[page_number])
            finally:  # 맞춰 보다 실패해도 대안이 실제 번역으로 남지 않게
                for c in trial:
                    c["element"].translated_text = c["original"]
            for c in trial:
                plan = plans.get(c["index"])
                if plan is None:
                    continue
                probe = _Probe(c["options"][k])
                score = _score(probe, plan, c["desired"])
                breaks = _breaks(probe, plan)
                better = (plan.font_size >= _ADOPT_GROWTH * c["size"]
                          or (breaks < c["breaks"] and plan.font_size >= c["size"]))
                if better and score >= c["score"] + _ADOPT_MARGIN:
                    c["qualified"].append((score, c["options"][k], int(plan.font_size),
                                           translator.new_words(c["original"], c["options"][k])))


def _confirm(candidates, session_factory):
    """새 낱말이 든 나은 대안만 모아 한 번 확인 — ({'쪽.번호/대안': 'ok'|'ng'|'timeout'|'failed'}, 요청 기록).

    확인할 대안이 없으면 요청하지 않는다 ({}, None).
    """
    items = []
    for c in candidates:
        for n, (_, text, _, words) in enumerate(c.get("qualified", [])):
            if words:
                items.append(dict(id=f"{c['page']}.{c['index']}/{n}", element=c["element"], short=text))
    if not items:
        return {}, None
    for c in candidates:
        c["element"].translated_text = c["original"]  # 확인은 지금 번역과 견준다
    session = session_factory()
    session.key = "짧은 번역 확인"
    try:
        answers, info = translator.confirm_shorter(session, items)
    finally:
        session.close()
    if answers is None:
        return {item["id"]: info["status"] for item in items}, info
    return {key: "ok" if ok else "ng" for key, ok in answers.items()}, info


def _decide(pages, candidates, verdicts, record):
    """나은 대안 가운데 새 낱말이 없거나 확인이 OK인 것 중 점수가 가장 높은 것을 번역으로 쓴다."""
    for c in candidates:
        key = f"{c['page']}.{c['index']}"
        checks, usable = [], []
        for n, (score, text, size, words) in enumerate(c.get("qualified", [])):
            verdict = verdicts.get(f"{key}/{n}") if words else None
            if words:
                checks.append(dict(text=text, new_words=words, verdict=verdict or "failed"))
            if not words or verdict == "ok":
                usable.append((score, text, size))
        c["best"] = max(usable, key=lambda item: item[0]) if usable else None
        entry = dict(page=pages[c["page"] - 1].source_page, bubble=c["index"], original=c["element"].original_text,
                     translation=c["original"], reasons=c["reasons"], size=c["size"], options=c.get("options", []),
                     room=c["room"])
        if checks:
            entry["checks"] = checks
        if c.get("best"):
            score, text, size = c["best"]
            c["element"].translated_text = text
            record["adopted"][key] = dict(entry, adopted=text, new_size=size)
            logger.info(f"짧은 번역 {key}: {c['original']} → {text} ({c['size']}→{size}px)")
        else:
            record["kept"][key] = entry


class _Probe:
    """점수 계산용 — 대안 번역을 단 요소 흉내."""

    def __init__(self, text):
        self.translated_text = text
