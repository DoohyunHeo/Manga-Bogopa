import hashlib
import json
import logging
import os
import shutil
import time
from typing import List, Optional

from src.data_models import PageData
from src.serialization import load_page_data_json, append_page_data_json, save_page_data_json
from src.utils import replace_file

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
META_FILENAME = "checkpoint_meta.json"


def _file_fingerprint(path: str) -> str:
    """입력 이미지 내용 지문 — 같은 이름에 다른 그림이 들어오면 달라진다."""
    digest = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointManager:
    """파이프라인 체크포인트 관리자. 중단된 작업을 재개할 수 있도록 메타데이터를 관리합니다."""

    def __init__(self, output_dir: str, input_dir: str):
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.meta_path = os.path.join(output_dir, META_FILENAME)
        self.json_path = os.path.join(output_dir, "translation_data.json")
        self._meta: Optional[dict] = None

    def load_or_create(self, total_images: int) -> dict:
        """기존 체크포인트를 로드하거나 새로 생성합니다.

        스키마 버전이 다른 체크포인트는 조용히 로드하면 미묘한 오동작을
        일으키므로, 백업해 두고 새로 시작한다.
        """
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("체크포인트 메타를 읽지 못했습니다 (%s) — 백업하고 새로 시작합니다.", e)
                meta = {"version": "unreadable"}
            if meta.get("version") == CHECKPOINT_VERSION:
                self._meta = meta
                logger.info(f"기존 체크포인트를 로드했습니다: {self.meta_path}")
                return self._meta
            self._backup_incompatible(meta.get("version"))

        self._meta = {
            "version": CHECKPOINT_VERSION,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "total_images": total_images,
            "pass1_completed_pages": [],
            "pass1_complete": False,
            "pass2_completed_pages": [],
            "pass2_complete": False,
        }
        self._save_meta()
        logger.info("새 체크포인트를 생성했습니다.")
        return self._meta

    def _backup_incompatible(self, found_version):
        """버전이 맞지 않는 체크포인트 파일들을 .bak으로 옮겨 수동 복구 여지를 남긴다."""
        suffix = f".v{found_version if found_version is not None else 'unknown'}.bak"
        for path in (self.meta_path, self.json_path):
            if os.path.exists(path):
                os.replace(path, path + suffix)
        logger.warning(
            "체크포인트 버전 불일치 (파일=%s, 현재=%s): 기존 체크포인트를 '%s' 백업으로 옮기고 새로 시작합니다.",
            found_version, CHECKPOINT_VERSION, suffix,
        )

    def _save_meta(self):
        """메타데이터를 디스크에 저장합니다."""
        os.makedirs(self.output_dir, exist_ok=True)
        tmp_path = self.meta_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)
        replace_file(tmp_path, self.meta_path)  # 웹 화면이 책 상태를 읽는 순간과 겹쳐도 실패하지 않게

    def sync_input_fingerprints(self, image_paths) -> List[str]:
        """입력 파일 내용 지문을 기록하고, 내용이 달라진 페이지를 체크포인트에서 뺀다.

        다른 책을 같은 출력 폴더로 돌리거나 파일을 바꿔 넣으면 이름이 같아도 지문이 달라져
        그 페이지는 처음부터 다시 처리된다. 지문이 없는 예전 체크포인트는 지금 파일을
        그대로 믿고 지문만 기록한다. 빠진 페이지 이름을 돌려준다.
        """
        current = {os.path.basename(path): _file_fingerprint(path) for path in image_paths}
        stored = self._meta.get("page_fingerprints") or {}
        changed = sorted(name for name, fingerprint in current.items()
                         if name in stored and stored[name] != fingerprint)
        if changed:
            dropped = set(changed)
            self._backup_translations()  # 파일을 다시 저장만 한 경우에도 번역을 되살릴 수 있게
            pages = [page for page in self.load_pass1_data() if page.source_page not in dropped]
            save_page_data_json(pages, self.json_path)
            for key in ("pass1_completed_pages", "pass2_completed_pages"):
                self._meta[key] = [name for name in self._meta.get(key, []) if name not in dropped]
            self._meta["pass1_complete"] = False
            self._meta["pass2_complete"] = False
            logger.warning("내용이 바뀐 입력 %d페이지는 처음부터 다시 처리합니다: %s", len(changed), changed[:5])
        self._meta["page_fingerprints"] = current
        self._meta["input_dir"] = self.input_dir
        self._save_meta()
        return changed

    # --- Pass 1 ---

    def mark_pass1_batch_complete(self, page_data_list: List[PageData]):
        """Pass 1 배치 완료를 기록하고 JSON에 증분 저장합니다."""
        new_pages = [pd.source_page for pd in page_data_list]
        self._meta["pass1_completed_pages"].extend(new_pages)
        append_page_data_json(page_data_list, self.json_path)
        self._save_meta()
        logger.info(f"체크포인트 업데이트: Pass 1 {len(self._meta['pass1_completed_pages'])}페이지 완료")

    def load_pass1_data(self) -> List[PageData]:
        """저장된 Pass 1 결과를 JSON에서 로드합니다. 읽을 수 없으면 옆으로 옮기고 빈 목록."""
        if not os.path.exists(self.json_path):
            return []
        try:
            return load_page_data_json(self.json_path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            broken = self.json_path + ".corrupt"
            os.replace(self.json_path, broken)
            logger.warning("대사집을 읽지 못해 '%s'로 옮기고 새로 시작합니다: %s", broken, e)
            return []

    def replace_pass1_data(self, page_data_list: List[PageData], complete: bool):
        """Pass 1 JSON을 현재 상태로 교체하고 메타데이터를 동기화합니다."""
        save_page_data_json(page_data_list, self.json_path)
        source_pages = [pd.source_page for pd in page_data_list]
        self._meta["pass1_completed_pages"] = source_pages
        self._meta["pass1_complete"] = complete
        self._meta["pass2_completed_pages"] = [
            page for page in self._meta.get("pass2_completed_pages", [])
            if page in set(source_pages)
        ]
        self._meta["pass2_complete"] = complete and (
            len(self._meta["pass2_completed_pages"]) == len(source_pages)
        )
        self._save_meta()
        logger.info(
            f"체크포인트 동기화: Pass 1 {len(source_pages)}페이지, complete={complete}"
        )

    def reset_for_new_run(self, clear_json: bool = True):
        """새 번역 실행을 위해 체크포인트 상태를 초기화합니다."""
        self._meta["pass1_completed_pages"] = []
        self._meta["pass1_complete"] = False
        self._meta["pass2_completed_pages"] = []
        self._meta["pass2_complete"] = False
        self._save_meta()

        if clear_json:
            self._backup_translations()
            save_page_data_json([], self.json_path)

        logger.info("체크포인트를 새 실행 상태로 초기화했습니다.")

    def _backup_translations(self):
        """새로 번역하기 전에 기존 대사집을 시각이 붙은 사본으로 남긴다 (번역에는 시간·사용량이 든다)."""
        if not os.path.exists(self.json_path) or os.path.getsize(self.json_path) <= 2:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = os.path.join(self.output_dir, f"translation_data.{stamp}.bak.json")
        shutil.copy2(self.json_path, backup)
        logger.info("기존 대사집을 '%s'로 백업했습니다.", backup)

    # --- Pass 2 ---

    def reset_pass2(self, pages=None):
        """Pass 2(식자)만 초기화 — 번역 데이터는 보존한 채 재식자를 위해. pages를 주면 그 쪽들만."""
        if pages:
            drop = set(pages)
            self._meta["pass2_completed_pages"] = [
                page for page in self._meta.get("pass2_completed_pages", []) if page not in drop
            ]
        else:
            self._meta["pass2_completed_pages"] = []
        self._meta["pass2_complete"] = False
        self._save_meta()
        logger.info("체크포인트: Pass 2 초기화%s (번역 데이터는 보존)", f" — {len(pages)}쪽" if pages else "")

    def _pass2_written(self, name) -> bool:
        """식자 완료로 적혀 있고 결과 그림도 결과 폴더에 있는지 — 결과를 지운 쪽은 다시 만든다."""
        return (name in self._meta.get("pass2_completed_pages", [])
                and os.path.exists(os.path.join(self.output_dir, name)))

    def is_pass2_done(self, names) -> bool:
        """이 쪽들이 모두 식자까지 끝났는지."""
        return all(self._pass2_written(name) for name in names)

    def get_pass2_remaining_pages(self, all_page_data: List[PageData]) -> List[PageData]:
        """Pass 2에서 아직 결과가 없는 페이지 — 완료로 적혀 있어도 결과 그림이 없으면 다시 만든다."""
        remaining = [pd for pd in all_page_data if not self._pass2_written(pd.source_page)]
        skipped = len(all_page_data) - len(remaining)
        if skipped > 0:
            logger.info(f"Pass 2: {skipped}페이지 스킵 (체크포인트), {len(remaining)}페이지 처리 예정")
        return remaining

    def mark_pass2_page_complete(self, source_page: str):
        if source_page not in self._meta["pass2_completed_pages"]:  # 결과를 지워 다시 만든 쪽은 이미 적혀 있다
            self._meta["pass2_completed_pages"].append(source_page)
        self._save_meta()

    def mark_complete(self):
        self._meta["pass2_complete"] = True
        self._save_meta()
        logger.info("체크포인트: 전체 파이프라인 완료")
