import json
import logging
import os
from typing import List, Optional

from src.data_models import PageData
from src.serialization import load_page_data_json, append_page_data_json, save_page_data_json

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
META_FILENAME = "checkpoint_meta.json"


class CheckpointManager:
    """파이프라인 체크포인트 관리자. 중단된 작업을 재개할 수 있도록 메타데이터를 관리합니다."""

    def __init__(self, output_dir: str, input_dir: str):
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.meta_path = os.path.join(output_dir, META_FILENAME)
        self.json_path = os.path.join(output_dir, "translation_data.json")
        self._meta: Optional[dict] = None

    def load_or_create(self, total_images: int) -> dict:
        """기존 체크포인트를 로드하거나 새로 생성합니다."""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self._meta = json.load(f)
            logger.info(f"기존 체크포인트를 로드했습니다: {self.meta_path}")
        else:
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

    def _save_meta(self):
        """메타데이터를 디스크에 저장합니다."""
        os.makedirs(self.output_dir, exist_ok=True)
        tmp_path = self.meta_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.meta_path)

    # --- Pass 1 ---

    def is_pass1_complete(self) -> bool:
        return self._meta and self._meta.get("pass1_complete", False)

    def get_pass1_remaining_paths(self, all_image_paths: List[str]) -> List[str]:
        """Pass 1에서 아직 처리되지 않은 이미지 경로를 반환합니다."""
        completed = set(self._meta.get("pass1_completed_pages", []))
        return [p for p in all_image_paths if os.path.basename(p) not in completed]

    def mark_pass1_batch_complete(self, page_data_list: List[PageData]):
        """Pass 1 배치 완료를 기록하고 JSON에 증분 저장합니다."""
        new_pages = [pd.source_page for pd in page_data_list]
        self._meta["pass1_completed_pages"].extend(new_pages)
        append_page_data_json(page_data_list, self.json_path)
        self._save_meta()
        logger.info(f"체크포인트 업데이트: Pass 1 {len(self._meta['pass1_completed_pages'])}페이지 완료")

    def mark_pass1_complete(self):
        self._meta["pass1_complete"] = True
        self._save_meta()
        logger.info("체크포인트: Pass 1 완료 표시")

    def load_pass1_data(self) -> List[PageData]:
        """저장된 Pass 1 결과를 JSON에서 로드합니다."""
        if os.path.exists(self.json_path):
            return load_page_data_json(self.json_path)
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
            save_page_data_json([], self.json_path)

        logger.info("체크포인트를 새 실행 상태로 초기화했습니다.")

    # --- Pass 2 ---

    def get_pass2_remaining_pages(self, all_page_data: List[PageData]) -> List[PageData]:
        """Pass 2에서 체크포인트에 완료로 기록되지 않은 페이지를 반환합니다."""
        completed = set(self._meta.get("pass2_completed_pages", []))
        remaining = [pd for pd in all_page_data if pd.source_page not in completed]
        skipped = len(all_page_data) - len(remaining)
        if skipped > 0:
            logger.info(f"Pass 2: {skipped}페이지 스킵 (체크포인트), {len(remaining)}페이지 처리 예정")
        return remaining

    def mark_pass2_page_complete(self, source_page: str):
        self._meta["pass2_completed_pages"].append(source_page)
        self._save_meta()

    def mark_complete(self):
        self._meta["pass2_complete"] = True
        self._save_meta()
        logger.info("체크포인트: 전체 파이프라인 완료")

    def cleanup(self):
        """완료된 체크포인트 메타데이터 파일을 삭제합니다."""
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
            logger.info("체크포인트 메타데이터 삭제 완료")
