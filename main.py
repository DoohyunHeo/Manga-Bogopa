import argparse

from src import config
from pipeline import MangaTranslationPipeline


def main():
    """어플리케이션의 메인 진입점"""
    parser = argparse.ArgumentParser(description="Manga-Bogopa: 일본 만화 자동 번역 및 식자 도구")
    parser.add_argument('--input', type=str, default=None, help="입력 이미지 폴더 경로")
    parser.add_argument('--output', type=str, default=None, help="출력 이미지 폴더 경로")
    parser.add_argument('--batch-size', type=int, default=None, help="번역 배치 크기")
    parser.add_argument('--debug', action='store_true', help="디버그 박스 표시")
    args = parser.parse_args()

    # CLI 인자로 config 오버라이드
    if args.input:
        config._config.INPUT_DIR = args.input
    if args.batch_size:
        config._config.TRANSLATION_BATCH_SIZE = args.batch_size
    if args.debug:
        config._config.DRAW_DEBUG_BOXES = True

    try:
        pipeline = MangaTranslationPipeline()
        if args.output:
            pipeline.output_dir = args.output
        pipeline.run()
    except Exception as e:
        print(f"\n오류가 발생하여 프로그램을 중단합니다: {e}")

if __name__ == '__main__':
    main()
