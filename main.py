from pipeline import MangaTranslationPipeline

def main():
    """어플리케이션의 메인 진입점"""
    try:
        pipeline = MangaTranslationPipeline()
        pipeline.run()
    except Exception as e:
        print(f"\n오류가 발생하여 프로그램을 중단합니다: {e}")
        # In a real application, you might want more robust logging or error handling.
        # For example, logging the full traceback.
        # import traceback
        # traceback.print_exc()

if __name__ == '__main__':
    main()
