"""
Примеры использования VideoTranscriber для транскрибации видео по ролям.
"""

from video_transcriber import VideoTranscriber


def example_1_basic_transcription():
    """
    Пример 1: Базовая транскрибация без диаризации спикеров.
    Используется только Whisper для распознавания речи.
    """
    print("\n" + "="*80)
    print("ПРИМЕР 1: Базовая транскрибация (без диаризации)")
    print("="*80 + "\n")

    # Создание транскрайбера без токена HuggingFace
    # В этом случае диаризация спикеров будет недоступна
    transcriber = VideoTranscriber(whisper_model="base")

    # Транскрибация видео
    results = transcriber.transcribe_video(
        video_path="video.mp4",
        output_path="transcription_basic.json",
        language="ru"
    )

    # Вывод результатов
    transcriber.print_results(results)


def example_2_with_diarization():
    """
    Пример 2: Транскрибация с диаризацией спикеров.
    Требует токен HuggingFace.
    """
    print("\n" + "="*80)
    print("ПРИМЕР 2: Транскрибация с диаризацией спикеров")
    print("="*80 + "\n")

    # ВАЖНО: Замените на ваш токен HuggingFace
    # Получить токен можно на https://huggingface.co/settings/tokens
    # После создания токена, примите условия использования:
    # https://huggingface.co/pyannote/speaker-diarization-3.1
    HF_AUTH_TOKEN = "your_huggingface_token_here"

    # Создание транскрайбера с диаризацией
    transcriber = VideoTranscriber(
        whisper_model="base",
        hf_auth_token=HF_AUTH_TOKEN
    )

    # Транскрибация видео с указанием количества спикеров (опционально)
    results = transcriber.transcribe_video(
        video_path="video.mp4",
        output_path="transcription_with_speakers.json",
        language="ru",
        num_speakers=2  # Если известно количество спикеров
    )

    # Вывод результатов
    transcriber.print_results(results)


def example_3_different_formats():
    """
    Пример 3: Сохранение результатов в различных форматах.
    """
    print("\n" + "="*80)
    print("ПРИМЕР 3: Сохранение в различных форматах")
    print("="*80 + "\n")

    HF_AUTH_TOKEN = "your_huggingface_token_here"

    transcriber = VideoTranscriber(
        whisper_model="base",
        hf_auth_token=HF_AUTH_TOKEN
    )

    # Транскрибация видео
    results = transcriber.transcribe_video(
        video_path="video.mp4",
        language="ru"
    )

    # Сохранение в разных форматах
    transcriber.save_results(results, "transcription.json")  # JSON
    transcriber.save_results(results, "transcription.txt")   # Текст
    transcriber.save_results(results, "transcription.srt")   # Субтитры

    print("Результаты сохранены в трех форматах!")


def example_4_multilingual():
    """
    Пример 4: Транскрибация видео на разных языках.
    """
    print("\n" + "="*80)
    print("ПРИМЕР 4: Мультиязычная транскрибация")
    print("="*80 + "\n")

    transcriber = VideoTranscriber(whisper_model="base")

    # Русский язык
    results_ru = transcriber.transcribe_video(
        video_path="video_ru.mp4",
        output_path="transcription_ru.json",
        language="ru"
    )

    # Английский язык
    results_en = transcriber.transcribe_video(
        video_path="video_en.mp4",
        output_path="transcription_en.json",
        language="en"
    )

    # Автоопределение языка (оставить language=None или использовать 'auto')
    results_auto = transcriber.transcribe_video(
        video_path="video_unknown.mp4",
        output_path="transcription_auto.json"
    )


def example_5_large_model():
    """
    Пример 5: Использование большой модели Whisper для лучшего качества.
    """
    print("\n" + "="*80)
    print("ПРИМЕР 5: Использование большой модели Whisper")
    print("="*80 + "\n")

    # Доступные модели: tiny, base, small, medium, large
    # Чем больше модель, тем лучше качество, но медленнее работа
    transcriber = VideoTranscriber(
        whisper_model="medium",  # Или "large" для максимального качества
        hf_auth_token="your_huggingface_token_here"
    )

    results = transcriber.transcribe_video(
        video_path="video.mp4",
        output_path="transcription_high_quality.json",
        language="ru"
    )

    transcriber.print_results(results)


def example_6_custom_workflow():
    """
    Пример 6: Пользовательский workflow с отдельными шагами.
    """
    print("\n" + "="*80)
    print("ПРИМЕР 6: Пользовательский workflow")
    print("="*80 + "\n")

    transcriber = VideoTranscriber(
        whisper_model="base",
        hf_auth_token="your_huggingface_token_here"
    )

    # Шаг 1: Извлечение аудио
    audio_path = transcriber.extract_audio("video.mp4", "extracted_audio.wav")

    # Шаг 2: Транскрибация
    transcription = transcriber.transcribe_audio(audio_path, language="ru")

    # Шаг 3: Диаризация спикеров
    speaker_segments = transcriber.diarize_speakers(audio_path, num_speakers=2)

    # Шаг 4: Объединение результатов
    results = transcriber.merge_transcription_and_diarization(
        transcription,
        speaker_segments
    )

    # Шаг 5: Сохранение
    transcriber.save_results(results, "custom_transcription.json")
    transcriber.print_results(results)


def example_7_batch_processing():
    """
    Пример 7: Пакетная обработка нескольких видео.
    """
    print("\n" + "="*80)
    print("ПРИМЕР 7: Пакетная обработка")
    print("="*80 + "\n")

    transcriber = VideoTranscriber(
        whisper_model="base",
        hf_auth_token="your_huggingface_token_here"
    )

    # Список видео для обработки
    videos = [
        "video1.mp4",
        "video2.mp4",
        "video3.mp4"
    ]

    for i, video_path in enumerate(videos, 1):
        print(f"\nОбработка видео {i}/{len(videos)}: {video_path}")

        try:
            results = transcriber.transcribe_video(
                video_path=video_path,
                output_path=f"transcription_{i}.json",
                language="ru"
            )
            print(f"Успешно обработано: {video_path}")

        except Exception as e:
            print(f"Ошибка при обработке {video_path}: {e}")
            continue


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        Примеры использования VideoTranscriber               ║
    ╚══════════════════════════════════════════════════════════════╝

    Доступные примеры:

    1. example_1_basic_transcription()       - Базовая транскрибация
    2. example_2_with_diarization()          - С диаризацией спикеров
    3. example_3_different_formats()         - Разные форматы вывода
    4. example_4_multilingual()              - Мультиязычная обработка
    5. example_5_large_model()               - Большая модель Whisper
    6. example_6_custom_workflow()           - Пользовательский workflow
    7. example_7_batch_processing()          - Пакетная обработка

    Раскомментируйте нужный пример ниже:
    """)

    # Раскомментируйте нужный пример:
    # example_1_basic_transcription()
    # example_2_with_diarization()
    # example_3_different_formats()
    # example_4_multilingual()
    # example_5_large_model()
    # example_6_custom_workflow()
    # example_7_batch_processing()

    print("\nДля запуска примера раскомментируйте его в коде!")
