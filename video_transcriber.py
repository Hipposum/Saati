"""
Модуль для транскрибации видео с разделением по ролям (спикерам).
Использует Whisper для распознавания речи и pyannote.audio для диаризации спикеров.
"""

import os
import json
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import whisper
from pyannote.audio import Pipeline
from moviepy.editor import VideoFileClip
import numpy as np

warnings.filterwarnings('ignore')


class VideoTranscriber:
    """
    Класс для транскрибации видео с разделением по спикерам.
    """

    def __init__(
        self,
        whisper_model: str = "base",
        hf_auth_token: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Инициализация транскрайбера.

        Args:
            whisper_model: Размер модели Whisper ('tiny', 'base', 'small', 'medium', 'large')
            hf_auth_token: Токен HuggingFace для pyannote (получить на https://huggingface.co/settings/tokens)
            device: Устройство для вычислений ('cuda', 'cpu', или None для автоопределения)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")

        # Загрузка модели Whisper
        print(f"Загрузка модели Whisper ({whisper_model})...")
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)

        # Загрузка модели для диаризации спикеров
        self.hf_auth_token = hf_auth_token
        if hf_auth_token:
            print("Загрузка модели диаризации спикеров...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_auth_token
            )
            if self.device == "cuda":
                self.diarization_pipeline.to(torch.device("cuda"))
        else:
            print("ВНИМАНИЕ: Токен HuggingFace не предоставлен. Диаризация спикеров будет недоступна.")
            print("Получите токен на https://huggingface.co/settings/tokens")
            self.diarization_pipeline = None

    def extract_audio(self, video_path: str, audio_path: str = None) -> str:
        """
        Извлечение аудио из видео файла.

        Args:
            video_path: Путь к видео файлу
            audio_path: Путь для сохранения аудио (если None, создается автоматически)

        Returns:
            Путь к извлеченному аудио файлу
        """
        if audio_path is None:
            audio_path = str(Path(video_path).with_suffix('.wav'))

        print(f"Извлечение аудио из {video_path}...")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()

        return audio_path

    def transcribe_audio(self, audio_path: str, language: str = "ru") -> Dict:
        """
        Транскрибация аудио с помощью Whisper.

        Args:
            audio_path: Путь к аудио файлу
            language: Язык аудио (по умолчанию 'ru' - русский)

        Returns:
            Результат транскрибации от Whisper
        """
        print(f"Транскрибация аудио...")
        result = self.whisper_model.transcribe(
            audio_path,
            language=language,
            verbose=False
        )
        return result

    def diarize_speakers(self, audio_path: str, num_speakers: Optional[int] = None) -> Dict:
        """
        Диаризация спикеров (определение, кто и когда говорит).

        Args:
            audio_path: Путь к аудио файлу
            num_speakers: Количество спикеров (если известно)

        Returns:
            Словарь с временными метками для каждого спикера
        """
        if self.diarization_pipeline is None:
            raise ValueError("Диаризация недоступна. Предоставьте токен HuggingFace при инициализации.")

        print(f"Диаризация спикеров...")

        # Выполнение диаризации
        diarization_params = {}
        if num_speakers is not None:
            diarization_params['num_speakers'] = num_speakers

        diarization = self.diarization_pipeline(audio_path, **diarization_params)

        # Преобразование результата в удобный формат
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end
            })

        return speaker_segments

    def merge_transcription_and_diarization(
        self,
        transcription: Dict,
        speaker_segments: List[Dict]
    ) -> List[Dict]:
        """
        Объединение результатов транскрибации и диаризации.

        Args:
            transcription: Результат от Whisper
            speaker_segments: Сегменты спикеров от pyannote

        Returns:
            Список сегментов с текстом и спикером
        """
        print("Объединение транскрибации и диаризации...")

        result = []

        for segment in transcription['segments']:
            segment_start = segment['start']
            segment_end = segment['end']
            segment_text = segment['text'].strip()

            # Находим спикера для этого временного отрезка
            # Берем спикера, который говорит большую часть времени в этом сегменте
            speaker_times = {}

            for sp_seg in speaker_segments:
                # Проверяем пересечение временных интервалов
                overlap_start = max(segment_start, sp_seg['start'])
                overlap_end = min(segment_end, sp_seg['end'])
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > 0:
                    speaker = sp_seg['speaker']
                    speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap_duration

            # Выбираем спикера с максимальным временем
            if speaker_times:
                main_speaker = max(speaker_times, key=speaker_times.get)
            else:
                main_speaker = "UNKNOWN"

            result.append({
                'start': segment_start,
                'end': segment_end,
                'speaker': main_speaker,
                'text': segment_text
            })

        return result

    def transcribe_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        language: str = "ru",
        num_speakers: Optional[int] = None,
        cleanup_audio: bool = True
    ) -> List[Dict]:
        """
        Полная транскрибация видео с разделением по ролям.

        Args:
            video_path: Путь к видео файлу
            output_path: Путь для сохранения результата (JSON)
            language: Язык видео
            num_speakers: Количество спикеров (если известно)
            cleanup_audio: Удалить временный аудио файл после обработки

        Returns:
            Список сегментов с транскрибацией по ролям
        """
        # Извлечение аудио
        audio_path = self.extract_audio(video_path)

        try:
            # Транскрибация
            transcription = self.transcribe_audio(audio_path, language=language)

            # Диаризация спикеров (если доступна)
            if self.diarization_pipeline is not None:
                speaker_segments = self.diarize_speakers(audio_path, num_speakers=num_speakers)
                result = self.merge_transcription_and_diarization(transcription, speaker_segments)
            else:
                # Без диаризации - просто транскрибация
                result = []
                for segment in transcription['segments']:
                    result.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'speaker': 'SPEAKER_00',
                        'text': segment['text'].strip()
                    })

            # Сохранение результата
            if output_path:
                self.save_results(result, output_path)

            return result

        finally:
            # Очистка временного аудио файла
            if cleanup_audio and os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"Временный аудио файл удален: {audio_path}")

    def save_results(self, results: List[Dict], output_path: str):
        """
        Сохранение результатов в различных форматах.

        Args:
            results: Список сегментов с транскрибацией
            output_path: Путь для сохранения (расширение определяет формат)
        """
        output_path = Path(output_path)

        if output_path.suffix == '.json':
            # JSON формат
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        elif output_path.suffix == '.txt':
            # Текстовый формат
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in results:
                    time_str = f"[{self._format_time(segment['start'])} -> {self._format_time(segment['end'])}]"
                    f.write(f"{time_str} {segment['speaker']}: {segment['text']}\n")

        elif output_path.suffix == '.srt':
            # SRT формат (субтитры)
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(results, 1):
                    f.write(f"{i}\n")
                    f.write(f"{self._format_time_srt(segment['start'])} --> {self._format_time_srt(segment['end'])}\n")
                    f.write(f"[{segment['speaker']}] {segment['text']}\n\n")

        else:
            # По умолчанию JSON
            with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Результаты сохранены в {output_path}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Форматирование времени в MM:SS формат."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    @staticmethod
    def _format_time_srt(seconds: float) -> str:
        """Форматирование времени в SRT формат."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def print_results(self, results: List[Dict]):
        """
        Красивый вывод результатов в консоль.

        Args:
            results: Список сегментов с транскрибацией
        """
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ ТРАНСКРИБАЦИИ")
        print("="*80 + "\n")

        for segment in results:
            time_str = f"{self._format_time(segment['start'])} -> {self._format_time(segment['end'])}"
            print(f"[{time_str}] {segment['speaker']}:")
            print(f"  {segment['text']}\n")


if __name__ == "__main__":
    # Пример использования
    import sys

    if len(sys.argv) < 2:
        print("Использование: python video_transcriber.py <путь_к_видео> [токен_HF]")
        sys.exit(1)

    video_path = sys.argv[1]
    hf_token = sys.argv[2] if len(sys.argv) > 2 else None

    # Создание транскрайбера
    transcriber = VideoTranscriber(
        whisper_model="base",
        hf_auth_token=hf_token
    )

    # Транскрибация видео
    results = transcriber.transcribe_video(
        video_path,
        output_path="transcription.json",
        language="ru"
    )

    # Вывод результатов
    transcriber.print_results(results)
