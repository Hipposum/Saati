# Saati

Python code that allows you to count eigenvectors, values, and the consistency index

## Транскрибация видео по ролям

Новая функциональность для транскрибации видео в текст с разделением по спикерам (ролям).

### Возможности

- 🎥 Извлечение аудио из видео файлов
- 🗣️ Распознавание речи с помощью OpenAI Whisper
- 👥 Диаризация спикеров (определение, кто и когда говорит)
- 🌍 Поддержка множества языков (русский, английский и др.)
- 📄 Экспорт результатов в JSON, TXT и SRT форматы
- ⚡ Поддержка GPU для ускорения обработки

### Установка

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Для использования диаризации спикеров:
   - Создайте аккаунт на [HuggingFace](https://huggingface.co/)
   - Получите токен на [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Примите условия использования модели: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

### Быстрый старт

#### Базовое использование (только транскрибация)

```python
from video_transcriber import VideoTranscriber

# Создание транскрайбера
transcriber = VideoTranscriber(whisper_model="base")

# Транскрибация видео
results = transcriber.transcribe_video(
    video_path="video.mp4",
    output_path="transcription.json",
    language="ru"
)

# Вывод результатов
transcriber.print_results(results)
```

#### С диаризацией спикеров

```python
from video_transcriber import VideoTranscriber

# Замените на ваш токен HuggingFace
HF_TOKEN = "your_token_here"

# Создание транскрайбера с диаризацией
transcriber = VideoTranscriber(
    whisper_model="base",
    hf_auth_token=HF_TOKEN
)

# Транскрибация с разделением по спикерам
results = transcriber.transcribe_video(
    video_path="video.mp4",
    output_path="transcription.json",
    language="ru",
    num_speakers=2  # Опционально: укажите количество спикеров
)

# Вывод результатов
transcriber.print_results(results)
```

#### Использование из командной строки

```bash
# Без диаризации
python video_transcriber.py video.mp4

# С диаризацией
python video_transcriber.py video.mp4 your_hf_token
```

### Форматы вывода

#### JSON
```json
[
  {
    "start": 0.0,
    "end": 3.5,
    "speaker": "SPEAKER_00",
    "text": "Привет, как дела?"
  },
  {
    "start": 3.5,
    "end": 6.2,
    "speaker": "SPEAKER_01",
    "text": "Отлично, спасибо!"
  }
]
```

#### TXT
```
[00:00 -> 00:03] SPEAKER_00: Привет, как дела?
[00:03 -> 00:06] SPEAKER_01: Отлично, спасибо!
```

#### SRT (субтитры)
```
1
00:00:00,000 --> 00:00:03,500
[SPEAKER_00] Привет, как дела?

2
00:00:03,500 --> 00:00:06,200
[SPEAKER_01] Отлично, спасибо!
```

### Модели Whisper

Доступные размеры моделей (от меньшего к большему):

| Модель | Размер | Качество | Скорость |
|--------|--------|----------|----------|
| tiny   | ~75 MB | Базовое  | Очень быстро |
| base   | ~150 MB| Хорошее  | Быстро |
| small  | ~500 MB| Отличное | Средне |
| medium | ~1.5 GB| Высокое  | Медленно |
| large  | ~3 GB  | Максимальное | Очень медленно |

Рекомендация: начните с `base`, для лучшего качества используйте `medium` или `large`.

### Примеры

Смотрите файл `example.py` для различных сценариев использования:

- Базовая транскрибация
- Транскрибация с диаризацией
- Сохранение в разных форматах
- Мультиязычная обработка
- Использование больших моделей
- Пользовательский workflow
- Пакетная обработка

### Требования

- Python 3.8+
- FFmpeg (для извлечения аудио из видео)
- CUDA (опционально, для GPU ускорения)

### Установка FFmpeg

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Скачайте с [ffmpeg.org](https://ffmpeg.org/download.html)

### Производительность

Для ускорения обработки:

1. Используйте GPU (CUDA):
```python
transcriber = VideoTranscriber(whisper_model="base", device="cuda")
```

2. Используйте меньшую модель Whisper для быстрой обработки
3. Обрабатывайте видео пакетами

### Лицензия

MIT License

### Поддержка

Если возникли вопросы или проблемы, создайте Issue в репозитории.