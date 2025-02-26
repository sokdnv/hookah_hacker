import os
from pathlib import Path
from dotenv import dotenv_values, find_dotenv
import logging


def load_env_config():
    """
    Достаем элементы из .env файла
    """
    dotenv_path = find_dotenv(usecwd=True)
    config = dotenv_values(dotenv_path=dotenv_path)
    return config


def ensure_log_directory():
    """
    Проверяет и создает директорию для логов, если она не существует.
    Работает как с абсолютными, так и с относительными путями.
    """
    # Получаем путь к текущей директории
    base_path = os.getcwd()

    # Для точного определения местоположения проекта, находим корневую директорию
    # (Поднимаемся на уровень выше, если запускаем из подпапки core)
    project_root = base_path
    if os.path.basename(base_path) == 'core':
        project_root = os.path.dirname(base_path)

    # Создаем путь к директории логов
    log_path = Path(project_root) / 'logs'

    # Проверяем и создаем директорию
    if not log_path.exists():
        try:
            log_path.mkdir(parents=True, exist_ok=True)
            print(f"Создана директория для логов: {log_path}")
        except Exception as e:
            print(f"Ошибка при создании директории логов: {e}")

    return log_path


def setup_logging():
    """
    Настройка логирования с выводом в консоль и файл.
    Создает папку logs если она не существует.
    """
    # Создаем папку для логов, если её нет
    log_path = ensure_log_directory()

    # Настройка логирования
    logger = logging.getLogger('telegram_scraper')
    logger.setLevel(logging.INFO)

    # Убедимся, что у логгера нет старых обработчиков
    if logger.handlers:
        logger.handlers = []

    # Создаем форматировщик логов
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Создаем обработчик для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Создаем обработчик для вывода в файл
    log_filename = f'telegram_scraper.log'
    log_file = log_path / log_filename
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Добавляем обработчики к логгеру
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Инициализация глобальных переменных
logger = setup_logging()
config = load_env_config()
BASE_DIR = Path(__file__).parent
