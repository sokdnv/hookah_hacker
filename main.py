#!/usr/bin/env python3
"""
Главный скрипт для запуска полного процесса обработки табачных новостей:
1. Парсинг новостей из Telegram-каналов
2. Фильтрация новостей с помощью LLM
3. Отправка уведомлений в Telegram
"""

import os
import importlib.util
import pandas as pd
import time
from pathlib import Path
from utils import logger, config

# Устанавливаем пути к модулям
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = BASE_DIR / 'core'
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'


def import_module_from_file(file_path):
    try:
        module_name = os.path.basename(file_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Ошибка при импорте модуля {file_path}: {e}")
        return None


def ensure_dirs():
    # Проверяем и создаем директорию для данных
    if not DATA_DIR.exists():
        logger.info(f"Создаем директорию для данных: {DATA_DIR}")
        DATA_DIR.mkdir(parents=True)

    # Проверяем и создаем директорию для логов
    LOGS_DIR = BASE_DIR / 'logs'
    if not LOGS_DIR.exists():
        logger.info(f"Создаем директорию для логов: {LOGS_DIR}")
        LOGS_DIR.mkdir(parents=True)


def main():
    try:
        # В первую очередь создаем необходимые директории
        ensure_dirs()

        start_time = time.time()
        logger.info("Запуск полного процесса обработки табачных новостей")

        # Шаг 1: Парсинг новостей из Telegram-каналов
        logger.info("Шаг 1: Парсинг новостей из Telegram-каналов")
        parse_module = import_module_from_file(CORE_DIR / 'parse_news.py')
        if parse_module:
            # Проверяем, является ли main корутиной
            import asyncio
            if asyncio.iscoroutinefunction(parse_module.main):
                asyncio.run(parse_module.main())
            else:
                parse_module.main()
        else:
            logger.error("Не удалось импортировать модуль парсинга")
            return

        # Проверяем, что файл с данными существует после парсинга
        telegram_posts_path = DATA_DIR / 'telegram_posts.csv'
        if not telegram_posts_path.exists():
            logger.error(f"Файл {telegram_posts_path} не найден после парсинга")
            return

        # Подсчитываем общее количество постов для последующего уведомления
        try:
            df = pd.read_csv(telegram_posts_path)
            total_posts = len(df)
            logger.info(f"Получено {total_posts} постов из Telegram-каналов")
        except Exception as e:
            logger.error(f"Ошибка при чтении файла с постами: {e}")
            return

        # Шаг 2: Фильтрация новостей с помощью LLM
        logger.info("Шаг 2: Фильтрация новостей с помощью LLM")
        filter_module = import_module_from_file(CORE_DIR / 'llm_filter.py')
        if filter_module:
            filter_module.main()
        else:
            logger.error("Не удалось импортировать модуль фильтрации")
            return

        # Проверяем, что отфильтрованные данные существуют
        filtered_posts_path = DATA_DIR / 'posts_filtered.csv'
        if not filtered_posts_path.exists():
            logger.error(f"Файл {filtered_posts_path} не найден после фильтрации")
            return

        # Шаг 3: Отправка уведомлений в Telegram
        logger.info("Шаг 3: Отправка уведомлений в Telegram")
        send_module = import_module_from_file(CORE_DIR / 'send_message.py')
        if send_module:
            send_module.main()
        else:
            logger.error("Не удалось импортировать модуль отправки уведомлений")
            return

        execution_time = round(time.time() - start_time, 2)
        logger.info(f"Полный процесс завершен успешно за {execution_time} секунд")

    except Exception as e:
        logger.error(f"Ошибка при выполнении полного процесса: {e}")


if __name__ == "__main__":
    main()
