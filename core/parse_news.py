import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from telethon import TelegramClient
from datetime import datetime, timedelta
import asyncio
import random
import re
import os


from utils import config, logger, BASE_DIR

# Конфигурация клиента Telegram
api_id = config['api_id']
api_hash = config['api_hash']
phone = config['phone']

# Пути к возможным расположениям файла сессии
SESSION_FILENAME = 'session_name.session'
SESSION_PATHS = [
    BASE_DIR / SESSION_FILENAME,  # В корневой директории проекта
    Path(__file__).parent / SESSION_FILENAME,  # В директории скрипта
    Path(SESSION_FILENAME)  # В текущей рабочей директории
]


# Функция для поиска существующего файла сессии
def find_session_file():
    for path in SESSION_PATHS:
        if path.exists():
            logger.info(f"Найден файл сессии: {path}")
            return path
    return None


# Путь для сохранения файла сессии (всегда в корневой директории)
SESSION_FILE = BASE_DIR / SESSION_FILENAME


# Функция для извлечения имени канала из URL
def extract_channel_name(url):
    match = re.search(r't\.me/([^/]+)', url)
    if match:
        return match.group(1)
    return url


# Функция для получения последних сообщений канала с продвинутой задержкой
async def get_last_day_messages(client, channel_name):
    try:
        channel = await client.get_entity(channel_name)

        # Добавляем случайную задержку между запросами (более естественная)
        delay = random.uniform(3, 7)  # Увеличенная задержка
        await asyncio.sleep(delay)

        # Определяем время один день назад
        import pytz
        local_tz = pytz.timezone('UTC')
        now = datetime.now(local_tz)
        one_day_ago = now - timedelta(days=1)

        # Получаем историю сообщений с ограниченной скоростью
        messages = []
        message_count = 0
        async for message in client.iter_messages(channel, limit=100):  # Уменьшаем лимит до 100
            # Добавляем микро-задержки после каждых 10 сообщений
            message_count += 1
            if message_count % 10 == 0:
                await asyncio.sleep(random.uniform(0.5, 1))

            if message.date >= one_day_ago:
                if message.text:
                    preview = message.text[:100] + "..." if len(message.text) > 100 else message.text
                    message_data = {
                        'text': preview,
                        'message_id': message.id,
                        'channel_id': channel.id,
                        'date': message.date
                    }
                    messages.append(message_data)
            else:
                break

        return messages
    except Exception as e:
        logger.error(f"Ошибка при получении сообщений канала {channel_name}: {e}")
        # Дополнительная задержка при ошибке, чтобы избежать блокировки
        await asyncio.sleep(random.uniform(10, 15))
        return []


# Основная функция
async def main():
    # Чтение списка каналов
    df = pd.read_csv(BASE_DIR / 'data' / 'news.csv')
    news_list = df['Ссылка ТГ'].to_list()
    news_list = [source for source in news_list if isinstance(source, str) and source.startswith('https://t.me/')]

    # Обрабатываем все каналы
    logger.info(f"Всего каналов для обработки: {len(news_list)}")

    session_name = str(SESSION_FILE)

    # Создание клиента Telegram с явным указанием полного пути к файлу сессии
    client = TelegramClient(session_name, api_id, api_hash)

    # Ищем существующий файл сессии в разных директориях
    session_file = find_session_file()
    session_exists = session_file is not None
    logger.info(f"Файл сессии {'найден: ' + str(session_file) if session_exists else 'не найден'}")

    # Если сессия существует, но в другом месте, создаем символическую ссылку или копируем в корневую директорию
    if session_exists and session_file != SESSION_FILE:
        try:
            # Пробуем создать директорию, если её нет
            os.makedirs(SESSION_FILE.parent, exist_ok=True)

            # Копируем файл сессии в целевую директорию
            import shutil
            shutil.copy2(session_file, SESSION_FILE)
            logger.info(f"Копировали файл сессии из {session_file} в {SESSION_FILE}")
        except Exception as e:
            logger.error(f"Ошибка при копировании файла сессии: {e}")

    # Определение функции для пользовательского запроса пароля
    async def custom_password_handler():
        logger.info("Пожалуйста, введите код подтверждения или пароль: ")
        return input()

    try:
        # Запускаем клиент с сохранением сессии
        if not session_exists:
            logger.info("Первый запуск, потребуется авторизация")
            await client.start(phone=phone, password=custom_password_handler)
            logger.info("Авторизация успешна, сессия сохранена")
        else:
            logger.info("Используем существующую сессию")
            await client.start(phone=phone)
            logger.info("Подключение успешно с использованием существующей сессии")

        # Проверяем, что мы действительно авторизованы
        if await client.is_user_authorized():
            logger.info("Пользователь авторизован успешно")
        else:
            logger.error("Пользователь не авторизован, требуется ручная авторизация")
            await client.start(phone=phone, password=custom_password_handler)

        # Словарь для хранения результатов
        results = {}
        all_posts = []

        # Разбиваем каналы на группы и вводим задержки между группами
        chunk_size = 5
        channel_chunks = [news_list[i:i + chunk_size] for i in range(0, len(news_list), chunk_size)]

        for chunk_idx, chunk in enumerate(channel_chunks):
            # Задержка между группами каналов
            if chunk_idx > 0:
                chunk_delay = random.uniform(30, 60)
                await asyncio.sleep(chunk_delay)

            logger.info(f"Обработка группы каналов {chunk_idx + 1}/{len(channel_chunks)}")

            for url in chunk:
                channel_name = extract_channel_name(url)
                try:
                    messages = await get_last_day_messages(client, channel_name)
                    results[channel_name] = [msg['text'] for msg in messages]

                    for msg in messages:
                        post_url = f"https://t.me/{channel_name}/{msg['message_id']}"
                        all_posts.append({
                            'channel_url': f'https://t.me/{channel_name}',
                            'post_text': msg['text'],
                            'post_url': post_url,
                            'post_date': msg['date'].strftime('%Y-%m-%d %H:%M:%S')
                        })

                    logger.info(f"Найдено {len(messages)} новых сообщений в канале {channel_name}")
                except Exception as e:
                    logger.error(f"Ошибка при обработке канала {channel_name}: {e}")
                    results[channel_name] = []
                    # Дополнительная задержка при ошибке
                    await asyncio.sleep(random.uniform(5, 10))

        # Сохранение результатов в CSV
        if all_posts:
            csv_filename = (BASE_DIR / 'data' / 'telegram_posts.csv')
            posts_df = pd.DataFrame(all_posts)
            posts_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            logger.info(f"Результаты сохранены в файл: {csv_filename}")
        else:
            logger.info("Нет новых постов для сохранения в CSV")

    except Exception as e:
        logger.error(f"Общая ошибка при выполнении скрипта: {e}")
    finally:
        # Корректное закрытие клиента
        await client.disconnect()
        logger.info("Клиент отключен")

    return results


# Запуск скрипта
if __name__ == "__main__":
    logger.info("Запуск скрипта для проверки новых постов в Telegram каналах")
    results = asyncio.run(main())
    logger.info("\nСловарь с результатами:")
    for channel, posts in results.items():
        logger.info(f"{channel}: {len(posts)} новых постов")