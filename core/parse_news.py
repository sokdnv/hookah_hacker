import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from telethon import TelegramClient, sync
from datetime import datetime, timedelta
import asyncio
import random
import re

from utils import config, logger, BASE_DIR


# Конфигурация клиента Telegram
api_id = config['api_id']
api_hash = config['api_hash']
phone = config['phone']


# Функция для извлечения имени канала из URL
def extract_channel_name(url):
    match = re.search(r't\.me/([^/]+)', url)
    if match:
        return match.group(1)
    return url


# Функция для получения последних сообщений канала
async def get_last_day_messages(client, channel_name):
    channel = await client.get_entity(channel_name)
    await asyncio.sleep(random.uniform(1, 2))

    # Определяем время один день назад c учетом часового пояса (aware datetime)
    # Используем tzlocal для получения локального часового пояса
    import pytz
    local_tz = pytz.timezone('UTC')  # Telegram использует UTC

    # Создаем aware datetime объект для текущего времени
    now = datetime.now(local_tz)
    one_day_ago = now - timedelta(days=1)

    # Получаем историю сообщений
    messages = []
    async for message in client.iter_messages(channel, limit=200):  # Увеличиваем лимит до 200
        # Проверяем, было ли сообщение отправлено в течение последнего дня
        if message.date >= one_day_ago:
            # Добавляем сообщение в список
            if message.text:
                # Ограничиваем длину сообщения для удобства отображения
                preview = message.text[:100] + "..." if len(message.text) > 100 else message.text

                # Создаем структуру с данными о сообщении
                message_data = {
                    'text': preview,
                    'message_id': message.id,
                    'channel_id': channel.id,
                    'date': message.date
                }

                messages.append(message_data)
        else:
            # Сообщения отсортированы по времени, поэтому как только мы достигаем
            # сообщения старше 1 дня, мы можем прекратить поиск
            break

    return messages


# Основная функция
async def main():
    # Чтение списка каналов
    df = pd.read_csv(BASE_DIR / 'data' / 'news.csv')
    news_list = df['Ссылка ТГ'].to_list()
    news_list = [source for source in news_list if isinstance(source, str) and source.startswith('https://t.me/')]

    session_name = f'session_name'

    # Создание клиента Telegram
    client = TelegramClient(session_name, api_id, api_hash)

    # Подавление предупреждений getpass путем использования обычного ввода
    import sys

    # Определение функции для пользовательского запроса пароля
    async def custom_password_handler():
        logger.info("Пожалуйста, введите код подтверждения или пароль: ")
        return input()

    # Использование пользовательского обработчика для запроса пароля
    await client.start(phone=phone, password=custom_password_handler)
    logger.info("Клиент Telegram успешно запущен")

    # Словарь для хранения результатов
    results = {}

    # Обработка каждого канала
    all_posts = []  # Список для хранения всех постов для CSV

    for url in news_list:
        channel_name = extract_channel_name(url)
        # logger.info(f"Проверка канала: {channel_name}...")

        try:
            messages = await get_last_day_messages(client, channel_name)

            # Форматируем результаты для словаря результатов (только тексты)
            results[channel_name] = [msg['text'] for msg in messages]

            # Добавляем посты в общий список для CSV с ссылками
            for msg in messages:
                # Формируем ссылку на пост
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

    # Сохранение результатов в CSV
    if all_posts:
        csv_filename = (BASE_DIR / 'data' / 'telegram_posts.csv')
        posts_df = pd.DataFrame(all_posts)
        posts_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        logger.info(f"Результаты сохранены в файл: {csv_filename}")
    else:
        logger.info("Нет новых постов для сохранения в CSV")

    # Закрытие клиента
    await client.disconnect()

    return results


# Запуск скрипта
if __name__ == "__main__":
    logger.info("Запуск скрипта для проверки новых постов в Telegram каналах")
    results = asyncio.run(main())
    logger.info("\nСловарь с результатами:")
    for channel, posts in results.items():
        logger.info(f"{channel}: {len(posts)} новых постов")
