import pandas as pd
import requests
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import logger, config, BASE_DIR
import time
from core.llm_filter import get_openai_answer, OPENAI_BIG
from core.prompts import GREETING_SYSTEM_PROMPT


GREETING_USER_PROMPT = """
Сгенерируй приветствие для уведомления о {count} новых постах о табаке. 
Включи упоминание о том, что ты проверил {total} сообщений.
"""


def generate_openai_greeting(count: int, total: int) -> str:
    try:
        # Подготавливаем пользовательский промпт с данными
        user_prompt = GREETING_USER_PROMPT.format(count=count, total=total)

        # Запрашиваем генерацию у OpenAI
        greeting, _ = get_openai_answer(
            data=user_prompt,
            prompt=GREETING_SYSTEM_PROMPT,
            temperature=1,
            model=OPENAI_BIG
        )

        # Убираем лишние кавычки, если они есть
        greeting = greeting.strip('"\'')

        logger.info(f"Сгенерировано новое приветствие через OpenAI")
        return greeting

    except Exception as e:
        logger.error(f"Ошибка при генерации приветствия: {e}")
        return f"Эй, пидрила! Нашел для тебя {count} новых табачных новостей из {total} проверенных постов! 🔥"


def send_telegram_message(message: str, format_type=None) -> bool:
    """
    Отправляет сообщение в Telegram с помощью бота.

    Args:
        message: Текст сообщения
        format_type: Тип форматирования ('markdown', 'html' или None)

    Returns:
        Успешность отправки
    """
    try:
        bot_token = config['TELEGRAM_BOT_TOKEN']
        chat_id = config['TELEGRAM_CHAT_ID']

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        logger.info(f"Пытаюсь отправить сообщение в чат с ID: {chat_id}")

        params = {
            "chat_id": chat_id,
            "text": message,
            "disable_web_page_preview": False
        }

        # Добавляем параметр форматирования, если указан
        if format_type == 'markdown':
            params["parse_mode"] = "MarkdownV2"

            # Экранируем специальные символы Markdown
            special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
            for char in special_chars:
                message = message.replace(char, f"\\{char}")

            params["text"] = message

        elif format_type == 'html':
            params["parse_mode"] = "HTML"

        response = requests.post(url, params=params)

        if response.status_code == 200:
            logger.info(f"Сообщение успешно отправлено в Telegram")
            return True
        else:
            logger.error(f"Ошибка при отправке сообщения в Telegram: {response.text}")

            # Если ошибка связана с форматированием, пробуем отправить без форматирования
            if format_type and "can't parse entities" in response.text:
                logger.warning(f"Ошибка при форматировании {format_type}, пробую отправить без форматирования")
                return send_telegram_message(message, format_type=None)

            return False

    except Exception as e:
        logger.error(f"Исключение при отправке сообщения в Telegram: {e}")
        return False


def notify_about_tobacco_news(filtered_df: pd.DataFrame, total_posts: int) -> None:
    """
    Отправляет уведомления о табачных новинках в Telegram.

    Args:
        filtered_df: Датафрейм с отфильтрованными новостями
        total_posts: Общее количество постов до фильтрации
    """
    if filtered_df.empty:
        # Генерируем приветствие для случая, когда новостей нет
        greeting = generate_openai_greeting(0, total_posts)
        # Отправляем приветствие с форматированием Markdown
        send_telegram_message(f"{greeting}\n\nНичего интересного сегодня не нашлось! Попробую снова позже.", format_type='markdown')
        return

    # Формируем приветственное сообщение с динамической генерацией
    greeting = generate_openai_greeting(len(filtered_df), total_posts)

    # Отправляем вступительное сообщение с форматированием Markdown
    send_telegram_message(greeting, format_type='markdown')

    # Для каждой найденной новости отправляем отдельное сообщение с HTML-форматированием
    for i, (idx, row) in enumerate(filtered_df.iterrows(), 1):
        try:
            # Формируем текст новости с использованием HTML и номером новости
            news_message = f"<b>🔥 ТАБАЧНАЯ НОВОСТЬ #{i}/{len(filtered_df)}:</b>\n\n"
            news_message += f"{row['summary']}\n\n"
            news_message += f"<a href='{row['post_url']}'>👉 Читать полный пост</a>"

            # Отправляем сообщение с HTML-форматированием
            send_telegram_message(news_message, format_type='html')

            # Небольшая пауза между сообщениями
            time.sleep(1)

        except Exception as e:
            logger.error(f"Ошибка при формировании сообщения для новости {idx}: {e}")


def main():
    """
    Основная функция для отправки уведомлений.
    """
    try:
        # Загружаем оригинальный датафрейм для подсчета всех постов
        original_df = pd.read_csv(BASE_DIR / 'data' / 'telegram_posts.csv')
        total_posts = len(original_df)

        # Загружаем отфильтрованный датафрейм
        filtered_df = pd.read_csv(BASE_DIR / 'data' / 'posts_filtered.csv')

        # Отправляем уведомления
        notify_about_tobacco_news(filtered_df, total_posts)

        logger.info(f"Уведомления отправлены. Всего новостей: {len(filtered_df)}")

    except Exception as e:
        logger.error(f"Ошибка при отправке уведомлений: {e}")


if __name__ == "__main__":
    main()
