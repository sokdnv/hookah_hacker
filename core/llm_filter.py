import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm.contrib.concurrent import thread_map
from utils import logger, config, BASE_DIR
from openai import OpenAI

USD_RUB_RATE = 85
TOKEN_LIMIT = 1000
OPENAI_BIG = "gpt-4o-2024-08-06"
OPENAI_MINI = "gpt-4o-mini-2024-07-18"

SYSTEM_PROMPT = """
Ты - ИИ-ассистент для анализа сообщений о новинках табаков и бестабачных смесей для кальяна.
Инструкции:

Определи, содержит ли текст информацию о новинках:

Новые табаки для кальяна (новые вкусы, линейки)
Новые бестабачные смеси (чаи, паста, камни и т.д.)
Лимитированные/сезонные выпуски табаков и смесей
Анонсы будущих релизов табачной продукции
Обновления существующих табаков (изменения вкуса, упаковки)

Если содержит информацию о табачных/бестабачных новинках:
Составь краткую выжимку (3-5 предложений)
Выжимка должна содержать в себе все названия вышедших продуктов

Если НЕ содержит информации о табачных/бестабачных новинках:
Верни только "0" без комментариев

Игнорируй информацию о кальянах, чашах, аксессуарах. Фокусируйся только на табаках и бестабачных смесях для кальяна.
"""


def evaluate_gpt_token_usage(chat_completion_usage: dict, model_version: str = 'gpt-4o-mini') -> float:
    """
    Функция для оценки стоимости генерации OPENAI.
    """

    prices_dict = {
        "gpt-3.5-turbo": {
            'output_tokens': 0.0000015,
            'prompt_tokens': 0.0000005
        },
        "gpt-4-turbo": {
            'output_tokens': 0.00003,
            'prompt_tokens': 0.00001
        },
        "gpt-4o": {
            'output_tokens': 0.000015,
            'prompt_tokens': 0.000005
        },
        "gpt-4o-2024-08-06": {
            'output_tokens': 0.00001,
            'prompt_tokens': 0.0000025
        },
        "gpt-4o-mini-2024-07-18": {
            'output_tokens': 0.0000006,
            'prompt_tokens': 0.00000015
        },
    }

    price_usd = round(chat_completion_usage.completion_tokens * prices_dict[model_version]['output_tokens']
                      + chat_completion_usage.prompt_tokens * prices_dict[model_version]['prompt_tokens'], 5)

    price_rub = float(round(price_usd * USD_RUB_RATE, 3))

    return price_rub


def get_openai_answer(data: str, prompt=SYSTEM_PROMPT, temperature: float = 0,
                      max_tokens: int = TOKEN_LIMIT, model: str = OPENAI_MINI) -> tuple[str, float]:
    """
    Функция для обращения к OPENAI по API и получения ответа.
    Возвращает ответ и стоимость генерации.
    """
    client = OpenAI(api_key=config['OPENAI_API_KEY'])

    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": data}]

    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    answer = chat_completion.choices[0].message.content
    price_rub = evaluate_gpt_token_usage(chat_completion_usage=chat_completion.usage,
                                         model_version=model)
    return answer, price_rub


def process_row(row):
    try:
        # Проверяем, что текст поста существует
        if pd.isna(row['post_text']) or row['post_text'] == '':
            return '0', 0

        # Получаем ответ от OpenAI
        answer, cost = get_openai_answer(data=row['post_text'], prompt=SYSTEM_PROMPT, model=OPENAI_BIG)

        return answer, cost

    except Exception as e:
        logger.error(f"Ошибка при обработке строки: {e}")
        return '0', 0


def process_dataframe(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['llm_output'] = ''

    # Используем thread_map для корректной работы с tqdm
    results = thread_map(process_row, [row for _, row in df_copy.iterrows()],
                         max_workers=max_workers,
                         desc="Обработка записей")

    results = list(results)

    total_cost = 0
    for i, (result, cost) in enumerate(results):
        df_copy.iloc[i, df_copy.columns.get_loc('llm_output')] = result
        total_cost += cost

    filtered_df = df_copy[df_copy['llm_output'] != '0']

    logger.info(f"Обработка завершена. Всего строк: {len(df_copy)}, отфильтровано: {len(filtered_df)}")
    logger.info(f"Общая стоимость запросов: {total_cost:.2f} рублей")

    return filtered_df


def deduplicate_tobacco_news(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаление дубликатов новостных статей с использованием одного вызова LLM.

    Аргументы:
        filtered_df (pd.DataFrame): DataFrame с новостными статьями для удаления дубликатов

    Возвращает:
        pd.DataFrame: DataFrame без дубликатов с наиболее информативными статьями
    """
    # Если 0 или 1 статья, дедупликация не требуется
    if len(filtered_df) <= 1:
        return filtered_df

    logger.info(f"Начало процесса дедупликации для {len(filtered_df)} новостных статей...")

    # Подготовка системного промпта для LLM
    dedup_system_prompt = """
    Вы - ИИ-эксперт по анализу новостей о табачных продуктах.

    Задача: Выявить и удалить дублирующиеся новостные статьи об одном и том же табачном продукте.

    Инструкции:
    1. Внимательно просмотрите все предоставленные новостные статьи
    2. Сгруппируйте дублирующиеся статьи
    3. Для каждой группы дубликатов выберите наиболее полную и информативную статью
    4. Верните список индексов статей, которые следует оставить

    Критерии определения дубликатов:
    - Один и тот же бренд и продукт

    Формат вывода: 
    Список индексов статей для сохранения через запятую (например, "0,3,5")
    """

    # Подготовка пользовательского промпта со всеми новостными статьями
    user_prompt = "\n\n---\n\n".join([
        f"Статья {i}:\n{row['llm_output']}"
        for i, row in enumerate(filtered_df.to_dict('records'))
    ])

    try:
        # Получение рекомендаций по дедупликации от LLM
        answer, _ = get_openai_answer(
            data=user_prompt,
            prompt=dedup_system_prompt,
            temperature=0,
            model=OPENAI_BIG
        )

        # Парсинг индексов для сохранения
        keep_indices = [int(idx.strip()) for idx in answer.split(',')]

        # Фильтрация DataFrame для сохранения только выбранных индексов
        result_df = filtered_df.iloc[keep_indices].reset_index(drop=True)

        logger.info(
            f"Дедупликация завершена. "
            f"Начали с {len(filtered_df)} статей, "
            f"осталось {len(result_df)} уникальных статей"
        )

        return result_df

    except Exception as e:
        logger.error(f"Ошибка при дедупликации: {e}")
        return filtered_df


def main():
    """
    Основная функция для запуска обработки.
    """
    # Загружаем датафрейм
    logger.info("Загрузка данных из файла...")
    try:
        df = pd.read_csv(BASE_DIR / 'data' / 'telegram_posts.csv')
        logger.info(f"Загружено {len(df)} записей")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        return

    # Шаг 1: Обрабатываем датафрейм с помощью LLM для фильтрации табачных новостей
    logger.info("Шаг 1: Первичная фильтрация новостей с помощью LLM...")
    filtered_df = process_dataframe(df)

    # Шаг 2: Выполняем дедупликацию новостей
    if len(filtered_df) > 1:
        logger.info("Шаг 2: Дедупликация табачных новостей...")
        unique_df = deduplicate_tobacco_news(filtered_df)
    else:
        unique_df = filtered_df
        logger.info("Шаг 2: Дедупликация не требуется (найдено менее 2 новостей)")

    # Сохраняем промежуточные результаты (после первичной фильтрации)
    interim_path = (BASE_DIR / 'data' / 'posts_filtered_all.csv')
    try:
        filtered_df.to_csv(interim_path, index=False, encoding='utf-8-sig')
        logger.info(f"Промежуточные результаты (до дедупликации) сохранены в файл: {interim_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении промежуточных результатов: {e}")

    # Сохраняем финальные результаты (после дедупликации)
    output_path = (BASE_DIR / 'data' / 'posts_filtered.csv')
    try:
        unique_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Финальные результаты сохранены в файл: {output_path}")
        logger.info(f"Итого найдено {len(unique_df)} уникальных табачных новостей из {len(df)} проверенных сообщений")
    except Exception as e:
        logger.error(f"Ошибка при сохранении финальных результатов: {e}")


if __name__ == "__main__":
    main()
