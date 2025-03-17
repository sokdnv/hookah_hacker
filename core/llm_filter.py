import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm.contrib.concurrent import thread_map
from utils import logger, config, BASE_DIR
from openai import OpenAI
from functools import partial
import json

from core.prompts import SYSTEM_PROMPT_FIRST_CHECK, DEDUP_SYSTEM_PROMPT, PARSE_SYSTEM_PROMPT, SUMMARY_SYSTEM_PROMPT, \
    SYSTEM_PROMPT_SECOND_CHECK
from utils import clear_json
from configs import *


def evaluate_gpt_token_usage(chat_completion_usage: dict, model_version: str = 'gpt-4o-mini') -> float:
    """
    Функция для оценки стоимости генерации OPENAI.
    """

    price_usd = round(chat_completion_usage.completion_tokens * prices_dict[model_version]['output_tokens']
                      + chat_completion_usage.prompt_tokens * prices_dict[model_version]['prompt_tokens'], 5)

    price_rub = float(round(price_usd * USD_RUB_RATE, 3))

    return price_rub


def get_openai_answer(data: str, prompt=SYSTEM_PROMPT_FIRST_CHECK, temperature: float = 0,
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


def process_row(row, type: str = 'first_check'):

    if type == 'first_check':
        system_prompt = SYSTEM_PROMPT_FIRST_CHECK
        model = OPENAI_MINI
        temperature = 0
    elif type == 'parse':
        system_prompt = PARSE_SYSTEM_PROMPT
        model = OPENAI_BIG
        temperature = 0
    elif type == 'summary':
        system_prompt = SUMMARY_SYSTEM_PROMPT
        model = OPENAI_BIG
        temperature = 1
    elif type == 'second_check':
        system_prompt = SYSTEM_PROMPT_SECOND_CHECK
        model = OPENAI_BIG
        temperature = 0

    try:
        # Проверяем, что текст поста существует
        if pd.isna(row['post_text']) or row['post_text'] == '':
            return '0', 0

        # Получаем ответ от OpenAI
        answer, cost = get_openai_answer(data=row['post_text'], prompt=system_prompt,
                                         model=model, temperature=temperature)

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
                         desc="Обработка 1 уровня")

    results = list(results)

    total_cost = 0

    for i, (result, cost) in enumerate(results):
        result = int(result.strip('"'))
        df_copy.iloc[i, df_copy.columns.get_loc('llm_output')] = result
        total_cost += cost

    filtered_df = df_copy[df_copy['llm_output'] != 0]

    logger.info(f"Обработка 1 уровня завершена. Всего строк: {len(df_copy)}, отфильтровано: {len(filtered_df)}")

    process_row_second= partial(process_row, type="second_check")
    results = thread_map(process_row_second, [row for _, row in filtered_df.iterrows()],
                         max_workers=max_workers,
                         desc="Обработка 2 уровня")

    results = list(results)

    for i, (result, cost) in enumerate(results):
        result = int(result.strip('"'))
        filtered_df.iloc[i, filtered_df.columns.get_loc('llm_output')] = result
        total_cost += cost

    filtered_df_2 = filtered_df[filtered_df['llm_output'] != 0]

    logger.info(f"Обработка 2 уровня завершена. Всего строк: {len(filtered_df)}, отфильтровано: {len(filtered_df_2)}")
    logger.info(f"Общая стоимость запросов: {total_cost:.2f} рублей")

    return filtered_df_2


def deduplicate_tobacco_news(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаление дубликатов новостных статей с использованием одного вызова LLM.
    """
    # Если 0 или 1 статья, дедупликация не требуется
    if len(filtered_df) <= 1:
        return filtered_df

    logger.info(f"Начало процесса дедупликации для {len(filtered_df)} новостных статей...")

    # Подготовка пользовательского промпта со всеми новостными статьями
    user_prompt = "\n\n---\n\n".join([
        f"Статья {i}:\n{row['post_text']}"
        for i, row in enumerate(filtered_df.to_dict('records'))
    ])

    try:
        # Получение рекомендаций по дедупликации от LLM
        answer, _ = get_openai_answer(
            data=user_prompt,
            prompt=DEDUP_SYSTEM_PROMPT,
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


def parse_info(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """Парсим информацию из текста с помощью llm и создаем новый датафрейм с отдельной строкой для каждого табака"""
    logger.info(f"Начало процесса парсинга информации для {len(df)} новостных статей...")
    df_copy = df.copy()
    df_copy['llm_output'] = ''

    process_row_with_parse = partial(process_row, type="parse")

    # Используем thread_map для корректной работы с tqdm
    results = thread_map(process_row_with_parse, [row for _, row in df_copy.iterrows()],
                         max_workers=max_workers,
                         desc="Обработка записей")

    results = list(results)

    # Обновляем датафрейм результатами
    for i, (result, _) in enumerate(results):
        result = clear_json(result)
        df_copy.iloc[i, df_copy.columns.get_loc('llm_output')] = result

    # Создаем новый датафрейм только с нужными колонками
    logger.info("Создание нового датафрейма с результатами...")

    # Список для хранения данных нового датафрейма
    new_data = []

    # Парсим JSON-результаты и создаем записи для нового датафрейма
    for idx, row in df_copy.iterrows():
        try:
            if row['llm_output'] and isinstance(row['llm_output'], str):
                json_data = json.loads(row['llm_output'])

                # Если результат - список объектов (несколько табаков)
                if isinstance(json_data, list):
                    for tobacco in json_data:
                        new_data.append({
                            'post_url': row['post_url'],
                            'brand': tobacco.get('brand') if tobacco.get('brand') != 0 else pd.NA,
                            'flavor': tobacco.get('flavor') if tobacco.get('flavor') != 0 else pd.NA,
                            'info': tobacco.get('info') if tobacco.get('info') != 0 else pd.NA
                        })
                # Если результат - один объект (один табак)
                else:
                    new_data.append({
                        'post_url': row['post_url'],
                        'brand': json_data.get('brand') if json_data.get('brand') != 0 else pd.NA,
                        'flavor': json_data.get('flavor') if json_data.get('flavor') != 0 else pd.NA,
                        'info': json_data.get('info') if json_data.get('info') != 0 else pd.NA
                    })
        except json.JSONDecodeError:
            logger.warning(f"Не удалось распарсить JSON для записи {idx}. Значение: {row['llm_output']}")
        except Exception as e:
            logger.error(f"Ошибка при обработке записи {idx}: {str(e)}, строка {row['llm_output']}")

    # Создаем новый датафрейм только с нужными колонками
    result_df = pd.DataFrame(new_data, columns=['post_url', 'brand', 'flavor', 'info'])

    logger.info(f"Процесс парсинга завершен. Создано {len(result_df)} записей из {len(df)} исходных статей.")

    return result_df


def add_summary(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """Добавляем краткое саммари к каждому посту"""
    logger.info(f"Начало процесса саммаризации информации для {len(df)} новостных статей...")
    df_copy = df.copy()
    df_copy['summary'] = ''

    process_row_with_summary = partial(process_row, type="summary")

    # Используем thread_map для корректной работы с tqdm
    results = thread_map(process_row_with_summary, [row for _, row in df_copy.iterrows()],
                         max_workers=max_workers,
                         desc="Обработка записей")

    results = list(results)

    for i, (result, _) in enumerate(results):
        df_copy.iloc[i, df_copy.columns.get_loc('summary')] = result

    return df_copy


def main():
    """
    Основная функция для запуска обработки.
    """
    # Загружаем датафрейм
    logger.info("Загрузка данных из файла...")
    df = pd.read_csv(BASE_DIR / 'data' / 'telegram_posts.csv')
    logger.info(f"Загружено {len(df)} записей")

    # Шаг 1: Обрабатываем датафрейм с помощью LLM для фильтрации табачных новостей
    logger.info("Шаг 1: Первичная фильтрация новостей с помощью LLM...")
    filtered_df = process_dataframe(df)

    # Сохраняем промежуточные результаты (после первичной фильтрации)
    interim_path = (BASE_DIR / 'data' / 'posts_filtered_all.csv')
    filtered_df.to_csv(interim_path, index=False, encoding='utf-8-sig')
    logger.info(f"Промежуточные результаты (до дедупликации) сохранены в файл: {interim_path}")


    # Шаг 2: Выполняем дедупликацию новостей
    if len(filtered_df) > 1:
        logger.info("Шаг 2: Дедупликация табачных новостей...")
        unique_df = deduplicate_tobacco_news(filtered_df)
    else:
        unique_df = filtered_df
        logger.info("Шаг 2: Дедупликация не требуется (найдено менее 2 новостей)")

    logger.info(f"Итого найдено {len(unique_df)} уникальных табачных новостей из {len(df)} проверенных сообщений")

    # Шаг 3. Парсинг информации из датафрейма
    logger.info("Шаг 3: Парсинг информации из новостей...")
    parsed_df = parse_info(unique_df)

    # Шаг 4. Пишем саммари
    logger.info("Шаг 4: Дописываем саммари...")
    final_df = add_summary(unique_df)

    # Сохраняем финальные результаты по вкусам
    output_path = (BASE_DIR / 'data' / 'new_flavors.csv')
    parsed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Финальные результаты по вкусам сохранены в файл: {output_path}")

    # Сохраняем результаты по постам
    output_path = (BASE_DIR / 'data' / 'posts_filtered.csv')
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Финальные результаты по постам сохранены в файл: {output_path}")

if __name__ == "__main__":
    main()
