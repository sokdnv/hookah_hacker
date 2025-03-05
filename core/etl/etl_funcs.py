import pandas as pd
import numpy as np
import faiss

from core.etl.db import get_brands_list, get_embeddings_by_brand
from utils import config, logger
from core.prompts import SYSTEM_PROMPT_BRAND_CHECK, SYSTEM_PROMPT_FLAVOR_CHECK
from core.llm_filter import get_openai_answer, OPENAI_BIG
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("ai-forever/ru-en-RoSBERTa")


def parse_vector_string(vector_str):
    # Удаляем квадратные скобки и разбиваем по запятой
    values = vector_str.strip('[]').split(',')
    # Преобразуем строки в числа
    return np.array([float(val) for val in values])


def check_brand_with_gpt(brands, brand_name):
    user_prompt = f"Бренд из описания {brand_name}, фирмы из база данных {brands}"
    final_brand = get_openai_answer(data=user_prompt, prompt=SYSTEM_PROMPT_BRAND_CHECK, model=OPENAI_BIG)
    return final_brand


def check_flavors_with_gpt(candidates, flavor_name):
    user_prompt = f"Новый табак для проверки: {flavor_name}\n\nТабаки из базы данных:\n"
    for i, candidate in enumerate(candidates, 1):
        user_prompt += f"{i}. {candidate}\n"
    llm_answer = get_openai_answer(data=user_prompt, prompt=SYSTEM_PROMPT_FLAVOR_CHECK, model=OPENAI_BIG)
    return llm_answer


def get_k_neighbours(query, brand, k=5):
    # Получаем эмбеддинги для бренда
    embeddings_df = get_embeddings_by_brand(config=config, logger=logger, brandname=brand)

    # Получаем эмбеддинг запроса
    query_embedding = model.encode(query, prompt_name="search_query").astype('float32')
    query_embedding = query_embedding.reshape(1, -1)

    embeddings_df['embedding'] = embeddings_df['embedding'].apply(parse_vector_string)
    embeddings = np.stack(embeddings_df['embedding'].values).astype('float32')

    # Создаем индекс FAISS для косинусного расстояния
    dimension = embeddings.shape[1]  # 1024 в вашем случае

    # Нормализуем векторы для косинусного сходства
    faiss.normalize_L2(embeddings)

    # Создаем простой плоский индекс (оптимален для ~3500 векторов)
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product для косинусного сходства
    index.add(embeddings)

    # Нормализуем вектор запроса
    faiss.normalize_L2(query_embedding)

    # Ищем k ближайших соседей
    distances, indices = index.search(query_embedding, k)

    # Получаем соответствующие строки из DataFrame
    top_k_flavors_df = embeddings_df.iloc[indices[0]]

    # Опционально: можно добавить значения сходства в результат
    top_k_flavors_df = top_k_flavors_df.copy()
    top_k_flavors_df['similarity'] = distances[0]

    logger.info(f"Топ {k} кандидатов {top_k_flavors_df['flavor'].tolist()}")

    return top_k_flavors_df


def check_and_update(brand_name, flavor_name, additional_info):
    brands = get_brands_list(config=config, logger=logger)
    final_brand, _ = check_brand_with_gpt(brands, brand_name)
    if final_brand == 0:
        # TODO запись в мусорку
        logger.info('Ничего не нашлось')
        pass

    logger.info(f'Определился бренд {final_brand}')

    query = flavor_name + additional_info
    candidates = get_k_neighbours(query=query, brand=final_brand)

    llm_answer = check_flavors_with_gpt(candidates['full_shit'].tolist(), query)
    if llm_answer[0] == "0":
        logger.info('Такой табак уже есть в базе')
    elif llm_answer[0] == "1":
        logger.info('Это новый табак, поздравляю!')
    return llm_answer
