from sqlalchemy import create_engine, text
import pandas as pd
from typing import List


def create_connector(config, logger):
    engine = create_engine(
        f"postgresql://{config['DB_USER']}:{config['DB_PASSWORD']}@{config['DB_HOST']}:{config['DB_PORT']}/{config['DATABASE']}",
        connect_args={'connect_timeout': 60},
        pool_size=5,
        max_overflow=10
    )
    if engine:
        logger.info("Подключились к базе!")
    return engine


def get_db_by_query(query, config, logger):
    """
    Выполняет SQL-запрос и возвращает результаты в виде pandas DataFrame.
    """
    try:
        sql = text(query)

        engine = create_connector(config, logger)

        with engine.connect() as connection:
            logger.info(f"Отправляем запрос: {query}")
            result = pd.read_sql(sql, connection)
            logger.info(f"Запрос успешен. Получили {len(result)} строк.")
            return result
    except Exception as e:
        logger.error(f"Ошибка выполнения sql запроса: {str(e)}")
        return None


def check_brand(brand_names: List[str], config, logger) -> List[bool]:
    query = """
    SELECT DISTINCT "Бренд итог" as brand_names
    FROM main
    """

    df = get_db_by_query(query, config, logger)

    brand_names_in_db = [brand.lower() for brand in df['brand_names'].tolist()]

    answer = [brand_name.lower() in brand_names_in_db for brand_name in brand_names]

    return answer
