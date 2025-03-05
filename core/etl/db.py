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


def get_brands_and_flavors(config, logger):
    query = """
    SELECT "УИН" as id, "Бренд итог" as brand_name, "Вкус итог" as flavor_name, "Описание короткое рус" as op_rus_short, 
    "Описание длинное рус" as op_rus_long
    FROM main
    """
    df = get_db_by_query(query, config, logger)

    tobacco_dict = {}

    for _, row in df.iterrows():
        total_shit = []
        brand = row['brand_name'].lower()
        total_shit.append(row['flavor_name'])
        total_shit.append(row['op_rus_short'] if row['op_rus_short'] else '')
        total_shit.append(row['op_rus_long'] if row['op_rus_long'] else '')

        total_shit = " ".join(total_shit).strip()

        if brand not in tobacco_dict:
            tobacco_dict[brand] = set()

        tobacco_dict[brand].add(total_shit)

    return tobacco_dict, df


def get_brands_list(config, logger):
    query = """
        SELECT DISTINCT "Бренд итог" as brand_name
        FROM main
        """
    df = get_db_by_query(query, config, logger)
    return df['brand_name'].tolist()


def get_embeddings_by_brand(config, logger, brandname):
    query = f"""
        SELECT DISTINCT ON ("Вкус итог") "embedding"::text as embedding,
        "Вкус итог" as flavor, 
        "Описание короткое рус" as op_rus_short, 
        "Описание длинное рус" as op_rus_long
        FROM main
        WHERE "Бренд итог" = '{brandname}'
        """

    df = get_db_by_query(query, config, logger)
    df.fillna('', inplace=True)
    df['full_shit'] = df['flavor'] + ' ' + df['op_rus_short'] + ' ' + df['op_rus_long']
    return df
