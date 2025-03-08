from sqlalchemy import create_engine, text
import pandas as pd
from typing import List
from datetime import date
import numpy as np


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

    df_main = get_db_by_query(query, config, logger)
    df_main.fillna('', inplace=True)
    df_main['info'] = df_main['flavor'] + ' ' + df_main['op_rus_short'] + ' ' + df_main['op_rus_long']
    df_main.drop(columns=['op_rus_short', 'op_rus_long'], inplace=True)
    logger.info(f'Колонки {df_main.columns}')

    query = f"""
    SELECT embedding::text as embedding,
    flavor,
    info
    FROM garbage
    WHERE brand = '{brandname}'
    """

    df_garbage = get_db_by_query(query, config, logger)
    df_garbage['info'] = df_garbage['flavor'] + ' ' + df_garbage['info']
    logger.info(f'Колонки {df_garbage.columns}')

    df_full = pd.concat([df_main, df_garbage], ignore_index=True)

    return df_full


def add_to_garbage_to_db(brand, flavor, info, embedding, config, logger):
    try:
        # Get the engine from the connector
        engine = create_connector(config, logger)

        # Get current date (without time)
        current_date = date.today()

        # Properly format embedding for pgvector
        # Ensure embedding is flattened if it's a 2D array
        if isinstance(embedding, np.ndarray):
            embedding = embedding.flatten()

        # Format as a PostgreSQL array string without any newlines or extra spaces
        embedding_str = f"[{','.join(str(float(x)) for x in embedding)}]"

        # Create a connection using the engine
        with engine.connect() as conn:
            # SQL query to insert data
            query = text("""
            INSERT INTO garbage (brand, flavor, info, timestamp, embedding)
            VALUES (:brand, :flavor, :info, :timestamp, :embedding)
            """)

            # Execute the query with parameters
            conn.execute(query, {
                "brand": brand,
                "flavor": flavor,
                "info": info,
                "timestamp": current_date,
                "embedding": embedding_str
            })

            # Commit the transaction
            conn.commit()

        logger.info(f"Successfully added data for brand '{brand}'")
        return True

    except Exception as e:
        logger.error(f"Error adding data to database: {e}")
        return False