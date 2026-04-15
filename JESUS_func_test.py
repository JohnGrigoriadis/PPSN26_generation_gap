# import sqlite3
# import pandas as pd

# from ariel.ec.a001 import Individual

# conn = sqlite3.connect("__data__/database.db")

# query = """
#     SELECT *
#     FROM individual
#     WHERE TIME_OF_DEATH BETWEEN ? AND ?
#     ORDER BY RANDOM()
#     LIMIT ?
# """

# x, y, n = 20, 40, 5

# df = pd.read_sql_query(query, conn, params=(x, y, n))
# conn.close()

# ind = Individual.model_validate_json(df.iloc[0])
# print(ind)

import json
import random
import sqlite3
import pandas as pd
from ariel.ec.a001 import Individual
import time


class JESUS:
    def __init__(self) -> None:
        
        self.conn = sqlite3.connect("__data__/database.db")

        self.query = """
            SELECT *
            FROM individual
            WHERE (TIME_OF_DEATH BETWEEN ? AND ?) AND (TIME_OF_BIRTH BETWEEN ? AND ?)
            ORDER BY RANDOM()
            LIMIT ?
        """

    def jesi(self,
             median_age: int,
             num_jesi: int,
             tournament_size: int,
            ) -> list[Individual]:
        
        assert tournament_size <= num_jesi, "Tournament size must be <= JESI individuals."

        time_death_low = median_age - 5
        time_death_high = median_age

        time_birth_low = 10
        time_birth_high = time_death_low - 5

        # Fetch a larger pool so tournaments have meaningful competition
        pool_size = num_jesi * tournament_size

        df = pd.read_sql_query(
            self.query, self.conn,
            params=(time_death_low, time_death_high, 
                    time_birth_low, time_birth_high, 
                    pool_size),
        )
        print(f"Fetched {len(df)} individuals from database for JESI tournaments.")

        # Parse JSON string columns back into Python objects
        for col in ("genotype_", "tags_"):
            df[col] = df[col].apply(lambda v: json.loads(v) if isinstance(v, str) else v)

        pool = [Individual.model_validate(row.to_dict()) for _, row in df.iterrows()]
        print(f"Pool size: {len(pool)}")

        # Run num_jesi tournaments, each of size tournament_size; return winners
        winners = []
        for _ in range(num_jesi):
            contestants = [random.choice(pool) for i in range(tournament_size)]
            winners.append(max(contestants, key=lambda ind: ind.fitness))

        return winners


if __name__ == "__main__":
    jesus = JESUS()
    start = time.time()
    population = jesus.jesi(median_age=40,
                            num_jesi=5,
                            tournament_size=5,
                            )
    end = time.time()
    print(f"JESI tournaments completed in {end - start:.2f} seconds.")

    for ind in population:
        print(f"""
    ID: {ind.id}
    Fitness: {ind.fitness}
    Time of Birth: {ind.time_of_birth}
    Time of Death: {ind.time_of_death}""")