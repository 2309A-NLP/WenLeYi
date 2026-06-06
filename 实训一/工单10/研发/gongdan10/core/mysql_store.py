"""Optional MySQL persistence for completed chat turns."""
import json
from typing import List, Dict


class MySQLStore:
    """Persist completed question/answer records when MySQL is enabled."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "rag_db",
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._enabled = False
        self._pymysql = None

        try:
            import pymysql

            self._pymysql = pymysql
            self._ensure_table()
            self._enabled = True
            print(f"[MYSQL] MySQLStore ready: {host}:{port}/{database}")
        except Exception as e:
            print(f"[MYSQL] MySQLStore disabled: {e}")

    @property
    def is_ready(self) -> bool:
        return self._enabled and self._pymysql is not None

    def _connect(self):
        return self._pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
            autocommit=True,
        )

    def _ensure_table(self):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_records (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(191) NOT NULL,
                        question TEXT NOT NULL,
                        answer MEDIUMTEXT NOT NULL,
                        sources_json MEDIUMTEXT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_session_created (session_id, created_at)
                    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
                    """
                )

    def save_chat(self, session_id: str, question: str, answer: str, sources: List[Dict] = None):
        if not self.is_ready:
            return
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chat_records
                            (session_id, question, answer, sources_json)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (session_id, question, answer, sources_json),
                    )
            print(f"[MYSQL] Saved chat record: session_id={session_id}, answer_len={len(answer)}")
        except Exception as e:
            print(f"[MYSQL] Save chat failed: {e}")
