"""
Database module for storing reviews with embeddings
Minimal storage layer - only stores accepted reviews
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import numpy as np
from typing import List, Dict, Optional, Tuple
import os


class ReviewDatabase:
    def __init__(
        self,
        db_name: str = "synthetic_reviews",
        db_user: str = "postgres",
        db_password: str = None,
        db_host: str = "localhost",
        db_port: str = "5432",
        embedding_dim: int = 384
    ):
        """
        Initialize the review database

        Args:
            db_name: Database name
            db_user: Database user
            db_password: Database password (defaults to env var DB_PASSWORD)
            db_host: Database host
            db_port: Database port
            embedding_dim: Dimension of embedding vectors
        """
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password or os.getenv('DB_PASSWORD', 'postgres')
        self.db_host = db_host
        self.db_port = db_port
        self.embedding_dim = embedding_dim

        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to the database"""
        try:
            self.conn = psycopg2.connect(
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            self.cursor = self.conn.cursor()
            print(f"✓ Connected to database '{self.db_name}'")
        except psycopg2.OperationalError as e:
            print(f"Database '{self.db_name}' doesn't exist. Creating it...")
            self._create_database()
            self.connect()

    def _create_database(self):
        """Create the database if it doesn't exist"""
        conn = psycopg2.connect(
            dbname='postgres',
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE {self.db_name}")
        print(f"✓ Database '{self.db_name}' created")
        cursor.close()
        conn.close()

    def setup_tables(self):
        """Create minimal tables"""
        # Enable pgvector extension
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("✓ pgvector extension enabled")

        # Create minimal reviews table with model_name
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS reviews (
            id SERIAL PRIMARY KEY,
            model_name TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
            review_text TEXT NOT NULL,
            embedding vector({self.embedding_dim}),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.cursor.execute(create_table_query)

        # Create index for similarity search
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS reviews_embedding_idx 
            ON reviews USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

        # Create index for model_name filtering
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS reviews_model_name_idx 
            ON reviews (model_name);
        """)

        self.conn.commit()
        print("✓ Tables created successfully")

    def get_review_count(self, model_name: Optional[str] = None) -> int:
        """
        Get total number of reviews, optionally filtered by model

        Args:
            model_name: Optional model name to filter by
        """
        if model_name:
            self.cursor.execute(
                "SELECT COUNT(*) FROM reviews WHERE model_name = %s", (model_name,))
        else:
            self.cursor.execute("SELECT COUNT(*) FROM reviews")
        return self.cursor.fetchone()[0]

    def get_all_reviews(self, model_name: Optional[str] = None) -> List[Dict]:
        """
        Get all reviews from database, optionally filtered by model

        Args:
            model_name: Optional model name to filter by

        Returns:
            List of dicts with 'id', 'model_name', 'rating', 'review_text', 'created_at'
        """
        if model_name:
            self.cursor.execute("""
                SELECT id, model_name, rating, review_text, created_at 
                FROM reviews 
                WHERE model_name = %s
                ORDER BY id
            """, (model_name,))
        else:
            self.cursor.execute("""
                SELECT id, model_name, rating, review_text, created_at 
                FROM reviews 
                ORDER BY id
            """)

        reviews = []
        for row in self.cursor.fetchall():
            reviews.append({
                'id': row[0],
                'model_name': row[1],
                'rating': row[2],
                'review_text': row[3],
                'created_at': row[4]
            })

        return reviews

    def find_similar_reviews(self, embedding: np.ndarray, limit: int = 5, model_name: Optional[str] = None) -> List[Tuple[int, float, str]]:
        """
        Find similar reviews using cosine similarity

        Args:
            embedding: Query embedding
            limit: Maximum number of similar reviews to return
            model_name: Optional model name to filter by

        Returns:
            List of tuples (review_id, similarity_score, review_text)
        """
        embedding_list = embedding.tolist()

        if model_name:
            query = """
                SELECT id, 1 - (embedding <=> %s::vector) as similarity, review_text
                FROM reviews
                WHERE model_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            self.cursor.execute(
                query, (embedding_list, model_name, embedding_list, limit))
        else:
            query = """
                SELECT id, 1 - (embedding <=> %s::vector) as similarity, review_text
                FROM reviews
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            self.cursor.execute(query, (embedding_list, embedding_list, limit))

        return self.cursor.fetchall()

    def insert_review(
        self,
        model_name: str,
        rating: int,
        review_text: str,
        embedding: np.ndarray
    ) -> Tuple[bool, Optional[int]]:
        """
        Insert a review (simple storage only)

        Args:
            model_name: Name of the model that generated this review
            rating: Rating (1-5)
            review_text: Review text
            embedding: Pre-computed embedding vector

        Returns:
            Tuple of (success, review_id)
        """
        try:
            embedding_list = embedding.tolist()

            self.cursor.execute("""
                INSERT INTO reviews (model_name, rating, review_text, embedding)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """, (model_name, rating, review_text, embedding_list))

            review_id = self.cursor.fetchone()[0]
            self.conn.commit()
            return True, review_id

        except Exception as e:
            self.conn.rollback()
            print(f"Database error: {str(e)}")
            return False, None

    def insert_batch(self, reviews: List[Dict], model_name: str) -> int:
        """
        Insert a batch of reviews

        Args:
            reviews: List of review dicts with 'rating', 'review_text', 'embedding'
            model_name: Name of the model that generated these reviews

        Returns:
            Number of reviews inserted
        """
        inserted = 0
        for review in reviews:
            success, _ = self.insert_review(
                model_name=model_name,
                rating=review['rating'],
                review_text=review['review_text'],
                embedding=review['embedding']
            )
            if success:
                inserted += 1

        return inserted

    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("✓ Database connection closed")
