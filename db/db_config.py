"""
Database configuration that works with Docker environment
"""

import os
from db.database import ReviewDatabase


def get_database_config():
    """Get database configuration from environment variables"""
    return {
        'db_name': os.getenv('DB_NAME', 'synthetic_reviews'),
        'db_user': os.getenv('DB_USER', 'postgres'),
        'db_password': os.getenv('DB_PASSWORD', 'postgres'),
        # 'postgres' for Docker, 'localhost' for local
        'db_host': os.getenv('DB_HOST', 'postgres'),
        'db_port': os.getenv('DB_PORT', '5432'),
        'embedding_dim': 384  # Default for all-MiniLM-L6-v2
    }


def initialize_database(embedding_dim: int = 384) -> ReviewDatabase:
    """
    Initialize and setup the review database

    Args:
        embedding_dim: Dimension of embedding vectors (default: 384 for all-MiniLM-L6-v2)

    Returns:
        Configured ReviewDatabase instance
    """
    config = get_database_config()

    db = ReviewDatabase(
        db_name=config['db_name'],
        db_user=config['db_user'],
        db_password=config['db_password'],
        db_host=config['db_host'],
        db_port=config['db_port'],
        embedding_dim=embedding_dim
    )

    db.connect()
    db.setup_tables()

    return db
