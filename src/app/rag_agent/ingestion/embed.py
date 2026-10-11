import psycopg
from pgvector.pg8000 import register_vector

from backend.infra.client import get_client

client = get_client()

conn = psycopg.connect("dbname=rag_db user=postgres password=passwd")
register_vector(conn)
with conn.cursor() as cur:
    
