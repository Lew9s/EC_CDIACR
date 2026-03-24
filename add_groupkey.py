from llama_index.graph_stores.neo4j import Neo4jPGStore
from config import settings

graph_store = Neo4jPGStore(
    username=settings.neo4j_username,
    password=settings.neo4j_password,
    url=settings.neo4j_url,
)
driver = graph_store.client
with driver.session() as session:
    result = session.run("""
    MATCH (co:CHANGE_ORDER)
    WHERE co.name IS NOT NULL
    WITH co,
         CASE 
           WHEN size(split(co.name, '-')) >= 3 THEN
             split(co.name, '-')[0] + '-' + split(co.name, '-')[1]
           ELSE
             co.name 
         END AS key
    SET co.group_key = key
    RETURN count(co) AS updated_count
    """)
    summary = result.single()
    print(f"成功为 {summary['updated_count']} 个变更单添加 group_key")

with driver.session() as session:
    session.run("""
    CREATE INDEX IF NOT EXISTS FOR (co:CHANGE_ORDER) ON (co.group_key)
    """)
    print("已创建 group_key 索引")