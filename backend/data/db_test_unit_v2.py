from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

rules = Chroma(persist_directory="./data/coc_rules_db", embedding_function=emb)
scen  = Chroma(persist_directory="./data/coc_scenario_db", embedding_function=emb)

def probe(db, q, k=5):
    print("\n" + "="*80)
    print("QUERY:", q)
    hits = db.similarity_search_with_score(q, k=k)
    for i, (doc, score) in enumerate(hits, 1):
        txt = doc.page_content.replace("\n", " ")
        print(f"\n[{i}] score={score:.4f}  meta={doc.metadata}")
        print(txt[:350], "...")
    return hits

# 1) Keyword sanity (must bring the exact section)
probe(rules, "pushing rolls retry once higher stakes", k=5)
probe(rules, "SAN roll when encounter unnatural failed roll sanity loss", k=5)

# 2) RU query sanity (should still work ok with multilingual embeddings)
probe(rules, "проверка рассудка при столкновении с необъяснимым", k=5)

# 3) Scenario atoms sanity
probe(scen, "isolation arctic signal", k=5)
probe(scen, "forbidden text disappearance institutional", k=5)
