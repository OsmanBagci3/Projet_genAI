from __future__ import annotations

import pandas as pd
import streamlit as st

from execute_sql import (
    ChainOfThoughtPlanner,
    DB_PATH,
    SQLExecutor,
    SQLGenerator,
    SQLValidator,
    SchemaInspector,
)
from hybrid_retrieval import HybridRetriever
from query_construction import QueryConstructor
from rerank_results import HeuristicReranker
from router import QueryRouter

st.set_page_config(page_title="NLQ to SQL", page_icon="🏥", layout="wide")
st.title("🏥 NLQ to SQL — Base Hospitalière")
st.caption("Posez une question en anglais sur la base de données hospitalière.")


@st.cache_resource
def load_pipeline():
    hybrid = HybridRetriever()
    reranker = HeuristicReranker()
    router = QueryRouter()
    constructor = QueryConstructor()
    generator = SQLGenerator()
    inspector = SchemaInspector(DB_PATH)
    inspector.load()
    planner = ChainOfThoughtPlanner(generator.generate)
    validator = SQLValidator(inspector)
    executor = SQLExecutor(DB_PATH)
    return hybrid, reranker, router, constructor, generator, planner, inspector, validator, executor


hybrid, reranker, router, constructor, generator, planner, inspector, validator, executor = load_pipeline()

query = st.text_input("Question :", placeholder="e.g. Which doctors work in Cardiology?")

if st.button("Exécuter", disabled=not query) and query:
    st.divider()

    # 0. Routing
    route = router.route(query)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Route", route)

    if route == "SCHEMA_HELP":
        st.info("Question orientée schéma détectée.")
        st.subheader("Schéma de la base de données")
        st.code(inspector.summary(), language="text")
        st.stop()
    elif route != "SQL_QUERY":
        st.error("Question hors périmètre SQL. Reformule avec une demande liée à la base hospitalière.")
        st.stop()

    # 1. Retrieval + reranking
    with st.spinner("Recherche de contexte..."):
        results = hybrid.search(query)
        reranked = reranker.rerank(query, results, top_k_final=8)

    # 2. Chain-of-Thought
    with st.spinner("Analyse Chain-of-Thought..."):
        cot_plan = planner.plan(query, inspector.summary())

    with st.expander("Plan CoT", expanded=True):
        st.text(cot_plan)

    # 3. Prompt
    prompt = constructor.build_context(query, reranked, cot_plan=cot_plan)

    # 4. Génération SQL avec boucle de correction
    sql = ""
    is_valid = False
    message = ""
    attempts_log = []

    with st.spinner("Génération SQL..."):
        for attempt in range(1, 4):
            raw = (
                generator.generate(prompt)
                if attempt == 1
                else generator.regenerate_with_feedback(
                    prompt, sql, message, inspector.summary()
                )
            )
            sql = validator.clean_sql(raw)
            is_valid, message = validator.validate(sql)
            attempts_log.append({"Tentative": attempt, "Valide": is_valid, "Message": message})
            if is_valid:
                break

    st.subheader("SQL généré")
    st.code(sql, language="sql")

    with st.expander("Détail des tentatives"):
        st.dataframe(pd.DataFrame(attempts_log), use_container_width=True)

    if not is_valid:
        st.error(f"Échec après 3 tentatives : {message}")
        st.stop()

    st.success(f"Validation : {message}")

    # 5. Exécution
    st.subheader("Résultats")
    try:
        cols, rows = executor.execute(sql)
        if rows:
            df = pd.DataFrame(rows, columns=cols)
            st.dataframe(df, use_container_width=True)
            st.caption(f"{len(rows)} ligne(s) retournée(s)")
        else:
            st.info("Aucune ligne retournée.")
    except Exception as e:
        st.error(f"Erreur lors de l'exécution : {e}")
