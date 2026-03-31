"""NLQ-to-SQL - Streamlit frontend application."""

import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from core.graph import run_nlq_pipeline
from core.langfuse_tracker import end_trace, log_event, start_trace
from core.llm_provider import list_providers
from core.memory import LongTermMemory, ShortTermMemory
from core.mlflow_model import log_pipeline_as_model
from core.mlflow_tracker import NLQTracker

st.set_page_config(
    page_title="NLQ-to-SQL",
    page_icon="DB",
    layout="wide",
)

if "short_memory" not in st.session_state:
    st.session_state.short_memory = ShortTermMemory()
if "long_memory" not in st.session_state:
    st.session_state.long_memory = LongTermMemory()

tracker = NLQTracker()

st.title("NLQ-to-SQL Multi-Agent")
st.markdown("*Systeme multi-agent de generation SQL " "a partir de langage naturel*")
st.markdown("---")

with st.sidebar:
    st.markdown("### Configuration")
    providers = list_providers()
    if not providers:
        providers = ["mistral", "claude"]
    provider = st.selectbox("LLM Provider", providers, index=0)

    st.markdown("---")
    st.markdown("### Historique des requetes")
    history = st.session_state.long_memory.get_history(limit=5)
    if history:
        for entry in history:
            status = "OK" if entry["is_valid"] else "FAIL"
            with st.expander("[" + status + "] " + entry["query"][:40] + "..."):
                st.code(entry["sql"], language="sql")
                st.markdown("**Provider:** " + entry["provider"])
                st.markdown("**Date:** " + entry["created_at"])
    else:
        st.markdown("*Aucune requete precedente.*")

    st.markdown("---")
    st.markdown("### Memoire de session")
    context = st.session_state.short_memory.get_context()
    st.text(context)

    if st.button("Effacer la memoire de session"):
        st.session_state.short_memory.clear()
        st.rerun()

query = st.text_area(
    "Posez votre question en langage naturel :",
    placeholder="Ex: Which patients received Paracetamol?",
    height=80,
)

if st.button(
    "Generer le SQL",
    type="primary",
    use_container_width=True,
):
    if not query.strip():
        st.warning("Veuillez saisir une question.")
    else:
        pipeline_start = time.time()
        trace = start_trace(query, provider)

        with tracker.track_run(query, provider):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### Pipeline")
                steps = {
                    "router": "[1/7] Routing",
                    "translator": "[2/7] Translation",
                    "retriever": "[3/7] Retrieval",
                    "reranker": "[4/7] Reranking",
                    "planner": "[5/7] CoT Planning",
                    "generator": "[6/7] SQL Generation",
                    "executor": "[7/7] Execution",
                }
                step_placeholders = {}
                for key, label in steps.items():
                    step_placeholders[key] = st.empty()
                    step_placeholders[key].markdown("[ ] " + label)

            with col2:
                for step_key, step_label in steps.items():
                    step_placeholders[step_key].markdown(
                        "**[...] " + step_label + "...**"
                    )

                start = time.time()
                result = run_nlq_pipeline(query, provider)
                elapsed = time.time() - start
                tracker.log_step("total_pipeline", elapsed)

                for key, label in steps.items():
                    step_placeholders[key].markdown("[OK] " + label)

            pipeline_duration = time.time() - pipeline_start

            # Log to Langfuse
            log_event(trace, "routing", {"route": result.get("route", "")})
            log_event(
                trace,
                "sql_generation",
                {
                    "sql": result.get("generated_sql", ""),
                    "valid": result.get("is_valid", False),
                    "attempts": result.get("attempts", 0),
                    "provider": provider,
                },
            )
            if result.get("rows"):
                log_event(
                    trace,
                    "execution",
                    {
                        "columns": result.get("columns", []),
                        "num_rows": len(result.get("rows", [])),
                    },
                )

            tracker.log_results(
                route=result.get("route", ""),
                is_valid=result.get("is_valid", False),
                num_rows=len(result.get("rows", [])),
                attempts=result.get("attempts", 0),
                sql=result.get("generated_sql", ""),
            )

            tracker.log_trace(
                query=query,
                provider=provider,
                route=result.get("route", ""),
                sql=result.get("generated_sql", ""),
                is_valid=result.get("is_valid", False),
                num_rows=len(result.get("rows", [])),
                duration=pipeline_duration,
            )
            log_pipeline_as_model()

            st.session_state.short_memory.add(
                query,
                result.get("generated_sql", ""),
                result.get("is_valid", False),
            )
            st.session_state.long_memory.save(
                query=query,
                route=result.get("route", ""),
                generated_sql=result.get("generated_sql", ""),
                is_valid=result.get("is_valid", False),
                num_rows=len(result.get("rows", [])),
                provider=provider,
                attempts=result.get("attempts", 0),
                duration_seconds=pipeline_duration,
            )

            # End Langfuse trace
            end_trace(
                trace,
                {
                    "status": "success" if result.get("is_valid") else "failed",
                    "sql": result.get("generated_sql", ""),
                    "num_rows": len(result.get("rows", [])),
                    "duration": round(pipeline_duration, 2),
                },
            )

        st.markdown("---")

        route = result.get("route", "")
        if route != "SQL_QUERY":
            st.warning("Route: " + route + " - Question hors perimetre SQL.")
        else:
            st.markdown("### SQL genere")
            st.code(result.get("generated_sql", ""), language="sql")

            if result.get("is_valid"):
                st.success(
                    "SQL valide - " + str(result.get("attempts", 0)) + " tentative(s)"
                )
            else:
                st.error("SQL invalide: " + result.get("validation_message", ""))

            if result.get("rows"):
                st.markdown("### Resultats")
                import pandas as pd

                df = pd.DataFrame(
                    result["rows"],
                    columns=result.get("columns", []),
                )
                st.dataframe(df, use_container_width=True)
                st.markdown(
                    "**" + str(len(result["rows"])) + " ligne(s) retournee(s)**"
                )
            elif result.get("is_valid"):
                st.info("Requete valide mais aucun resultat.")

            if result.get("error"):
                st.error("Erreur: " + result["error"])

        with st.sidebar:
            st.markdown("---")
            st.markdown("### Metriques")
            st.markdown("**Duree totale:** " + str(round(pipeline_duration, 1)) + "s")
            st.markdown("**Provider:** " + provider)
            st.markdown("**Route:** " + result.get("route", ""))
            st.markdown("**Tentatives:** " + str(result.get("attempts", 0)))
            st.markdown(
                "**SQL valide:** " + ("Oui" if result.get("is_valid") else "Non")
            )
            st.markdown("---")
            st.markdown("### Observabilite")
            st.markdown("[MLflow](http://localhost:5001)")
            st.markdown("[Langfuse](https://cloud.langfuse.com)")
