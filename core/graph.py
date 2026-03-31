"""LangGraph orchestration for the NLQ-to-SQL pipeline."""

import sys
from pathlib import Path

from langgraph.graph import END, StateGraph

from core.llm_provider import generate
from core.state import NLQState

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def router_node(state: NLQState) -> dict:
    """Route the query to the appropriate handler."""
    from router import QueryRouter

    router = QueryRouter()
    route = router.route(state["query"])
    return {"route": route, "status": "routed"}


def translator_node(state: NLQState) -> dict:
    """Translate query into multiple variants for retrieval."""
    from query_translation import QueryTranslator

    translator = QueryTranslator()
    variants = translator.translate(state["query"])
    return {
        "translated_queries": variants,
        "status": "translated",
    }


def retriever_node(state: NLQState) -> dict:
    """Retrieve relevant context using hybrid search."""
    from hybrid_retrieval import HybridRetriever

    hybrid = HybridRetriever()
    try:
        results = hybrid.search(state["query"])
        return {
            "retrieval_results": results,
            "status": "retrieved",
        }
    finally:
        hybrid.close()


def reranker_node(state: NLQState) -> dict:
    """Rerank retrieval results."""
    from rerank_results import HeuristicReranker

    reranker = HeuristicReranker()
    reranked = reranker.rerank(
        state["query"],
        state["retrieval_results"],
        top_k_final=8,
    )
    return {
        "reranked_results": reranked,
        "status": "reranked",
    }


def planner_node(state: NLQState) -> dict:
    """Chain-of-thought planning before SQL generation."""
    from execute_sql import DB_PATH, SchemaInspector

    inspector = SchemaInspector(DB_PATH)
    inspector.load()

    planning_prompt = (
        "You are a SQL analyst. Before writing any SQL, "
        "analyze the question carefully.\n\n"
        "DATABASE SCHEMA (only these exist):\n"
        f"{inspector.summary()}\n\n"
        f"QUESTION: {state['query']}\n\n"
        "Answer ONLY these 4 points:\n"
        "1. TABLES NEEDED: Which tables are required?\n"
        "2. COLUMNS NEEDED: Which exact columns?\n"
        "3. FILTERS: What WHERE conditions?\n"
        "4. JOINS: Which foreign keys to join on?\n\n"
        "Use ONLY tables and columns from the schema.\n"
        "Be concise, one line per point."
    )
    plan = generate(planning_prompt, state.get("provider", ""))
    return {"cot_plan": plan, "status": "planned"}


def constructor_node(state: NLQState) -> dict:
    """Build the SQL generation prompt."""
    from query_construction import QueryConstructor

    constructor = QueryConstructor()
    prompt = constructor.build_context(
        state["query"],
        state["reranked_results"],
        cot_plan=state.get("cot_plan", ""),
    )
    return {"prompt": prompt, "status": "prompt_built"}


def generator_node(state: NLQState) -> dict:
    """Generate SQL using the configured LLM provider."""
    from execute_sql import DB_PATH, SchemaInspector, SQLValidator

    inspector = SchemaInspector(DB_PATH)
    inspector.load()
    validator = SQLValidator(inspector)

    provider = state.get("provider", "")
    prompt = state["prompt"]
    max_attempts = 3
    sql = ""
    is_valid = False
    message = ""

    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            raw_sql = generate(prompt, provider)
        else:
            repair_prompt = (
                f"{prompt}\n\n---\n\n"
                f"PREVIOUS SQL (FAILED):\n{sql}\n\n"
                f"VALIDATION ERROR:\n{message}\n\n"
                f"AVAILABLE SCHEMA:\n{inspector.summary()}\n\n"
                "Fix the SQL. Output ONLY SQL or NO_SQL."
            )
            raw_sql = generate(repair_prompt, provider)

        sql = validator.clean_sql(raw_sql)
        is_valid, message = validator.validate(sql)

        if is_valid:
            break

    return {
        "generated_sql": sql,
        "is_valid": is_valid,
        "validation_message": message,
        "attempts": attempt,
        "status": "generated",
    }


def executor_node(state: NLQState) -> dict:
    """Execute the validated SQL query."""
    from execute_sql import DB_PATH, SQLExecutor

    if not state.get("is_valid", False):
        return {
            "columns": [],
            "rows": [],
            "error": state.get("validation_message", "SQL invalide"),
            "status": "failed",
        }

    executor = SQLExecutor(DB_PATH)
    try:
        columns, rows = executor.execute(state["generated_sql"])
        return {
            "columns": columns,
            "rows": [list(r) for r in rows],
            "error": "",
            "status": "executed",
        }
    except Exception as e:
        return {
            "columns": [],
            "rows": [],
            "error": str(e),
            "status": "execution_error",
        }


def should_execute(state: NLQState) -> str:
    """Decide if SQL should be executed or pipeline should stop."""
    if state.get("route") != "SQL_QUERY":
        return "end"
    return "continue"


def build_graph():
    """Build and compile the NLQ-to-SQL multi-agent graph."""
    graph = StateGraph(NLQState)

    graph.add_node("router", router_node)
    graph.add_node("translator", translator_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("planner", planner_node)
    graph.add_node("constructor", constructor_node)
    graph.add_node("generator", generator_node)
    graph.add_node("executor", executor_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        should_execute,
        {"continue": "translator", "end": END},
    )
    graph.add_edge("translator", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "planner")
    graph.add_edge("planner", "constructor")
    graph.add_edge("constructor", "generator")
    graph.add_edge("generator", "executor")
    graph.add_edge("executor", END)

    return graph.compile()


def run_nlq_pipeline(query: str, provider: str = "") -> dict:
    """Run the full NLQ-to-SQL pipeline.

    Args:
        query: Natural language question.
        provider: LLM provider ('mistral' or 'claude').

    Returns:
        Final state dict with SQL and results.
    """
    app = build_graph()
    initial_state = {
        "query": query,
        "route": "",
        "translated_queries": [],
        "retrieval_results": [],
        "reranked_results": [],
        "cot_plan": "",
        "prompt": "",
        "generated_sql": "",
        "is_valid": False,
        "validation_message": "",
        "columns": [],
        "rows": [],
        "attempts": 0,
        "provider": provider,
        "error": "",
        "status": "started",
    }
    return app.invoke(initial_state)
