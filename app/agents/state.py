import operator
from typing import Annotated, List, TypedDict


class RetrievedDocument(TypedDict):
    content: str
    source: str
    score: float


class AgentState(TypedDict):
    # Using Annotated with operator.add ensures that messages
    # are appended to the history rather than replaced.
    messages: Annotated[List[dict], operator.add]
    current_query: str
    documents: List[RetrievedDocument]
    plan: List[str]
    status: str
    final_answer: str
