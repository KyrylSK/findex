from typing import Optional
from findex.index import InvertedIndex
from findex.semantic import SemanticIndex


class AppState:
    index: Optional[InvertedIndex] = None
    semantic_index: Optional[SemanticIndex] = None


app_state = AppState()