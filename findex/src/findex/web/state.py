from typing import Optional
from findex.index import InvertedIndex

class AppState:
    index: Optional[InvertedIndex] = None

app_state = AppState()