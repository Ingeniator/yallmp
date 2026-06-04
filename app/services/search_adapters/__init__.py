from app.services.search_adapters.base import SearchAdapter
from app.services.search_adapters.tavily import TavilyAdapter
from app.services.search_adapters.exa import ExaAdapter
from app.services.search_adapters.brave import BraveAdapter
from app.services.search_adapters.gigasearch import GigaSearchAdapter

__all__ = ["SearchAdapter", "TavilyAdapter", "ExaAdapter", "BraveAdapter", "GigaSearchAdapter"]


def get_adapter(provider_type: str) -> SearchAdapter:
    """Return the adapter instance for the given provider type string."""
    adapters: dict[str, SearchAdapter] = {
        "tavily": TavilyAdapter(),
        "exa": ExaAdapter(),
        "brave": BraveAdapter(),
        "gigasearch": GigaSearchAdapter(),
    }
    adapter = adapters.get(provider_type)
    if adapter is None:
        raise ValueError(f"Unknown search provider type: {provider_type!r}. "
                         f"Available: {list(adapters)}")
    return adapter
