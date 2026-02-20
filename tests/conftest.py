import os
import sys
from unittest.mock import MagicMock

# Ensure PROMETHEUS_MULTIPROC_DIR exists before any prometheus_client multiprocess import
metrics_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "/tmp/metrics")
os.makedirs(metrics_dir, exist_ok=True)
os.environ["PROMETHEUS_MULTIPROC_DIR"] = metrics_dir

# Stub out langchain_classic (not installed in dev) so chain_manager can be imported
for mod_name in [
    "langchain_classic",
    "langchain_classic.chains",
    "langchain_classic.chains.loading",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()
