"""
imf_weo_fetcher.py (credit) -- compatibility shim
==================================================
The canonical implementation has moved to app/modules/core/imf_weo_fetcher.py
so it can be shared by all modules (credit, market, liquidity, climate).

This shim re-exports everything from the new location to avoid breaking
any existing import that still uses the old path.  Update imports to:
    from app.modules.core.imf_weo_fetcher import ...
or the relative equivalent:
    from ..core.imf_weo_fetcher import ...
"""
from ..core.imf_weo_fetcher import *          # noqa: F401,F403
from ..core.imf_weo_fetcher import (          # noqa: F401  (explicit for IDEs)
    MacroIndicator,
    CREDIT_MACRO_UNIVERSE,
    fetch_credit_macro,
    universe_index,
    fetch_fred_series,
    _fetch_fred,
)