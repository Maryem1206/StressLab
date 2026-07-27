from .credit.wrapper    import CreditModuleWrapper
from .climate.wrapper   import ClimateModuleWrapper
from .market.wrapper    import MarketModuleWrapper
from .liquidity.wrapper import LiquidityModuleWrapper

REGISTRY = {
    "credit":    CreditModuleWrapper,
    "climate":   ClimateModuleWrapper,
    "market":    MarketModuleWrapper,
    "liquidity": LiquidityModuleWrapper,
}

MODULE_META = [
    {"id":"credit",    "label":"Risque de Crédit",        "icon":"📊","color":"#FD5108","placeholder":False},
    {"id":"climate",   "label":"Risque Climatique / ESG", "icon":"🌍","color":"#10B981","placeholder":False},
    {"id":"market",    "label":"Risque de Marché",        "icon":"📈","color":"#8B5CF6","placeholder":False},
    {"id":"liquidity", "label":"Risque de Liquidité",     "icon":"💧","color":"#06B6D4","placeholder":False},
]

def get_module_class(mid): return REGISTRY[mid]
def list_modules(): return MODULE_META
