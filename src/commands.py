from dataclasses import dataclass
from agents_utils import create_financial_agent

#----------------------------------------------------------------------------

agents = {
    "general"             : create_financial_agent("general"),
    "technical"           : create_financial_agent("technical"),
    "fundamental"         : create_financial_agent("fundamental"),
    "pairs_and_volatility": create_financial_agent("pairs_and_volatility")
}

#----------------------------------------------------------------------------

@dataclass
class CommandConfig:
    """Configuration for a bot command."""
    description: str
    agent: object
    query_template: str
    requires_symbol: bool = True
    required_symbols_count: int = 1
    no_args_message: str = "Por favor, especifica un símbolo válido. Ejemplo: /{} $AAPL"

#----------------------------------------------------------------------------

# Command configurations
COMMANDS = {
    "start": CommandConfig(
        description="Welcome message",
        agent=None,
        query_template="",
        requires_symbol=False,
        required_symbols_count=0
    ),
    "precio": CommandConfig(
        description="Get current stock price",
        agent=agents["general"],
        query_template="Dame el precio actual de {symbol}",
        no_args_message="Por favor, especificá el símbolo de la acción. Ejemplo: /precio $AAPL"
    ),
    "noticias": CommandConfig(
        description="Get company or market news",
        agent=agents["general"],
        query_template="Dame un resumen de las últimas noticias de {symbol}",
        requires_symbol=False,
        required_symbols_count=0,
        no_args_message=None
    ),
    "tecnicos": CommandConfig(
        description="Get technical analysis",
        agent=agents["technical"],
        query_template="Dame el resumen de los análisis técnicos de {symbol}",
        requires_symbol=True,
        required_symbols_count=1
    ),
    "fundamentales": CommandConfig(
        description="Get fundamental analysis",
        agent=agents["fundamental"],
        query_template="Dame el resumen detallado de análisis fundamental de {symbol}",
        requires_symbol=True,
        required_symbols_count=1
    ),
    "correlacion": CommandConfig(
        description="Get correlation between two assets",
        agent=agents["pairs_and_volatility"],
        query_template="Dame la correlación entre {symbol1} y {symbol2}",
        requires_symbol=True,
        required_symbols_count=2,
        no_args_message="Mandáme dos tickers, loco. Ejemplo: /correlacion $SPY $AAPL"
    ),
    "volatilidad": CommandConfig(
        description="Get volatility for the asset",
        agent=agents["pairs_and_volatility"],
        query_template="Dame resumen de volatilidad de {symbol}",
        requires_symbol=True,
        required_symbols_count=1,
        no_args_message="Mandáme un ticker, loco. Ejemplo: /volatilidad $AAPL"
    )
}