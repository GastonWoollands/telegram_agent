from dataclasses import dataclass
from agents_utils import create_financial_agent

#----------------------------------------------------------------------------

agents = {
    "general"             : create_financial_agent("general"),
    "technical"           : create_financial_agent("technical"),
    "fundamental"         : create_financial_agent("fundamental"),
    "pairs_and_volatility": create_financial_agent("pairs_and_volatility"),
    "options"             : create_financial_agent("options")
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
    required_symbols_min: int = 1
    required_symbols_max: int = None
    no_args_message: str = "Por favor, especifica un símbolo válido. Ejemplo: /{} $AAPL"

#----------------------------------------------------------------------------

# Command configurations
COMMANDS = {
    "start": CommandConfig(
        description="Welcome message",
        agent=None,
        query_template="",
        requires_symbol=False,
        required_symbols_min=0
    ),
    "precio": CommandConfig(
        description="Get current stock price",
        agent=agents["general"],
        query_template="Dame el precio actual del ticker: {symbol}",
        no_args_message="Por favor, especificá el símbolo de la acción. Ejemplo: /precio $AAPL"
    ),
    "noticias": CommandConfig(
        description="Get company or market news",
        agent=agents["general"],
        query_template="Dame un resumen de las últimas noticias del ticker: {symbol}",
        requires_symbol=True,
        required_symbols_min=1,
        no_args_message=None
    ),

    "noticias_general": CommandConfig(
        description="Get company or market news",
        agent=agents["general"],
        query_template="Dame un resumen de las noticias del mercado financiero del ticker: ^GSPC y el ticker ^SPX",
        requires_symbol=False,
        required_symbols_min=0,
        no_args_message=None
    ),
    "tecnicos": CommandConfig(
        description="Get technical analysis",
        agent=agents["technical"],
        query_template="Dame el resumen de los análisis técnicos del ticker: {symbol}",
        requires_symbol=True,
        required_symbols_min=1
    ),
    "fundamentales": CommandConfig(
        description="Get fundamental analysis",
        agent=agents["fundamental"],
        query_template="Dame el resumen detallado de análisis fundamental del ticker: {symbol}",
        requires_symbol=True,
        required_symbols_min=1
    ),
    "correlacion": CommandConfig(
        description="Get correlation between two assets",
        agent=agents["pairs_and_volatility"],
        query_template="Dame la correlacion entre los siguientes tickers: {symbols}",
        requires_symbol=True,
        required_symbols_min=2,
        no_args_message="Mandáme dos tickers. Ejemplo: /correlacion $^SPY $AAPL"
    ),
    "volatilidad": CommandConfig(
        description="Get volatility for the asset",
        agent=agents["pairs_and_volatility"],
        query_template="Dame resumen de volatilidad del ticker: {symbol} con '^SPX' como benchmark",
        requires_symbol=True,
        required_symbols_min=1,
        no_args_message="Mandáme un ticker. Ejemplo: /volatilidad $AAPL"
    ),
    "opciones": CommandConfig(
        description="Get options analysis for the asset",
        agent=agents["options"],
        query_template="Dame analisis de opciones del ticker: {symbol}",
        requires_symbol=True,
        required_symbols_min=1,
        no_args_message="Mandáme un ticker. Ejemplo: /opciones $AAPL"
    ),

}