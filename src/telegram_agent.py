import os
import json
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
# from textwrap import dedent
from agno.agent import Agent
from agents_utils import create_financial_agent
from agents_utils import DEFAULT_RESPONSE, WELCOME_MESSAGE, PARSE_MODE
from commands import CommandConfig, COMMANDS

#----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------

def extract_symbol(text: str) -> str:
    """
    Extracts the stock symbol from a user's message.
    - If '$' is present, extracts the ticker after '$'.
    - If '$' is not present, uses the raw text as the ticker.
    """
    try:
        dollar_index = text.find("$")
        if dollar_index != -1 and dollar_index + 1 < len(text):
            ticker = text[dollar_index + 1:].split()[0].upper()
            return ticker
        else:
            return text.strip().upper()
    except Exception as e:
        logger.error(f"Error extracting symbol: {e}")
        return None

#----------------------------------------------------------------------------

async def get_agent_response(agent: Agent, query: str) -> str:
    """Runs a query through the financial agent and returns the response."""
    try:
        response = agent.run(query)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"Error running agent query: {e}")
        return DEFAULT_RESPONSE


#----------------------------------------------------------------------------

async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE, config: CommandConfig) -> None:
    """Generic handler for bot commands supporting variable symbol counts."""
    if config.agent is None:  # Special case for /start
        await update.message.reply_text(WELCOME_MESSAGE, parse_mode=PARSE_MODE)
        return

    query = None
    args = " ".join(context.args) if context.args else ""
    symbols = [extract_symbol(arg) for arg in context.args if extract_symbol(arg)] if args else []

    logger.info(f"Command: {config.description}, Extracted Symbols: {symbols}, Len Symbols: {len(symbols)}")

    # Validate symbol count
    if config.requires_symbol:
        if len(symbols) < config.required_symbols_min:
            message = f"Che, mandaste pocos tickers. Necesito al menos {config.required_symbols_min}. Ejemplo: /{update.message.text.split()[0][1:]} {' '.join(['$SYM'] * config.required_symbols_min)}"
            await update.message.reply_text(message)
            return
        if config.required_symbols_max and len(symbols) > config.required_symbols_max:
            message = f"Che, mandaste demasiados tickers. Máximo {config.required_symbols_max}."
            await update.message.reply_text(message)
            return
        
    if config.query_template:
        try:
            if config.required_symbols_min == 0 and not symbols:  # Handle /noticias without ticker
                query = config.query_template

            elif config.required_symbols_min == 1 and len(symbols) == 1:
                query = config.query_template.format(symbol=symbols[0])

            elif config.required_symbols_min >= 2 and len(symbols) >= 2:
                query = config.query_template.format(symbols=" ".join(symbols))

            else:
                query = config.query_template  # Fallback

            response_text = await get_agent_response(config.agent, query)

        except IndexError:
            response_text = "Che, algo salió mal con los tickers. Asegurate de mandarlos bien."
        except KeyError as e:
            response_text = f"Error en el formato: {str(e)}. Usá el ejemplo del comando."
    else:
        response_text = "Comando no implementado correctamente, che."

    await update.message.reply_text(response_text)

#----------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["start"])

#----------------------------------------------------------------------------

async def price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["precio"])

#----------------------------------------------------------------------------

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["noticias"])

#----------------------------------------------------------------------------

async def news_general(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["noticias_general"])

#----------------------------------------------------------------------------

async def technical_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["tecnicos"])

#----------------------------------------------------------------------------

async def fundamental_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["fundamentales"])

#----------------------------------------------------------------------------

async def correlation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["correlacion"])

#----------------------------------------------------------------------------

async def volatility(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["volatilidad"])

#----------------------------------------------------------------------------

async def options(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["opciones"])

#----------------------------------------------------------------------------

def setup_application() -> ApplicationBuilder:
    """Initialize and configure the Telegram application."""
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("TELEGRAM_TOKEN not found. Please set the TELEGRAM_TOKEN environment variable.")
        raise ValueError("Telegram token not found.")
    return ApplicationBuilder().token(token)

#----------------------------------------------------------------------------

def register_handlers(app):
    """Register command handlers with the application."""
    handlers = {
        "start": start,
        "precio": price,
        "noticias": news,
        "noticias_general": news_general,
        "tecnicos": technical_analysis,
        "fundamentales": fundamental_analysis,
        "correlacion": correlation,
        "volatilidad": volatility,
        "opciones": options
    }
    for command, handler in handlers.items():
        app.add_handler(CommandHandler(command, handler))

#----------------------------------------------------------------------------

def main():
    """Initialize and run the bot."""
    app = setup_application().build()
    register_handlers(app)
    logger.info("Bot started and running...")
    app.run_polling()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()