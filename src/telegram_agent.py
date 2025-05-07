import os
import json
import logging
import asyncio
import time
from telegram.helpers import escape_markdown
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
# from textwrap import dedent
from agno.agent import Agent
from agents_utils import create_financial_agent
from agents_utils import DEFAULT_RESPONSE, WELCOME_MESSAGE, PARSE_MODE
from commands import CommandConfig, COMMANDS
from progress_indicator import ProgressIndicator

#----------------------------------------------------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
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
            logger.debug(f"Extracted symbol with $: {ticker}")
            return ticker
        else:
            ticker = text.strip().upper()
            logger.debug(f"Extracted symbol without $: {ticker}")
            return ticker
    except Exception as e:
        logger.error(f"Error extracting symbol from text '{text}': {str(e)}")
        return None

#----------------------------------------------------------------------------

async def get_agent_response(agent: Agent, query: str, progress: ProgressIndicator = None) -> str:
    """Runs a query through the financial agent and returns the response."""
    try:
        logger.info(f"Running agent query: {query}")
        if progress:
            await progress.update_text("Consultando datos financieros")
        response = agent.run(query)
        logger.debug(f"Agent response received: {response.content[:100]}...")
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"Error running agent query '{query}': {str(e)}")
        return DEFAULT_RESPONSE

#----------------------------------------------------------------------------

# Add conversation states
WAITING_FOR_SYMBOL = 1

# Add rate limiting
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.user_requests = defaultdict(list)
        logger.info(f"Rate limiter initialized: {max_requests} requests per {time_window} seconds")
    
    def is_allowed(self, user_id: int) -> bool:
        now = datetime.now()
        user_timestamps = self.user_requests[user_id]
        user_timestamps = [ts for ts in user_timestamps if now - ts < timedelta(seconds=self.time_window)]
        self.user_requests[user_id] = user_timestamps
        
        if len(user_timestamps) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return False
        
        user_timestamps.append(now)
        logger.debug(f"Request allowed for user {user_id}, current count: {len(user_timestamps)}")
        return True

rate_limiter = RateLimiter(max_requests=10, time_window=60)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message with all available commands."""
    logger.info(f"Help command requested by user {update.effective_user.id}")
    help_text = """📊 Comandos disponibles:

/precio - Te tiro el precio de una acción. Ejemplo: /precio $AAPL
/noticias - Las últimas novedades de una empresa. Ejemplo: /noticias $TSLA
/noticias_general - Un repasito rápido de cómo viene la mano en el mercado.
/tecnicos - Análisis técnico del activo que gustes. Ejemplo: /tecnicos $GOOGL
/fundamentales - Los números pesados de una empresa. Ejemplo: /fundamentales $AAPL
/correlacion - Te cuento cómo se llevan una lista de acciones. Ejemplo: /correlacion $AAPL $MELI
/volatilidad - Te analizo la volatilidad de una acción. Ejemplo: /volatilidad $MELI
/opciones - Te analizo opciones financieras de una acción. Ejemplo: /opciones $MELI

💡 Tips:
Usá MAYÚSCULAS para los tickers
Podés usar el símbolo $ o no
Para más info, usá /ayuda"""

    await update.message.reply_text(help_text)
    logger.debug("Help message sent successfully")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the bot."""
    logger.error(f"Update {update} caused error {context.error}")
    
    if isinstance(context.error, ValueError):
        error_msg = "Che, algo salió mal con los datos. Asegurate de usar un ticker válido."
    elif isinstance(context.error, TimeoutError):
        error_msg = "Se me colgó la conexión. Intentá de nuevo en un ratito."
    else:
        error_msg = "Ups, algo salió mal. Intentá de nuevo más tarde."
    
    logger.error(f"Error details: {str(context.error)}")
    await update.message.reply_text(error_msg)

async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE, config: CommandConfig) -> None:
    """Generic handler for bot commands supporting variable symbol counts."""
    user_id = update.effective_user.id
    command = update.message.text.split()[0]
    logger.info(f"Command '{command}' received from user {user_id}")

    # Check rate limit
    if not rate_limiter.is_allowed(user_id):
        logger.warning(f"Rate limit exceeded for user {user_id}")
        await update.message.reply_text("Che, estás haciendo muchas consultas. Esperá un minuto y volvé a intentar.")
        return
    
    # Initialize progress indicator
    progress = ProgressIndicator(update, context)
    await progress.start("Iniciando análisis")
    
    try:
        if config.agent is None:  # Special case for /start and /help
            logger.debug(f"Handling special command: {command}")
            if update.message.text.startswith("/help"):
                await progress.stop()
                await help_command(update, context)
            else:
                await progress.stop()
                await update.message.reply_text(WELCOME_MESSAGE)
            return

        query = None
        args = " ".join(context.args) if context.args else ""
        symbols = [extract_symbol(arg) for arg in context.args if extract_symbol(arg)] if args else []

        logger.info(f"Command: {config.description}, Extracted Symbols: {symbols}, Len Symbols: {len(symbols)}")

        # Validate symbol count
        if config.requires_symbol:
            if len(symbols) < config.required_symbols_min:
                message = f"Che, mandaste pocos tickers. Necesito al menos {config.required_symbols_min}. Ejemplo: /{update.message.text.split()[0][1:]} {' '.join(['$SYM'] * config.required_symbols_min)}"
                logger.warning(f"Invalid symbol count for user {user_id}: {len(symbols)} < {config.required_symbols_min}")
                await progress.stop()
                await update.message.reply_text(message)
                return
            if config.required_symbols_max and len(symbols) > config.required_symbols_max:
                message = f"Che, mandaste demasiados tickers. Máximo {config.required_symbols_max}."
                logger.warning(f"Too many symbols for user {user_id}: {len(symbols)} > {config.required_symbols_max}")
                await progress.stop()
                await update.message.reply_text(message)
                return
        
        if config.query_template:
            try:
                if config.required_symbols_min == 0 and not symbols:  # Handle /noticias without ticker
                    query = config.query_template
                    logger.debug("Using template without symbols")

                elif config.required_symbols_min == 1 and len(symbols) == 1:
                    query = config.query_template.format(symbol=symbols[0])
                    logger.debug(f"Formatted query with single symbol: {symbols[0]}")

                elif config.required_symbols_min >= 2 and len(symbols) >= 2:
                    query = config.query_template.format(symbols=" ".join(symbols))
                    logger.debug(f"Formatted query with multiple symbols: {symbols}")

                else:
                    query = config.query_template  # Fallback
                    logger.debug("Using template as fallback")

                logger.info(f"Executing query: {query}")
                await progress.update_text("Procesando datos")
                response_text = await get_agent_response(config.agent, query, progress)
                logger.debug(f"Response length: {len(response_text)} characters")

            except IndexError:
                logger.error(f"Index error processing symbols: {symbols}")
                response_text = "Che, algo salió mal con los tickers. Asegurate de mandarlos bien."
            except KeyError as e:
                logger.error(f"Key error in template formatting: {str(e)}")
                response_text = f"Error en el formato: {str(e)}. Usá el ejemplo del comando."
        else:
            logger.warning(f"No query template defined for command: {command}")
            response_text = "Comando no implementado correctamente, che."

        # Stop progress indicator and send response
        await progress.stop()
        await update.message.reply_text(response_text)
        logger.info(f"Command '{command}' completed for user {user_id}")

    except Exception as e:
        logger.error(f"Unexpected error in handle_command: {str(e)}")
        await progress.stop()
        await update.message.reply_text("Ups, algo salió mal. Intentá de nuevo más tarde.")

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
        logger.error("TELEGRAM_TOKEN not found in environment variables")
        raise ValueError("Telegram token not found.")
    logger.info("Telegram application setup completed")
    return ApplicationBuilder().token(token)

#----------------------------------------------------------------------------

def register_handlers(app):
    """Register command handlers with the application."""
    logger.info("Registering command handlers")
    
    # Add error handler
    app.add_error_handler(error_handler)
    logger.debug("Error handler registered")
    
    # Add help command handler
    app.add_handler(CommandHandler("help", help_command))
    logger.debug("Help command handler registered")
    
    # Register other command handlers
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
        logger.debug(f"Handler registered for command: {command}")
    
    logger.info("All handlers registered successfully")

#----------------------------------------------------------------------------

def main():
    """Initialize and run the bot."""
    logger.info("Starting bot initialization")
    app = setup_application().build()
    register_handlers(app)
    logger.info("Bot started and running...")
    app.run_polling()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()