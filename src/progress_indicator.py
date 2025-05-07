import logging
from telegram import Update
from telegram.ext import ContextTypes

# Configure logging
logger = logging.getLogger(__name__)

class ProgressIndicator:
    """Handles progress messages for long-running operations efficiently."""
    
    def __init__(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Initialize the progress indicator.
        
        Args:
            update (Update): The Telegram update object
            context (ContextTypes.DEFAULT_TYPE): The Telegram context object
        """
        self.update = update
        self.context = context
        self.message = None
        self.is_running = False
        self.base_text = "Procesando"
    
    async def start(self, initial_text: str = "Procesando"):
        """
        Start the progress indicator with minimal resource usage.
        
        Args:
            initial_text (str): The initial message to display
        """
        if self.is_running:
            return
            
        self.is_running = True
        self.base_text = initial_text
        try:
            self.message = await self.update.message.reply_text(self.base_text)
        except Exception as e:
            logger.error(f"Error starting progress indicator: {str(e)}")
            self.is_running = False
    
    async def update_text(self, new_text: str):
        """
        Update text with error handling.
        
        Args:
            new_text (str): The new message to display
        """
        if not self.message or not self.is_running:
            return
            
        try:
            self.base_text = new_text
            await self.message.edit_text(self.base_text)
        except Exception as e:
            logger.error(f"Error updating progress text: {str(e)}")
    
    async def stop(self, final_text: str = None):
        """
        Clean shutdown of the progress indicator.
        
        Args:
            final_text (str, optional): The final message to display before stopping
        """
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.message:
            try:
                if final_text:
                    await self.message.edit_text(final_text)
                else:
                    await self.message.delete()
            except Exception as e:
                logger.error(f"Error stopping progress indicator: {str(e)}")
        
        self.message = None 