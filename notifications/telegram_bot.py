"""Telegram bot with python-telegram-bot v20+ async support."""

from __future__ import annotations

import os
import asyncio
from typing import Optional

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from notifications.pre_match import run_pre_match_notifications
from notifications.final_results import run_result_notifications
from notifications.daily_summary import run_daily_summary
from notifications.operational_alerts import run_operational_alerts
from config.settings import get_settings


# Command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "🏆 Football Analysis Bot\n\n"
        "Available commands:\n"
        "/bugun - Today's matches with predictions\n"
        "/banko - Best value bets (banker picks)\n"
        "/performans - Model performance statistics\n"
        "/yardim - Show this help message"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /yardim command."""
    await update.message.reply_text(
        "🏆 Football Analysis Bot - Commands\n\n"
        "/bugun - Today's matches with predictions\n"
        "/banko - Best value bets (banker picks)\n"
        "/performans - Model performance statistics\n"
        "/yardim - Show this help message"
    )


async def bugun_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /bugun command - today's matches."""
    await update.message.reply_text("📅 Fetching today's matches...")
    # This would integrate with the existing pre_match_notifications logic
    # For now, return a placeholder
    await update.message.reply_text(
        "📅 Bugünün maçları getiriliyor...\n"
        "Bu özellik yakında eklenecek."
    )


async def banko_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /banko command - best value bets."""
    await update.message.reply_text("💰 Banko tahminler aranıyor...")
    # This would integrate with the existing value_bets logic
    await update.message.reply_text(
        "💰 Banko tahminler getiriliyor...\n"
        "Bu özellik yakında eklenecek."
    )


async def performans_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /performans command - model performance."""
    await update.message.reply_text("📊 Performans istatistikleri getiriliyor...")
    # This would integrate with the existing track_performance logic
    await update.message.reply_text(
        "📊 Performans istatistikleri getiriliyor...\n"
        "Bu özellik yakında eklenecek."
    )


async def unknown_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unknown messages."""
    await update.message.reply_text(
        "❓ Anlaşılmayan komut. /yardim yazın."
    )


def create_bot_application() -> Application:
    """Create and configure the Telegram bot application."""
    settings = get_settings()
    bot_token = settings.api_football_key  # Reuse or add TELEGRAM_BOT_TOKEN to settings
    
    # For now, use environment variable directly
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

    # Create application
    application = Application.builder().token(bot_token).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("yardim", help_command))
    application.add_handler(CommandHandler("bugun", bugun_command))
    application.add_handler(CommandHandler("banko", banko_command))
    application.add_handler(CommandHandler("performans", performans_command))

    # Handle unknown messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, unknown_message))

    return application


async def run_bot_polling() -> None:
    """Run the bot with polling (for development)."""
    application = create_bot_application()
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    # Keep running until interrupted
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        await application.updater.stop_polling()
        await application.stop()
        await application.shutdown()


async def run_bot_webhook(webhook_url: str, port: int = 8080) -> None:
    """Run the bot with webhook (for production)."""
    application = create_bot_application()
    
    await application.initialize()
    await application.start()
    
    # Set webhook
    await application.bot.set_webhook(webhook_url)
    
    # Run webhook server
    await application.updater.start_webhook(
        listen="0.0.0.0",
        port=port,
        url_path="/webhook",
        webhook_url=webhook_url,
    )
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(run_bot_polling())