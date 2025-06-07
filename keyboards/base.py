from aiogram.utils.keyboard import InlineKeyboardBuilder

class BaseKeyboardManager:
    @staticmethod
    def create_main_kb():
        builder = InlineKeyboardBuilder()
        builder.button(text="♟ Шахматы", callback_data="chess")
        builder.button(text="🌊 Морской бой", callback_data="naval_battle")
        return builder.as_markup()
    
    @staticmethod
    def create_opponent_kb():
        builder = InlineKeyboardBuilder()
        builder.button(text="👥 С другом", callback_data="friend")
        builder.button(text="🤖 С ИИ", callback_data="llm")
        return builder.as_markup() 
    
    @staticmethod
    def create_llm_kb():
        builder = InlineKeyboardBuilder()
        builder.button(text="Stockfish", callback_data="stockfish")
        builder.button(text="Ollama", callback_data="ollama")
        return builder.as_markup()