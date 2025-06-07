# states.py
from aiogram.fsm.state import StatesGroup, State

class GameFSM(StatesGroup):
    choosing_game = State()
    choosing_opponent = State()
    choosing_llm = State()
    playing = State()  # Добавляем состояние игры
    promotion = State()