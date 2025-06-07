# init_game.py
from abc import ABC, abstractmethod
from aiogram import types
from states import GameFSM
from game_field import ChessKeyboard
from aiogram.fsm.context import FSMContext

class BaseGame(ABC):
    def __init__(self, settings: dict, state: FSMContext):
        self.settings = settings
        self.state = state
        self.message = None
    
    @abstractmethod
    async def start(self, message: types.Message):
        pass
    
    @abstractmethod
    async def handle_move(self, callback: types.CallbackQuery):
        pass