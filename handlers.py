from aiogram.filters import Command
from aiogram import Router, types, F
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from keyboards.base import BaseKeyboardManager
import random
from aiogram import Bot
from states import GameFSM
from aiogram.fsm.storage.base import StorageKey
from game_field import ChessKeyboard
from aiogram.fsm.storage.redis import RedisStorage

from bot import bot
import uuid



router = Router()

from aiogram.filters import CommandStart
import redis.asyncio as redis
from games.chess import *

@router.message(CommandStart())
async def start_cmd(message: types.Message, state: FSMContext):
    
    args = None
    if message.text:
        parts = message.text.split(maxsplit=1)
        if len(parts) > 1:
            args = parts[1]

    if args and args.startswith("chess_"):
        code = args #id партии uuid
        print(code)
        
        redis_client = redis.Redis(host="localhost", port=6379, db=0)

        first_user_id = int(str(await redis_client.get(code))[2:-1])

        await redis_client.delete(code)
        if first_user_id is None:
            await message.answer("Игра не найдена или уже завершена.")
            return
        
        user_ids = [first_user_id, int(message.from_user.id)]
        random.shuffle(user_ids)
        
        storage = RedisStorage(redis=redis_client)
        
        game_id = code

        await state.set_state(GameFSM.playing)
        await state.update_data(game_id=game_id)


        
        game = ChessPvP(user_ids[0], user_ids[1], storage, game_id)
        await game.start(message) 

        

    else:
        await state.clear()

        await state.update_data(usr_id=message.from_user.id)
        await message.answer("Выберите игру:", reply_markup=BaseKeyboardManager.create_main_kb())




####Основное меню игры, коллбэки для выбора игры, противника и LLM
@router.callback_query(F.data.in_(["chess", "naval_battle"]))
async def choose_game(callback: types.CallbackQuery, state: FSMContext):
    
    await state.set_state(GameFSM.choosing_opponent)
    await state.update_data(choosing_game=callback.data)
    await callback.message.edit_text(
        "Выберите противника:",
        reply_markup=BaseKeyboardManager.create_opponent_kb()
    )

@router.callback_query(F.data.in_(["llm", "friend"]), GameFSM.choosing_opponent)
async def choose_llm(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(choosing_opponent=callback.data)
    await state.set_state(GameFSM.choosing_llm)
    data = await state.get_data()

    if callback.data == 'friend':
        await init_game(callback.message, state, data)
    else:
        await callback.message.edit_text(
            "Выберите модель:",
            reply_markup=BaseKeyboardManager.create_llm_kb()
        )

@router.callback_query(F.data.in_(["stockfish", "ollama"]), GameFSM.choosing_llm)
async def start_game(callback: CallbackQuery, state: FSMContext):
    await state.update_data(choosing_llm=callback.data)
    
    data = await state.get_data()
    await callback.message.answer(f'Итак, вы выбрали:\n'
                                  f'Игра: {data["choosing_game"]}\n'
                                  f'Противник: {data["choosing_opponent"]}\n'
                                  f'Модель: {data["choosing_llm"]}')
    
    await init_game(callback.message, state, data)

async def init_game(message: types.Message, state: FSMContext, settings: dict):
    '''Функция инициализации игры'''
    try:
        if settings['choosing_game'] == 'chess':
            if settings['choosing_opponent'] == 'llm':
                if settings['choosing_llm'] == 'stockfish':
                    print('Играю со стокфиш')
                    game = ChessWithStockfish(settings, state)
                else:
                    print('Играю с оламой')
                    game = ChessWithOllama(settings, state)
            else:
                print('Играю с другом')
                link = await GameRoom.create_room(settings['usr_id'])
                await state.set_state(GameFSM.playing)
                await state.update_data(game_id=link[28:])
                await message.answer(link)
                return
        else:
            await message.answer('Игра в морской бой пока не реализована')
            return
        
        await game.start(message)
    
    except Exception as e:
        await message.answer(f"Ошибка запуска игры: {str(e)}")


@router.callback_query(GameFSM.playing)
async def handle_chess_move(callback: types.CallbackQuery, state: FSMContext):

    data = await state.get_data()

    game: ChessPvP | None = data.get("game")

    if not game:
        game_id = data.get("game_id")
        if not game_id:
            await callback.answer("Игра не найдена.")
            return
        game = await ChessPvP.load_from_redis(game_id, state.storage)
        if not game:
            await callback.answer("Игра не найдена.")
            return

    await game.handle_move(callback)


@router.callback_query()
async def debug_all_callbacks(callback: types.CallbackQuery, state: FSMContext):
    current_state = await state.get_state()
    print(f"[DEBUG CALLBACK] Data: {callback.data}, State: {current_state}")
