import chess
from .init_game import BaseGame
from aiogram import types
from states import GameFSM
from game_field import ChessKeyboard
from aiogram.fsm.context import FSMContext
from aiogram import Bot
from aiogram.fsm.storage.base import StorageKey
from bot import bot

from .phrases import *
from bot import bot
import random
import chess.engine
import aiohttp
import json
import uuid
import redis.asyncio as redis
import asyncio


class BaseChessGame(BaseGame):
    def __init__(self, settings: dict, state: FSMContext):
        super().__init__(settings, state)
        self.board = chess.Board()
        self.player_color = chess.WHITE
        self.selected_cell = None
        self.board_message = None
        self.awaiting_ai_move = False
        self.promotion_move = None
        self.promotion_message = None

    async def start(self, message: types.Message):
        self.message = message
        board_markup = ChessKeyboard.create_board(self.board.fen())
        self.board_message = await message.answer(
            f"Игра началась! Ваш цвет: {'белые' if self.player_color == chess.WHITE else 'черные'}",
            reply_markup=board_markup
        )
        await self.state.set_state(GameFSM.playing)
        await self.state.update_data(game=self)

    async def handle_move(self, callback: types.CallbackQuery):
        current_state = await self.state.get_state()
        if current_state == GameFSM.promotion:
            await self.cancel_promotion()
            await self.state.set_state(GameFSM.playing)

        if self.awaiting_ai_move:
            await callback.answer("Подождите, пока ИИ сделает свой ход.")
            return

        if self.board.turn != self.player_color:
            await callback.answer("Сейчас не ваш ход! Дождитесь хода противника.")
            return

        _, row_str, col_str = callback.data.split('_')
        row, col = int(row_str), int(col_str)
        square = chr(ord('a') + col) + str(8 - row)

        try:
            square_index = chess.parse_square(square)
            piece = self.board.piece_at(square_index)
        except ValueError:
            await callback.answer("Неверные координаты клетки!")
            return

        if self.selected_cell is None:
            if not piece or piece.color != self.player_color:
                await callback.answer("Выберите свою фигуру для хода!")
                return
            self.selected_cell = square
            await callback.answer(f"Фигура выбрана: {square}")
        else:
            move_uci = self.selected_cell + square
            move = chess.Move.from_uci(move_uci)

            if move in self.board.legal_moves:
                # Проверка на превращение пешки
                piece = self.board.piece_at(chess.parse_square(self.selected_cell))
                if piece and piece.piece_type == chess.PAWN:
                    to_rank = chess.square_rank(chess.parse_square(square))
                    if (piece.color == chess.WHITE and to_rank == 7) or \
                       (piece.color == chess.BLACK and to_rank == 0):
                        self.promotion_move = move
                        await self.prompt_promotion(callback.message)
                        return
                
                self.apply_move(move)
                await self.update_board_ui()

                if self.board.is_game_over():
                    await self.finish_game(callback.message)
            else:
                await callback.answer("Недопустимый ход! Попробуйте другой.")
                self.selected_cell = None

    async def prompt_promotion(self, message: types.Message):
        """Запрос превращения у игрока"""
        buttons = [
            [
                types.InlineKeyboardButton(text="♕ Ферзь", callback_data="promote_q"),
                types.InlineKeyboardButton(text="♖ Ладья", callback_data="promote_r"),
            ],
            [
                types.InlineKeyboardButton(text="♗ Слон", callback_data="promote_b"),
                types.InlineKeyboardButton(text="♘ Конь", callback_data="promote_n"),
            ]
        ]
        markup = types.InlineKeyboardMarkup(inline_keyboard=buttons)
        self.promotion_message = await message.answer(
            "Выберите фигуру для превращения:",
            reply_markup=markup
        )
        await self.state.set_state(GameFSM.promotion)

    async def handle_promotion(self, callback: types.CallbackQuery):
        """Обработка выбора фигуры для превращения"""
        if not self.promotion_move:
            await callback.answer("Ошибка превращения!")
            return

        promo_type = callback.data.split('_')[1]
        move = chess.Move(
            from_square=self.promotion_move.from_square,
            to_square=self.promotion_move.to_square,
            promotion={'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}[promo_type]
        )

        if move in self.board.legal_moves:
            self.apply_move(move)
            await self.update_board_ui()
            await callback.answer(f"Пешка превращена в {promo_type.upper()}!")
        else:
            await callback.answer("Недопустимое превращение!")

        if self.promotion_message:
            await self.promotion_message.delete()
            self.promotion_message = None
        self.promotion_move = None
        await self.state.set_state(GameFSM.playing)

        if self.board.is_game_over():
            await self.finish_game(callback.message)

    def apply_move(self, move: chess.Move):
        """Применяет ход к доске и сбрасывает состояние"""
        self.board.push(move)
        self.selected_cell = None

    async def cancel_promotion(self):
        """Отмена превращения"""
        if self.promotion_message:
            await self.promotion_message.delete()
            self.promotion_message = None
        self.promotion_move = None
        self.selected_cell = None

    async def update_board_ui(self):
        board_markup = ChessKeyboard.create_board(self.board.fen())
        await self.board_message.edit_reply_markup(reply_markup=board_markup)

    async def finish_game(self, msg):
        await self.cancel_promotion()
        
        result = self.get_result()
        await msg.answer(f"Игра окончена! {result}")
        await self.state.clear()

    def get_result(self) -> str:
        if self.board.is_checkmate():
            winner = "чёрные" if self.board.turn == chess.WHITE else "белые"
            return f"Мат! Победили {winner}."
        if self.board.is_stalemate():
            return "Пат! Ничья."
        if self.board.is_insufficient_material():
            return "Ничья (недостаточно материала)."
        if self.board.is_seventyfive_moves():
            return "Ничья (правило 75 ходов)."
        if self.board.is_fivefold_repetition():
            return "Ничья (пятикратное повторение позиции)."
        return "Игра завершена."

# Stockfish
class ChessWithStockfish(BaseChessGame):
    def __init__(self, settings: dict, state: FSMContext):
        super().__init__(settings, state)
        self.engine = chess.engine.SimpleEngine.popen_uci(settings.get('engine_path', 'stockfish'))
        self.engine_level = settings.get('engine_level', 2)

    async def start(self, message: types.Message):
        await super().start(message)
        if self.player_color == chess.BLACK:
            await self.make_ai_move()

    async def make_ai_move(self):
        self.awaiting_ai_move = True
        try:
            result = self.engine.play(
                self.board,
                chess.engine.Limit(time=0.1),
                options={"Skill Level": self.engine_level}
            )
            self.apply_move(result.move)
            await self.update_board_ui()
        finally:
            self.awaiting_ai_move = False

    async def handle_move(self, callback: types.CallbackQuery):
        current_fen = self.board.fen()
        await super().handle_move(callback)

        if await self.state.get_state() == GameFSM.promotion:
            return

        if self.board.fen() != current_fen and not self.board.is_game_over():
            if self.board.turn != self.player_color:
                await callback.answer("ИИ думает...")
                await self.make_ai_move()
                if self.board.is_game_over():
                    await self.finish_game(callback.message)

    def __del__(self):
        self.engine.quit()

# Ollama
class ChessWithOllama(BaseChessGame):
    def __init__(self, settings: dict, state: FSMContext):
        super().__init__(settings, state)
        self.ollama_model = settings.get('ollama_model', 'llama3')
        self.ollama_url = settings.get('ollama_url', 'http://localhost:1234/v1/chat/completions')
        self.algebraic_notation = []

    def apply_move(self, move: chess.Move):
        fen_before = self.board.fen()
        self.board.push(move)
        algebraic = to_algebraic(fen_before, move.uci())
        self.algebraic_notation.append(algebraic)

    def build_prompt(self) -> str:
        moves_str = " ".join(self.algebraic_notation)
        legal_moves = [to_algebraic(self.board.fen(), move.uci()) for move in self.board.legal_moves]
        legal_str = " ".join(legal_moves)

        return f"""
            ### Instruction:
            Given the current state of a chess game and the list of legal moves, choose the single best next move for the current position. 
            You MUST follow these rules:
            1. Output ONLY the move in algebraic notation
            2. NEVER include any explanations, evaluations or additional text
            3. Provide EXACTLY one move (the absolute best you see)
            4. The move must be from the Legal Moves list

            ### Input:
            Moves so far: {moves_str}
            Legal Moves: {legal_str}

            ### Response (ONLY algebraic move):
            """.strip()


    async def handle_move(self, callback: types.CallbackQuery):
        current_fen = self.board.fen()
        await super().handle_move(callback)
        
        if await self.state.get_state() == GameFSM.promotion:
            return
            
        if self.board.fen() != current_fen and not self.board.is_game_over():
            if self.board.turn != self.player_color:
                await callback.answer("ИИ думает...")
                await self.make_ai_move()
                if self.board.is_game_over():
                    await self.finish_game(callback.message)

    async def make_ai_move(self):
        MAX_ATTEMPTS = 5
        CURRENT_ATTEMPT = 0


        self.awaiting_ai_move = True

        prompt = self.build_prompt()


        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt.strip()}],
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                while CURRENT_ATTEMPT < MAX_ATTEMPTS:
                    print(f'Попытка {CURRENT_ATTEMPT}')
                    try:
                        async with session.post(
                            self.ollama_url,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=30)) as response:
                            
                            response.raise_for_status()
                            data = await response.json()

                            move_str = data['choices'][0]['message']['content'].strip()
                            print(f"Оллама сходила: {move_str}")

                            if len(move_str.split()) > 1:
                                move_str = move_str.split()[0]
            

                            move_str = algebraic_to_uci(self.board, move_str)
        
                            try:
                                move = chess.Move.from_uci(move_str)
                            except ValueError:
                                raise ValueError(f"Неправильный формат хода: {move_str}")

                            if move in self.board.legal_moves:
                                self.apply_move(move)
                                await self.update_board_ui()
                                return
                            else:
                                raise ValueError(f"Недопустимый ход: {move_str}")
                    except:
                        ...
                    CURRENT_ATTEMPT += 1
                raise('Оллама дурная, за 5 попыток ни одного нормального хода не сделала. Завершаю игру =(')

        except Exception as e:
            error_msg = f"Ошибка ollama: {str(e)}"
            await self.message.answer(error_msg)
            # Сдаёмся от имени ИИ
            await self.message.answer("ИИ сдаётся. Вы победили!")
            await self.finish_game(self.message)
        finally:
            self.awaiting_ai_move = False


def to_algebraic(fen: str, move_str: str) -> str:
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_str)

    if move not in board.legal_moves:
        raise ValueError("Недопустимый ход в данной позиции")

    piece = board.piece_at(move.from_square)

    if piece is None:
        raise ValueError("На указанной клетке нет фигуры")

    piece_symbols = {
        chess.PAWN: "",
        chess.KNIGHT: "N",
        chess.BISHOP: "B",
        chess.ROOK: "R",
        chess.QUEEN: "Q",
        chess.KING: "K",
    }

    symbol = piece_symbols[piece.piece_type]
    target_square = chess.square_name(move.to_square)

    return f"{symbol}{target_square}"

def algebraic_to_uci(board: chess.Board, algebraic_move: str) -> str:
    """
    Находит ход в формате UCI по алгебраической нотации и позиции доски.
    
    :param board: объект chess.Board с текущей позицией
    :param algebraic_move: ход в алгебраической нотации (например, "Nf3", "O-O")
    :return: ход в формате UCI (например, "g1f3")
    :raises ValueError: если ход не найден или неоднозначен
    """

    candidates = []
    for move in board.legal_moves:
        san = board.san(move)
        if san == algebraic_move:
            candidates.append(move)
    
    if len(candidates) == 1:
        return candidates[0].uci()
    elif len(candidates) == 0:
        raise ValueError(f"Ход {algebraic_move} не найден среди легальных ходов.")
    else:
        raise ValueError(f"Ход {algebraic_move} неоднозначен.")


class ChessPvP:
    def __init__(self, player_white_id, player_black_id, storage, game_id=None):
        self.storage = storage
        self.player_white_id = player_white_id
        self.player_black_id = player_black_id

        self.board = chess.Board()
        self.current_player_id = int(self.player_white_id)

        self.board_message_white = None
        self.board_message_black = None

        self.selected_cell = None
        
        self.selected_square = {}

        self.promotion_move = None
        self.promotion_message = None

        self.game_id = game_id

        self.redis = redis.Redis(host='localhost', port=6379, db=0)

        self.board_message_white_id = None
        self.board_message_black_id = None
        self.board_message_white_chat_id = None
        self.board_message_black_chat_id = None


        print(f"Игра инициализирована: белые — {self.player_white_id}, черные — {self.player_black_id}, game_id={self.game_id}")

    async def start(self, message: types.Message):

        self.board_message_white = await message.bot.send_message(
            self.player_white_id,
            "Игра началась! Вы играете за белых.",
            reply_markup=ChessKeyboard.create_board(self.board.fen())
        )
        self.board_message_white_id = self.board_message_white.message_id
        self.board_message_white_chat_id = self.player_white_id  # Обычно chat_id == player_id

        self.board_message_black = await message.bot.send_message(
            self.player_black_id,
            "Игра началась! Вы играете за чёрных.",
            reply_markup=ChessKeyboard.create_board(self.board.fen(), inverted=True)
        )
        self.board_message_black_id = self.board_message_black.message_id
        self.board_message_black_chat_id = self.player_black_id


        await self.save_to_redis()

        # Устанавливаем состояние FSM для обоих игроков (Этот блок скорее всего ничего не делает)
        for player_id in [self.player_white_id, self.player_black_id]:
            ctx = FSMContext(
                storage=self.storage,
                key=StorageKey(bot_id=message.bot.id, chat_id=player_id, user_id=player_id)
            )
            await ctx.set_state(GameFSM.playing)
            await ctx.update_data(game_id=self.game_id)

    async def handle_move(self, callback: types.CallbackQuery):
        ctx = FSMContext(
            storage=self.storage,
            key=StorageKey(bot_id=callback.bot.id, chat_id=callback.from_user.id, user_id=callback.from_user.id)
        )
        data = await ctx.get_data()
        game_id = data.get("game_id")
        game = await ChessPvP.load_from_redis(game_id, self.storage)

        if not game:
            await callback.answer("Игра не найдена или завершена.")
            return

        user_id = callback.from_user.id
        print(f"user_id={user_id} ({type(user_id)}), current_player_id={game.current_player_id} ({type(game.current_player_id)})")

        if user_id != int(game.current_player_id):
            print('Вот тут это говно будет проходить, сравниваю нахуй')
            print(user_id)
            print(game.current_player_id)
            await callback.answer("Сейчас не ваш ход!")
            return

        try:
            _, row_str, col_str = callback.data.split('_')
            row, col = int(row_str), int(col_str)
            square = chr(ord('a') + col) + str(8 - row)
        except Exception:
            await callback.answer("Некорректные данные кнопки!")
            return

        # Если не выбрана начальная клетка, запоминаем её
        if game.selected_cell is None:
            piece = game.board.piece_at(chess.parse_square(square))

            game.selected_cell = square
            await callback.answer(f"Выбрано {square}")
            await game.save_to_redis()
            return
        else:
            move_uci = game.selected_cell + square
            move = chess.Move.from_uci(move_uci)

            if move not in game.board.legal_moves:
                await callback.answer("Недопустимый ход! Попробуйте выбрать фигуру заново.")
                game.selected_cell = None
                await game.save_to_redis()
                return

            piece = game.board.piece_at(chess.parse_square(game.selected_cell))
            to_rank = chess.square_rank(chess.parse_square(square))
            if piece and piece.piece_type == chess.PAWN:
                if (piece.color == chess.WHITE and to_rank == 7) or (piece.color == chess.BLACK and to_rank == 0):
                    game.promotion_move = move
                    game.selected_cell = None
                    await game.prompt_promotion(callback.message)
                    await game.save_to_redis()
                    return

            game.board.push(move)
            game.selected_cell = None
            await game.update_boards()

            if game.board.is_game_over():
                await game.finish_game(callback.message)
                return

            # Передаём ход сопернику
            game.current_player_id = int(game.player_black_id if user_id == game.player_white_id else game.player_white_id)
        
            await callback.answer("Ход выполнен.")
            await game.save_to_redis()

    async def prompt_promotion(self, message: types.Message):
        buttons = [
            [
                types.InlineKeyboardButton(text="♕ Ферзь", callback_data="promote_q"),
                types.InlineKeyboardButton(text="♖ Ладья", callback_data="promote_r"),
            ],
            [
                types.InlineKeyboardButton(text="♗ Слон", callback_data="promote_b"),
                types.InlineKeyboardButton(text="♘ Конь", callback_data="promote_n"),
            ],
        ]
        markup = types.InlineKeyboardMarkup(inline_keyboard=buttons)
        self.promotion_message = await message.answer("Выберите фигуру для превращения:", reply_markup=markup)

    async def handle_promotion(self, callback: types.CallbackQuery):
        if not self.promotion_move:
            await callback.answer("Ошибка превращения!")
            return

        promo_type = callback.data.split('_')[1]
        move = chess.Move(
            from_square=self.promotion_move.from_square,
            to_square=self.promotion_move.to_square,
            promotion={'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}[promo_type]
        )

        if move not in self.board.legal_moves:
            await callback.answer("Недопустимое превращение!")
            return

        self.board.push(move)
        await self.update_boards()

        if self.promotion_message:
            await self.promotion_message.delete()
            self.promotion_message = None
        self.promotion_move = None

        self.current_player_id = self.player_black_id if self.current_player_id == self.player_white_id else self.player_white_id

        await callback.answer(f"Пешка превращена в {promo_type.upper()}!")

        if self.board.is_game_over():
            await self.finish_game(callback.message)

        await self.save_to_redis()

    def _generate_caption(self, idd: int, last_move: str | None = None) -> str:
        if int(idd) != int(self.current_player_id):
            status = random.choice(COMMAND_PHRASES)
        else:
            status = random.choice(WAIT_PHRASES)
        move_info = f"♟️ <b>Последний ход:</b> <code>{last_move}</code>" if last_move else ""
        return f"{status}\n\n{move_info}"

    async def update_boards(self, last_move=None):
        white_markup = ChessKeyboard.create_board(self.board.fen())
        black_markup = ChessKeyboard.create_board(self.board.fen(), inverted=True)

        white_text = self._generate_caption(self.player_white_id, last_move)
        black_text = self._generate_caption(self.player_black_id, last_move)


        

        await bot.edit_message_text(
            chat_id=self.board_message_white_chat_id,
            message_id=self.board_message_white_id,
            text=white_text,
            reply_markup=white_markup,
            parse_mode="HTML"
        )
        await bot.edit_message_text(
            chat_id=self.board_message_black_chat_id,
            message_id=self.board_message_black_id,
            text=black_text,
            reply_markup=black_markup,
            parse_mode="HTML"
        )

    async def finish_game(self, message: types.Message):
        result = self.get_result()
        await message.answer(f"Игра окончена! {result}")

        for player_id in [self.player_white_id, self.player_black_id]:
            ctx = FSMContext(storage=self.storage, key=StorageKey(bot_id=message.bot.id, chat_id=player_id, user_id=player_id))
            await ctx.clear()

        await self.redis.delete(f"game:{self.game_id}") 

    def get_result(self):
        if self.board.is_checkmate():
            winner = "чёрные" if self.board.turn == chess.WHITE else "белые"
            return f"Мат! Победили {winner}."
        if self.board.is_stalemate():
            return "Пат! Ничья."
        if self.board.is_insufficient_material():
            return "Ничья (недостаточно материала)."
        if self.board.is_seventyfive_moves():
            return "Ничья (правило 75 ходов)."
        if self.board.is_fivefold_repetition():
            return "Ничья (пятикратное повторение позиции)."
        return "Игра завершена."

    def serialize(self) -> str:
        data = {


            "player_white_id": self.player_white_id,
            "player_black_id": self.player_black_id,
            "fen": self.board.fen(),
            "current_player_id": self.current_player_id,
            "selected_cell": self.selected_cell,
            "board_message_white_id": self.board_message_white_id,
            "board_message_white_chat_id": self.board_message_white_chat_id,
            "board_message_black_id": self.board_message_black_id,
            "board_message_black_chat_id": self.board_message_black_chat_id,
        }
        return json.dumps(data)

    @classmethod
    def deserialize(cls, data_str: str, storage, game_id=None):
        data = json.loads(data_str)
        obj = cls(data["player_white_id"], data["player_black_id"], storage, game_id=game_id)
        obj.board.set_fen(data["fen"])
        obj.current_player_id = data["current_player_id"]
        obj.selected_cell = data.get("selected_cell")
        obj.board_message_white_id = data.get("board_message_white_id")
        obj.board_message_white_chat_id = data.get("board_message_white_chat_id")
        obj.board_message_black_id = data.get("board_message_black_id")
        obj.board_message_black_chat_id = data.get("board_message_black_chat_id")

        return obj

    async def save_to_redis(self):
        key = f"game:{self.game_id}"
        data_str = self.serialize()
        await self.redis.set(key, data_str)
        await self.redis.expire(key, 86400)

    @classmethod
    async def load_from_redis(cls, game_id, storage):
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        key = f"game:{game_id}"
        data_bytes = await redis_client.get(key)
        if not data_bytes:
            return None
        return cls.deserialize(data_bytes.decode(), storage, game_id)

class GameRoom:
    redis_conn = redis.Redis(host="localhost", port=6379, db=0)

    @staticmethod
    async def create_room(player_id: int) -> str:
        game_id = str(uuid.uuid4())[:8]
        await GameRoom.redis_conn.set(f"chess_{game_id}", str(player_id))
        
        return f"https://t.me/plarubot?start=chess_{game_id}"

    @staticmethod
    async def get_players(game_id: str) -> tuple:
        key = f"room:{game_id}"
        data = await GameRoom.redis_conn.hgetall(key)
        white = int(data.get(b"white", 0) or 0)
        black = int(data.get(b"black", 0) or 0)
        return white, black
