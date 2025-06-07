from aiogram.utils.keyboard import InlineKeyboardBuilder

class ChessKeyboard:
    @staticmethod
    def parse_fen(fen: str):
        board = []
        rows = fen.split()[0].split('/')
        for row in rows:
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend([' '] * int(char))
                else:
                    board_row.append(char)
            board.append(board_row)
        return board

    @classmethod
    def create_board(cls, fen: str, inverted: bool = False):
        PIECE_SYMBOLS = {
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
            ' ': ' ', '.': '⬛'
        }

        builder = InlineKeyboardBuilder()
        board = cls.parse_fen(fen)

        if inverted:
            board = board[::-1]
            for row_idx, row in enumerate(board):
                row = row[::-1]
                for col_idx, piece in enumerate(row):
                    text = PIECE_SYMBOLS.get(piece, ' ')
                    builder.button(
                        text=text,
                        callback_data=f"cell_{7 - row_idx}_{7 - col_idx}"
                    )
        else:
            for row_idx, row in enumerate(board):
                for col_idx, piece in enumerate(row):
                    text = PIECE_SYMBOLS.get(piece, ' ')
                    builder.button(
                        text=text,
                        callback_data=f"cell_{row_idx}_{col_idx}"
                    )

        builder.adjust(8)
        return builder.as_markup()
