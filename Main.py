import pygame
import os
from typing import Dict, List, Tuple, Optional
import time
import random

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 800
BOARD_SIZE = 650
SQUARE_SIZE = BOARD_SIZE // 8
BOARD_X = (WINDOW_WIDTH - BOARD_SIZE - 100) // 2
BOARD_Y = (WINDOW_HEIGHT - BOARD_SIZE) // 2

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (186, 202, 68)
CAPTURE_HIGHLIGHT = (255, 0, 0, 50)
CHECK_HIGHLIGHT = (255, 165, 0)
BACKGROUND = (49, 46, 43)

# Setup display
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Chess with AI')


class ChessGame:
    def __init__(self):
        self.reset_game()
        self.load_images()
        self.font = pygame.font.Font(None, 36)
        self.first_black_move = True

    def reset_game(self):
        self.board = self.create_initial_board()
        self.current_turn = 'white'
        self.selected_piece = None
        self.valid_moves = []
        self.game_over = False
        self.winner = None
        self.evaluation = 0
        self.move_history = []
        self.ai_thinking = False
        self.ai_move = None
        # Add tracking for castling
        self.king_moved = {'white': False, 'black': False}
        self.rooks_moved = {
            'white': {'kingside': False, 'queenside': False},
            'black': {'kingside': False, 'queenside': False}
        }

    def create_initial_board(self) -> Dict:
        """Create the initial chess board setup."""
        return {
            'white-pawn': [(6, i) for i in range(8)],
            'white-rook': [(7, 0), (7, 7)],
            'white-knight': [(7, 1), (7, 6)],
            'white-bishop': [(7, 2), (7, 5)],
            'white-queen': [(7, 3)],
            'white-king': [(7, 4)],
            'black-pawn': [(1, i) for i in range(8)],
            'black-rook': [(0, 0), (0, 7)],
            'black-knight': [(0, 1), (0, 6)],
            'black-bishop': [(0, 2), (0, 5)],
            'black-queen': [(0, 3)],
            'black-king': [(0, 4)]
        }

    def load_images(self):
        """Load chess piece images."""
        self.images = {}
        pieces = ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']
        colors = ['black', 'white']

        for color in colors:
            for piece in pieces:
                image_path = os.path.join('Pieces', f'{color}-{piece}1.png')
                if os.path.isfile(image_path):
                    try:
                        self.images[f'{color}-{piece}'] = pygame.transform.scale(
                            pygame.image.load(image_path),
                            (SQUARE_SIZE, SQUARE_SIZE)
                        )
                    except pygame.error as e:
                        print(f"Could not load image {image_path}: {e}")
                        # Fall back to the basic shapes if image loading fails
                        self._create_basic_piece(color, piece)
                else:
                    print(f"Image file not found: {image_path}")
                    # Fall back to the basic shapes if image is missing
                    self._create_basic_piece(color, piece)

    def _create_basic_piece(self, color, piece):
        """Create basic piece shapes as fallback if images are missing."""
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        piece_color = (255, 255, 255) if color == 'white' else (0, 0, 0)

        if piece == 'pawn':
            pygame.draw.circle(surface, piece_color, (SQUARE_SIZE // 2, SQUARE_SIZE // 3), SQUARE_SIZE // 4)
            pygame.draw.polygon(surface, piece_color, [
                (SQUARE_SIZE // 3, SQUARE_SIZE // 2),
                (2 * SQUARE_SIZE // 3, SQUARE_SIZE // 2),
                (SQUARE_SIZE // 2, SQUARE_SIZE * 3 // 4)
            ])
        elif piece == 'rook':
            pygame.draw.rect(surface, piece_color,
                             (SQUARE_SIZE // 4, SQUARE_SIZE // 4, SQUARE_SIZE // 2, SQUARE_SIZE // 2))
        elif piece == 'knight':
            pygame.draw.polygon(surface, piece_color, [
                (SQUARE_SIZE // 4, SQUARE_SIZE * 3 // 4),
                (SQUARE_SIZE // 2, SQUARE_SIZE // 4),
                (SQUARE_SIZE * 3 // 4, SQUARE_SIZE * 3 // 4)
            ])
        elif piece == 'bishop':
            pygame.draw.polygon(surface, piece_color, [
                (SQUARE_SIZE // 4, SQUARE_SIZE * 3 // 4),
                (SQUARE_SIZE // 2, SQUARE_SIZE // 4),
                (SQUARE_SIZE * 3 // 4, SQUARE_SIZE * 3 // 4)
            ])
            pygame.draw.circle(surface, piece_color, (SQUARE_SIZE // 2, SQUARE_SIZE // 3), SQUARE_SIZE // 6)
        elif piece == 'queen':
            pygame.draw.polygon(surface, piece_color, [
                (SQUARE_SIZE // 4, SQUARE_SIZE * 3 // 4),
                (SQUARE_SIZE // 2, SQUARE_SIZE // 4),
                (SQUARE_SIZE * 3 // 4, SQUARE_SIZE * 3 // 4)
            ])
            pygame.draw.circle(surface, piece_color, (SQUARE_SIZE // 2, SQUARE_SIZE // 3), SQUARE_SIZE // 4)
        elif piece == 'king':
            pygame.draw.rect(surface, piece_color,
                             (SQUARE_SIZE // 3, SQUARE_SIZE // 3, SQUARE_SIZE // 3, SQUARE_SIZE // 3))
            pygame.draw.line(surface, piece_color, (SQUARE_SIZE // 2, SQUARE_SIZE // 6),
                             (SQUARE_SIZE // 2, SQUARE_SIZE * 5 // 6), 4)
            pygame.draw.line(surface, piece_color, (SQUARE_SIZE // 6, SQUARE_SIZE // 2),
                             (SQUARE_SIZE * 5 // 6, SQUARE_SIZE // 2), 4)

        self.images[f'{color}-{piece}'] = surface

    def is_valid_move(self, piece: str, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if a move is valid."""
        piece_type = piece.split('-')[1]
        color = piece.split('-')[0]
        start_row, start_col = start
        end_row, end_col = end

        # Basic checks
        if start == end:
            return False

        # Check if destination has friendly piece
        for p, positions in self.board.items():
            if end in positions and p.startswith(color):
                return False

        # Pawn movement
        if piece_type == 'pawn':
            direction = -1 if color == 'white' else 1
            if start_col == end_col:  # Forward movement
                if start_row + direction == end_row:
                    if not any(end in pos for pos in self.board.values()):
                        return True
                elif (color == 'white' and start_row == 6) or (color == 'black' and start_row == 1):
                    if start_row + 2 * direction == end_row:
                        if not any((start_row + direction, start_col) in pos for pos in self.board.values()):
                            if not any(end in pos for pos in self.board.values()):
                                return True
            # Capture
            elif abs(start_col - end_col) == 1 and start_row + direction == end_row:
                for p, positions in self.board.items():
                    if end in positions and not p.startswith(color):
                        return True
            return False

        # Knight movement
        if piece_type == 'knight':
            row_diff = abs(start_row - end_row)
            col_diff = abs(start_col - end_col)
            return (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)

        # Bishop movement
        if piece_type == 'bishop':
            if abs(start_row - end_row) != abs(start_col - end_col):
                return False
            return self.is_path_clear(start, end)

        # Rook movement
        if piece_type == 'rook':
            if start_row != end_row and start_col != end_col:
                return False
            return self.is_path_clear(start, end)

        # Queen movement
        if piece_type == 'queen':
            if start_row != end_row and start_col != end_col:
                if abs(start_row - end_row) != abs(start_col - end_col):
                    return False
            return self.is_path_clear(start, end)

        # King movement
        if piece_type == 'king':
            row_diff = abs(start_row - end_row)
            col_diff = abs(start_col - end_col)

            # Normal move
            if row_diff <= 1 and col_diff <= 1:
                # Check if the move would put the king in check
                temp_board = {k: v[:] for k, v in self.board.items()}
                king_index = temp_board[piece].index(start)
                temp_board[piece][king_index] = end

                # Check if any opponent piece can attack the king at the new position
                opponent_color = 'black' if color == 'white' else 'white'
                for opp_piece, positions in temp_board.items():
                    if opp_piece.startswith(opponent_color):
                        for pos in positions:
                            if self.is_valid_move_without_check(opp_piece, pos, end, temp_board):
                                return False
                return True

            # Castling move
            if row_diff == 0 and col_diff == 2:
                if self.can_castle(color, 'kingside' if end_col > start_col else 'queenside'):
                    return True

            return False

    def can_castle(self, color: str, side: str) -> bool:
        """Check if castling is possible for the given color and side."""
        if self.king_moved[color]:
            return False

        if self.rooks_moved[color][side]:
            return False

        if self.is_in_check(color):
            return False

        king_pos = self.board[f'{color}-king'][0]
        king_row = king_pos[0]
        king_col = king_pos[1]

        if side == 'kingside':
            # Check path is clear
            if not all(not any((king_row, col) in positions for positions in self.board.values())
                       for col in range(king_col + 1, 7)):
                return False

            # Check if squares king moves through are attacked
            for col in range(king_col, king_col + 3):
                if self.is_square_attacked((king_row, col), color):
                    return False

        else:  # queenside
            # Check path is clear
            if not all(not any((king_row, col) in positions for positions in self.board.values())
                       for col in range(1, king_col)):
                return False

            # Check if squares king moves through are attacked
            for col in range(king_col, king_col - 3, -1):
                if self.is_square_attacked((king_row, col), color):
                    return False

        return True

    def is_square_attacked(self, square: Tuple[int, int], color: str) -> bool:
        """Check if a square is attacked by any opponent piece."""
        opponent_color = 'black' if color == 'white' else 'white'
        for piece, positions in self.board.items():
            if piece.startswith(opponent_color):
                for pos in positions:
                    if self.is_valid_move_without_check(piece, pos, square, self.board):
                        return True
        return False

    def is_valid_move_without_check(self, piece: str, start: Tuple[int, int], end: Tuple[int, int],
                                    board: Dict) -> bool:
        """Check if a move is valid without considering check (for internal use)."""
        piece_type = piece.split('-')[1]
        color = piece.split('-')[0]
        start_row, start_col = start
        end_row, end_col = end

        # Basic checks
        if start == end:
            return False

        # Check if destination has friendly piece
        for p, positions in board.items():
            if end in positions and p.startswith(color):
                return False

        # Pawn movement
        if piece_type == 'pawn':
            direction = -1 if color == 'white' else 1
            if start_col == end_col:  # Forward movement
                if start_row + direction == end_row:
                    if not any(end in pos for pos in board.values()):
                        return True
                elif (color == 'white' and start_row == 6) or (color == 'black' and start_row == 1):
                    if start_row + 2 * direction == end_row:
                        if not any((start_row + direction, start_col) in pos for pos in board.values()):
                            if not any(end in pos for pos in board.values()):
                                return True
            # Capture
            elif abs(start_col - end_col) == 1 and start_row + direction == end_row:
                for p, positions in board.items():
                    if end in positions and not p.startswith(color):
                        return True
            return False

        # Knight movement
        if piece_type == 'knight':
            row_diff = abs(start_row - end_row)
            col_diff = abs(start_col - end_col)
            return (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)

        # Bishop movement
        if piece_type == 'bishop':
            if abs(start_row - end_row) != abs(start_col - end_col):
                return False
            return self.is_path_clear_custom(start, end, board)

        # Rook movement
        if piece_type == 'rook':
            if start_row != end_row and start_col != end_col:
                return False
            return self.is_path_clear_custom(start, end, board)

        # Queen movement
        if piece_type == 'queen':
            if start_row != end_row and start_col != end_col:
                if abs(start_row - end_row) != abs(start_col - end_col):
                    return False
            return self.is_path_clear_custom(start, end, board)

        # King movement (simplified for checking attacks)
        if piece_type == 'king':
            row_diff = abs(start_row - end_row)
            col_diff = abs(start_col - end_col)
            return row_diff <= 1 and col_diff <= 1

        return False

    def is_path_clear_custom(self, start: Tuple[int, int], end: Tuple[int, int], board: Dict) -> bool:
        """Check if the path between start and end is clear of pieces using custom board."""
        start_row, start_col = start
        end_row, end_col = end

        row_step = 0 if start_row == end_row else (end_row - start_row) // abs(end_row - start_row)
        col_step = 0 if start_col == end_col else (end_col - start_col) // abs(end_col - start_col)

        current_row = start_row + row_step
        current_col = start_col + col_step

        while (current_row, current_col) != end:
            if any((current_row, current_col) in positions for positions in board.values()):
                return False
            current_row += row_step
            current_col += col_step

        return True

    def is_path_clear(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if the path between start and end is clear of pieces."""
        return self.is_path_clear_custom(start, end, self.board)

    def get_valid_moves(self, piece: str, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all valid moves for a piece at a position."""
        valid_moves = []
        color = piece.split('-')[0]

        # Handle castling for kings
        if piece.endswith('king'):
            king_row, king_col = position
            # Kingside castling
            if self.can_castle(color, 'kingside'):
                valid_moves.append((king_row, king_col + 2))
            # Queenside castling
            if self.can_castle(color, 'queenside'):
                valid_moves.append((king_row, king_col - 2))

        # Regular moves
        for row in range(8):
            for col in range(8):
                if self.is_valid_move(piece, position, (row, col)):
                    # Check if the move would leave the king in check
                    temp_board = {k: v[:] for k, v in self.board.items()}
                    piece_index = temp_board[piece].index(position)
                    temp_board[piece][piece_index] = (row, col)

                    # Handle capture in the temporary board
                    for p, positions in list(temp_board.items()):
                        if (row, col) in positions and p != piece:
                            positions.remove((row, col))

                    # Check if king is in check after the move
                    king_pos = position if piece.endswith('king') else temp_board[f'{color}-king'][0]
                    if piece.endswith('king'):
                        king_pos = (row, col)

                    is_safe = True
                    opponent_color = 'black' if color == 'white' else 'white'
                    for opp_piece, positions in temp_board.items():
                        if opp_piece.startswith(opponent_color):
                            for pos in positions:
                                if self.is_valid_move_without_check(opp_piece, pos, king_pos, temp_board):
                                    is_safe = False
                                    break
                            if not is_safe:
                                break

                    if is_safe:
                        valid_moves.append((row, col))

        return valid_moves

    def is_in_check(self, color: str) -> bool:
        """Check if the specified color's king is in check."""
        king_pos = self.board[f'{color}-king'][0]
        opponent_color = 'black' if color == 'white' else 'white'

        for piece, positions in self.board.items():
            if piece.startswith(opponent_color):
                for pos in positions:
                    if self.is_valid_move_without_check(piece, pos, king_pos, self.board):
                        return True
        return False

    def is_checkmate(self, color: str) -> bool:
        """Modified to never return True - game never ends."""
        return False



    def is_stalemate(self, color: str) -> bool:
        """Modified to never return True - game never ends."""
        return False

    def make_move(self, piece: str, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Make a move on the board."""
        if not self.is_valid_move(piece, start, end):
            return False

        # Handle capture
        for p, positions in self.board.items():
            if end in positions:
                positions.remove(end)
                break

        # Move piece
        piece_index = self.board[piece].index(start)
        self.board[piece][piece_index] = end

        # Handle pawn promotion
        if piece.endswith('pawn'):
            if (piece.startswith('white') and end[0] == 0) or (piece.startswith('black') and end[0] == 7):
                self.board[piece].remove(end)
                new_queen = f"{piece.split('-')[0]}-queen"
                if new_queen in self.board:
                    self.board[new_queen].append(end)
                else:
                    self.board[new_queen] = [end]

        return True

    def evaluate_position(self) -> int:
        """Evaluate the current position."""
        piece_values = {
            'pawn': 100,
            'knight': 320,
            'bishop': 330,
            'rook': 500,
            'queen': 900,
            'king': 20000
        }

        score = 0
        for piece, positions in self.board.items():
            value = piece_values[piece.split('-')[1]]
            multiplier = 1 if piece.startswith('white') else -1
            score += len(positions) * value * multiplier

        # Add bonus for check
        if self.is_in_check('black'):
            score += 50
        if self.is_in_check('white'):
            score -= 50

        # Add bonus for checkmate
        if self.is_checkmate('black'):
            score += 99999
        if self.is_checkmate('white'):
            score -= 99999

        return score

    def get_all_moves(self, color: str) -> List[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
        """Get all possible moves for a color."""
        moves = []
        for piece, positions in self.board.items():
            if piece.startswith(color):
                for start in positions:
                    valid_moves = self.get_valid_moves(piece, start)
                    for move in valid_moves:
                        moves.append((piece, start, move))
        return moves

    def minimax(self, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[int, Optional[Tuple]]:
        """Minimax algorithm with alpha-beta pruning."""
        if depth == 0:
            return self.evaluate_position(), None

        color = 'white' if maximizing_player else 'black'
        moves = self.get_all_moves(color)

        if not moves:
            if self.is_in_check(color):
                return (-99999 if maximizing_player else 99999), None
            return 0, None

        best_move = None
        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                # Save board state
                old_board = {k: v[:] for k, v in self.board.items()}
                old_king_moved = dict(self.king_moved)
                old_rooks_moved = {c: dict(r) for c, r in self.rooks_moved.items()}

                # Make move
                self.make_move(*move)

                eval_score, _ = self.minimax(depth - 1, alpha, beta, False)

                # Restore board state
                self.board = old_board
                self.king_moved = old_king_moved
                self.rooks_moved = old_rooks_moved

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break

            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in moves:
                # Save board state
                old_board = {k: v[:] for k, v in self.board.items()}
                old_king_moved = dict(self.king_moved)
                old_rooks_moved = {c: dict(r) for c, r in self.rooks_moved.items()}

                # Make move
                self.make_move(*move)

                eval_score, _ = self.minimax(depth - 1, alpha, beta, True)

                # Restore board state
                self.board = old_board
                self.king_moved = old_king_moved
                self.rooks_moved = old_rooks_moved

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break

            return min_eval, best_move

    def ai_make_move(self):
        """Modified AI move function to ensure first move is middle pawn."""
        if self.first_black_move:
            # Move the middle pawn (at position 1, 4) forward one square
            piece = 'black-pawn'
            start = (1, 4)
            end = (2, 4)
            self.make_move(piece, start, end)
            self.first_black_move = False
        else:
            # Normal AI move for subsequent turns
            _, best_move = self.minimax(3, float('-inf'), float('inf'), self.current_turn == 'white')
            if best_move:
                self.make_move(*best_move)

        self.current_turn = 'white'
        self.evaluation = self.evaluate_position()
        return True

    def draw(self):
        """Modified draw function to remove game over state."""
        # Draw background
        window.fill(BACKGROUND)

        # Draw board squares
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(window, color,
                                 (BOARD_X + col * SQUARE_SIZE,
                                  BOARD_Y + row * SQUARE_SIZE,
                                  SQUARE_SIZE, SQUARE_SIZE))

        # Draw valid moves
        for move in self.valid_moves:
            row, col = move
            is_capture = any(move in positions for p, positions in self.board.items()
                             if not p.startswith(self.current_turn))
            color = CAPTURE_HIGHLIGHT if is_capture else HIGHLIGHT
            pygame.draw.rect(window, color,
                             (BOARD_X + col * SQUARE_SIZE,
                              BOARD_Y + row * SQUARE_SIZE,
                              SQUARE_SIZE, SQUARE_SIZE), 4)

        # Draw pieces
        for piece, positions in self.board.items():
            for pos in positions:
                if piece in self.images and pos != self.selected_piece:
                    row, col = pos
                    window.blit(self.images[piece],
                                (BOARD_X + col * SQUARE_SIZE,
                                 BOARD_Y + row * SQUARE_SIZE))

        # Draw selected piece at mouse position if dragging
        if self.selected_piece:
            mouse_pos = pygame.mouse.get_pos()
            for piece, positions in self.board.items():
                if self.selected_piece in positions:
                    window.blit(self.images[piece],
                                (mouse_pos[0] - SQUARE_SIZE // 2,
                                 mouse_pos[1] - SQUARE_SIZE // 2))

        # Draw evaluation bar
        self.draw_evaluation_bar()

        # Draw turn indicator and evaluation text
        turn_text = self.font.render(f"Turn: {self.current_turn.capitalize()}", True, WHITE)
        eval_text = self.font.render(f"Eval: {self.evaluation / 100:.2f}", True, WHITE)
        window.blit(turn_text, (10, 10))
        window.blit(eval_text, (WINDOW_WIDTH - 150, 10))

        # Update display
        pygame.display.flip()

    def draw_evaluation_bar(self):
        """Draw the evaluation bar on the right side of the board."""
        bar_width = 20
        bar_height = 400
        bar_x = WINDOW_WIDTH - 40
        bar_y = (WINDOW_HEIGHT - bar_height) // 2

        # Normalize evaluation to [-1, 1] range
        max_eval = 2000
        normalized_eval = max(min(self.evaluation / max_eval, 1), -1)

        # Calculate heights for white and black portions
        white_height = int((0.5 + normalized_eval / 2) * bar_height)
        black_height = bar_height - white_height

        # Draw background
        pygame.draw.rect(window, GRAY, (bar_x, bar_y, bar_width, bar_height))

        # Draw white and black portions
        pygame.draw.rect(window, WHITE, (bar_x, bar_y + black_height, bar_width, white_height))
        pygame.draw.rect(window, BLACK, (bar_x, bar_y, bar_width, black_height))

        # Draw center line
        pygame.draw.line(window, GRAY,
                         (bar_x, bar_y + bar_height // 2),
                         (bar_x + bar_width, bar_y + bar_height // 2))

    def draw_game_over(self):
        """Draw the game over message (always a draw now)."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        window.blit(overlay, (0, 0))

        text = "Draw!"  # Always display "Draw"

        game_over_text = self.font.render(text, True, WHITE)
        text_rect = game_over_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))  # Shift up for buttons
        window.blit(game_over_text, text_rect)

        # Replay Button
        replay_button = pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 25, 200, 50)
        pygame.draw.rect(window, GRAY, replay_button)
        replay_text = self.font.render("Replay", True, WHITE)
        replay_rect = replay_text.get_rect(center=replay_button.center)
        window.blit(replay_text, replay_rect)

        # Undo Button
        undo_button = pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 100, 200, 50)  # Below Replay
        pygame.draw.rect(window, GRAY, undo_button)
        undo_text = self.font.render("Undo", True, WHITE)
        undo_rect = undo_text.get_rect(center=undo_button.center)
        window.blit(undo_text, undo_rect)

    def run(self):
        """Main game loop."""
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    board_pos = (
                        (mouse_pos[1] - BOARD_Y) // SQUARE_SIZE,
                        (mouse_pos[0] - BOARD_X) // SQUARE_SIZE
                    )

                    if 0 <= board_pos[0] < 8 and 0 <= board_pos[1] < 8:
                        for piece, positions in self.board.items():
                            if (board_pos in positions and
                                    piece.startswith(self.current_turn)):
                                self.selected_piece = board_pos
                                self.valid_moves = self.get_valid_moves(piece, board_pos)
                                break

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self.selected_piece:
                        mouse_pos = pygame.mouse.get_pos()
                        board_pos = (
                            (mouse_pos[1] - BOARD_Y) // SQUARE_SIZE,
                            (mouse_pos[0] - BOARD_X) // SQUARE_SIZE
                        )

                        if 0 <= board_pos[0] < 8 and 0 <= board_pos[1] < 8:
                            for piece, positions in self.board.items():
                                if self.selected_piece in positions:
                                    if board_pos in self.valid_moves:
                                        self.make_move(piece, self.selected_piece, board_pos)
                                        self.current_turn = 'black'
                                        self.evaluation = self.evaluate_position()

                                        # AI move
                                        if self.current_turn == 'black':
                                            self.ai_thinking = True
                                            self.draw()
                                            pygame.display.flip()
                                            self.ai_make_move()
                                            self.ai_thinking = False
                                    break

                        self.selected_piece = None
                        self.valid_moves = []

            # Draw the current state
            self.draw()

            # Cap the frame rate
            clock.tick(60)

        pygame.quit()


# Create and run the game
if __name__ == "__main__":
    game = ChessGame()
    game.run()