#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)


import copy
from typing import List, Dict, Tuple, Optional
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):

    # Constants for transposition table flags
    EXACT = 0       # Exact minimax value
    LOWER_BOUND = 1 # Value is at least this (beta cutoff occurred)
    UPPER_BOUND = 2 # Value is at most this (alpha cutoff occurred)

    def __init__(self):
        super().__init__()
        self.transposition_table: Dict[int, Tuple[float, int, int, Optional[Move]]] = {}
        # Format: hash -> (value, depth, flag, best_move)


    def compute_best_move(self, game_state: GameState) -> None:

        # Clear transposition table for new search
        self.transposition_table = {}
        
        # Get all legal moves
        legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            return
        
        # SAFETY: Propose the first legal move immediately
        # This ensures we always have a move even if terminated instantly
        self.propose_move(legal_moves[0])
        
        # Order moves by immediate score (greedy) for initial proposal
        ordered_moves = self.order_moves(game_state, legal_moves)
        
        # Propose best greedy move as improvement
        if ordered_moves:
            self.propose_move(ordered_moves[0])
        
        # Iterative deepening - search deeper and deeper until killed
        max_depth = 100  # Practical upper bound (will be time-limited anyway)
        our_player = game_state.current_player
        
        for depth in range(1, max_depth + 1):
            try:
                best_move = None
                best_value = float('-inf')
                alpha = float('-inf')
                beta = float('inf')
                
                # Re-order moves using best move from previous iteration (from TT)
                tt_best = self.get_tt_best_move(game_state)
                ordered_moves = self.order_moves(game_state, legal_moves, tt_best)
                
                # Search each move
                for move in ordered_moves:
                    new_state = self.apply_move(game_state, move)
                    # After our move, it's opponent's turn (minimizing)
                    value = self.minimax(new_state, depth - 1, alpha, beta, False, our_player)
                    
                    if value > best_value:
                        best_value = value
                        best_move = move
                    
                    alpha = max(alpha, value)
                
                # Propose the best move found at this depth
                if best_move:
                    self.propose_move(best_move)
                    # Store root position in transposition table
                    self.store_tt(game_state, best_value, depth, self.EXACT, best_move)
                    
            except Exception:
                # If anything goes wrong (including timeout), we already have a proposed move
                break


    def get_legal_moves(self, game_state: GameState) -> List[Move]:

        board = game_state.board
        N = board.N
        m, n = board.m, board.n
        
        # Get allowed squares for current player
        player_squares = game_state.player_squares()
        
        # Determine which empty squares we can play on
        if player_squares is None:
            # Classic mode: all empty squares allowed
            empty_squares = []
            for i in range(N):
                for j in range(N):
                    if board.get((i, j)) == SudokuBoard.empty:
                        empty_squares.append((i, j))
        else:
            # Non-classic: only allowed squares that are empty
            empty_squares = [sq for sq in player_squares if board.get(sq) == SudokuBoard.empty]
        
        # Build taboo set for O(1) lookup
        taboo_set = set()
        for taboo in game_state.taboo_moves:
            taboo_set.add((taboo.square, taboo.value))
        
        legal_moves = []
        
        for square in empty_squares:
            row, col = square
            
            # Collect values already present in row, column, and block (C0 constraint)
            used_values = set()
            
            # Row values
            for j in range(N):
                val = board.get((row, j))
                if val != SudokuBoard.empty:
                    used_values.add(val)
            
            # Column values
            for i in range(N):
                val = board.get((i, col))
                if val != SudokuBoard.empty:
                    used_values.add(val)
            
            # Block values
            block_row_start = (row // m) * m
            block_col_start = (col // n) * n
            for i in range(block_row_start, block_row_start + m):
                for j in range(block_col_start, block_col_start + n):
                    val = board.get((i, j))
                    if val != SudokuBoard.empty:
                        used_values.add(val)
            
            # Try each value that doesn't violate C0 and isn't taboo
            for value in range(1, N + 1):
                if value not in used_values and (square, value) not in taboo_set:
                    legal_moves.append(Move(square, value))
        
        return legal_moves



    def calculate_move_score(self, board: SudokuBoard, move: Move) -> int:

        N = board.N
        m, n = board.m, board.n
        row, col = move.square
        
        regions_completed = 0
        
        # Check if row will be completed (only this cell empty in row)
        row_empty = sum(1 for j in range(N) if board.get((row, j)) == SudokuBoard.empty)
        if row_empty == 1:
            regions_completed += 1
        
        # Check if column will be completed
        col_empty = sum(1 for i in range(N) if board.get((i, col)) == SudokuBoard.empty)
        if col_empty == 1:
            regions_completed += 1
        
        # Check if block will be completed
        block_row_start = (row // m) * m
        block_col_start = (col // n) * n
        block_empty = 0
        for i in range(block_row_start, block_row_start + m):
            for j in range(block_col_start, block_col_start + n):
                if board.get((i, j)) == SudokuBoard.empty:
                    block_empty += 1
        if block_empty == 1:
            regions_completed += 1
        
        # Scoring table
        score_table = {0: 0, 1: 1, 2: 3, 3: 7}
        return score_table[regions_completed]

    def count_region_empty_cells(self, board: SudokuBoard, row: int, col: int) -> Tuple[int, int, int]:

        N = board.N
        m, n = board.m, board.n
        
        row_empty = sum(1 for j in range(N) if board.get((row, j)) == SudokuBoard.empty)
        col_empty = sum(1 for i in range(N) if board.get((i, col)) == SudokuBoard.empty)
        
        block_row_start = (row // m) * m
        block_col_start = (col // n) * n
        block_empty = 0
        for i in range(block_row_start, block_row_start + m):
            for j in range(block_col_start, block_col_start + n):
                if board.get((i, j)) == SudokuBoard.empty:
                    block_empty += 1
        
        return row_empty, col_empty, block_empty


    def evaluate(self, game_state: GameState, our_player: int) -> float:

        opponent = 3 - our_player
        
        # Component 1: Actual score difference (most important)
        our_score = game_state.scores[our_player - 1]
        opp_score = game_state.scores[opponent - 1]
        score_diff = (our_score - opp_score) * 10.0
        
        # Component 2: Potential from almost-complete regions
        board = game_state.board
        N = board.N
        m, n = board.m, board.n
        
        potential = 0.0
        
        # Rows
        for i in range(N):
            empty = sum(1 for j in range(N) if board.get((i, j)) == SudokuBoard.empty)
            if empty == 1:
                potential += 0.8  # About to complete
            elif empty == 2:
                potential += 0.3
            elif empty == 3:
                potential += 0.1
        
        # Columns
        for j in range(N):
            empty = sum(1 for i in range(N) if board.get((i, j)) == SudokuBoard.empty)
            if empty == 1:
                potential += 0.8
            elif empty == 2:
                potential += 0.3
            elif empty == 3:
                potential += 0.1
        
        # Blocks
        for block_row in range(n):  # Number of block rows
            for block_col in range(m):  # Number of block columns
                empty = 0
                for i in range(block_row * m, (block_row + 1) * m):
                    for j in range(block_col * n, (block_col + 1) * n):
                        if board.get((i, j)) == SudokuBoard.empty:
                            empty += 1
                if empty == 1:
                    potential += 0.8
                elif empty == 2:
                    potential += 0.3
                elif empty == 3:
                    potential += 0.1
        
        # Component 3: Mobility (number of legal moves)
        # More moves = more flexibility (slight bonus)
        # Note: This is computed for current player, which alternates
        # We give a small bonus for having more options
        mobility_bonus = 0.0
        if game_state.current_player == our_player:
            legal_moves = self.get_legal_moves(game_state)
            mobility_bonus = len(legal_moves) * 0.01
        
        return score_diff + potential + mobility_bonus


    def apply_move(self, game_state: GameState, move: Move) -> GameState:

        new_state = copy.deepcopy(game_state)
        player = new_state.current_player
        
        # Calculate and add score for this move
        move_score = self.calculate_move_score(new_state.board, move)
        new_state.scores[player - 1] += move_score
        
        # Place the value on the board
        new_state.board.put(move.square, move.value)
        new_state.moves.append(move)
        
        # Update occupied squares (for non-classic modes)
        if not new_state.is_classic_game():
            if player == 1:
                new_state.occupied_squares1.append(move.square)
            else:
                new_state.occupied_squares2.append(move.square)
        
        # Switch to other player
        new_state.current_player = 3 - player
        
        return new_state


    def order_moves(self, game_state: GameState, moves: List[Move],
                    tt_best_move: Optional[Move] = None) -> List[Move]:

        board = game_state.board
        
        def move_priority(move: Move) -> float:
            priority = 0.0
            
            # Highest priority: TT best move
            if tt_best_move and move.square == tt_best_move.square and move.value == tt_best_move.value:
                priority += 10000
            
            # High priority: moves that score points
            score = self.calculate_move_score(board, move)
            priority += score * 100
            
            # Medium priority: moves in almost-complete regions
            row, col = move.square
            row_empty, col_empty, block_empty = self.count_region_empty_cells(board, row, col)
            
            # Prefer cells that are in regions with fewer empty cells
            if row_empty <= 3:
                priority += (4 - row_empty) * 5
            if col_empty <= 3:
                priority += (4 - col_empty) * 5
            if block_empty <= 3:
                priority += (4 - block_empty) * 5
            
            return priority
        
        return sorted(moves, key=move_priority, reverse=True)


    def compute_hash(self, game_state: GameState) -> int:
        """
        Computes a hash for the game state for transposition table lookup.
        
        @param game_state: The game state to hash.
        @return: Hash value.
        """
        # Hash combines: board state, current player, and scores
        # This ensures we dont confuse different game states
        board_tuple = tuple(game_state.board.squares)
        return hash((board_tuple, game_state.current_player,
                     game_state.scores[0], game_state.scores[1]))

    def store_tt(self, game_state: GameState, value: float, depth: int,
                 flag: int, best_move: Optional[Move]) -> None:

        h = self.compute_hash(game_state)
        
        # Only overwrite if new entry has greater or equal depth
        if h in self.transposition_table:
            _, stored_depth, _, _ = self.transposition_table[h]
            if stored_depth > depth:
                return  # Keep the deeper search result
        
        self.transposition_table[h] = (value, depth, flag, best_move)

    def lookup_tt(self, game_state: GameState, depth: int, alpha: float, beta: float) \
            -> Tuple[bool, float, Optional[Move]]:

        h = self.compute_hash(game_state)
        
        if h not in self.transposition_table:
            return False, 0.0, None
        
        stored_value, stored_depth, flag, best_move = self.transposition_table[h]
        
        # Only use stored value if it was searched to at least our required depth
        if stored_depth >= depth:
            if flag == self.EXACT:
                return True, stored_value, best_move
            elif flag == self.LOWER_BOUND and stored_value >= beta:
                return True, stored_value, best_move
            elif flag == self.UPPER_BOUND and stored_value <= alpha:
                return True, stored_value, best_move
        
        # Can't use value, but return best_move for move ordering
        return False, 0.0, best_move

    def get_tt_best_move(self, game_state: GameState) -> Optional[Move]:

        h = self.compute_hash(game_state)
        if h in self.transposition_table:
            return self.transposition_table[h][3]
        return None

    def minimax(self, game_state: GameState, depth: int, alpha: float, beta: float,
                is_maximizing: bool, our_player: int) -> float:

        # Save original alpha for TT flag determination
        original_alpha = alpha
        
        # Transposition table lookup
        found, tt_value, tt_best_move = self.lookup_tt(game_state, depth, alpha, beta)
        if found:
            return tt_value
        
        # Terminal: depth limit reached
        if depth == 0:
            return self.evaluate(game_state, our_player)
        
        # Get legal moves
        legal_moves = self.get_legal_moves(game_state)
        
        # Terminal: no legal moves (game effectively over for this player)
        if not legal_moves:
            return self.evaluate(game_state, our_player)
        
        # Order moves for better pruning
        ordered_moves = self.order_moves(game_state, legal_moves, tt_best_move)
        
        best_move = ordered_moves[0]  # Default to first move
        
        if is_maximizing:
            max_eval = float('-inf')
            
            for move in ordered_moves:
                new_state = self.apply_move(game_state, move)
                eval_score = self.minimax(new_state, depth - 1, alpha, beta, False, our_player)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff - opponent won't allow this path
            
            # Determine TT flag
            if max_eval <= original_alpha:
                flag = self.UPPER_BOUND  # Failed low
            elif max_eval >= beta:
                flag = self.LOWER_BOUND  # Failed high (cutoff)
            else:
                flag = self.EXACT
            
            self.store_tt(game_state, max_eval, depth, flag, best_move)
            return max_eval
            
        else:  # Minimizing
            min_eval = float('inf')
            
            for move in ordered_moves:
                new_state = self.apply_move(game_state, move)
                eval_score = self.minimax(new_state, depth - 1, alpha, beta, True, our_player)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff - we won't choose this path
            
            # Determine TT flag
            if min_eval <= original_alpha:
                flag = self.UPPER_BOUND  # Failed low (cutoff)
            elif min_eval >= beta:
                flag = self.LOWER_BOUND  # Failed high
            else:
                flag = self.EXACT
            
            self.store_tt(game_state, min_eval, depth, flag, best_move)
            return min_eval