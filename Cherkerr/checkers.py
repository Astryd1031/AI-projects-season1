import argparse
import copy


cache = {}
MAX_SEARCH_DEPTH = 10


class State:
    # This class is used to represent a state.
    # board : a list of lists that represents the 8*8 board

    def __init__(self, board):

        self.board = board
        self.state_history = set()
        self.width = 8
        self.height = 8

    def display(self):
        for i in self.board:
            for j in i:
                print(j, end="")
            print("")
        print("")

    def get_successors(self, player):
        """Return all possible successor states from the current state for a given player, enforcing mandatory jumps."""
        jump_successors = set()  # To store states with jumps
        move_successors = set()  # To store states with simple moves (only if no jumps are available)

        checkers = self.checks(player)  # Get all pieces for the current player

        for checker_piece in checkers:
            if checker_piece.symbol in ['R', 'B']:  # Handle king pieces (R and B)
                king_jump_moves = checker_piece.get_king_jumps(self)
                if king_jump_moves:
                    # Add king jumps to jump_successors (since jumps are mandatory)
                    for jump_sequence, captured_pieces, initial_pos in king_jump_moves:
                        new_state = copy.deepcopy(self)
                        new_state.apply_king_jump(jump_sequence, initial_pos)

                        # Remove captured pieces
                        for pos in captured_pieces:
                            new_state.board[pos[0]][pos[1]] = '.'

                        # Add new state with king jump to successors
                        jump_successors.add(new_state)

                # If no jumps, get simple king moves
                if not king_jump_moves:
                    king_simple_moves = checker_piece.get_king_simple_moves(self)
                    for move in king_simple_moves:
                        start_pos, end_pos = move
                        new_state = copy.deepcopy(self)
                        new_state.apply_king_move(start_pos, end_pos)
                        move_successors.add(new_state)

            else:  # Handle regular pieces (r and b)
                jump_moves = checker_piece.get_jumps(self)
                if jump_moves:
                    # Add regular jumps to jump_successors (since jumps are mandatory)
                    for jump_sequence, captured_pieces, initial_pos in jump_moves:
                        new_state = copy.deepcopy(self)
                        new_state.apply_jump(jump_sequence, initial_pos)

                        # Remove captured pieces
                        for pos in captured_pieces:
                            new_state.board[pos[0]][pos[1]] = '.'

                        # Add new state with regular jump to successors
                        jump_successors.add(new_state)

                # If no jumps, get simple moves
                if not jump_moves:
                    simple_moves = checker_piece.get_simple_moves(self)
                    for move in simple_moves:
                        start_pos, end_pos = move
                        new_state = copy.deepcopy(self)
                        new_state.apply_move(start_pos, end_pos)
                        move_successors.add(new_state)
        # Enforce mandatory jumps: If any jumps are available, only return jump_successors
        if jump_successors:
            return set(jump_successors)
        else:
            return set(move_successors)

    def checks(self, player):
        """Return all checkers of the player on the  board
        Helper for get_succcessor method"""
        checks = []
        if player == 'r':
            for i in range(8):
                for j in range(8):

                    if self.board[i][j] == 'r':
                        checks.append(checker(i, j, 'r'))
                    elif self.board[i][j] == 'R':
                        checks.append(checker(i, j, 'R'))
        elif player == 'b':
            for i in range(8):
                for j in range(8):
                    if self.board[i][j] == 'b':
                        checks.append(checker(i, j, 'b'))
                    elif self.board[i][j] == 'B':
                        checks.append(checker(i, j, 'B'))

        return checks

    def apply_move(self, start_pos, end_pos):
        """Apply a simple move to the board (without jumping) for a regular piece."""
        x_start, y_start = start_pos
        x_end, y_end = end_pos
        self.board[x_end][y_end] = self.board[x_start][y_start]
        self.board[x_start][y_start] = '.'

        if self.promote_if_king((x_end, y_end)):
            return  # Terminate the move if the piece is promoted to a king

    def apply_jump(self, jump_sequence, initial):
        """Apply a sequence of jumps of regular piece to the board."""
        # Start from the initial position of the piece
        start_pos = initial

        for jump_pos in jump_sequence:
            # Move the piece to the end position
            end_pos = jump_pos
            self.apply_move(start_pos, end_pos)

            if self.promote_if_king(end_pos):
                break  # Stop if promotion happens

            # Update the starting position for the next jump
            start_pos = end_pos

    def promote_if_king(self, position):
        """ Promote a regular piece to a king """
        x, y = position
        if self.board[x][y] == 'r' and x == 0:  # Red piece reaches the top
            self.board[x][y] = 'R'  # Promote to Red King
            return True  # Promotion occurred
        elif self.board[x][y] == 'b' and x == 7:  # Black piece reaches the bottom
            self.board[x][y] = 'B'  # Promote to Black King
            return True  # Promotion occurred
        return False  # No promotion

    def apply_king_move(self, start_pos, end_pos):
        """Apply king's non-jump movement to the board"""
        x_start, y_start = start_pos
        x_end, y_end = end_pos
        self.board[x_end][y_end] = self.board[x_start][y_start]
        self.board[x_start][y_start] = '.'

    def apply_king_jump(self, jump_sequence, initial):
        """Apply a sequence of jumps of king to the board."""
        # Start from the initial position of the piece
        start_pos = initial

        for jump_pos in jump_sequence:
            # Move the piece to the end position
            end_pos = jump_pos
            self.apply_move(start_pos, end_pos)
            # Update the starting position for the next jump
            start_pos = end_pos

    def utility(self):
        """Evaluate if it's a terminal state (win/loss/tie)."""
        red_pieces = sum(row.count('r') + row.count('R') for row in self.board)
        black_pieces = sum(row.count('b') + row.count('B') for row in self.board)

        # Check for win/loss conditions
        if red_pieces == 0:
            return -1  # Red loses
        elif black_pieces == 0:
            return 1  # Black loses

        # Check for stalemate conditions
        red_has_moves = any(checker.get_moves(self) for checker in self.checks('r'))
        black_has_moves = any(checker.get_moves(self) for checker in self.checks('b'))

        if not red_has_moves and not black_has_moves:
            return 0

        if not red_has_moves and red_pieces > 0:
            return -1  # Red has no legal moves, black wins2
        elif not black_has_moves and black_pieces > 0:
            return 1  # Black has no legal moves, red wins

        return 0  # Continue playing

    def evaluation(self, player):
        """score of each state"""
        # Weights for different types of pieces and factors
        king_weight = 3  # Kings are more valuable
        piece_weight = 1  # Normal pieces are less valuable
        adjacency_weight = 5  # Weight for being adjacent to opponent's pieces
        blocking_weight = 1000  # Weight for blocking opponent's moves  # Extra reward for preventing opponent from becoming a king

        # Scores for red and black
        red_score = 0
        black_score = 0
        # Check guaranteed jumps
        guaranteed_jumps = self.has_guaranteed_jumps(player)
        if guaranteed_jumps != 0:
            if player == 'r':
                red_score += guaranteed_jumps * 10
            else:
                black_score += guaranteed_jumps * 10

        # Iterate through the board to evaluate the position
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                piece = self.board[row][col]

                # Determine movement directions based on piece type
                directions = []
                if piece in ['R', 'B']:
                    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Kings can move in all diagonal directions
                elif piece == 'r':
                    directions = [(-1, -1), (-1, 1)]  # Regular red pieces move up
                elif piece == 'b':
                    directions = [(1, -1), (1, 1)]  # Regular black pieces move down

                    # Check for adjacent opponent pieces and award higher scores for adjacency
                    for dx, dy in directions:
                        new_row, new_col = row + dx, col + dy
                        jump_row, jump_col = row + 2 * dx, col + 2 * dy
                        if checker(row, col, piece).is_in_bounds(new_row, new_col):
                            adjacent_piece = self.board[new_row][new_col]
                            # Reward for adjacency to opponent's pieces
                            if checker(new_row, new_col, piece).is_in_bounds(jump_row, jump_col):
                                if adjacent_piece in ['b', 'B'] and self.board[jump_row][jump_col] == '.':
                                    if piece in ['r', 'R']:
                                        red_score += adjacency_weight  # Higher priority for adjacency for red
                                    elif piece in ['b', 'B']:
                                        black_score += adjacency_weight  # Higher priority for adjacency for black

                            # Check if the move blocks an opponent's movement
                            if piece in ['r', 'R'] and adjacent_piece in ['b', 'B']:
                                red_score += blocking_weight  # Reward for blocking black
                            elif piece in ['b', 'B'] and adjacent_piece in ['r', 'R']:
                                black_score += blocking_weight  # Reward for blocking red


                if piece == 'r':  # Regular red piece
                    red_score += piece_weight
                    red_score += (len(self.board) - row - 1)  # Encourage advancing upwards for regular pieces
                elif piece == 'R':  # Red king
                    red_score += king_weight  # No positional weight for kings
                elif piece == 'b':  # Regular black piece
                    black_score += piece_weight
                    black_score += row  # Encourage advancing downwards for regular pieces
                elif piece == 'B':  # Black king
                    black_score += king_weight  # No positional weight for kings


        # Return the evaluation score from the perspective of the current player
        return red_score - black_score if player == 'r' else black_score - red_score

    def has_guaranteed_jumps(self, player):
        """Determine if the current player has guaranteed jumps available.
        Helper for evaluation method """
        # Check each piece for possible jumps
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                piece = self.board[row][col]
                if (player == 'r' and piece in ('r', 'R')) or (player == 'b' and piece in ('b', 'B')):
                    checker_piece = checker(row, col, piece)  # Create a checker instance
                    if piece in 'RB':
                        jump_moves = checker_piece.get_king_jumps(self)
                        if jump_moves is None:
                            return 0
                        else:
                            jump_moves = len(checker_piece.get_king_jumps(self))
                    else:
                        jump_moves = len((checker_piece.get_jumps(self)))

                    if jump_moves > 0:
                        return jump_moves
        return 0

    def cache_value(self, state, depth, alpha, beta, utility):
        """Cache value generator for optimizing alpha_beta method"""
        serialized_state = self.serialize_state(state)
        cache[serialized_state] = (depth, alpha, beta, utility)

    def serialize_state(self, state):
        """Helper function for cache_value method"""
        # Convert the state representation to a hashable type
        # If needed, you can add more attributes to uniquely represent the state
        return str(state.board) # Return just the board for now

    def get_cached_value(self, state, depth, alpha, beta):
        """Return cached utility"""
        serialized_state = self.serialize_state(state)
        if serialized_state in cache:
            dc, ac, bc, utility = cache[serialized_state]
            if dc <= depth and ac < alpha and bc > beta:
                return utility
        return None

class checker:
    def __init__(self, coord_x, coord_y, symbol):
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.symbol = symbol

    def get_moves(self, state):
        """Return all possible moves, including mandatory jumps for regular pieces."""
        if self.symbol in ['R', 'B']:  # King piece
            return self.get_king_moves(state)

        moves = []
        jump_moves = self.get_jumps(state)

        if jump_moves:
            return jump_moves  # Mandatory jumps take precedence

        return self.get_simple_moves(state)

    def get_simple_moves(self, state):
        """Return simple diagonal moves (without jumps) for regular moves."""
        moves = []
        x, y = self.coord_x, self.coord_y
        board = state.board

        # Explicitly define the directions for each type of piece
        if self.symbol == 'r':  # Red piece, can only move up (diagonal top-left or top-right)
            directions = [(-1, -1), (-1, 1)]
        elif self.symbol == 'b':  # Black piece, can only move down (diagonal bottom-left or bottom-right)
            directions = [(1, -1), (1, 1)]
        else:  # King piece (R or B), handled separately
            directions = []

        # Check for simple (non-jumping) moves
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if self.is_in_bounds(new_x, new_y) and board[new_x][new_y] == '.':
                moves.append(((x, y), (new_x, new_y)))  # Normal move

        return moves

    def get_jumps(self, state, last_pos=None, captured_pieces=None):
        """Return all possible jump moves for regular piece."""

        if last_pos is None:
            last_pos = (self.coord_x, self.coord_y)
        if captured_pieces is None:
            captured_pieces = []

        jumps = []  # List to store jump paths and captured pieces
        x, y = last_pos
        board = state.board

        # Explicitly define the directions for jumps
        if self.symbol == 'r':  # Red piece, can only jump up
            directions = [(-2, -2), (-2, 2)]
        elif self.symbol == 'b':  # Black piece, can only jump down
            directions = [(2, -2), (2, 2)]
        else:  # King piece (R or B), can jump in all diagonal directions
            directions = [(-2, -2), (-2, 2), (2, -2), (2, 2)]

        # Try all jump directions
        for dx, dy in directions:
            jump_x, jump_y = x + dx, y + dy  # Target position after jumping
            mid_x, mid_y = x + dx // 2, y + dy // 2  # Piece being jumped over

            if self.is_in_bounds(jump_x, jump_y) and board[jump_x][jump_y] == '.':
                middle_piece = board[mid_x][mid_y]
                if self.is_opponent_piece(middle_piece) and (mid_x, mid_y) not in captured_pieces:
                    new_captured_pieces = captured_pieces + [(mid_x, mid_y)]
                    jumps.append(([(jump_x, jump_y)], new_captured_pieces, last_pos))

                    # Recursive search for further jumps, starting from this new position
                    subsequent_jumps = self.get_jumps(state, (jump_x, jump_y), new_captured_pieces)
                    for sj, sc, _ in subsequent_jumps:
                        jumps.append(([(jump_x, jump_y)] + sj, sc, last_pos))

        return jumps

    def get_king_moves(self, state):
        """Return all possible moves for a king, including simple moves and jumps."""
        moves = []
        jump_moves = self.get_king_jumps(state)

        # If there are jump moves, they take priority
        if jump_moves:
            return jump_moves

        # Otherwise, check for simple diagonal moves
        moves += self.get_king_simple_moves(state)
        return moves

    def get_king_simple_moves(self, state):
        """Return all possible diagonal moves for a king at (row, col)."""
        x, y = self.coord_x,self.coord_y
        moves = []

        # Check diagonal moves
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Up-left, Up-right, Down-left, Down-right

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                if state.board[new_x][new_y] == '.':  # Check if the square is empty
                    moves.append(((x,y),(new_x, new_y)))
        return moves

    def get_king_jumps(self, state, last_pos=None, captured_pieces=None):
        """Return all possible jump moves, including for kings."""

        if last_pos is None:
            last_pos = (self.coord_x, self.coord_y)
        if captured_pieces is None:
            captured_pieces = []

        jumps = []  # List to store jump paths and captured pieces
        x, y = last_pos
        board = state.board

        directions = [(-2, -2), (-2, 2), (2, -2), (2, 2)]

        # Try all jump directions
        for dx, dy in directions:
            jump_x, jump_y = x + dx, y + dy  # Target position after jumping
            mid_x, mid_y = x + dx // 2, y + dy // 2  # Piece being jumped over

            # Check if jump target is within bounds and the landing square is empty
            if self.is_in_bounds(jump_x, jump_y) and board[jump_x][jump_y] == '.':
                middle_piece = board[mid_x][mid_y]

                # Check if the middle piece is an opponent's piece and not already jumped
                if self.is_opponent_piece(middle_piece) and (mid_x, mid_y) not in captured_pieces:
                    new_captured_pieces = captured_pieces + [(mid_x, mid_y)]

                    # Add this jump to the list of jumps
                    jumps.append(([(jump_x, jump_y)], new_captured_pieces, last_pos))

                    # Recursive search for further jumps from the new position
                    subsequent_jumps = self.get_king_jumps(state, (jump_x, jump_y), new_captured_pieces)
                    for sj, sc, _ in subsequent_jumps:
                        jumps.append(([(jump_x, jump_y)] + sj, sc, last_pos))

        return jumps

    def is_in_bounds(self, x, y):
        """Check if the coordinates are within the board's boundaries."""
        return 0 <= x < 8 and 0 <= y < 8

    def is_opponent_piece(self, piece):
        """Check if the piece belongs to the opponent."""
        if self.symbol in ['r', 'R']:
            return piece in ['b', 'B']  # Red pieces' opponents are black pieces
        else:
            return piece in ['r', 'R']  # Black pieces' opponents are red pieces

def alpha_beta(state, player, depth, alpha=float('-inf'), beta=float('inf')):
    """Perform the alpha-beta pruning algorithm."""
    # Check for cached value
    cached_utility = state.get_cached_value(state, depth, alpha, beta)
    if cached_utility is not None:
        return cached_utility , state


    # Terminal state or max depth reached
    if depth == 0 or state.utility() != 0:
        return state.evaluation(player), None  # Return evaluation score and a placeholder for move

    best_move = None
    if player == 'r':  # Maximizing for red
        max_eval = float('-inf')
        successors = state.get_successors(player)
        if not successors:  # No successors means no valid moves
            return state.utility(), None

        for successor in successors:
            evaluation, _ = alpha_beta(successor, 'b', depth - 1, alpha, beta)
            if evaluation > max_eval:
                max_eval = evaluation
                best_move = successor  # Keep track of the best move
            alpha = max(alpha, evaluation)
            if alpha >= beta:
                break  # Beta cutoff

        state.cache_value(state, depth, alpha, beta, max_eval)
        return max_eval, best_move

    else:  # Minimizing for black
        min_eval = float('inf')
        successors = state.get_successors(player)
        if not successors:  # No successors means no valid moves
            return state.utility(), None

        for successor in successors:
            evaluation, _ = alpha_beta(successor, 'r', depth - 1, alpha, beta)
            if evaluation < min_eval:
                min_eval = evaluation
                best_move = successor  # Keep track of the best move
            beta = min(beta, evaluation)

            if beta <= alpha:
                break  # Alpha cutoff

        state.cache_value(state, depth, alpha, beta, min_eval)
        return min_eval, best_move

def get_next_turn(curr_turn):
    """Return opponent"""
    if curr_turn == 'r':
        return 'b'
    else:
        return 'r'

def play_game(state, output_file):
    turn = 'r'  # Start with red

    with open(output_file, 'w') as f:
        # Write the initial state to the output file
        for row in state.board:
            f.write(''.join(row) + '\n')
        f.write('\n')  # Add an empty line after the initial state

        move_count = 0  # Track the number of moves made

        while True:

            # Check for terminal state
            if state.utility() != 0:
                break  # Exit if the game is over

            # Get the best move for the current player
            _, move = alpha_beta(state, turn, MAX_SEARCH_DEPTH, alpha=float('-inf'),
                                 beta=float('inf'))  # Get the best move (state)
            if move is None:
                break  # If no valid moves, exit the loop

            # Apply the optimal move
            state = move

            # Write the current board state after the move
            for row in state.board:
                f.write(''.join(row) + '\n')  # Write to file
            f.write('\n')  # Add an empty line between moves

            turn = get_next_turn(turn)  # Switch players
            move_count += 1  # Increment the move count

def read_from_file(filename):
    f = open(filename)
    lines = f.readlines()
    board = [[str(x) for x in l.rstrip()] for l in lines]
    f.close()
    return board

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    args = parser.parse_args()

    initial_board = read_from_file(args.inputfile)  # Read initial board state from file
    state = State(initial_board)  # Create the initial state

    play_game(state, args.outputfile)
