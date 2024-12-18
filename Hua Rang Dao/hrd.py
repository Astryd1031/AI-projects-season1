import argparse
import sys
import heapq
from copy import deepcopy
from typing import Optional

#====================================================================================

char_single = '2'
moves = ["right", "left", "up", "down"]

class Piece:
    """Represents a piece on the Hua Rong Dao puzzle."""

    def __init__(self, is_2_by_2, is_single, coord_x, coord_y, symbol: Optional[str]=None , orientation=None):
        self.is_2_by_2 = is_2_by_2
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.symbol = symbol
        self.orientation = orientation

    def set_coords(self, coord_x, coord_y):
        """Move the piece to the new coordinates."""
        self.coord_x = coord_x
        self.coord_y = coord_y

    def get_coordinates(self):
        """Return the occupied coordinates of the piece on the board."""
        if self.is_2_by_2:
            return [
                (self.coord_x, self.coord_y),
                (self.coord_x, self.coord_y + 1),
                (self.coord_x + 1, self.coord_y),
                (self.coord_x + 1, self.coord_y + 1)
            ]
        elif self.is_single:
            return [(self.coord_x, self.coord_y)]
        elif self.orientation == 'h':  # Horizontal 1x2 piece
            return [(self.coord_x, self.coord_y), (self.coord_x + 1, self.coord_y)]
        elif self.orientation == 'v':  # Vertical 1x2 piece
            return [(self.coord_x, self.coord_y), (self.coord_x, self.coord_y + 1)]
        return []

    def __repr__(self):
        return f"Piece(is_2_by_2={self.is_2_by_2}, is_single={self.is_single}, coord=({self.coord_x}, {self.coord_y}), orientation={self.orientation})"


class Board:
    """Board class for setting up the playing board."""

    def __init__(self, height, pieces):
        self.width = 4
        self.height = height
        self.pieces = pieces
        self.grid = []
        self.__construct_grid()

    def __eq__(self, other):
        if isinstance(other, Board):
            return self.grid == other.grid
        return False

    def __construct_grid(self):
        """Set up a 2D grid based on the piece location information."""
        for i in range(self.height):
            line = ['.' for _ in range(self.width)]
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_2_by_2:
                self.grid[piece.coord_y][piece.coord_x] = '1'
                self.grid[piece.coord_y][piece.coord_x + 1] = '1'
                self.grid[piece.coord_y + 1][piece.coord_x] = '1'
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = '1'
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

    def display(self):
        """Print out the current board."""
        for line in self.grid:
            print(''.join(line))


    def isempty(self):
        """
        Find the two empty spaces on the board, return their positions
        [(x1, y1), (x2, y2)]
        """
        res = []
        for i, line in enumerate(self.grid):
            for x, ch in enumerate(line):
                if ch == ".":
                    res.append((i, x))
        return res

    def get_neighbors(self, blanks):
        """
        Get all pieces adjacent to the given blank space(s) that can move into it.

        :param blanks: A list of coordinates representing blank spaces [(y1, x1), (y2, x2), ...].
        :return: A list of pieces that can fully move into the blank space(s).
        """
        neighbors = set()  # Use a set to avoid duplicate pieces

        # Possible directions for adjacent pieces to move into blank(s)
        directions = [
            (0, 1),  # Right
            (1, 0),  # Down
            (0, -1),  # Left
            (-1, 0)  # Up
        ]
        for blank in blanks:
            blank_y, blank_x = blank
            for dy, dx in directions:
                neighbor_y = blank_y + dy
                neighbor_x = blank_x + dx

                # Check if the neighbor coordinates are within bounds
                if 0 <= neighbor_y < self.height and 0 <= neighbor_x < self.width:
                    piece = self.grid[neighbor_y][neighbor_x]
                    if piece != '.' and piece is not None:  # If there's a piece in the neighboring position
                        # Find the corresponding Piece object
                        target = self.get_piece_at(neighbor_x,neighbor_y)
                        if target is not None:
                            neighbors.add(target)
        return list(neighbors)  # Convert the set to a list before returning

    def get_successors(self):
        """
        Generate all valid successor states by moving pieces into blank spaces.
        :return: A list of successor board states.
        """
        successors = []
        board = self
        grid = board.grid
        pieces = board.pieces

        for i in range(len(pieces)):
            piece = pieces[i]
            px = piece.coord_x
            py = piece.coord_y

            # Handle 1x1 Piece Logic
            if piece.is_single:
                if py + 1 < board.height and (grid[py+1][px] == '.' ):
                    new_board = self.move_piece(i, "2", "down")
                    if new_board:
                        successors.append(new_board)
                if py - 1 >= 0 and (grid[py - 1][px] == '.'):
                    new_board = self.move_piece(i, "2", "up")
                    if new_board:
                        successors.append(new_board)
                if px - 1 >= 0 and (grid[py][px-1] == '.'):
                    new_board = self.move_piece(i, "2", "left")
                    if new_board:
                        successors.append(new_board)
                if px + 1 < board.width and (grid[py][px + 1] == '.') :
                    new_board = self.move_piece(i, "2", "right")
                    if new_board:
                        successors.append(new_board)

            # Handle Horizontal 1x2 Tile Logic
            if piece.orientation == "h" and grid[py][px + 1] == ">" and  px + 1 < board.height:
                if px - 1 >= 0 and (grid[py][px -1] == '.') :
                    new_board = self.move_piece(i, "h", "left")
                    if new_board:
                        successors.append(new_board)
                if px + 2 < board.width and (grid[py][px+2] == '.') :
                    new_board = self.move_piece(i, "h", "right")
                    if new_board:
                        successors.append(new_board)
                if py + 1 < board.height and (grid[py+1][px] == '.' and grid[py + 1][px+1] == '.') :
                    new_board = self.move_piece(i, "h", "down")
                    if new_board:
                        successors.append(new_board)
                if py - 1 >= 0 and (grid[py-1][px] == '.' and grid[py - 1][px+1] == '.') :
                    new_board = self.move_piece(i, "h", "up")
                    if new_board:
                        successors.append(new_board)

            # Handle Vertical 2x1 Tile Logic
            if piece.orientation == "v" and grid[py + 1][px] == "v" and py + 1 < board.height:
                if py + 2 < board.height and (grid[py+2][px]== '.') :
                    new_board = self.move_piece(i, "v", "down")
                    if new_board:
                        successors.append(new_board)
                if py - 1 >= 0 and (grid[py - 1][px] == '.') :
                    new_board = self.move_piece(i, "v", "up")
                    if new_board:
                        successors.append(new_board)
                if px - 1 >= 0 and (grid[py][px-1] == '.' and grid[py + 1][px-1] == '.') :
                    new_board = self.move_piece(i, "v", "left")
                    if new_board:
                        successors.append(new_board)
                if px + 1 < board.width and (grid[py][px+1] == '.' and grid[py + 1][px+1] == '.') :
                    new_board = self.move_piece(i, "v", "right")
                    if new_board:
                        successors.append(new_board)

            # Handle 2x2 Tile Logic
            if piece.is_2_by_2:  # Assuming you have a way to identify 2x2 pieces
                # Check if piece can move right
                if px + 1 < board.width and py + 1 < board.height and grid[py][px + 1] == "1" and grid[py + 1][px] == "1" and grid[py + 1][px + 1] == "1":
                    if px + 2 < board.width and (grid[py][px+2] == '.' and grid[py+1][px+2] =='.'):
                        new_board = self.move_piece(i, "1", "right")
                        if new_board:
                            successors.append(new_board)
                    if px - 1 >= 0 and (grid[py][px-1] == '.' and grid[py+1][px-1] =='.') :
                        new_board = self.move_piece(i, "1", "left")
                        if new_board:
                            successors.append(new_board)
                    if py + 2 < board.height and (grid[py+2][px] == '.' and grid[py+2][px+1] =='.') :
                        new_board = self.move_piece(i, "1", "down")
                        if new_board:
                            successors.append(new_board)
                    if py - 1 >= 0 and (grid[py-1][px] == '.' and grid[py-1][px+1] == '.') :
                        new_board = self.move_piece(i, "1", "up")
                        if new_board:
                            successors.append(new_board)
        return successors

    def move_piece(self, i,piece_type ,direction):
        """
        Move a piece based on its type and direction.
        Returns the new board state after moving.
        """
        new_board = deepcopy(self) # Make a deep copy of the current board
        piece = new_board.pieces[i]  # Get the piece to move

        # Calculate current coordinates
        x = piece.coord_x
        y = piece.coord_y

        if piece.is_2_by_2:
            # 2x2 piece movement
            if direction == "right":
                new_board.grid[y][x] = "."
                new_board.grid[y][x + 1] = "1"
                new_board.grid[y + 1][x] = "."
                new_board.grid[y + 1][x + 1] = "1"
                piece.coord_x += 1  # Update x coordinate
                new_board.grid[y][x + 2] = "1"
                new_board.grid[y + 1][x + 2] = "1"
            elif direction == "left":
                new_board.grid[y][x + 1] = "."
                new_board.grid[y + 1][x + 1] = "."
                piece.coord_x -= 1  # Update x coordinate
                new_board.grid[y][x - 1] = "1"
                new_board.grid[y + 1][x - 1] = "1"
                new_board.grid[y][x] = "1"
                new_board.grid[y+1][x] = "1"
            elif direction == "down":
                new_board.grid[y][x] = "."
                new_board.grid[y][x + 1] = "."
                piece.coord_y += 1  # Update y coordinate
                new_board.grid[y + 2][x] = "1"
                new_board.grid[y + 2][x + 1] = "1"
                new_board.grid[y + 1][x] = "1"
                new_board.grid[y + 1][x + 1] = "1"
            elif direction == "up":
                new_board.grid[y + 1][x] = "."
                new_board.grid[y + 1][x + 1] = "."
                piece.coord_y -= 1  # Update y coordinate
                new_board.grid[y - 1][x] = "1"
                new_board.grid[y - 1][x + 1] = "1"
                new_board.grid[y][x] = "1"
                new_board.grid[y][x + 1] = "1"

        elif piece.orientation == "h":
            # Horizontal piece movement
            if direction == "left":
                new_board.grid[y][x - 1] = "<"
                new_board.grid[y][x] = ">"
                new_board.grid[y][x + 1] = "."
                piece.coord_x -= 1
            elif direction == "right":
                new_board.grid[y][x] = "."
                new_board.grid[y][x + 1] = "<"
                new_board.grid[y][x + 2] = ">"
                piece.coord_x += 1
            elif direction == "down":
                new_board.grid[y][x] = "."
                new_board.grid[y][x + 1] = "."
                piece.coord_y += 1
                new_board.grid[y + 1][x] = "<"
                new_board.grid[y + 1][x + 1] = ">"
            elif direction == "up":
                new_board.grid[y][x] = "."
                new_board.grid[y][x + 1] = "."
                piece.coord_y -= 1
                new_board.grid[y - 1][x] = "<"
                new_board.grid[y - 1][x + 1] = ">"

        elif piece.orientation == "v":
            # Vertical piece movement
            if direction == "down":
                new_board.grid[y][x] = "."
                new_board.grid[y + 1][x] = "^"
                new_board.grid[y + 2][x] = "v"
                piece.coord_y += 1
            elif direction == "up":
                new_board.grid[y + 1][x] = "."
                new_board.grid[y][x] = "v"
                new_board.grid[y - 1][x] = "^"
                piece.coord_y -= 1
            elif direction == "left":
                new_board.grid[y][x] = "."
                new_board.grid[y + 1][x] = "."
                new_board.grid[y][x - 1] = "^"
                new_board.grid[y + 1][x - 1] = "v"
                piece.coord_x -= 1
            elif direction == "right":
                new_board.grid[y][x] = "."
                new_board.grid[y + 1][x] = "."
                new_board.grid[y][x + 1] = "^"
                new_board.grid[y + 1][x + 1] = "v"
                piece.coord_x += 1

        elif piece.is_single:
            # 1x1 piece movement
            if direction == "down":
                new_board.grid[y][x] = "."
                new_board.grid[y + 1][x] = "2"
                piece.coord_y += 1
            elif direction == "up":
                new_board.grid[y - 1][x] = "2"
                new_board.grid[y][x] = "."
                piece.coord_y -= 1
            elif direction == "left":
                new_board.grid[y][x] = "."
                new_board.grid[y][x - 1] = "2"
                piece.coord_x -= 1
            elif direction == "right":
                new_board.grid[y][x] = "."
                new_board.grid[y][x + 1] = "2"
                piece.coord_x += 1

        return new_board

    def get_piece_at(self, x, y):
        """
        Retrieve the piece located at the specified coordinates on the board.
        :param x: The x-coordinate of the desired position.
        :param y: The y-coordinate of the desired position.
        :return: The piece at the given coordinates, or None if there is no piece.
        """
        for piece in self.pieces:
            # Check if the piece occupies the given coordinates
            if (piece.coord_x == x and piece.coord_y == y) or (
                    piece.is_2_by_2 and (
                    (piece.coord_x == x and piece.coord_y == y) or
                    (piece.coord_x + 1 == x and piece.coord_y == y) or
                    (piece.coord_x == x and piece.coord_y + 1 == y) or
                    (piece.coord_x + 1 == x and piece.coord_y + 1 == y)
            )
            ):
                return piece
        return

    def __hash__(self):
        # Convert the grid to a tuple of tuples for hashing
        return hash(tuple(tuple(row) for row in self.grid))

    def get_target_position(self,piecee):
        """helper for manhattan distance"""
        for piece in self.pieces:
            # Check if the piece type matches and other relevant properties
            if (piece.is_2_by_2 == piecee.is_2_by_2 and
                    piece.is_single == piecee.is_single and
                    piece.orientation == piecee.orientation):
                return (piece.coord_y, piece.coord_x)

        return  # If no matching piece is found

class State:
    """State class wrapping a Board with extra current state information."""

    def __init__(self, board, hfn, f, depth, parent=None):
        self.board = board
        self.hfn = hfn  # Heuristic function value
        self.f = f      # Total cost (g + h)
        self.depth = depth  # Depth in the search tree
        self.parent = parent  # Parent state for backtracking

    def __eq__(self, other):
        # Compare states by their board's grid
        return isinstance(other, State) and self.board.grid == other.board.grid

    def __lt__(self, other):
        # Compare states by their f values (for the priority queue)
        return self.f < other.f

    def __hash__(self):
        """Generate a hash based on the board's grid."""
        return hash(tuple(map(tuple, self.board.grid)))  # Convert grid to a tuple of tuples for hashing


def is_goal_state(board, goal_board):
    """Check if the current board state is the goal state."""
    # Check if the number of pieces is the same
    if len(board.pieces) != len(goal_board.pieces):
        return False

    # Compare each piece's position on the current board with the goal board
    for i in range(board.height):
        for j in range(board.width):
            if board.grid[i][j] != goal_board.grid[i][j]:
                return False
    return True

class Solver:
    def __init__(self, initial_board, goal_board):
        self.goal_board = goal_board
        self.initial_state = State(initial_board, self.heuristic(initial_board), 0, 0)
         # Ensure goal_board is defined

    def is_in_target_position(self, piece):
        """Check if the piece is in its target position based on the goal board."""
        target_position = self.goal_board.get_target_position(piece)
        if target_position:
            return (piece.coord_x, piece.coord_y) == target_position
        return False

    def heuristic(self, board):
        """Calculate the heuristic value for the A* algorithm."""
        total_distance = 0
        for piece in board.pieces:  # Assume board.pieces contains all pieces with their positions
            current_y, current_x = piece.coord_y, piece.coord_x  # Current position
            target_y, target_x = goal_board.get_target_position(piece)# Target position
            total_distance +=  abs(current_y - target_y) + abs(current_x - target_x)
        return total_distance

    def a_star_search(self):
        """Implement A* search algorithm with closed_set as a set."""
        open_set = []  # Priority queue
        open_dict = {}  # For quick lookups of boards in open_set
        closed_set = set()  # Use a set for faster lookups
        heapq.heappush(open_set, self.initial_state)
        open_dict[self.initial_state.board] = self.initial_state.f
        parent = {}
        self.initial_state.depth = 0

        while open_set:
            current_state = heapq.heappop(open_set)
            open_dict.pop(current_state.board, None)  # Remove from open_dict

            # Check for goal state
            if is_goal_state(current_state.board, self.goal_board):
                return self.reconstruct_path_astar(current_state, parent)

            closed_set.add(current_state.board)  # Add to closed_set

            # Explore successors
            for successor in current_state.board.get_successors():
                g = current_state.depth + 1  # Increment depth (g value)
                h = self.heuristic(successor)  # Heuristic value (h)
                f = g + h  # Total cost (f = g + h)
                successor_state = State(successor, h, f, g, current_state)

                # Skip if this board has already been explored (is in closed_set)
                if successor_state.board in closed_set:
                    continue

                # Check if the successor is already in the open set
                if successor_state.board not in open_dict or f <= open_dict[successor_state.board]:
                    parent[successor_state.board] = current_state.board
                    open_dict[successor_state.board] = f
                    heapq.heappush(open_set, successor_state)
        return "No solution"  # No solution found

    def dfs(self):
        """Implement Depth-First Search (DFS) algorithm."""
        stack = [self.initial_state.board]
        visited = set()
        parent = {}
        while stack:
            current_state = stack.pop()

            # Check for goal state
            if is_goal_state(current_state,self.goal_board):
                return self.reconstruct_path(current_state, parent)

            visited.add(current_state)

            for successor in current_state.get_successors():
                if successor not in visited:
                    stack.append(successor)
                    parent[successor] = current_state
        return "No solution"



    def reconstruct_path(self, board, parent):
        """Backtrack to reconstruct the solution path, returning the path from initial to goal."""
        path = []

        # Backtrack from the goal state to the initial state
        while board in parent:
            print(board)# Continue until there is no parent (initial state)
            path.append(board)  # Append the current board to the path
            board = parent[board]  # Move to the parent state

        # Add the initial state, which has no parent, to the pathp
        path.append(self.initial_state.board)  # Assuming initial_state has a 'board' attribute

        # Return the path in the order from initial to goal
        return path[::-1]  # Reverse the path to show from initial to goal

    def reconstruct_path_astar(self, goal_state, parent):
        """Backtrack to reconstruct the solution path from initial to goal for A* search."""
        path = []

        # Backtrack from the goal state to the initial state
        while goal_state.board in parent:  # Continue until there is no parent (initial state)
            path.append(goal_state.board)  # Append the current board to the path
            goal_state.board = parent[goal_state.board]  # Move to the parent state

        # Append the initial state, which has no parent, to the path
        path.append(self.initial_state.board)  # Assuming initial_state has a 'board' attribute

        # Return the path in the order from initial to goal
        return path[::-1]  # Reverse the path to show from initial to goal

def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    final_pieces = []
    final = False
    found_2by2 = False
    finalfound_2by2 = False
    height_ = 0

    for line in puzzle_file:
        height_ += 1
        if line == '\n':
            if not final:
                height_ = 0
                final = True
                line_index = 0
            continue
        if not final:  # initial board
            for x, ch in enumerate(line):
                if ch == '^':  # found vertical piece
                    pieces.append(Piece(False, False, x, line_index, '^','v'))
                elif ch == '<':  # found horizontal piece
                    pieces.append(Piece(False, False, x, line_index, '<','h'))
                elif ch == char_single:
                    pieces.append(Piece(False, True, x, line_index,'2', None))
                elif ch == '1':
                    if found_2by2 is False:
                        pieces.append(Piece(True, False, x, line_index,'1', None,))
                        found_2by2 = True
        else:  # goal board
            for x, ch in enumerate(line):
                if ch == '^':  # found vertical piece
                    final_pieces.append(Piece(False, False, x, line_index,'^', 'v'))
                elif ch == '<':  # found horizontal piece
                    final_pieces.append(Piece(False, False, x, line_index, '<','h'))
                elif ch == char_single:
                    final_pieces.append(Piece(False, True, x, line_index,'2', None))
                elif ch == '1':
                    if finalfound_2by2 is False:
                        final_pieces.append(Piece(True, False, x, line_index,'1', None))
                        finalfound_2by2 = True
        line_index += 1

    puzzle_file.close()
    board = Board(height_, pieces)
    goal_board = Board(height_, final_pieces)
    return board, goal_board


def write_solution_to_file(path, outputfile):
    """Write the solution path to the output file."""
    with open(outputfile, 'w') as f:
        if not path:  # Check if the path is empty
            f.write("No solution\n")
        else:
            for state in path:
                # Access the board from the state object and convert it to a string
                f.write(grid_to_string(state.grid) + "\n")  # Add newline for separation

def grid_to_string(grid):
    string = ""
    for i, line in enumerate(grid):
        for ch in line:
            string += ch
        string += "\n"
    return string


if __name__ == "__main__":
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
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()

    # read the board from the file
    board, goal_board = read_from_file(args.inputfile)
    # Initialize the solver with both boards
    solver = Solver(board, goal_board)
    # Initialize the solver with the initial and goal boards

    if args.algo == 'astar':
        solution_path = solver.a_star_search()
    elif args.algo == 'dfs':
        solution_path = solver.dfs()
    else:
        print("Invalid algorithm choice.")
        sys.exit(1)

    # Write solution
    if solution_path == "No solution":
        print("No solution found.")
        with open(args.outputfile, 'w') as f:
            f.write("No solution\n")
    else:
        write_solution_to_file(solution_path, args.outputfile)

