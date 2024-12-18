import unittest
from hrsmine import read_from_file
from hrd import Piece, Board, State,  write_solution_to_file



class TestHuaRongDao(unittest.TestCase):

    def setUp(self):
        # Set up an initial and goal board for testing
        initial_pieces = [
            Piece(True, False, 1, 1,'1'),  # 2x2 piece at (1, 1)
            Piece(False, True, 0, 0,'2'),  # Single piece at (0, 0)
            Piece(False, False, 2, 0,'<', 'h'),  # Horizontal 1x2 piece at (2, 0)
            Piece(False, False, 0, 2,'^', 'v'),  # Vertical 1x2 piece at (0, 2)
        ]
        goal_pieces = [
            Piece(True, False, 2, 2,'1',),  # 2x2 piece at goal position (2, 2)
            Piece(False, True, 0, 0,'2',),  # Single piece at (0, 0)
            Piece(False, False, 2, 0,'<' ,'h'),  # Horizontal 1x2 piece at (2, 0)
            Piece(False, False, 0, 2, '^','v'),  # Vertical 1x2 piece at (0, 2)
        ]
        self.initial_board = Board(5, initial_pieces)
        self.goal_board = Board(5, goal_pieces)


    def test_valid_move_2x2(self):
        # Test if moving the 2x2 piece down is valid
        piece = self.initial_board.pieces[0]  # 2x2 piece
        print(self.initial_board.display())
        valid_move = self.initial_board.is_valid_move(piece, 0, 1)
        self.assertTrue(valid_move)

    def test_invalid_move_2x2(self):
        # Test if moving the 2x2 piece out of bounds is invalid
        piece = self.initial_board.pieces[0]  # 2x2 piece
        invalid_move = self.initial_board.is_valid_move(piece, 0, -2)  # Move out of bounds
        self.assertFalse(invalid_move)

    def test_valid_move_1x1(self):
        # Test if moving the 1x1 piece is valid
        piece = self.initial_board.pieces[1]  # 1x1 piece
        valid_move = self.initial_board.is_valid_move(piece, 1, 0)  # Move right
        self.assertTrue(valid_move)

    def test_dfs_solution(self):
        # Test if DFS finds the correct solution path
        solution_path = get_dfs_solution(self.initial_board, self.goal_board)
        self.assertIsNotNone(solution_path)
        self.assertTrue(self.goal_board in solution_path)  # Goal should be reached

    def test_a_star_solution(self):
        # Test if A* finds the correct solution path
        solution_path = a_star_search(self.initial_board, self.goal_board)
        self.assertIsNotNone(solution_path)
        self.assertTrue(self.goal_board in solution_path)  # Goal should be reached

    def test_manhattan_distance(self):
        # Test the Manhattan distance heuristic
        distance = manhattan_distance(self.initial_board, self.goal_board)
        print(self.goal_board.display())
        self.assertEqual(distance, 6)  # Manhattan distance for the 2x2 piece from (1, 1) to (2, 2)



    def test_valid_move_up(self):
        """Test moving a piece upwards."""
        new_board = self.initial_board.move(self.initial_board.pieces[0], 'up')
        self.assertEqual(new_board.get_piece_at(0, 1), self.initial_board.pieces[0])

    def test_valid_move_down(self):
        """Test moving a piece downwards."""
        new_board = self.initial_board.move(self.initial_board.pieces[1], 'down')
        self.assertEqual(new_board.get_piece_at(2, 1).symbol, self.initial_board.pieces[1].symbol)

    def test_valid_move_left(self):
        """Test moving a piece left."""
        new_board = self.initial_board.move(self.initial_board.pieces[0], 'left')
        self.assertEqual(new_board.get_piece_at(1, 0),self.initial_board.pieces[0])

    def test_valid_move_right(self):
        """Test moving a piece right."""
        new_board = self.initial_board.move(self.initial_board.pieces[1], 'right')
        self.assertEqual(new_board.get_piece_at(1, 2), self.initial_board.pieces[1])






if __name__ == '__main__':
    unittest.main()

