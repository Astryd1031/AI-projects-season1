import unittest
from checkers import State, checker,minimax  # Replace 'your_module_name' with the actual name of your module

class TestCheckers(unittest.TestCase):

    def setUp(self):
        """Set up a standard initial board state for testing."""
        self.initial_board = [
            ['b', '.', 'b', '.', 'b', '.', 'b', '.'],
            ['.', 'b', '.', 'b', '.', 'b', '.', 'b'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['r', '.', 'r', '.', 'r', '.', 'r', '.'],
            ['.', 'r', '.', 'r', '.', 'r', '.', 'r'],
            ['r', '.', 'r', '.', 'r', '.', 'r', '.']
        ]
        self.state = State(self.initial_board)

    def test_apply_move(self):
        """Test applying a simple move."""
        start_pos = (5, 0)  # r at (5, 0)
        end_pos = (4, 1)  # Move to (4, 1)

        self.state.apply_move(start_pos, end_pos)

        # Verify the board state after the move
        expected_board = [
            ['b', '.', 'b', '.', 'b', '.', 'b', '.'],
            ['.', 'b', '.', 'b', '.', 'b', '.', 'b'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', 'r', '.', '.', '.', '.', '.', '.'],
            ['.', '.', 'r', '.', 'r', '.', 'r', '.'],
            ['.', 'r', '.', 'r', '.', 'r', '.', 'r'],
            ['r', '.', 'r', '.', 'r', '.', 'r', '.']
        ]

        self.assertEqual(self.state.board, expected_board)

    def test_get_successors(self):
        """Test the get_successors method."""
        successors = self.state.get_successors('r')  # Get successors for red

        # Check if the number of successors is correct (for example purposes)
        self.assertGreater(len(successors), 0)  # Ensure at least one successor exists


    def test_single_jump(self):
        """Test applying a single jump move for a red checker."""
        red_checker = checker(5, 2, 'r')  # Starting at (5, 2)
        jump_path = [(5, 2), (3, 4)]  # Jump over (4, 3)

        # Apply the jump path
        self.state.apply_jump_path(jump_path)

        # Verify the board state after the jump
        expected_board = [
            ['b', '.', 'b', '.', 'b', '.', 'b', '.'],
            ['.', 'b', '.', 'b', '.', 'b', '.', 'b'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', 'r', '.', '.', '.'],  # Red piece should be at (3, 4)
            ['.', '.', '.', '.', '.', '.', '.', '.'],  # The opponent's piece at (4, 3) should be removed
            ['r', '.', '.', '.', 'r', '.', 'r', '.'],
            ['.', 'r', '.', 'r', '.', 'r', '.', 'r'],
            ['r', '.', 'r', '.', 'r', '.', 'r', '.']
        ]

        self.assertEqual(self.state.board, expected_board)

    def test_checks_black(self):
        """Test checks method for the black player."""
        black_checkers = self.state.checks('black')

        # Expecting black checkers at (0, 0), (0, 2), (0, 4), (0, 6), (1, 1), (1, 3), (1, 5), (1, 7)
        expected_positions = [(0, 0), (0, 2), (0, 4), (0, 6),
                              (1, 1), (1, 3), (1, 5), (1, 7)]

        actual_positions = [checker.get_coordinates() for checker in black_checkers]
        self.assertCountEqual(actual_positions, expected_positions)

    def test_checks_red(self):
        """Test checks method for the red player."""
        red_checkers = self.state.checks('red')

        # Expecting red checkers at (5, 0), (5, 2), (5, 4), (5, 6), (6, 1), (6, 3), (6, 5), (6, 7), (7, 0), (7, 2), (7, 4), (7, 6)
        expected_positions = [(5, 0), (5, 2), (5, 4), (5, 6),
                              (6, 1), (6, 3), (6, 5), (6, 7),
                              (7, 0), (7, 2), (7, 4), (7, 6)]

        actual_positions = [checker.get_coordinates() for checker in red_checkers]
        self.assertCountEqual(actual_positions, expected_positions)

    def test_multiple_jumps(self):
        """Test applying a sequence of multiple jumps for a red checker."""
        jump_path = [(5, 2), (3, 4), (1, 2)]  # Jump over (5, 2) and (3, 4)

        # Apply the jump path
        self.state.apply_jump_path(jump_path)

        # Verify the board state after the jumps
        expected_board = [
            ['b', '.', 'b', '.', 'b', '.', 'b', '.'],
            ['.', 'b', 'r', 'b', '.', 'b', '.', 'b'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],  # Red piece should be at (2, 5)
            ['.', '.', '.', '.', '.', '.', '.', '.'],  # The opponent's piece at (3, 4) should be removed
            ['.', '.', '.', '.', '.', '.', '.', '.'],  # The opponent's piece at (5, 2) should be removed
            ['r', '.', '.', '.', 'r', '.', 'r', '.'],
            ['.', 'r', '.', 'r', '.', 'r', '.', 'r'],
            ['r', '.', 'r', '.', 'r', '.', 'r', '.']
        ]

        self.assertEqual(self.state.board, expected_board)

    def test_checker_get_jumps(self):
            """Test the get_jumps method for jump move generation."""
            test_checker = checker(5, 2, 'r')  # A red checker at (5, 2)

            # Create a board state where jumping is possible
            jump_board = [
                ['b', '.', 'b', '.', 'b', '.', 'b', '.'],
                ['.', 'b', '.', 'b', '.', 'b', '.', 'b'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', 'b', '.', '.', '.', '.'],  # Jump over the black piece at (4, 2)
                ['r', '.', 'r', '.', 'r', '.', 'r', '.'],
                ['.', 'r', '.', 'r', '.', 'r', '.', 'r'],
                ['r', '.', 'r', '.', 'r', '.', 'r', '.']
            ]
            jump_state = State(jump_board)

            jumps = test_checker.get_jumps(jump_state)  # Get jumps for the checker

            # Check if the expected jump is generated correctly
            expected_jumps = [(3, 4)]  # The position after jumping over (4, 2)
            self.assertIn((3, 4), jumps)


    def test_utility(self):
        """Test the utility method for terminal state evaluation."""
        # Create a terminal state where red has lost
        terminal_board = [
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', 'b', '.', 'b', '.', 'b', '.', 'b']
        ]
        terminal_state = State(terminal_board)

        self.assertEqual(terminal_state.utility(), -1)  # Red loses

import unittest

class TestStateGetSuccessorsMethod(unittest.TestCase):

    def setUp(self):
        """Set up a standard initial board state for testing."""
        self.initial_board = [
            ['b', '.', 'b', '.', 'b', '.', 'b', '.'],
            ['.', 'b', '.', 'b', '.', 'b', '.', 'b'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['r', '.', 'r', '.', 'r', '.', 'r', '.'],
            ['.', 'r', '.', 'r', '.', 'r', '.', 'r'],
            ['r', '.', 'r', '.', 'r', '.', 'r', '.']
        ]
        self.state = State(self.initial_board)

    def test_get_successors_red(self):
        """Test get_successors method for the red player."""
        successors = self.state.get_successors('r')

        # There are expected successors for red checkers.
        self.assertEqual(len(successors), 6)  # Assuming there are 6 possible moves

        # Example check for one of the successor states
        # Check the first successor state for a known move
        first_successor = successors[0]

        # Check that a specific move was made
        expected_board = [
            ['b', '.', 'b', '.', 'b', '.', 'b', '.'],
            ['.', 'b', '.', 'b', '.', 'b', '.', 'b'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', 'r', '.', '.', '.'],  # A piece moved from (5, 4) to (4, 4)
            ['r', '.', 'r', '.', '.', '.', 'r', '.'],
            ['.', 'r', '.', 'r', '.', 'r', '.', 'r'],
            ['r', '.', 'r', '.', 'r', '.', 'r', '.']
        ]

        self.assertEqual(first_successor.board, expected_board)

    def test_get_successors_black(self):
        """Test get_successors method for the black player."""
        successors = self.state.get_successors('b')

        # Assuming there are valid moves for black
        self.assertEqual(len(successors), 4)  # Assuming there are 4 possible moves

        # Example check for one of the successor states
        # Check the first successor state for a known move
        first_successor = successors[0]

        # Check that a specific move was made
        expected_board = [
            ['.', '.', 'b', '.', 'b', '.', 'b', '.'],
            ['.', 'b', '.', 'b', '.', 'b', '.', 'b'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['r', '.', 'r', '.', 'r', '.', 'r', '.'],
            ['.', 'r', '.', 'r', '.', 'r', '.', 'r'],
            ['r', '.', 'r', '.', 'r', '.', 'r', '.']
        ]

        self.assertEqual(first_successor.board, expected_board)

if __name__ == '__main__':
    unittest.main()

