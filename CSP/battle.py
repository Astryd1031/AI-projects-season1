import time
from backtracking import *
import argparse
from csp import CSP,Variable
from constraints import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--inputfile",
    type=str,
    required=True,
    help="The input file that contains the puzzle."
)
parser.add_argument(
    "--outputfile",
    type=str,
    required=True,
    help="The output file that contains the solution."
)

args = parser.parse_args()
file = open(args.inputfile, 'r')
b = file.read()
b2 = b.split()
ship_size = b2[2]
size = len(b2[0])
size = size + 2
b3 = []
b3 += ['0' + b2[0] + '0']
b3 += ['0' + b2[1] + '0']
b3 += [b2[2] + ('0' if len(b2[2]) == 3 else '')]
b3 += ['0' * size]
for i in range(3, len(b2)):
    b3 += ['0' + b2[i] + '0']
b3 += ['0' * size]
board = "\n".join(b3)

varlist = []
varn = {}
conslist = []

elements = []

final_board = board.split()[3:]
# 1/0 variables
for i in range(0, size):
    for j in range(0, size):
        v = None
        if i == 0 or i == size - 1 or j == 0 or j == size - 1:
            v = Variable(str(-1 - (i * size + j)), [0])
        else:
            ch = final_board[i][j]
            v = Variable(str(-1 - (i * size + j)), [0, 1])
            if ch != "0":
                elements.append((i, j, ch))

        varlist.append(v)
        varn[str(-1 - (i * size + j))] = v

# make 1/0 variables match board info
ii = 0
for i in board.split()[3:]:
    jj = 0
    for j in i:
        if j != '0' and j != '.':  # must be ship parts
            conslist.append(TableConstraint('boolean_match', [varn[str(-1 - (ii * size + jj))]], [[1]]))
        elif j == '.':
            conslist.append(TableConstraint('boolean_match', [varn[str(-1 - (ii * size + jj))]], [[0]]))
        jj += 1
    ii += 1

# row and column constraints on 1/0 variables
row_constraint = []
for i in board.split()[0]:
    row_constraint += [int(i)]
for row in range(0, size):
    conslist.append(NValuesConstraint('row', [varn[str(-1 - (row * size + col))] for col in range(0, size)], [1],
                                      row_constraint[row], row_constraint[row]))

col_constraint = []
for i in board.split()[1]:
    col_constraint += [int(i)]
# print(col_constraint)
for col in range(0, size):
    conslist.append(NValuesConstraint('col', [varn[str(-1 - (col + row * size))] for row in range(0, size)], [1],
                                      col_constraint[col], col_constraint[col]))

# diagonal constraints on 1/0 variables
for i in range(1, size - 1):
    for j in range(1, size - 1):
        for k in range(9):
            conslist.append(
                NValuesConstraint('diag', [varn[str(-1 - (i * size + j))], varn[str(-1 - ((i - 1) * size + (j - 1)))]],
                                  [1], 0, 1))
            conslist.append(
                NValuesConstraint('diag', [varn[str(-1 - (i * size + j))], varn[str(-1 - ((i - 1) * size + (j + 1)))]],
                                  [1], 0, 1))


# ./S/</>/v/^/M variables
# these would be added to the csp as well, before searching,
# along with other constraints
for i in range(0, size):
    for j in range(0, size):
        v = Variable(str(i * size + j), ['.', 'S'])
        varlist.append(v)
        varn[str(i * size + j)] = v
        # connect 1/0 variables to ./S/L/R/B/T/M variables
        # make positive pieces ship part when 1, not ship part when 0
        conslist.append(
            TableConstraint('connect', [varn[str(-1 - (i * size + j))], varn[str(i * size + j)]], [[0, '.'], [1, 'S']]))

# find all solutions and check which one has right ship #'s
csp = CSP('battleship', varlist, conslist)
t_start = time.time()
solutions, num_nodes = bt_search('GAC', csp, 'mrv', False, False, ship_size, final_board, size, elements)
for i in range(len(solutions)):
        # output_to_file(filename=args.outputfile, sol=solutions[i])
    #print_solution(solutions[i], size)
    #print("--------------")
    output_to_file(filename=args.outputfile, sol=solutions[i], size=size)