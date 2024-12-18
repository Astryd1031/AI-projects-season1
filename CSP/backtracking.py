import random
from csp import Variable

class UnassignedVars:
    '''class for holding the unassigned variables of a CSP. We can extract
       from, re-initialize it, and return variables to it.  Object is
       initialized by passing a select_criteria (to determine the
       order variables are extracted) and the CSP object.

       select_criteria = ['random', 'fixed', 'mrv'] with
       'random' == select a random unassigned variable
       'fixed'  == follow the ordering of the CSP variables (i.e.,
                   csp.variables()[0] before csp.variables()[1]
       'mrv'    == select the variable with minimum values in its current domain
                   break ties by the ordering in the CSP variables.
    '''
    def __init__(self, select_criteria, csp):
        if select_criteria not in ['random', 'fixed', 'mrv']:
            pass #print "Error UnassignedVars given an illegal selection criteria {}. Must be one of 'random', 'stack', 'queue', or 'mrv'".format(select_criteria)
        self.unassigned = list(csp.variables())
        self.csp = csp
        self._select = select_criteria
        if select_criteria == 'fixed':
            #reverse unassigned list so that we can add and extract from the back
            self.unassigned.reverse()

    def extract(self):
        if not self.unassigned:
            pass #print "Warning, extracting from empty unassigned list"
            return None
        if self._select == 'random':
            i = random.randint(0,len(self.unassigned)-1)
            nxtvar = self.unassigned[i]
            self.unassigned[i] = self.unassigned[-1]
            self.unassigned.pop()
            return nxtvar
        if self._select == 'fixed':
            return self.unassigned.pop()
        if self._select == 'mrv':
            nxtvar = min(self.unassigned, key=lambda v: v.curDomainSize())
            self.unassigned.remove(nxtvar)
            return nxtvar

    def empty(self):
        return len(self.unassigned) == 0

    def insert(self, var):
        if not var in self.csp.variables():
            pass #print "Error, trying to insert variable {} in unassigned that is not in the CSP problem".format(var.name())
        else:
            self.unassigned.append(var)

def bt_search(algo, csp, variableHeuristic, allSolutions, trace, ship_size, final_board, size, elements):
    '''Main interface routine for calling different forms of backtracking search
       algorithm is one of ['BT', 'FC', 'GAC']
       csp is a CSP object specifying the csp problem to solve
       variableHeuristic is one of ['random', 'fixed', 'mrv']
       allSolutions True or False. True means we want to find all solutions.
       trace True of False. True means turn on tracing of the algorithm

       bt_search returns a list of solutions. Each solution is itself a list
       of pairs (var, value). Where var is a Variable object, and value is
       a value from its domain.
    '''
    varHeuristics = ['random', 'fixed', 'mrv']
    algorithms = ['BT', 'FC', 'GAC']
    size = size
    bt_search.nodesExplored = 0

    if variableHeuristic not in varHeuristics:
        pass
    if algo not in algorithms:
        pass

    uv = UnassignedVars(variableHeuristic, csp)
    Variable.clearUndoDict()
    for v in csp.variables():
        v.reset()
    if algo == 'BT':
        solutions = BT(uv, csp, allSolutions, trace, ship_size, size)
    elif algo == 'GAC':
        GacEnforce(csp.constraints(), csp, None, None,size)  # GAC at the root
        solutions = GAC(uv, csp, final_board, ship_size, size, elements)

    return solutions, bt_search.nodesExplored


def BT(unAssignedVars, csp, allSolutions, trace, ship_size, size):
    '''Backtracking Search. unAssignedVars is the current set of
       unassigned variables.  csp is the csp problem, allSolutions is
       True if you want all solutionss trace if you want some tracing
       of variable assignments tried and constraints failed. Returns
       the set of solutions found.

      To handle finding 'allSolutions', at every stage we collect
      up the solutions returned by the recursive  calls, and
      then return a list of all of them.

      If we are only looking for one solution we stop trying
      further values of the variable currently being tried as
      soon as one of the recursive calls returns some solutions.
    '''
    if unAssignedVars.empty():
        if trace: pass
        soln = []
        for v in csp.variables():
            if int(v._name) > 0:
                # print("yes")
                soln.append((v, v.getValue()))
        print_solution(soln, 8)
        print("\n")
        return [soln]
    bt_search.nodesExplored += 1
    solns = []
    nxtvar = unAssignedVars.extract()
    if trace: pass
    for val in nxtvar.domain():
        if trace: pass
        nxtvar.setValue(val)
        constraintsOK = True
        for cnstr in csp.constraintsOf(nxtvar):
            if cnstr.numUnassigned() == 0:
                if not cnstr.check():
                    constraintsOK = False
                    if trace: pass
                    break
        if constraintsOK:
            new_solns = BT(unAssignedVars, csp, allSolutions, trace, ship_size, size)
            if new_solns:
                five, four, three, two, one, s_ = count_ship_numbers(new_solns[0], size)
                if one == int(ship_size[0]) and two == int(ship_size[1]) and three == int(
                        ship_size[2]) and four == int(ship_size[3] and five == int(ship_size[4])):
                    solns.extend(new_solns)
                    if len(solns) > 0 and not allSolutions:
                        break
    nxtvar.unAssign()
    unAssignedVars.insert(nxtvar)
    return solns

def GacEnforce(cnstrs, csp, assignedvar, assignedval,size):
    cnstrs = csp.constraints()
    while len(cnstrs) != 0:
        cnstr = cnstrs.pop()
        for var in cnstr.scope():
            for val in var.curDomain():
                if not cnstr.hasSupport(var, val):
                    var.pruneValue(val, assignedvar, assignedval)
                    if var.curDomainSize() == 0:
                        return "DWO"
                    for recheck in csp.constraintsOf(var):
                        if recheck != cnstr and recheck not in cnstrs:
                            cnstrs.append(recheck)
    return "OK"

def removed(csp, ship_size, size):
    """
    This function checks if the number of ships of each size
    exceeds the given ship_size limits. A helper function for GAC.
    """
    nsoln = []
    for var in csp.variables():
        if int(var._name) > 0:
            nsoln.append((var, var.getValue()))

    n5, n4, n3, n2, n1, ns_ = remove_ships(nsoln, size)
    if n1 > int(ship_size[0]) or n2 > int(ship_size[1]) or n3 > int(ship_size[2]) or n4 > int(
            ship_size[3] or n5 > int(ship_size[4])):
        return True
    return False

def remove_ships(solution, size):
    """
    This function counts the number of ships placed on the board and update solution.
    A helper function for GAC .
    """
    five = four = three = two = one = 0
    s_ = {}

    for (var, val) in solution:
        s_[int(var.name())] = val

    # Check vertical ships
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            belowP = None
            abovP = None
            if (i < (size - 5) and s_[(i * size + j)] == "S" and s_[((i + 1) * size + j)] == "S" and s_[
                ((i + 2) * size + j)] == "S" and s_[((i + 3) * size + j)] == "S" and s_[((i + 4) * size + j)] == "S"):
                five += 1
                s_[((i) * size + j)] = "^"
                s_[((i + 1) * size + j)] = "M"
                s_[((i + 2) * size + j)] = "M"
                s_[((i + 3) * size + j)] = "M"
                s_[((i + 4) * size + j)] = "v"
            elif (i < (size - 4) and s_[(i * size + j)] == "S" and s_[((i + 1) * size + j)] == "S" and s_[
                ((i + 2) * size + j)] == "S" and s_[((i + 3) * size + j)] == "S"):
                four += 1
                s_[((i) * size + j)] = "^"
                s_[((i + 1) * size + j)] = "M"
                s_[((i + 2) * size + j)] = "M"
                s_[((i + 3) * size + j)] = "v"
            elif (i < (size - 3) and s_[(i * size + j)] == "S" and s_[((i + 1) * size + j)] == "S" and s_[
                ((i + 2) * size + j)] == "S"):
                if i != size - 4:
                    belowP = s_[((i + 3) * size + j)]
                if i != 1:
                    abovP = s_[((i - 1) * size + j)]
                if belowP == "." and abovP == ".":
                    three += 1
                    s_[((i) * size + j)] = "^"
                    s_[((i + 1) * size + j)] = "M"
                    s_[((i + 2) * size + j)] = "v"
            elif (i < (size - 2) and s_[(i * size + j)] == "S" and s_[((i + 1) * size + j)] == "S"):
                if i != size - 3:
                    belowP = s_[((i + 2) * size + j)]
                if i != 1:
                    abovP = s_[((i - 1) * size + j)]
                if belowP == "." and abovP == ".":
                    two += 1
                    s_[((i) * size + j)] = "^"
                    s_[((i + 1) * size + j)] = "v"

    # Check horizontal ships
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            rightP = None
            leftP = None
            if (j < (size - 5) and s_[(i * size + j)] == "S" and s_[(i * size + j + 1)] == "S" and s_[
                (i * size + j + 2)] == "S" and s_[(i * size + j + 3)] == "S" and s_[(i * size + j + 4)] == "S"):
                five += 1
                s_[(i * size + j)] = "<"
                s_[(i * size + j + 1)] = "M"
                s_[(i * size + j + 2)] = "M"
                s_[(i * size + j + 3)] = "M"
                s_[(i * size + j + 4)] = ">"
            elif (j < (size - 4) and s_[(i * size + j)] == "S" and s_[(i * size + j + 1)] == "S" and s_[
                (i * size + j + 2)] == "S" and s_[(i * size + j + 3)] == "S"):
                four += 1
                s_[(i * size + j)] = "<"
                s_[(i * size + j + 1)] = "M"
                s_[(i * size + j + 2)] = "M"
                s_[(i * size + j + 3)] = ">"
            elif (j < (size - 3) and s_[(i * size + j)] == "S" and s_[(i * size + j + 1)] == "S" and s_[
                (i * size + j + 2)] == "S"):
                if j != size - 4:
                    rightP = s_[(i * size + j + 3)]
                if j != 1:
                    leftP = s_[(i * size + j - 1)]
                if rightP == "." and leftP == ".":
                    three += 1
                    s_[(i * size + j)] = "<"
                    s_[(i * size + j + 1)] = "M"
                    s_[(i * size + j + 2)] = ">"
            elif (j < (size - 2) and s_[(i * size + j)] == "S" and s_[(i * size + j + 1)] == "S"):
                if j != size - 3:
                    rightP = s_[(i * size + j + 2)]
                if j != 1:
                    leftP = s_[(i * size + j - 1)]
                if rightP == "." and leftP == ".":
                    two += 1
                    s_[(i * size + j)] = "<"
                    s_[(i * size + j + 1)] = ">"

    # Check for single "S" ships
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            lone = False  # if S is surrounded by water
            if s_[(i * size + j)] == "S":
                if i == 1 and j == 1:  # upper left corner
                    belowP = s_[((i + 1) * size + j)]
                    rightP = s_[((i) * size + j + 1)]
                    if belowP == "." and rightP == ".":
                        lone = True
                elif i == 1 and j == size - 2:  # upper right corner
                    belowP = s_[((i + 1) * size + j)]
                    leftP = s_[((i) * size + j - 1)]
                    if belowP == "." and leftP == ".":
                        lone = True
                elif i == size - 2 and j == 1:  # lower left corner
                    abovP = s_[((i - 1) * size + j)]
                    rightP = s_[((i) * size + j + 1)]
                    if abovP == "." and rightP == ".":
                        lone = True
                elif i == size - 2 and j == size - 2:  # lower right corner
                    abovP = s_[((i - 1) * size + j)]
                    leftP = s_[((i) * size + j - 1)]
                    if abovP == "." and leftP == ".":
                        lone = True
                elif i == 1:  # upper border
                    belowP = s_[((i + 1) * size + j)]
                    leftP = s_[((i) * size + j - 1)]
                    rightP = s_[((i) * size + j + 1)]
                    if belowP == "." and leftP == "." and rightP == ".":
                        lone = True
                elif j == 1:  # left border
                    abovP = s_[((i - 1) * size + j)]
                    belowP = s_[((i + 1) * size + j)]
                    rightP = s_[((i) * size + j + 1)]
                    rightP = s_[((i) * size + j + 1)]
                    if abovP == "." and belowP == "." and rightP == ".":
                        lone = True
                elif j == size - 2:  # right border
                    abovP = s_[((i - 1) * size + j)]
                    belowP = s_[((i + 1) * size + j)]
                    leftP = s_[((i) * size + j - 1)]
                    if abovP == "." and belowP == "." and leftP == ".":
                        lone = True
                elif i == size - 2:  # bottom border
                    abovP = s_[((i - 1) * size + j)]
                    leftP = s_[((i) * size + j - 1)]
                    rightP = s_[((i) * size + j + 1)]
                    if abovP == "." and leftP == "." and rightP == ".":
                        lone = True
                elif s_[(i * size + j)] == "S":
                    abovP = s_[((i - 1) * size + j)]
                    belowP = s_[((i + 1) * size + j)]
                    leftP = s_[((i) * size + j - 1)]
                    rightP = s_[((i) * size + j + 1)]
                    if abovP == "." and belowP == "." and leftP == "." and rightP == ".":
                        lone = True
                if lone:
                    one += 1

    return five, four, three, two, one, s_

def remove_check(csp, size, elements):
    """
    This function checks whether ships are placed correctly .
    A helper function for GAC .
    """
    soln_m = []
    for var in csp.variables():
        if int(var._name) > 0:
            soln_m.append((var, var.getValue()))

    s_val = {}
    for (var, val) in soln_m:
        s_val[int(var.name())] = val

    for (i, j, ch) in elements:
        if ch == "M":
            if j == 1 and s_val[int(i * size + j + 1)] == "S":
                return True
            elif i == 1 and s_val[int((i + 1) * size + j)] == "S":
                return True
            elif j == size - 2 and s_val[int(i * size + j - 1)] == "S":
                return True
            elif i == size - 2 and s_val[int((i - 1) * size + j)] == "S":
                return True
        elif ch == "<" and s_val[int(i * size + j - 1)] == "S":
            return True
        elif ch == ">" and s_val[int(i * size + j + 1)] == "S":
            return True
        elif ch == "^" and s_val[int((i - 1) * size + j)] == "S":
            return True
        elif ch == "v" and s_val[int((i + 1) * size + j)] == "S":
            return True
    return False

def GAC(unAssignedVars, csp, final_board, ship_size, size, elements):
    if unAssignedVars.empty():
        soln = []
        for var in csp.variables():
            if int(var._name) > 0:
                soln.append((var, var.getValue()))
        return [soln]
    bt_search.nodesExplored += 1
    solns = []
    nxtvar = unAssignedVars.extract()
    for val in nxtvar.curDomain():
        nxtvar.setValue(val)
        noDWO = True
        if GacEnforce(csp.constraintsOf(nxtvar), csp, nxtvar, val, elements) == "DWO":
            noDWO = False
        if noDWO and not removed(csp, ship_size, size) and not remove_check(csp, size, elements):
            new_solns = GAC(unAssignedVars, csp, final_board, ship_size, size, elements)
            if new_solns:
                five, four, three, two, one, s_ = count_ship_numbers(new_solns[0], size)
                if one == int(ship_size[0]) and two == int(ship_size[1]) and three == int(
                        ship_size[2]) and four == int(ship_size[3]) and five == int(ship_size[4]):
                    match = verify(final_board, s_, size)
                    if (match):
                        solns.extend(new_solns)
                        if len(solns) > 0:
                            break
        nxtvar.restoreValues(nxtvar, val)
    nxtvar.unAssign()
    unAssignedVars.insert(nxtvar)
    return solns

def verify(final_board, s_, size):
    """
    This function verifies if the solution matches the original board.
    """
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            sol_val = s_[(i * size + j)]
            orig_val = final_board[i][j]
            if orig_val != "0" and sol_val != orig_val:
                return False
    return True

def count_ship_numbers(solution, size):
    five = four = three = two = one = 0
    s_ = {}
    for (var, val) in solution:
        s_[int(var.name())] = val

    # First, check for 5-length ships horizontally and vertically
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            # Check for 5-length vertical ships
            if i < (size - 5) and s_[(i * size + j)] == "S" and s_[(i + 1) * size + j] == "S" and s_[(i + 2) * size + j] == "S" and s_[(i + 3) * size + j] == "S" and s_[(i + 4) * size + j] == "S":
                five += 1
                s_[(i * size + j)] = "^"
                s_[(i + 1) * size + j] = "M"
                s_[(i + 2) * size + j] = "M"
                s_[(i + 3) * size + j] = "M"
                s_[(i + 4) * size + j] = "v"
            # Check for 5-length horizontal ships
            elif j < (size - 5) and s_[(i * size + j)] == "S" and s_[(i * size + j + 1)] == "S" and s_[(i * size + j + 2)] == "S" and s_[(i * size + j + 3)] == "S" and s_[(i * size + j + 4)] == "S":
                five += 1
                s_[(i * size + j)] = "<"
                s_[(i * size + j + 1)] = "M"
                s_[(i * size + j + 2)] = "M"
                s_[(i * size + j + 3)] = "M"
                s_[(i * size + j + 4)] = ">"

    # Then, check for 4-length ships
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            # Check for 4-length vertical ships
            if i < (size - 4) and s_[(i * size + j)] == "S" and s_[(i + 1) * size + j] == "S" and s_[(i + 2) * size + j] == "S" and s_[(i + 3) * size + j] == "S":
                four += 1
                s_[(i * size + j)] = "^"
                s_[(i + 1) * size + j] = "M"
                s_[(i + 2) * size + j] = "M"
                s_[(i + 3) * size + j] = "v"
            # Check for 4-length horizontal ships
            elif j < (size - 4) and s_[(i * size + j)] == "S" and s_[(i * size + j + 1)] == "S" and s_[(i * size + j + 2)] == "S" and s_[(i * size + j + 3)] == "S":
                four += 1
                s_[(i * size + j)] = "<"
                s_[(i * size + j + 1)] = "M"
                s_[(i * size + j + 2)] = "M"
                s_[(i * size + j + 3)] = ">"

    # Now, check for 3-length ships
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            # Check for 3-length vertical ships
            if i < (size - 3) and s_[(i * size + j)] == "S" and s_[(i + 1) * size + j] == "S" and s_[(i + 2) * size + j] == "S":
                three += 1
                s_[(i * size + j)] = "^"
                s_[(i + 1) * size + j] = "M"
                s_[(i + 2) * size + j] = "v"
            # Check for 3-length horizontal ships
            elif j < (size - 3) and s_[(i * size + j)] == "S" and s_[(i * size + j + 1)] == "S" and s_[(i * size + j + 2)] == "S":
                three += 1
                s_[(i * size + j)] = "<"
                s_[(i * size + j + 1)] = "M"
                s_[(i * size + j + 2)] = ">"

    # Then, check for 2-length ships
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            # Check for 2-length vertical ships
            if i < (size - 2) and s_[(i * size + j)] == "S" and s_[(i + 1) * size + j] == "S":
                two += 1
                s_[(i * size + j)] = "^"
                s_[(i + 1) * size + j] = "v"
            # Check for 2-length horizontal ships
            elif j < (size - 2) and s_[(i * size + j)] == "S" and s_[(i * size + j + 1)] == "S":
                two += 1
                s_[(i * size + j)] = "<"
                s_[(i * size + j + 1)] = ">"

    # Finally, count individual 'S' for 1-length ships
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            if s_[(i * size + j)] == "S":
                one += 1

    return five, four, three, two, one, s_

def mark_ship(s_, i, j, size, length, direction):
    """
    Marks a ship of the given length and direction on the board (s_).
    A helper function for print and output file methods.
    """
    if direction == "horizontal":
        for k in range(length):
            s_[(i * size + j + k)] = "M"
        s_[(i * size + j)] = "<"
        s_[(i * size + j + length - 1)] = ">"
    elif direction == "vertical":
        for k in range(length):
            s_[((i + k) * size + j)] = "M"
        s_[(i * size + j)] = "^"
        s_[((i + length - 1) * size + j)] = "v"

def check_and_mark_ships(s_, i, j, size, length):
    """
    Checks if the ship can be placed. A helper for print and output methods
    """
    if length == 5:
        if j < (size - 4) and all(s_[(i * size + j + k)] == "S" for k in range(5)):
            mark_ship(s_, i, j, size, 5, "horizontal")
        elif i < (size - 4) and all(s_[(i + k) * size + j] == "S" for k in range(5)):
            mark_ship(s_, i, j, size, 5, "vertical")
    elif length == 4:
        if j < (size - 3) and all(s_[(i * size + j + k)] == "S" for k in range(4)):
            mark_ship(s_, i, j, size, 4, "horizontal")
        elif i < (size - 3) and all(s_[(i + k) * size + j] == "S" for k in range(4)):
            mark_ship(s_, i, j, size, 4, "vertical")
    elif length == 3:
        if j < (size - 2) and all(s_[(i * size + j + k)] == "S" for k in range(3)):
            mark_ship(s_, i, j, size, 3, "horizontal")
        elif i < (size - 2) and all(s_[(i + k) * size + j] == "S" for k in range(3)):
            mark_ship(s_, i, j, size, 3, "vertical")
    elif length == 2:
        if j < (size - 1) and s_[(i * size + j)] == "S" and s_[(i * size + j + 1)] == "S":
            mark_ship(s_, i, j, size, 2, "horizontal")
        elif i < (size - 1) and s_[(i * size + j)] == "S" and s_[(i + 1) * size + j] == "S":
            mark_ship(s_, i, j, size, 2, "vertical")

def output_to_file(filename, sol, size):
    f = open(filename, "a")
    f.seek(0)
    f.truncate()

    s_ = {int(var.name()): val for var, val in sol}

    for i in range(1, size - 1):
        for j in range(1, size - 1):
            for length in [5, 4, 3, 2]:  # Check larger ships first
                check_and_mark_ships(s_, i, j, size, length)

    for i in range(1, size - 1):
        for j in range(1, size - 1):
            f.write(s_[(i * size + j)])
        if i != (size - 2):
            f.write("\n")
    f.close()

def print_solution(s, size):
    """Print solution to the terminal """
    s_ = {int(var.name()): val for var, val in s}

    for i in range(1, size - 1):
        for j in range(1, size - 1):
            for length in [5, 4, 3, 2]:  # Check larger ships first
                check_and_mark_ships(s_, i, j, size, length)

    # Output the grid
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            print(s_[(i * size + j)], end="")
        if i != (size - 2):
            print()

