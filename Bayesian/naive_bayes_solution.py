from bnetbase import Variable, Factor, BN
import csv
import itertools

def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object. 
    :return: a new Factor object resulting from normalizing factor.
    '''
    #Name change

    total_sum = sum(factor.values)
    if total_sum == 0:
        raise ValueError("Cannot normalize a factor with total value sum of 0.")

    # Step 2: Normalize each value in the factor
    normalized_values = []
    for value in factor.values:
        normalized_values.append(value / total_sum )

    # Step 3: Create a new factor with the same scope and normalized values
    normalized_factor = Factor(f"P({factor.name}_normalized)", factor.get_scope())
    normalized_factor.values = normalized_values

    return normalized_factor

def restrict(factor, variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.

    '''
    # Create the new scope without the restricted variable
    scope = factor.get_scope()
    restrict_idx = scope.index(variable)
    new_scope = [var for var in scope if var != variable]

    # Name the restricted factor
    new_factor_name = f"P({factor.name}|{variable.name}={value})"
    restricted_factor = Factor(new_factor_name, new_scope)

    # Prepare domains and assignments
    full_assignments = itertools.product(*(var.domain() for var in scope))
    restricted_values = []

    for assignment in full_assignments:
        if assignment[restrict_idx] == value:
            # Remove the restricted variable's value from the assignment
            restricted_assignment = assignment[:restrict_idx] + assignment[restrict_idx + 1:]
            restricted_values.append((restricted_assignment, factor.get_value(assignment)))

    # Add the restricted values to the new factor
    values = []
    for restricted_assignment, factor_value in restricted_values:
        values.append(list(restricted_assignment) + [factor_value])
    restricted_factor.add_values(values)
    return restricted_factor

def sum_out(factor, variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''
    values = factor.values
    scope = factor.get_scope()
    var_idx = scope.index(variable)

    # Calculate the gap between consecutive values of `variable` in the flattened list
    gap = 1
    for i in range(var_idx + 1, len(scope)):
        gap *= scope[i].domain_size()

    new_values = []
    offset = 0

    # Marginalize over all values of `variable`
    while offset < len(values):
        # Sum out `variable` for one unit and append to `new_values`
        new_values.extend(
            sum(values[i + j * gap] for j in range(variable.domain_size()))
            for i in range(offset, gap + offset)
        )
        offset += gap * variable.domain_size()

    # Create a new scope without the summed-out variable
    new_scope = scope[:var_idx] + scope[var_idx + 1:]

    # Construct the new factor
    new_factor_name = f"P({factor.name}_sum_{variable.name})"
    marginalized_factor = Factor(new_factor_name, new_scope)
    marginalized_factor.values = new_values


    return marginalized_factor

def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors. 

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''
    result_factor = factor_list[0]

    # Iterate through the remaining factors and multiply them with the result_factor
    for factor in factor_list[1:]:
        result_factor = _multiply(result_factor, factor)

    return result_factor

def _multiply(f1: Factor, f2: Factor) -> Factor:
    # Get the scope for the new factor (merge scopes from both f1 and f2)
    new_scope = f1.get_scope()

    # Get the common variables between f1 and f2
    f1cdx, f2cdx = [], []
    for i, v1 in enumerate(f1.get_scope()):
        for j, v2 in enumerate(f2.get_scope()):
            if v1 == v2:
                f1cdx.append(i)
                f2cdx.append(j)

    # Append variables from f2's scope to new_scope if not already present in f1's scope
    for v in f2.get_scope():
        if v not in new_scope:
            new_scope.append(v)

    # Find all assignments of f1 and f2
    f1_all_assignments = tuple(itertools.product(*(v.domain() for v in f1.get_scope())))
    f2_all_assignments = tuple(itertools.product(*(v.domain() for v in f2.get_scope())))

    new_assignments = []


    for f1_a in f1_all_assignments:
        for f2_a in f2_all_assignments:
            # Skip if the common variables in f1 and f2 have different values
            if any(f1_a[f1cdx[i]] != f2_a[f2cdx[i]] for i in range(len(f1cdx))):
                continue

            res = f1.get_value(f1_a) * f2.get_value(f2_a)

            new_a = list(f1_a) + [f2_a[i] for i in range(len(f2.get_scope())) if i not in f2cdx] + [res]
            new_assignments.append(new_a)

    # Create a new Factor with the merged scope and computed assignments
    new_factor_name = f'({f1.name} * {f2.name})'
    new_factor = Factor(new_factor_name, new_scope)
    new_factor.add_values(new_assignments)

    return new_factor

def ve(bayes_net, var_query, EvidenceVars):
    '''

    Execute the variable elimination algorithm on the Bayesian network bayes_net
        to compute a distribution over the values of var_query given the
        evidence provided by EvidenceVars.

        :param bayes_net: a BN object.
        :param var_query: the query variable. we want to compute a distribution
                         over the values of the query variable.
        :param EvidenceVars: the evidence variables. Each evidence variable has
                             its evidence set to a value from its domain
                             using set_evidence.
        :return: a Factor object representing a distribution over the values
                 of var_query. that is a list of numbers, one for every value
                 in var_query's domain. These numbers sum to 1. The i-th number
                 is the probability that var_query is equal to its i-th value given
                 the settings of the evidence variables.

        For example, assume that
            var_query = A with Dom[A] = ['a', 'b', 'c'],
            EvidenceVars = [B, C], and
            we have called B.set_evidence(1) and C.set_evidence('c'),
        then VE would return a list of three numbers, e.g. [0.5, 0.24, 0.26].
        These numbers would mean that
            Pr(A='a'|B=1, C='c') = 0.5,
            Pr(A='a'|B=1, C='c') = 0.24, and
            Pr(A='a'|B=1, C='c') = 0.26.

    '''
    # Step 1: Restrict factors based on evidence variables
    factors = bayes_net.factors()
    for evidence_var in EvidenceVars:
        for i in range(len(factors)):
            if evidence_var not in factors[i].get_scope():
                continue
            factors[i] = restrict(factors[i], evidence_var, evidence_var.get_evidence())

    scopes = [set(f.get_scope()) for f in factors]
    hidden_variables = set().union(*scopes) - {var_query}
    for hidden_var in hidden_variables:

        related_factors = [f for f in factors if hidden_var in f.get_scope()]
        if not related_factors:
            continue

        combined_factor = multiply(related_factors)

        summed_factor = sum_out(combined_factor, hidden_var)

        factors.append(summed_factor)
        factors = [f for f in factors if f not in related_factors]

    # Step 3: Multiply remaining factors to form the final factor
    final_factor = multiply(factors)

    # Step 4: Normalize the factor to get probabilities for the query variable
    normalized_factor = normalize(final_factor)

    return normalized_factor

def naive_bayes_model(data_file):
    '''
       NaiveBayesModel returns a BN that is a Naive Bayes model that
       represents the joint distribution of value assignments to
       variables in the Adult Dataset from UCI.  Remember a Naive Bayes model
       assumes P(X1, X2,.... XN, Class) can be represented as
       P(X1|Class)*P(X2|Class)* .... *P(XN|Class)*P(Class).
       When you generated your Bayes bayes_net, assume that the values
       in the SALARY column of the dataset are the CLASS that we want to predict.
       @return a BN that is a Naive Bayes model and which represents the Adult Dataset.
    '''
    input_data = []
    with open('data/adult-train.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)  # skip header row
        # each row is a list of str
        for row in reader:
            input_data.append(row)

    class_var = Variable("Salary", ['<50K', '>=50K'])
    ### DOMAIN INFORMATION REFLECTS ORDER OF COLUMNS IN THE DATA SET
    variable_domains = {
        "Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
        "Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
        "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
        "Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
        "Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
        "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
        "Gender": ['Male', 'Female'],
        "Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
        "Salary": ['<50K', '>=50K']
    }
    variables = []
    for key, domain in variable_domains.items():
        variables.append(Variable(key, domain))


    # create factors
    f1 = Factor('Work', [variables[0], variables[8]])
    f2 = Factor('Education', [variables[1], variables[8]])
    f3 = Factor('MartialStatus', [variables[2], variables[8]])
    f4 = Factor('Occupation', [variables[3], variables[8]])
    f5 = Factor('Relationship', [variables[4], variables[8]])
    f6 = Factor('Race', [variables[5], variables[8]])
    f7 = Factor('Gender', [variables[6], variables[8]])
    f8 = Factor('Country', [variables[7], variables[8]])
    f9 = Factor('Salary', [variables[8]])
    factors = [f1,f2,f3,f4,f5,f6,f7,f8]

    # salary count
    n = len(input_data)
    salary_count = {}  # Standard dictionary
    for work, edu, marry, occup, rell, race, gender, country, salary in input_data:
        salary_count[salary] = salary_count.get(salary, 0) + 1
    # other
    other_count = [{} for _ in range(8)]
    for work, edu, marry, occup, rell, race, gender, country, s in input_data:
        other_count[0][(work, s)] = other_count[0].get((work, s), 0) + 1
        other_count[1][(edu, s)] = other_count[1].get((edu, s), 0) + 1
        other_count[2][(marry, s)] = other_count[2].get((marry, s), 0) + 1
        other_count[3][(occup, s)] = other_count[3].get((occup, s), 0) + 1
        other_count[4][(rell, s)] = other_count[4].get((rell, s), 0) + 1
        other_count[5][(race, s)] = other_count[5].get((race, s), 0) + 1
        other_count[6][(gender, s)] = other_count[6].get((gender, s), 0) + 1
        other_count[7][(country, s)] = other_count[7].get((country, s), 0) + 1

    for i in range(8):
        values = [[key[0], key[1], value / salary_count[key[1]]] for key, value in
                  other_count[i].items()]
        factors[i].add_values(values)

    # initialize salary factor
    f9.add_values([[key, value / n] for key, value in salary_count.items()])
    factors.append(f9)

    # create BN
    bn = BN('bayes_net',variables, factors)
    return bn

def explore(bayes_net, question):
    ''' Input: bayes_net---a BN object (a Bayes bayes_net)
        question---an integer indicating the question in HW4 to be calculated. Options are:
        1. What percentage of the women in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
        2. What percentage of the men in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
        3. What percentage of the women in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
        4. What percentage of the men in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
        5. What percentage of the women in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
        6. What percentage of the men in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
        @return a percentage (between 0 and 100)
    '''

    # Evidence variable groups
    E1v = [bayes_net.get_variable("Work"),
           bayes_net.get_variable("Occupation"),
           bayes_net.get_variable("Education"),
           bayes_net.get_variable("Relationship")]

    E2v = E1v + [bayes_net.get_variable("Gender")]

    # Read input data
    input_data = []
    with open("data/adult-test.csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)  # Skip header row
        input_data = list(reader)  # Load all rows into memory

    # Count the total number of men and women
    n_man = sum(1 for row in input_data if row[6] == 'Male')
    n_woman = len(input_data) - n_man

    # Helper function to set evidence
    def set_evidence(row):
        bayes_net.get_variable("Work").set_evidence(row[0])
        bayes_net.get_variable("Occupation").set_evidence(row[3])
        bayes_net.get_variable("Education").set_evidence(row[1])
        bayes_net.get_variable("Relationship").set_evidence(row[4])
        if len(row) > 6:  # Set Gender only when needed
            bayes_net.get_variable("Gender").set_evidence(row[6])

    # Question-specific computations
    count, total = 0, 0
    for row in input_data:
        gender, salary = row[6], row[8]
        if (question in [1, 3, 5] and gender != 'Female') or (question in [2, 4, 6] and gender != 'Male'):
            continue

        set_evidence(row)

        p1 = ve(bayes_net, bayes_net.get_variable("Salary"), E1v).values[1]
        if question in [1, 2]:  # Compare P(S|E1) > P(S|E2)
            p2 = ve(bayes_net, bayes_net.get_variable("Salary"), E2v).values[1]
            if p1 > p2:
                count += 1
        elif question in [3, 4]:  # P(S|E1) > 0.5 and actual salary check
            if p1 > 0.5:
                total += 1
                count += int(salary == '>=50K')
        elif question in [5, 6]:  # P(S|E1) > 0.5 only
            if p1 > 0.5:
                count += 1

    # Calculate percentage for the given question
    if question in [1, 2]:  # Percentage with P(S|E1) > P(S|E2)
        return count / (n_woman if question == 1 else n_man) * 100
    elif question in [3, 4]:  # Percentage with actual salary >=50K
        return count / total * 100 if total > 0 else 0
    elif question in [5, 6]:  # Percentage assigned P(S|E1) > 0.5
        return count / (n_woman if question == 5 else n_man) * 100

if __name__ == '__main__':
    nb = naive_bayes_model('data/adult-train.csv')
    for i in range(1,7):
        print("explore(nb,{}) = {}".format(i, explore(nb, i)))
