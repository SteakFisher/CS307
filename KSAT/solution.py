import random

def create_k_sat_formula(k, m, n):
    formula = []

    for i in range(m):

        clause = []

        variables = random.sample(range(1, n+1), k)

        for variable in variables:
            if random.random() < 0.5:
                clause.append(variable)
            else:
                clause.append(-variable)

        formula.append(clause)

    return formula

print("Enter k, m, and n respectively: ")

k, m, n = map(int, input().split())

formula = create_k_sat_formula(k, m, n)

print(formula)
