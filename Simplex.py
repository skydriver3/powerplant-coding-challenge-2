import numpy as np
from scipy.optimize import linprog
import unittest

def simplex(c, A_ub=None, b_ub=None, bounds=None, A_eq=None, b_eq=None):
    '''
    The function does not return the correct answer all of the time, it still needs further debugging  
    '''
    n = c.shape[0]
    m_ub = A_ub.shape[0] if A_ub is not None else 0
    m_eq = A_eq.shape[0] if A_eq is not None else 0
    m = m_ub + m_eq
    
    # Add slack variables to turn inequality constraints into equality constraints
    if A_ub is not None:
        A_ub = np.hstack([A_ub, np.eye(m_ub)])
    
    # Combine inequality and equality constraints
    if A_ub is not None and A_eq is not None:
        A = np.vstack([A_ub, np.hstack([A_eq, np.zeros((m_eq, m_ub))])])
        b = np.hstack([b_ub, b_eq])
    elif A_ub is not None:
        A = A_ub
        b = b_ub
    elif A_eq is not None:
        A = A_eq
        b = b_eq
    else:
        A = np.empty((0, n + m_ub))
        b = np.empty(0)
    
    # Add bounds to the problem
    if bounds is not None:
        for i in range(n):
            if bounds[i][0] is not None:
                # Add lower bound
                new_row = -np.eye(n + m_ub)[i, :]
                A = np.vstack([A, new_row])
                b = np.hstack([b, -bounds[i][0]])
                m += 1
            if bounds[i][1] is not None:
                # Add upper bound
                new_row = np.eye(n + m_ub)[i, :]
                A = np.vstack([A, new_row])
                b = np.hstack([b, bounds[i][1]])
                m += 1
 
    # Phase 1: Solve auxiliary problem to find initial basic feasible solution
    
    # Add artificial variables to turn problem into standard form
    A_aux = np.hstack([A, np.eye(m)])
    
    # Define objective function for auxiliary problem
    c_aux = np.hstack([np.zeros(n + m_ub), np.ones(m)])
    
    # Initialize the tableau for the auxiliary problem
    tableau_aux = np.vstack([np.hstack([A_aux, b.reshape(-1, 1)]), np.hstack([c_aux, 0])])
    
    # Iterate until an optimal solution is found for the auxiliary problem
    while True:
        # Find the column with the most negative reduced cost
        pivot_col_aux = np.argmin(tableau_aux[-1,:-1])

        # If all reduced costs are non-negative, we have found an optimal solution for the auxiliary problem
        if tableau_aux[-1,pivot_col_aux] >= 0:
            break

        # Find the row with the smallest ratio of b to the corresponding element in the pivot column
        ratios_aux = tableau_aux[:-1,-1] / tableau_aux[:-1,pivot_col_aux]
        ratios_aux[tableau_aux[:-1,pivot_col_aux] <= 0] = np.inf
        pivot_row_aux = np.argmin(ratios_aux)

        # Perform a pivot operation to move the entering variable into the basis
        pivot_element_aux = tableau_aux[pivot_row_aux,pivot_col_aux]
        tableau_aux[pivot_row_aux] /= pivot_element_aux
        for i in range(tableau_aux.shape[0]):
            if i != pivot_row_aux:
                tableau_aux[i] -= tableau_aux[pivot_row_aux] * tableau_aux[i,pivot_col_aux]
    
    # Check if original problem is infeasible
    if not np.isclose(tableau_aux[-1,-1], 0):
        raise ValueError("Problem is infeasible")
    
    # Remove artificial variables from tableau
#    tableau = tableau_aux[:,:n+m_ub +1]
    tableau = np.hstack([tableau_aux[:,:n+m_ub], tableau_aux[:, -1:]]) 

    # Phase 2: Solve original problem using simplex method
    
    # Compute initial values of reduced costs for original problem
    c_padded = np.hstack([c, np.zeros(m_ub)])
    for j in range(n+m_ub):
        if (tableau[:-1,j] == 0).all():
            continue
        
        pivot_row = np.where(tableau[:-1,j] == 1)[0][0]
        
        pivot_element = tableau[pivot_row,j]
        tableau[pivot_row] /= pivot_element
        
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i] -= tableau[pivot_row] * tableau[i,j]
    
    tableau[-1,:-1] = c_padded - np.dot(tableau[:-1,:-1].T, tableau[:-1,-1])
    
    # Iterate until an optimal solution is found for the original problem
    while True:
        # Find the column with the most negative reduced cost
        pivot_col = np.argmin(tableau[-1,:-1])

        # If all reduced costs are non-negative, we have found an optimal solution for the original problem
        if tableau[-1,pivot_col] >= 0:
            break

        # Find the row with the smallest ratio of b to the corresponding element in the pivot column
        ratios = tableau[:-1,-1] / tableau[:-1,pivot_col]
        ratios[tableau[:-1,pivot_col] <= 0] = np.inf
        pivot_row = np.argmin(ratios)

        # Perform a pivot operation to move the entering variable into the basis
        pivot_element = tableau[pivot_row,pivot_col]
        tableau[pivot_row] /= pivot_element
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i] -= tableau[pivot_row] * tableau[i,pivot_col]

    # Extract the solution from the final tableau
    solution = np.zeros(n)
    for i in range(n):
        col = tableau[:,i]
        if (col[:-1] == 1).sum() == 1 and (col[:-1] == 0).sum() == m - 1:
            solution[i] = tableau[np.where(col[:-1] == 1)[0][0],-1]

    return solution


class TestSimplex(unittest.TestCase):
    def test_simplex_case1(self):
        # Define the problem
        c = np.array([3, 2])
        A = np.array([[2, 1], [1, 1], [1, 0]])
        b = np.array([4, 3, 2])
        bounds = [(0, 1.2), (1.8, 2.5)]
        A_eq = None
        b_eq = None

        # Solve the problem using the simplex method
        solution_simplex = simplex(c=c,A_ub=A,b_ub=b,bounds=bounds,A_eq=A_eq,b_eq=b_eq)

        # Solve using linprog function
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds,A_eq=A_eq,b_eq=b_eq)
        solution_linprog = res.x

        # Check if the solutions are equal
        np.testing.assert_allclose(solution_simplex, solution_linprog, rtol=1e-5)

    def test_simplex_case2(self):
        # Define the problem
        c = np.array([-1, 4])
        A = np.array([[-3, 1], [1, 2]])
        b = np.array([6, 4])
        bounds = [(None, None), (-3, None)]
        A_eq = None
        b_eq = None

        # Solve the problem using the simplex method
        solution_simplex = simplex(c=c,A_ub=A,b_ub=b,bounds=bounds,A_eq=A_eq,b_eq=b_eq)

        # Solve using linprog function
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds,A_eq=A_eq,b_eq=b_eq)
        solution_linprog = res.x

        # Check if the solutions are equal
        np.testing.assert_allclose(solution_simplex, solution_linprog, rtol=1e-5)

    def test_simplex_case3(self):
        # Define the problem
        c = np.array([2,-3])
        A = None
        b = None
        bounds = [(0,None),(0,None)]
        A_eq = np.array([[3,-2]])
        b_eq = np.array([-4])

        # Solve the problem using the simplex method
        solution_simplex = simplex(c=c,A_ub=A,b_ub=b,bounds=bounds,A_eq=A_eq,b_eq=b_eq)

        # Solve using linprog function
        res = linprog(c=c,A_ub=A,b_ub=b,bounds=bounds,A_eq=A_eq,b_eq=b_eq)
        solution_linprog=res.x

         # Check if the solutions are equal
        np.testing.assert_allclose(solution_simplex,solution_linprog)

if __name__ == '__main__':
    unittest.main()