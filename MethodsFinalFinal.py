import streamlit as st

class Problem: # A class to store problem's information
    def __init__(self):
        self.goal = 0  # Goal = 1: Minimize, Goal = 2: Maximize
        self.var_amount = 0  # Number of variables
        self.vars = []  # Variables
        self.obj_idx = []  # Objective index
        self.var_constrs = []  # Variable constraints
        self.eq_sign_constrs = []  # Equality/inequality's sign constraints
        self.f_coeffs = []  # Free coefficients
        self.sign_constrs = []  # Variable's sign constraints

    def set_goal(self, goal):
        self.goal = goal

    def get_goal(self):
        return self.goal

    def set_var_amount(self):
        self.var_amount = len(self.obj_idx)

    def get_var_amount(self):
        return self.var_amount

    def set_object_index(self, obj_idx):
        self.obj_idx = obj_idx

    def get_object_index(self):
        return self.obj_idx

    def set_var_constraints(self, var_constrs):
        self.var_constrs = var_constrs

    def get_var_constrs(self):
        return self.var_constrs

    def set_eq_sign_constrs(self, eq_sign_constrs):
        self.eq_sign_constrs = eq_sign_constrs

    def get_eq_sign_constrs(self):
        return self.eq_sign_constrs

    def set_f_coeffs(self, f_coeffs):
        self.f_coeffs = f_coeffs

    def get_f_coeffs(self):
        return self.f_coeffs

    def set_sign_constrs(self, sign_constrs):
        self.sign_constrs = sign_constrs

    def get_sign_constrs(self):
        return self.sign_constrs

    def add_variables(self):
        for i in range(self.var_amount):
            self.vars.append(f"x{i + 1}")

    def set_variables(self, vars):
        self.vars = vars

    def get_variables(self):
        return self.vars

    def display(self):
        # Print problem's objective function
        st.header("Problem:")
        obj_function_str = f"{'Minimize' if self.goal == 1 else 'Maximize'} "
        obj_function_str += " ".join([
            f"{'' if i == 0 and coeff > 0 else '+' if coeff > 0 else ''}{coeff:.4f}{var}"
            for i, (coeff, var) in enumerate(zip(self.obj_idx, self.vars)) if coeff != 0
        ])
        st.write(f"**{obj_function_str}**")
        st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

        # Print constraints
        constraints = []
        for i, (constr, sign, free_coeff) in enumerate(zip(self.var_constrs, self.eq_sign_constrs, self.f_coeffs)):
            constraint_str = " ".join([
                f"{'' if j == 0 and coeff > 0 else '+' if coeff > 0 else ''}{coeff:.4f}{var}"
                for j, (coeff, var) in enumerate(zip(constr, self.vars)) if coeff != 0
            ])
            constraint_str += f" {sign} {free_coeff:.4f}"
            constraints.append(constraint_str)
        
        # Use tables to align constraints
        st.write("### Constraints")
        constraint_table = "| Constraints |\n|-------------|\n"
        for constraint in constraints:
            constraint_table += f"| {constraint} |\n"
        st.markdown(constraint_table)

        # Print sign constraints
        sign_constraints_str = ", ".join([
            f"{var} {sign} 0"
            for var, sign in zip(self.vars, self.sign_constrs) if sign in ["<=", ">="]
        ])
        st.write(f"**Variable Sign Constraints:** {sign_constraints_str}")


    def normalize_problem(self):
        nor_prob = Problem()
        nor_prob.set_goal(1)

        nor_prob.obj_idx = self.get_object_index()
        nor_prob.set_var_amount()
        nor_prob.var_constrs = self.get_var_constrs()
        nor_prob.eq_sign_constrs = self.get_eq_sign_constrs()
        nor_prob.f_coeffs = self.get_f_coeffs()
        nor_prob.sign_constrs = self.get_sign_constrs()
        nor_prob.vars = self.get_variables()

        #Add the amount of variables
        for sign in nor_prob.sign_constrs:
            if sign == "f":
                nor_prob.var_amount += 1

        #Handle with free variables
        idx = 0
        while idx < nor_prob.var_amount:
            if nor_prob.sign_constrs[idx] == "f":
                nor_prob.vars[idx] = f"x{idx + 1}+"
                temp = f"x{idx + 1}-"
                nor_prob.vars.insert(idx + 1, temp)
                nor_prob.sign_constrs[idx] = ">="
                nor_prob.sign_constrs.insert(idx + 1, ">=")
                nor_prob.obj_idx.insert(idx + 1, -nor_prob.obj_idx[idx])

                for i in range(len(nor_prob.var_constrs)):
                    nor_prob.var_constrs[i].insert(idx + 1, -nor_prob.var_constrs[i][idx])
            idx += 1

        #Handle with "<=" sign constraints
        for i in range(len(nor_prob.sign_constrs)):
            if nor_prob.sign_constrs[i] == "<=":
                nor_prob.vars[i] = f"y{i + 1}"
                nor_prob.sign_constrs[i] = ">="
                nor_prob.obj_idx[i] = -nor_prob.obj_idx[i]
                for j in range(len(nor_prob.var_constrs)):
                    nor_prob.var_constrs[j][i] = -nor_prob.var_constrs[j][i]

        #Handle with ">=" inequality constraints
        for i in range(len(nor_prob.eq_sign_constrs)):
            if nor_prob.eq_sign_constrs[i] == ">=":
                for j in range(len(nor_prob.var_constrs[i])):
                    nor_prob.var_constrs[i][j] = -nor_prob.var_constrs[i][j]
                nor_prob.eq_sign_constrs[i] = "<="
                nor_prob.f_coeffs[i] = -nor_prob.f_coeffs[i]

        #Handle with "=" equality constraints
        temp = []
        for i in range(len(nor_prob.eq_sign_constrs)):
            if nor_prob.eq_sign_constrs[i] == "=":
                temp = nor_prob.var_constrs[i].copy()
                for j in range(len(temp)):
                    temp[j] = -temp[j]
                nor_prob.eq_sign_constrs[i] = "<="

                nor_prob.var_constrs.append(temp)
                nor_prob.eq_sign_constrs.append("<=")
                nor_prob.f_coeffs.append(-self.f_coeffs[i])

        # Handle with the objective function
        if self.get_goal() == 2:
            for i in range(len(nor_prob.obj_idx)):
                nor_prob.obj_idx[i] = -nor_prob.obj_idx[i]

        return nor_prob


class Dict: # A class to store dictionary information
    def __init__(self):
        self.dep_var = []  # Basic variables
        self.indep_var = []  # Non-basic variables
        self.indep_var_obj = []  # Objective function values
        self.eq_constrs_matrix = []  # Equality constraints matrix

    def set_dep_var(self, dep_var):
        self.dep_var = dep_var

    def get_dep_var(self):
        return self.dep_var

    def set_indep_var(self, indep_var):
        self.indep_var = indep_var

    def get_indep_var(self):
        return self.indep_var

    def set_indep_var_obj(self, indep_var_obj):
        self.indep_var_obj = indep_var_obj

    def get_indep_var_obj(self):
        return self.indep_var_obj

    def set_eq_constrs_matrix(self, eq_constrs_matrix):
        self.eq_constrs_matrix = eq_constrs_matrix

    def get_eq_constrs_matrix(self):
        return self.eq_constrs_matrix
    
    def choose_pivot(self): # Choose the pivot row and column
        col = -1
        var = ""
        # Get the first variable which has the c_j negative value
        for i in range(1, len(self.indep_var_obj)):
            if self.indep_var_obj[i] < 0:
                col = i
                var = self.indep_var[i]
                break

        # Get the actual min variable which has the c_j negative value, which means to choose the pivot column
        # Decending variable priority order: x0 -> x1 -> x1+ -> x1- -> ... xn- -> w1 -> w2 -> ... wn
        for i in range(col + 1, len(self.indep_var_obj)):
            if self.indep_var_obj[i] < 0:
                if self.indep_var[i][0] != var[0]:
                    if self.indep_var[i] > var:
                        row = i
                        var = self.indep_var[i]
                if self.indep_var[i][0] == var[0]:
                    if self.indep_var[i] < var:
                        row = i
                        var = self.indep_var[i]
        
        row = -1
        min_val = float("inf")
        # Get the row of the first value of b_i / a_ij, a_ij is negative
        for i in range(len(self.eq_constrs_matrix)):
            if self.eq_constrs_matrix[i][col] >= 0: # Skip if a_ij >= 0
                continue
            min_val = self.eq_constrs_matrix[i][0] / -self.eq_constrs_matrix[i][col]
            row = i
            var = self.dep_var[i + 1] # Get the variable name which depends on pivot column's variable
            break
            
        # Get the actual min b_i / a_ij value, a_ij is negative, which means to choose the pivot row
        for i in range(row + 1, len(self.eq_constrs_matrix)):
            if self.eq_constrs_matrix[i][col] >= 0:
                continue
            if self.eq_constrs_matrix[i][0] / -self.eq_constrs_matrix[i][col] < min_val:
                row = i
                min_val = self.eq_constrs_matrix[i][0] / -self.eq_constrs_matrix[i][col]
                var = self.dep_var[i + 1]

        # If many rows have the same b_i / a_ij value, choose the one which has the smallest variable name
        # Decending variable priority order: x0 -> x1 -> x1+ -> x1- -> ... xn- -> w1 -> w2 -> ... wn
        for i in range(len(self.eq_constrs_matrix)):
            if self.eq_constrs_matrix[i][col] >= 0:
                continue
            if self.eq_constrs_matrix[i][0] / -self.eq_constrs_matrix[i][col] == min_val:
                if self.dep_var[i + 1][0] != var[0]:
                    if self.dep_var[i + 1] > var:
                        row = i
                        var = self.dep_var[i + 1]
                if self.dep_var[i + 1][0] == var[0]:
                    if self.dep_var[i + 1] < var:
                        row = i
                        var = self.dep_var[i + 1]
         
        # Return the chosen row and column
        pos = [row, col]
        return pos

    def pivot_dict(self, row, col): # Pivot the dictionary
        self.dep_var[row + 1], self.indep_var[col] = self.indep_var[col], self.dep_var[row + 1]  # Swap variables
        value = self.eq_constrs_matrix[row][col]  # Get pivot value
        self.eq_constrs_matrix[row][col] = -1  # Set pivot value to -1
        self.eq_constrs_matrix[row] = [x / -value for x in self.eq_constrs_matrix[row]]  # Update the row

        # Update the objective function
        value = self.indep_var_obj[col]
        self.indep_var_obj[col] = 0
        self.indep_var_obj = [x + value * y for x, y in zip(self.indep_var_obj, self.eq_constrs_matrix[row])]

        # Update the equality constraints matrix
        for i in range(len(self.eq_constrs_matrix)):
            if i == row:
                continue
            value = self.eq_constrs_matrix[i][col]
            self.eq_constrs_matrix[i][col] = 0
            self.eq_constrs_matrix[i] = [x + value * y for x, y in zip(self.eq_constrs_matrix[i], self.eq_constrs_matrix[row])]

    def display(self):
        # Print the objective function
        obj_function_str = f"{self.dep_var[0]} = " + " ".join([
            f"{'' if i == 0 and val >= 0 else '+' if val >= 0 else ''}{val:.4f}{var}"
            for i, (val, var) in enumerate(zip(self.indep_var_obj, self.indep_var))
        ])

        # Use HTML to align table
        # <hr>  style='margin:10px 0;' Change the distance between the objective function and the constraints
        st.markdown("### Hàm mục tiêu")
        st.markdown(f"<div style='margin:0;'>{obj_function_str}</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

        # Print constraints
        st.markdown("### Các ràng buộc")
        
        # Use table to align constraints
        constraints_table = "<table style='width:100%; border-collapse: collapse;'>"
        for i, row in enumerate(self.eq_constrs_matrix):
            constraint_str = f"{self.dep_var[i + 1]} = " + " ".join([
                f"{'' if j == 0 and val >= 0 else '+' if val >= 0 else ''}{val:.4f}{var:<15}"
                for j, (val, var) in enumerate(zip(row, self.indep_var))
            ])
            constraint_str = f"<pre>{constraint_str}</pre>"  # Keep the original format
            constraints_table += f"<tr><td style='padding: 5px; border: 1px solid black;'>{constraint_str}</td></tr>"
        constraints_table += "</table>"
        st.markdown(constraints_table, unsafe_allow_html=True)


class Phase1(Dict):
    def __init__(self, nor_Prob = None):
        super().__init__()
        self.status = 1  # 1: Feasible, 2: Infeasible
        self.times = 0
        if nor_Prob:
            self._initialize(nor_Prob)

    def _initialize(self, nor_Prob):
        self.indep_var = [''] + nor_Prob.get_variables() + ['x0']
        self.indep_var_obj = [0] * (len(self.indep_var) - 1) + [1]
        self.dep_var = ['*'] + [f'w{i + 1}' for i in range(len(nor_Prob.get_var_constrs()))]
        eq_matrix = [row + [-1] for row in nor_Prob.get_var_constrs()]
        f_coeffs = nor_Prob.get_f_coeffs()
        self.eq_constrs_matrix = [[f_coeff] + [-val for val in row] for f_coeff, row in zip(f_coeffs, eq_matrix)]

    def set_status(self, status):
        self.status = status

    def set_times(self,times):
        self.times = times

    def get_status(self):
        return self.status
    
    def get_times(self):
        return self.times

    def solve(self):
        result = Dict()
        times = 0

        st.header("\n--Phase 1--\n")
        st.header(f"\nTimes: {times}")
        self.display()
        times += 1
        
        all_positive = True
        
        # Check if all b_i values are not negative, which means the problem is feasible and we can go to Phase 2
        for row in self.eq_constrs_matrix:
            if row[0] < 0:
                all_positive = False
                break

        if all_positive:
            result.set_dep_var(self.get_dep_var())
            result.set_indep_var(self.get_indep_var())
            result.set_indep_var_obj(self.get_indep_var_obj())
            result.set_eq_constrs_matrix(self.get_eq_constrs_matrix())
            self.set_status(1)
            self.set_times(times)
            return

        # If there're not, begin Phase 1
        # Pivot for the first time
        min_val = float("inf")
        row = -1
        for i, row_values in enumerate(self.eq_constrs_matrix):
            if row_values[0] < min_val:
                min_val = row_values[0]
                row = i # Get min value of b_i

        self.pivot_dict(row, len(self.get_indep_var()) - 1) # Pivot at that position
        st.header(f"\nTimes: {times}")
        self.display()
        times += 1

        #Pivot from the second time
        eps =1e-5
        while True:
            all_positive = all(val >= 0 for val in self.indep_var_obj[1:]) # Check if all values of c_j are not negative
            if all_positive:
                self.set_status(1 if abs(self.indep_var_obj[0]) < eps else 2) # If c_0 = 0, the problem is feasible, otherwise it's infeasible
                self.set_times(times)
                return
            
            # Get pivot position
            [row, col] = self.choose_pivot()

            self.pivot_dict(row, col) # Pivot at that the chosen row and column
            st.header(f"\nTimes: {times}")
            self.display()
            times += 1


class Phase2(Dict):
    def __init__(self, result = None):
        super().__init__()
        self.status = 1  # 1: Optimal, 2: Feasible, 3: Unbounded
        self.times = 0
        if result:
            self.dep_var = result.get_dep_var()
            self.indep_var = result.get_indep_var()
            self.indep_var_obj = result.get_indep_var_obj()
            self.eq_constrs_matrix = result.get_eq_constrs_matrix()

    def set_status(self, status):
        self.status = status

    def set_times(self,times):
        self.times = times

    def get_status(self):
        return self.status
    
    def get_times(self):
        return self.times

    def handle_vars(self, nor_Prob): # Get the dictionary from the result of Phase 1, suppose the problem is feasible
        self.dep_var[0] = "z"  # Change the objective function name to "z"

        # Remove the variables which are not basic variables and their objective function values in Phase 1 are not 0
        for i in range(len(self.indep_var_obj) - 1, 0, -1): # Remove from the end to the beginning
            if self.indep_var_obj[i] != 0:
                self.indep_var.pop(i)
                self.indep_var_obj.pop(i)
                for j in range(len(self.eq_constrs_matrix)):
                    self.eq_constrs_matrix[j].pop(i)

        # Combine the constraints in Phase 1 with the objective function in the normalized problem
        vars_ = nor_Prob.get_variables()
        obj_idx = nor_Prob.get_object_index()

        # Handle with the non-basic variables
        for i in range(len(vars_)):
            for j in range(1, len(self.indep_var)):
                if vars_[i] == self.indep_var[j]:
                    self.indep_var_obj[j] += obj_idx[i]

        # Handle with the basic variables
        for i in range(len(vars_)):
            for j in range(1, len(self.dep_var)):
                if vars_[i] == self.dep_var[j]:
                    for k in range(len(self.eq_constrs_matrix[j - 1])):
                        self.indep_var_obj[k] += obj_idx[i] * self.eq_constrs_matrix[j - 1][k]

    def solve(self):
        times = 0
        st.header("\n--Phase 2--")
        st.header(f"\nTimes: {times}")
        self.display()
        times += 1

        all_positive = True

        while True:
            all_positive = True

            all_positive = all(val >= 0 for val in self.indep_var_obj[1:]) # Check if all values of c_j are not negative, which means we can stop the algorithm

            if all_positive:
                inf_sol_check = False
                pos = []
                
                # Check if there's a variable which is not basic, has a form as "x..." or "y..." and has its objective function value is 0
                for i in range(1, len(self.indep_var_obj)):
                    if self.indep_var_obj[i] == 0 and (self.indep_var[i][0] == 'x' or self.indep_var[i][0] == 'y'):
                        for j in range(len(self.eq_constrs_matrix)):
                            if self.eq_constrs_matrix[j][i] != 0:
                                pos.append(i) # If at least one constraint has a non-zero value at that variable, add it to the list

                amount = 0
                for i in range(len(pos)): 
                    if not (self.indep_var[pos[i]][-1] == '+' or self.indep_var[pos[i]][-1] == '-'):
                        inf_sol_check = True
                        break # If the variable are not in form as "...+" or "...-", the problem is feasible

                    for j in range(len(self.eq_constrs_matrix)): # Check if at least two variables depends on that variable
                        if self.eq_constrs_matrix[j][pos[i]] != 0:
                            amount += 1
                    if amount >= 2:
                        inf_sol_check = True
                        break
                    amount = 0

                for i in range(len(pos)): 
                    if not (self.indep_var[pos[i]][-1] == '+' or self.indep_var[pos[i]][-1] == '-'):
                        inf_sol_check = True
                        break # If the variable are not in form as "...+" or "...-", the problem is feasible

                    for j in range(1, len(self.dep_var)):
                        temp1 = self.dep_var[j][:-1]
                        temp2 = self.indep_var[pos[i]][:-1]
                        if temp1 == temp2:
                            if self.eq_constrs_matrix[j - 1][pos[i]] != 1:
                                inf_sol_check = True
                                break # If the two variables transform from the free variables are not in form
                                        # x_i+ = b_i + x_i- or x_i- = b_i + x_i+, the problem is feasible

                if inf_sol_check:
                    self.set_status(2)
                else:
                    self.set_status(1)
                return
            
            # Get pivot position
            [row, col] = self.choose_pivot()

            # If there's no row to pivot, the problem is unbounded
            if row == -1:
                self.set_status(3)
                return

            self.pivot_dict(row, col) # Pivot at that the chosen row and column
            st.header(f"\nTimes: {times}")
            self.display()
            times += 1
            self.set_times(times)

class S2P_Method:
    def __init__(self, Prob):
        self.Prob = Prob
        self.Phase1 = None
        self.Phase2 = None
        self.opt_vars_list = []
        self.opt_vars_val = []
        self.opt_val = 0
        self.status = 1  # 1: Optimal, 2: Infeasible, 3: Feasible, 4: Unbounded

    def solve(self):
        nor_Prob = self.Prob.normalize_problem()
        st.header("Normalize Problem: ")
        nor_Prob.display()
        self.Phase1 = Phase1(nor_Prob)
        self.Phase1.solve()
        if self.Phase1.get_status() == 2:
            self.Phase2 = Phase2(self.Phase1)
            self.set_status(2) # The problem is infeasible
            return

        self.Phase2 = Phase2(self.Phase1)
        self.Phase2.handle_vars(nor_Prob)
        self.Phase2.solve()
        if self.Phase2.get_status() == 3:
            self.set_status(4) # The problem is unbounded
        elif self.Phase2.get_status() == 2:
            self.set_status(3) # The problem is feasible
        elif self.Phase2.get_status() == 1:
            self.set_status(1) # The problem is optimal
        self.get_opt_vals() # Get the optimal values

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status
 
    def get_opt_vals(self): # Get the optimal values
        if self.get_status() in (2, 4):  # If the problem is infeasible or unbounded, quit the function
            return

        vals = self.Phase2.get_indep_var_obj()
        self.opt_val = vals[0] # Get the optimal value

        if self.get_status() == 3: # If the problem is feasible, quit the function
            return

        vars_1 = self.Phase2.get_indep_var()[1:]  # Remove the first element, which is null
        vars_2 = self.Phase2.get_dep_var()[1:]  # Remove the first element, which is "z"

        eq_matrix = self.Phase2.get_eq_constrs_matrix()

        # Get the variables and their values
        self.opt_vars_list = vars_1 + vars_2
        self.opt_vars_val = [0] * len(vars_1) + [eq_matrix[i][0] for i in range(len(vars_2))]

        # Handle with the "y" variables
        for i in range(len(self.opt_vars_list)):
            if self.opt_vars_list[i][0] == 'y':
                self.opt_vars_list[i] = 'x' + self.opt_vars_list[i][1:]
                self.opt_vars_val[i] *= -1
        
        # Sort the variables
        sorted_vars = sorted(zip(self.opt_vars_list, self.opt_vars_val))
        self.opt_vars_list, self.opt_vars_val = zip(*sorted_vars)

        self.opt_vars_list = list(self.opt_vars_list)
        self.opt_vars_val = list(self.opt_vars_val)

        # Remove the "w" - slack variables
        while self.opt_vars_list and self.opt_vars_list[0][0] == 'w':
            self.opt_vars_list.pop(0)
            self.opt_vars_val.pop(0)

        # Transfrom back to the free variables
        index = 0
        while index < len(self.opt_vars_list):
            if self.opt_vars_list[index].endswith('+'):
                self.opt_vars_list[index] = self.opt_vars_list[index][:-1]
                self.opt_vars_val[index] -= self.opt_vars_val.pop(index + 1)
                self.opt_vars_list.pop(index + 1)
            index += 1
            
    def display(self):
        total_times_phase1 = 0
        total_times_phase2 = 0
        if(self.Phase1.get_times()):
            total_times_phase1=self.Phase1.get_times() -1
        if(self.Phase2.get_times()):
            total_times_phase2=self.Phase2.get_times() -1

        total_times = total_times_phase1 + total_times_phase2
        if self.get_status() == 1:
            st.header("Optimal Problem")

            # Print the optimal solution in a table
            st.markdown("### Optimal Solution")
            opt_solution_table = "<table style='width:100%; border-collapse: collapse;'>"
            opt_solution_table += "<tr><th style='padding: 5px; border: 1px solid black;'>Variable</th><th style='padding: 5px; border: 1px solid black;'>Value</th></tr>"
            for var, val in zip(self.opt_vars_list, self.opt_vars_val):
                opt_solution_table += f"<tr><td style='padding: 5px; border: 1px solid black;'>{var}</td><td style='padding: 5px; border: 1px solid black;'>{val}</td></tr>"
            opt_solution_table += "</table>"
            st.markdown(opt_solution_table, unsafe_allow_html=True)

            # Print the optimal value
            st.markdown("### Optimal Value")
            opt_val = self.opt_val if self.Prob.get_goal() == 1 else -self.opt_val
            st.markdown(f"<p style='margin:0;'>z = {opt_val}</p>", unsafe_allow_html=True)

        elif self.get_status() == 2:
            st.header("Infeasible Problem")
            st.write("Optimal solution: None")
            st.write("Optimal value: z = +∞" if self.Prob.get_goal() == 1 else "Optimal value: z = -∞")

        elif self.get_status() == 3:
            st.header("Feasible Problem")
            st.write("Optimal solution: None")
            opt_val = self.opt_val if self.Prob.get_goal() == 1 else -self.opt_val
            st.write(f"Optimal value: z = {opt_val}")

        elif self.get_status() == 4:
            st.header("Unbounded Problem")
            st.write("Optimal solution: None")
            st.write("Optimal value: z = -∞" if self.Prob.get_goal() == 1 else "Optimal value: z = +∞")
        # Print the total iterations
        st.markdown("### Iteration Details")
        st.markdown(f"<p style='margin:0;'>Total Iterations in Phase 1: {total_times_phase1}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:0;'>Total Iterations in Phase 2: {total_times_phase2}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:0;'>Total Iterations: {total_times}</p>", unsafe_allow_html=True)