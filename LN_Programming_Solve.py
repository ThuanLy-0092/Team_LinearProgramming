import streamlit as st
class Problem:
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

    def add_object_index(self):
        print("Enter objective indices (type '*' to end):")
        while True:
            temp = input()
            if temp == "*":
                break
            self.obj_idx.append(float(temp))
        self.set_var_amount()

    def set_object_index(self, obj_idx):
        self.obj_idx = obj_idx

    def get_object_index(self):
        return self.obj_idx

    def add_constraints(self):
        print("Enter constraints (type '*' to end):")
        while True:
            constr = []
            for _ in range(self.var_amount):
                temp = input()
                if temp == "*":
                    return
                constr.append(float(temp))
            self.var_constrs.append(constr)
            self.eq_sign_constrs.append(input())
            self.f_coeffs.append(float(input()))

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

    def add_sign_constrs(self):
        print("Enter variable sign constraints:")
        for _ in range(self.var_amount):
            self.sign_constrs.append(input())

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
        # Xây dựng hàm mục tiêu dưới dạng chuỗi
        st.header("Problem:")
        obj_function_str = f"{'Minimize' if self.goal == 1 else 'Maximize'} "
        obj_function_str += " ".join([
            f"{'' if i == 0 and coeff > 0 else '+' if coeff > 0 else ''}{coeff:.4f}{var}"
            for i, (coeff, var) in enumerate(zip(self.obj_idx, self.vars)) if coeff != 0
        ])
        st.write(obj_function_str)

        st.write("--------------------------------")

        # Xây dựng và in ra các ràng buộc
        for i, (constr, sign, free_coeff) in enumerate(zip(self.var_constrs, self.eq_sign_constrs, self.f_coeffs)):
            constraint_str = " ".join([
                f"{'' if j == 0 and coeff > 0 else '+' if coeff > 0 else ''}{coeff:.4f}{var}"
                for j, (coeff, var) in enumerate(zip(constr, self.vars)) if coeff != 0
            ])
            constraint_str += f" {sign} {free_coeff:.4f}"
            st.write(constraint_str)

        # Hiển thị các ràng buộc dấu của biến
        sign_constraints_str = ", ".join([
            f"{var} {sign} 0"
            for var, sign in zip(self.vars, self.sign_constrs) if sign in ["<=", ">="]
        ])
        st.write(sign_constraints_str)

        st.write()

    def add_problem(self):
        goal = int(input("Input the object: 1 for Minimize, 2 for Maximize\n"))
        self.set_goal(goal)
        self.add_object_index()
        self.add_constraints()
        self.add_sign_constrs()
        self.add_variables()

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

        for sign in nor_prob.sign_constrs:
            if sign == "f":
                nor_prob.var_amount += 1

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

        for i in range(len(nor_prob.sign_constrs)):
            if nor_prob.sign_constrs[i] == "<=":
                nor_prob.vars[i] = f"y{i + 1}"
                nor_prob.sign_constrs[i] = ">="
                nor_prob.obj_idx[i] = -nor_prob.obj_idx[i]
                for j in range(len(nor_prob.var_constrs)):
                    nor_prob.var_constrs[j][i] = -nor_prob.var_constrs[j][i]

        for i in range(len(nor_prob.eq_sign_constrs)):
            if nor_prob.eq_sign_constrs[i] == ">=":
                for j in range(len(nor_prob.var_constrs[i])):
                    nor_prob.var_constrs[i][j] = -nor_prob.var_constrs[i][j]
                nor_prob.eq_sign_constrs[i] = "<="
                nor_prob.f_coeffs[i] = -nor_prob.f_coeffs[i]

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

        if self.get_goal() == 2:
            for i in range(len(nor_prob.obj_idx)):
                nor_prob.obj_idx[i] = -nor_prob.obj_idx[i]

        return nor_prob


class Dict:
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

    def pivot_dict(self, row, col):
        self.dep_var[row + 1], self.indep_var[col] = self.indep_var[col], self.dep_var[row + 1]  # Swap variables
        value = self.eq_constrs_matrix[row][col]  # Get pivot value
        self.eq_constrs_matrix[row][col] = -1  # Set pivot value to -1
        self.eq_constrs_matrix[row] = [x / -value for x in self.eq_constrs_matrix[row]]  # Update the row

        value = self.indep_var_obj[col]
        self.indep_var_obj[col] = 0
        self.indep_var_obj = [x + value * y for x, y in zip(self.indep_var_obj, self.eq_constrs_matrix[row])]

        for i in range(len(self.eq_constrs_matrix)):
            if i == row:
                continue
            value = self.eq_constrs_matrix[i][col]
            self.eq_constrs_matrix[i][col] = 0
            self.eq_constrs_matrix[i] = [x + value * y for x, y in zip(self.eq_constrs_matrix[i], self.eq_constrs_matrix[row])]

    def display(self):
        # Xây dựng hàm mục tiêu dưới dạng chuỗi
        obj_function_str = f"{self.dep_var[0]} = " + " ".join([
            f"{'' if i == 0 and val > 0 else '+' if val > 0 else ''}{val:.4f}{var}"
            for i, (val, var) in enumerate(zip(self.indep_var_obj, self.indep_var))
        ])
        st.write(obj_function_str)
        st.write("--------------------------------")
        # Xây dựng và in ra các ràng buộc
        for i, row in enumerate(self.eq_constrs_matrix):
            constraint_str = f"{self.dep_var[i + 1]} = " + " ".join([
                f"{'' if j == 0 and val > 0 else '+' if val > 0 else ''}{val:.4f}{var}"
                for j, (val, var) in enumerate(zip(row, self.indep_var))
            ])
            st.write(constraint_str)





class Phase1(Dict):
    def __init__(self, nor_Prob=None):
        super().__init__()
        self.status = 1  # 1: Feasible, 2: Infeasible
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

    def get_status(self):
        return self.status

    def solve(self):
        all_positive = True
        result = Dict()
        times = 0

        st.header("\n--Phase 1--\n")
        st.header(f"\nTimes: {times}")
        self.display()
        times += 1

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
            return

        # Phase 1
        min_val = 0
        row = 0
        for i, row_values in enumerate(self.eq_constrs_matrix):
            if row_values[0] < min_val:
                min_val = row_values[0]
                row = i

        self.pivot_dict(row, len(self.get_indep_var()) - 1)
        st.header(f"\nTimes: {times}")
        self.display()
        times += 1

        while True:
            all_positive = all(val >= 0 for val in self.indep_var_obj[1:])
            if all_positive:
                self.set_status(1 if self.indep_var_obj[0] == 0 else 2)
                return

            col = next((i for i, val in enumerate(self.indep_var_obj[1:], 1) if val < 0), None)
            min_val = float('inf')
            row = None
            var = None

            for i, row_values in enumerate(self.eq_constrs_matrix):
                if row_values[col] >= 0:
                    continue
                quotient = row_values[0] / -row_values[col]
                if quotient < min_val or (quotient == min_val and self.dep_var[i + 1] < var):
                    min_val = quotient
                    row = i
                    var = self.dep_var[i + 1]

            self.pivot_dict(row, col)
            st.header(f"\nTimes: {times}")
            self.display()
            times += 1


class Phase2(Dict):
    def __init__(self, result=None):
        super().__init__()
        self.status = 1  # 1: Optimal, 2: Feasible, 3: Unbounded
        if result:
            self.dep_var = result.get_dep_var()
            self.indep_var = result.get_indep_var()
            self.indep_var_obj = result.get_indep_var_obj()
            self.eq_constrs_matrix = result.get_eq_constrs_matrix()

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def handle_vars(self, nor_Prob):
        self.dep_var[0] = "z"  # Change the objective function name to "z"

        for i in range(len(self.indep_var_obj) - 1, 0, -1):
            if self.indep_var_obj[i] != 0:
                self.indep_var.pop(i)
                self.indep_var_obj.pop(i)
                for j in range(len(self.eq_constrs_matrix)):
                    self.eq_constrs_matrix[j].pop(i)

        vars_ = nor_Prob.get_variables()
        obj_idx = nor_Prob.get_object_index()

        for i in range(len(vars_)):
            for j in range(1, len(self.indep_var)):
                if vars_[i] == self.indep_var[j]:
                    self.indep_var_obj[j] += obj_idx[i]

        for i in range(len(vars_)):
            for j in range(1, len(self.dep_var)):
                if vars_[i] == self.dep_var[j]:
                    for k in range(len(self.eq_constrs_matrix[j - 1])):
                        self.indep_var_obj[k] += obj_idx[i] * self.eq_constrs_matrix[j - 1][k]

    def solve(self):
        all_positive = True
        times = 0

        st.header("\n--Phase 2--\n")
        st.header(f"\nTimes: {times}")
        self.display()
        times += 1

        while True:
            all_positive = all(val >= 0 for val in self.indep_var_obj[1:])

            if all_positive:
                inf_sol_check = any(
                    all(self.eq_constrs_matrix[j][i] == 0 for j in range(len(self.eq_constrs_matrix)))
                    for i in range(1, len(self.indep_var_obj))
                    if self.indep_var_obj[i] == 0
                )

                if inf_sol_check:
                    self.set_status(2)
                else:
                    self.set_status(1)
                return

            for i in range(1, len(self.indep_var_obj)):
                if self.indep_var_obj[i] < 0:
                    col = i
                    break

            row = -1
            min_val = float('inf')
            for i in range(len(self.eq_constrs_matrix)):
                if self.eq_constrs_matrix[i][col] >= 0:
                    continue
                quotient = self.eq_constrs_matrix[i][0] / -self.eq_constrs_matrix[i][col]
                if quotient < min_val:
                    min_val = quotient
                    row = i
                    var = self.dep_var[i + 1]

            if row == -1:
                self.set_status(3)
                return

            self.pivot_dict(row, col)
            st.header(f"\nTimes: {times}")
            self.display()
            times += 1


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

        self.Phase1 = Phase1(nor_Prob)
        self.Phase1.solve()
        if self.Phase1.get_status() == 2:
            self.set_status(2)
            return

        self.Phase2 = Phase2(self.Phase1)
        self.Phase2.handle_vars(nor_Prob)
        self.Phase2.solve()
        if self.Phase2.get_status() == 3:
            self.set_status(4)

        self.get_opt_vals()

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def get_opt_vals(self):
        if self.get_status() in (2, 3, 4):
            return

        vals = self.Phase2.get_indep_var_obj()
        self.opt_val = vals[0]

        vars_1 = self.Phase2.get_indep_var()[1:]  # Remove the first element, which is null
        vars_2 = self.Phase2.get_dep_var()[1:]  # Remove the first element, which is "z"

        eq_matrix = self.Phase2.get_eq_constrs_matrix()

        self.opt_vars_list = vars_1 + vars_2
        self.opt_vars_val = [0] * len(vars_1) + [eq_matrix[i][0] for i in range(len(vars_2))]

        sorted_vars = sorted(zip(self.opt_vars_list, self.opt_vars_val))
        self.opt_vars_list, self.opt_vars_val = zip(*sorted_vars)

        self.opt_vars_list = list(self.opt_vars_list)
        self.opt_vars_val = list(self.opt_vars_val)

        while self.opt_vars_list and self.opt_vars_list[0][0] == 'w':
            self.opt_vars_list.pop(0)
            self.opt_vars_val.pop(0)

        for i in range(len(self.opt_vars_list)):
            if self.opt_vars_list[i][0] == 'y':
                self.opt_vars_list[i] = 'x' + self.opt_vars_list[i][1:]
                self.opt_vars_val[i] *= -1

        index = 0
        while index < len(self.opt_vars_list):
            if self.opt_vars_list[index].endswith('+'):
                self.opt_vars_list[index] = self.opt_vars_list[index][:-1]
                self.opt_vars_val[index] -= self.opt_vars_val.pop(index + 1)
                self.opt_vars_list.pop(index + 1)
            index += 1

            
    def display(self):
        if self.get_status() == 1:
            st.header("\nOptimal problem\n")
            st.write("Optimal solution: ", end="")
            for var, val in zip(self.opt_vars_list, self.opt_vars_val):
                st.write(f"{var} = {val}, ", end="")
            st.write("\nOptimal value: z = ", end="")
            st.write(self.opt_val if self.Prob.get_goal() == 1 else -self.opt_val)

        elif self.get_status() == 2:
            st.header("\nInfeasible problem\n")
            st.write("Optimal solution: None")
            st.write("Optimal value: z = +∞" if self.Prob.get_goal() == 1 else "Optimal value: z = -∞")

        elif self.get_status() == 3:
            st.header("\nFeasible problem\n")
            st.write("Optimal solution: None")
            st.write(f"Optimal value: z = {self.opt_val}" if self.Prob.get_goal() == 1 else f"Optimal value: z = {-self.opt_val}")

        elif self.get_status() == 4:
            st.header("\nUnbounded problem\n")
            st.write("Optimal solution: None")
            st.write("Optimal value: z = -∞" if self.Prob.get_goal() == 1 else "Optimal value: z = +∞")
