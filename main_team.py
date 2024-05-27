import sys
from LN_Programming_Solve import Problem, Dict, Phase1, Phase2, S2P_Method
import numpy as np
import pandas as pd
import streamlit as st

st.title("Nhập đối số cho bài toán Quy hoạch tuyến tính")

goal = st.radio("Chọn mục tiêu của bài toán", ("Minimize", "Maximize"))
objective_indices = st.text_input("Nhập chỉ số của hàm mục tiêu (cách nhau bằng dấu cách):")
var_constraints = st.text_area("Nhập các ràng buộc (mỗi dòng là một ràng buộc, các phần tử cách nhau bằng dấu cách):")
eq_sign_constraints = st.text_input("Nhập dấu của các ràng buộc (cách nhau bằng dấu cách):")
f_coeffs = st.text_input("Nhập các hệ số tự do (cách nhau bằng dấu cách):")
sign_constraints = st.text_input("Nhập dấu của ràng buộc biến (cách nhau bằng dấu cách):")

if st.button("Tạo bài toán"):
    #min max
    goal_value = 1 if goal == "Minimize" else 2
        #hệ số của f
    objective_indices = list(map(int, objective_indices.split()))
        #hệ số vế trái ràng buộc
    var_constraints = [list(map(float, line.split())) for line in var_constraints.split('\n')]
        #dấu ràng buộc
    eq_sign_constraints = eq_sign_constraints.split()
        #hệ số tự do
    f_coeffs = list(map(float, f_coeffs.split()))
        #ràng buôcj biến
    sign_constraints = sign_constraints.split()

    prob = Problem()
    prob.set_goal(goal_value)
    prob.set_object_index(objective_indices)
    prob.set_var_constraints(var_constraints)
    prob.set_eq_sign_constrs(eq_sign_constraints)
    prob.set_f_coeffs(f_coeffs)
    prob.set_sign_constrs(sign_constraints)
    prob.set_var_amount()
    prob.add_variables()
    prob.display()
    solver = S2P_Method(prob)
    solver.solve()
        # Hiển thị kết quả
    solver.display()
