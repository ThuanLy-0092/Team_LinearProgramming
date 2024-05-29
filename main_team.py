import sys
from Methods import Problem, Dict, Phase1, Phase2, S2P_Method
import numpy as np
import streamlit as st

# # Thiết lập tiêu đề trang và logo
st.set_page_config(page_title="Chương Trình Giải Bài Toán Quy Hoạch Tuyến Tính", page_icon=r"logo.jpg")

  # Hiển thị logo
logo_path = "LN_Programming.jpg"  # Đường dẫn tới tệp logo
st.image(logo_path)  # Hiển thị logo với chiều rộng 200 pixel (bạn có thể điều chỉnh chiều rộng theo ý muốn)

st.title("Chương Trình Giải Bài Toán Quy Hoạch Tuyến Tính")
def convert_constraints(constr):
    lhs = []
    rhs = []
    sign = []
    constraints = constr.split('\n')

    for constr in constraints:
        if constr:
            constr = constr.strip()
            constr = ' '.join(constr.split())
            constraint_parts = constr.split()

            sign.append(constraint_parts[-2])

            constraint_values = [float(val) if val != constraint_parts[-2] else val for val in constraint_parts]
            lhs.append(constraint_values[:-2])
            rhs.append(constraint_values[-1])

    return lhs, rhs, sign

goal = st.radio("Chọn mục tiêu của bài toán", ("Minimize", "Maximize"))
objective_indices = st.text_input("Nhập chỉ số của hàm mục tiêu (cách nhau bằng dấu cách):",placeholder="5 4 0 0")
var_constraints = st.text_area("Nhập các ràng buộc (mỗi dòng là một ràng buộc, các phần tử cách nhau bằng dấu cách):",placeholder="""1 1 -1 0 =1
1 -1 0 -1 = 5""")
sign_constraints = st.text_input("Nhập dấu của ràng buộc biến (cách nhau bằng dấu cách):",placeholder=">= >= >= f")
lhs ,rhs ,sign =convert_constraints(var_constraints)

if st.button("Tạo bài toán"):
    #min max
    goal_value = 1 if goal == "Minimize" else 2
        #hệ số của f
    objective_indices = list(map(int, objective_indices.split()))
        #hệ số vế trái ràng buộc
    var_constraints = lhs
        #dấu ràng buộc
    eq_sign_constraints = sign
        #hệ số tự do
    f_coeffs = list(map(float, rhs))
        #ràng buộc biến
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
