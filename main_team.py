from MethodsFinalFinal import Problem, Dict, Phase1, Phase2, S2P_Method
import numpy as np
import streamlit as st

hide_elements_css = """
<style>
/* Ẩn biểu tượng GitHub và các lớp liên quan */
.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK {
  display: none !important;
}

/* Ẩn menu chính (MainMenu) */
#MainMenu {
  visibility: hidden !important;
}

/* Ẩn footer */
footer {
  visibility: hidden !important;
}

/* Ẩn header */
header {
  visibility: hidden !important;
}
</style>
"""

# # Thiết lập tiêu đề trang và logo
st.set_page_config(page_title="Chương Trình Giải Bài Toán Quy Hoạch Tuyến Tính", page_icon=r"logo.jpg")
st.markdown(hide_elements_css, unsafe_allow_html=True)
# # Hiển thị logo
logo_path = "LN_Programming.jpg"  # Đường dẫn tới tệp logo
st.image(logo_path)  # Hiển thị logo với chiều rộng 200 pixel (bạn có thể điều chỉnh chiều rộng theo ý muốn)

st.title("Chương Trình Giải Bài Toán Quy Hoạch Tuyến Tính")

# Sidebar với hướng dẫn sử dụng
st.sidebar.title("Hướng dẫn sử dụng")
st.sidebar.markdown("""
### Cách sử dụng:
1. **Chọn mục tiêu của bài toán**: Chọn "Minimize" để tối thiểu hóa hoặc "Maximize" để tối đa hóa hàm mục tiêu.
2. **Nhập chỉ số của hàm mục tiêu**: Nhập các hệ số của hàm mục tiêu, cách nhau bằng dấu cách.
3. **Nhập các ràng buộc**:
   - Mỗi ràng buộc trên một dòng.
   - Các hệ số của biến và dấu ràng buộc cách nhau bằng dấu cách.
   - Ví dụ: `1 1 -1 0 = 1` hoặc `1 -1 0 -1 = 5`
4. **Nhập dấu của ràng buộc biến**: Nhập dấu của các ràng buộc biến, cách nhau bằng dấu cách.
5. **Nhấn nút "Tạo bài toán"**: Để tạo và giải bài toán.

### Ví dụ:
**Testcase 1:**
min
- **Hàm mục tiêu**: `5 4 0 0`
- **Ràng buộc**:
  - `1 1 -1 0 = 1`
  - `1 -1 0 -1 >= 5`
- **Dấu ràng buộc biến**: `>= >= >= f`

**Testcase 2:**
min
- **Hàm mục tiêu**: `1 2 3 4`
- **Ràng buộc**:
  - `0 1 2 3 <= 9`
  - `7 6 1 -5 >= 13`
  - `12 1 4 9 <= 14`
- **Dấu ràng buộc biến**: `>= <= f >=`

**Testcase 3**:
max
- **Hàm mục tiêu**: `-1 -3 -1`
- **Ràng buộc**:
  - `2 -5 1 <= -5`
  - `2 -1 2 <= 4`
- **Dấu ràng buộc biến**: `>= >= >=`

### Nhóm tác giả:
- **Lý Vĩnh Thuận**
- **Trịnh Ngọc Mạnh Hùng**
- **Nguyễn Trần Lê Hoàng**
""")

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
objective_indices = st.text_input("Nhập chỉ số của hàm mục tiêu (cách nhau bằng dấu cách):", placeholder="5 4 0 0")
var_constraints = st.text_area("Nhập các ràng buộc (mỗi dòng là một ràng buộc, các phần tử cách nhau bằng dấu cách):", placeholder="""1 1 -1 0 =1
1 -1 0 -1 = 5""")
sign_constraints = st.text_input("Nhập dấu của ràng buộc biến (cách nhau bằng dấu cách):", placeholder=">= >= >= f")
lhs, rhs, sign = convert_constraints(var_constraints)

if st.button("Tạo bài toán"):
  goal_value = 1 if goal == "Minimize" else 2
  objective_indices = list(map(int, objective_indices.split()))
  var_constraints = lhs
  eq_sign_constraints = sign
  f_coeffs = list(map(float, rhs))
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
  solver.display()
