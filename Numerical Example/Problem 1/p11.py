import numpy as np
from scipy.sparse import diags
import pandas as pd

# =====================================================
# Exact Solution
# =====================================================
def exact_solution(x, t):
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

# =====================================================
# Numerical Solver
# =====================================================
def solve_heat_equation(h):

    k = h**2

    x0 = 0.0
    xm = 1.0
    T = 1.0
    alpha = 1.0

    m = int((xm-x0)/h)
    n = int(T/k)

    x = np.linspace(x0, xm, m+1)
    t = np.linspace(0, n*k, n+1)

    # -------------------------------------------------
    # Scheme coefficients
    # -------------------------------------------------
    R  = 1/(12*k) - alpha/(2*h**2)
    Q  = 10/(12*k) + alpha/(h**2)
    P  = 1/(12*k) - alpha/(2*h**2)

    Rb = 1/(12*k) + alpha/(2*h**2)
    Qb = 10/(12*k) - alpha/(h**2)
    Pb = 1/(12*k) + alpha/(2*h**2)

    # -------------------------------------------------
    # Initialize solution
    # -------------------------------------------------
    u = np.zeros((m+1, n+1))

    # Initial condition
    u[:,0] = np.sin(np.pi*x)

    # Boundary conditions
    u[0,:] = 0
    u[m,:] = 0

    # -------------------------------------------------
    # Matrix A
    # -------------------------------------------------
    li = np.full(m-2, R)
    di = np.full(m-1, Q)
    ui = np.full(m-2, P)

    A = diags([li, di, ui], [-1,0,1],
              shape=(m-1,m-1)).toarray()

    b = np.zeros(m-1)
    C = np.zeros(m-1)

    # -------------------------------------------------
    # Time Loop
    # -------------------------------------------------
    for j in range(n):

        C[:] = 0.0
        C[0]  = R*u[0,j]
        C[-1] = P*u[m,j]

        for i in range(1,m):
            b[i-1] = (
                Pb*u[i+1,j]
                + Qb*u[i,j]
                + Rb*u[i-1,j]
                - C[i-1]
            )

        vi = np.linalg.solve(A,b)

        u[1:m,j+1] = vi

    # -------------------------------------------------
    # Exact Solution
    # -------------------------------------------------
    X, TT = np.meshgrid(x,t,indexing='ij')
    v = exact_solution(X,TT)

    error = np.max(np.abs(u-v))

    return x, u[:,-1], v[:,-1], error

# =====================================================
# Mesh sizes
# =====================================================
h_values = [1/5, 1/10, 1/20, 1/40, 1/80, 1/160]

errors = []

results = {}

# =====================================================
# Run all cases
# =====================================================
for h in h_values:

    x, u_num, u_ex, err = solve_heat_equation(h)

    errors.append(err)

    results[h] = (x, u_num, u_ex)

# =====================================================
# Convergence order
# =====================================================
orders = [np.nan]

for i in range(1,len(errors)):
    p = np.log(errors[i-1]/errors[i])/np.log(2)
    orders.append(p)

# =====================================================
# Error Table
# =====================================================
conv_table = pd.DataFrame({
    "h": h_values,
    "Maximum Error": errors,
    "Order": orders
})

pd.set_option("display.float_format","{:.12e}".format)

print("\n")
print("="*70)
print("CONVERGENCE TABLE")
print("="*70)
print(conv_table.to_string(index=False))

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(14,6)
)

plt.subplots_adjust(wspace=0.3)
num_line, = ax1.plot(
    [], [],
    'ro-',
    linewidth=2,
    markersize=7,
    label='Numerical'
)

exact_line, = ax1.plot(
    [], [],
    'b^--',
    linewidth=2,
    markersize=5,
    label='Exact'
)

info_text = ax1.text(
    0.02,
    0.95,
    '',
    transform=ax1.transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(facecolor='white')
)

ax1.set_xlabel(r'$x$', fontsize=13)
ax1.set_ylabel(r'$u(x,T)$', fontsize=13)

ax1.grid(True)
ax1.legend()


ax2.set_xscale('log')
ax2.set_yscale('log')

ax2.invert_xaxis()

ax2.grid(True, which='both')

ax2.set_xlabel(r'$h$', fontsize=13)
ax2.set_ylabel('Maximum Error', fontsize=13)

error_line, = ax2.plot(
    [],
    [],
    'ro-',
    linewidth=2,
    markersize=8,
    label='Computed Error'
)

h_array = np.array(h_values)

reference = errors[0]*(h_array/h_array[0])**4

ax2.plot(
    h_array,
    reference,
    'k--',
    linewidth=2,
    label=r'$O(h^4)$'
)

ax2.legend()

def update(frame):

    h = h_values[frame]

    x, u_num, u_ex = results[h]

    # --------------------------------
    # Left panel
    # --------------------------------

    num_line.set_data(x, u_num)

    exact_line.set_data(x, u_ex)

    ax1.set_title(
        f'h = {h:.5f}'
    )

    if frame == 0:
        order_str = "---"
    else:
        order_str = f"{orders[frame]:.4f}"

    info_text.set_text(
        f"Grid points = {len(x)}\n"
        f"Error = {errors[frame]:.3e}\n"
        f"Order = {order_str}"
    )

    ax1.relim()
    ax1.autoscale_view()
    # ax1.set_xlim(0,1)
    # ax1.set_ylim(0, np.exp(-np.pi**2)+0.00009)

    # --------------------------------
    # Right panel
    # --------------------------------

    error_line.set_data(
        h_array[:frame+1],
        errors[:frame+1]
    )

    return (
        num_line,
        exact_line,
        error_line,
        info_text
    )


ani = FuncAnimation(
    fig,
    update,
    frames=len(h_values),
    interval=2000,
    repeat=True
)

ani.save(
    "p11Fourth_Order_Convergence.gif",
    writer=PillowWriter(fps=1)
)

print("GIF SAVED!")

plt.show()

# print("="*70)

# # =====================================================
# # Numerical and Exact Solution Tables
# # =====================================================
# for h in h_values:

#     x, u_num, u_ex = results[h]

#     sol_table = pd.DataFrame({
#         "x": x,
#         "Numerical Solution": u_num,
#         "Exact Solution": u_ex,
#         "Abs Error": np.abs(u_num-u_ex)
#     })

#     print("\n\n")
#     print("="*90)
#     print(f"SOLUTION TABLE FOR h = {h}")
#     print("="*90)
#     print(sol_table.to_string(index=False))