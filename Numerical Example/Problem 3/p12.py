import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags
from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# PDE Solver
# ==========================================================
def solve_pde(h):

    k = h**2

    x0 = 0.0
    xm = 1.0
    Tfinal = 2.0

    alpha = 0.1
    beta = 1.0

    m = int((xm - x0)/h)
    n = int(Tfinal/k)

    # ------------------------------------------------------
    # Grid
    # ------------------------------------------------------
    x = np.linspace(x0, xm, m+1)
    t = np.linspace(0, n*k, n+1)

    # ------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------
    S1 = -((alpha/h**2) + (beta**2/(12*alpha)))
    S2 = beta/(2*h)
    S3 = 1/12
    S4 = -(beta*h)/(24*alpha)

    P  = -((S1+S2)/2) - ((S3+S4)/k)
    Pb = ((S1+S2)/2) - ((S3+S4)/k)

    Q  = S1 - ((1-2*S3)/k)
    Qb = -S1 - ((1-2*S3)/k)

    R  = -(S1-S2)/2 - ((S3-S4)/k)
    Rb = (S1-S2)/2 - ((S3-S4)/k)

    # ------------------------------------------------------
    # Numerical Solution
    # ------------------------------------------------------
    u = np.zeros((m+1, n+1))

    # Initial condition
    for i in range(m+1):
        u[i,0] = np.exp(5*x[i])*(
                 np.cos((np.pi/2)*x[i])
                 + 0.25*np.sin((np.pi/2)*x[i]))

    # Boundary conditions
    for j in range(1,n+1):

        u[0,j] = (
            np.exp(5*(0 - t[j]/2))
            * np.exp(-(np.pi**2/40)*t[j])
            * (np.cos(0)+0.25*np.sin(0))
        )

        u[m,j] = (
            np.exp(5*(1 - t[j]/2))
            * np.exp(-(np.pi**2/40)*t[j])
            * (np.cos(np.pi/2)+0.25*np.sin(np.pi/2))
        )

    # ------------------------------------------------------
    # Matrix A
    # ------------------------------------------------------
    li = np.full(m-2, R)
    di = np.full(m-1, Q)
    ui = np.full(m-2, P)

    A = diags(
        [li, di, ui],
        [-1, 0, 1],
        shape=(m-1, m-1)
    ).toarray()

    b = np.zeros(m-1)
    C = np.zeros(m-1)

    # ------------------------------------------------------
    # Time marching
    # ------------------------------------------------------
    for j in range(n):

        C[:] = 0.0
        C[0]  = R*u[0,j+1]
        C[-1] = P*u[m,j+1]

        for i in range(1,m):

            b[i-1] = (
                Pb*u[i+1,j]
                + Qb*u[i,j]
                + Rb*u[i-1,j]
                - C[i-1]
            )

        vi = np.linalg.solve(A,b)

        u[1:m,j+1] = vi

    # ------------------------------------------------------
    # Exact solution
    # ------------------------------------------------------
    v = np.zeros((m+1,n+1))

    for i in range(m+1):
        for j in range(n+1):

            v[i,j] = (
                np.exp(5*(x[i]-t[j]/2))
                * np.exp(-(np.pi**2/40)*t[j])
                * (
                    np.cos((np.pi/2)*x[i])
                    + 0.25*np.sin((np.pi/2)*x[i])
                )
            )

    # ------------------------------------------------------
    # Error
    # ------------------------------------------------------
    err = np.abs(u-v)
    max_error = np.max(err)

    return x,t,u,v,max_error


# ==========================================================
# Mesh sizes
# ==========================================================
h_values = [1/4, 1/8, 1/16, 1/32, 1/64, 1/128]

errors = []
results = {}

# ==========================================================
# Solve for each h
# ==========================================================
for h in h_values:

    # print("\n")
    # print("="*80)
    # print(f"Running for h = {h}")
    # print("="*80)

    x,t,u,v,error = solve_pde(h)

    errors.append(error)

    results[h] = (x,t,u,v,error)

    print(f"Maximum Error = {error:.12e}")

    # ------------------------------------------------------
    # Final Time Solution Table
    # ------------------------------------------------------
    table = pd.DataFrame({
        "x"               : x,
        "Numerical"       : u[:,-1],
        "Exact"           : v[:,-1],
        "Absolute Error"  : np.abs(u[:,-1]-v[:,-1])
    })

    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format',
                  lambda x: f'{x:.12e}')

    # print("\n")
    # print("="*80)
    # print(f"FINAL TIME SOLUTION TABLE (h = {h})")
    # print("="*80)
    # print(table.to_string(index=False))

    # ------------------------------------------------------
    # # 3D Surface Plots
    # # ------------------------------------------------------
    # X,Tmesh = np.meshgrid(x,t)

    # fig = plt.figure(figsize=(14,6))

    # # Numerical
    # ax1 = fig.add_subplot(121, projection='3d')

    # ax1.plot_surface(
    #     X,Tmesh,u.T,
    #     cmap='hot',
    #     edgecolor='k',
    #     alpha=0.7
    # )

    # ax1.set_title(f'Numerical Solution (h={h})')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('t')
    # ax1.set_zlabel('u(x,t)')
    # ax1.view_init(elev=30, azim=100)

    # # Exact
    # ax2 = fig.add_subplot(122, projection='3d')

    # ax2.plot_surface(
    #     X,Tmesh,v.T,
    #     cmap='coolwarm',
    #     edgecolor='k',
    #     alpha=0.7
    # )

    # ax2.set_title(f'Exact Solution (h={h})')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('t')
    # ax2.set_zlabel('u(x,t)')
    # ax2.view_init(elev=30, azim=100)

    # plt.tight_layout()
    # plt.show()


# ==========================================================
# Convergence Table
# ==========================================================
orders = [np.nan]

for i in range(1,len(errors)):

    p = np.log(errors[i-1]/errors[i])/np.log(2)

    orders.append(p)

conv_table = pd.DataFrame({
    "h"          : h_values,
    "Max Error"  : errors,
    "Order"      : orders
})

# print("\n\n")
# print("="*80)
# print("CONVERGENCE TABLE")
# print("="*80)
# print(conv_table.to_string(index=False))
# print("="*80)



# =====================================================
# PROFESSIONAL 3D GIF
# =====================================================

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# Figure
# =====================================================

fig = plt.figure(figsize=(15,7))

ax1 = fig.add_subplot(
    121,
    projection='3d'
)

ax2 = fig.add_subplot(
    122,
    projection='3d'
)

# =====================================================
# Global z limits
# =====================================================

all_values = []

for h in h_values:

    x,t,u,v,error = results[h]

    all_values.extend(u.flatten())
    all_values.extend(v.flatten())

zmin = min(all_values)
zmax = max(all_values)

# =====================================================
# Main Title
# =====================================================

main_title = fig.text(
    0.5,
    0.96,
    "",
    ha='center',
    fontsize=15,
    fontweight='bold',
    bbox=dict(
        facecolor='white',
        edgecolor='black',
        boxstyle='round,pad=0.4'
    )
)

# =====================================================
# Update Function
# =====================================================

def update(frame):

    ax1.clear()
    ax2.clear()

    # ---------------------------------------
    # Mesh refinement every 30 frames
    # ---------------------------------------

    mesh_frame = frame // 15

    if mesh_frame >= len(h_values):
        mesh_frame = len(h_values)-1

    h = h_values[mesh_frame]

    x,t,u,v,error = results[h]

    X,Tmesh = np.meshgrid(x,t)

    # ---------------------------------------
    # Camera rotation
    # ---------------------------------------

    angle = 120 + frame*2

    # ---------------------------------------
    # Numerical Surface
    # ---------------------------------------

    ax1.plot_surface(
        X,
        Tmesh,
        u.T,
        cmap='hot',
        edgecolor='k',
        linewidth=0.4,
        alpha=0.85
    )

    ax1.set_title(
        "Numerical Solution",
        fontsize=14,
        fontweight='bold'
    )

    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x,t)')

    ax1.set_zlim(
        zmin,
        zmax
    )

    ax1.view_init(
        elev=25,
        azim=angle
    )

    # ---------------------------------------
    # Exact Surface
    # ---------------------------------------

    ax2.plot_surface(
        X,
        Tmesh,
        v.T,
        cmap='coolwarm',
        edgecolor='k',
        linewidth=0.4,
        alpha=0.85
    )

    ax2.set_title(
        "Exact Solution",
        fontsize=14,
        fontweight='bold'
    )

    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u(x,t)')

    ax2.set_zlim(
        zmin,
        zmax
    )

    ax2.view_init(
        elev=25,
        azim=angle
    )

    # ---------------------------------------
    # Order
    # ---------------------------------------

    if mesh_frame == 0:
        order_text = "---"
    else:
        order_text = f"{orders[mesh_frame]:.4f}"

    main_title.set_text(
        f"Mesh Size h = {h:.6f}   |   "
        f"Maximum Error = {error:.3e}   |   "
        f"Order = {order_text}"
    )

    return []

# =====================================================
# Animation
# =====================================================

ani = FuncAnimation(
    fig,
    update,
    frames=90,
    interval=50,
    repeat=True,
    blit=False
)

# =====================================================
# Layout
# =====================================================

plt.tight_layout(
    rect=[0,0,1,0.92]
)

# =====================================================
# Save GIF
# =====================================================

ani.save(
    "PDE_3D_Professional.gif",
    writer=PillowWriter(fps=8)
)

print("GIF SAVED SUCCESSFULLY!")

plt.show()







# from matplotlib.animation import FuncAnimation, PillowWriter
# import matplotlib.pyplot as plt
# import numpy as np

# # =====================================================
# # Figure
# # =====================================================

# fig = plt.figure(figsize=(15,7))

# ax1 = fig.add_subplot(
#     121,
#     projection='3d'
# )

# ax2 = fig.add_subplot(
#     122,
#     projection='3d'
# )

# # =====================================================
# # Global z-limits
# # =====================================================

# all_values = []

# for h in h_values:

#     x,t,u,v,error = results[h]

#     all_values.extend(u.flatten())
#     all_values.extend(v.flatten())

# zmin = min(all_values)
# zmax = max(all_values)

# # =====================================================
# # Main Title
# # =====================================================

# main_title = fig.text(
#     0.5,
#     0.95,
#     "",
#     ha='center',
#     fontsize=16,
#     fontweight='bold',
#     bbox=dict(
#         facecolor='white',
#         edgecolor='black',
#         boxstyle='round,pad=0.4'
#     )
# )

# # =====================================================
# # Update Function
# # =====================================================

# def update(frame):

#     ax1.clear()
#     ax2.clear()

#     h = h_values[frame]

#     x,t,u,v,error = results[h]

#     X, Tmesh = np.meshgrid(x,t)

#     # ------------------------------------
#     # Numerical Surface
#     # ------------------------------------

#     ax1.plot_surface(
#         X,
#         Tmesh,
#         u.T,
#         cmap='hot',
#         edgecolor='k',
#         alpha=0.8
#     )

#     ax1.set_title(
#         "Numerical Solution",
#         fontsize=14,
#         fontweight='bold'
#     )

#     ax1.set_xlabel('x')
#     ax1.set_ylabel('t')
#     ax1.set_zlabel('u(x,t)')

#     ax1.set_zlim(zmin,zmax)

#     ax1.view_init(
#         elev=40,
#         azim=100
#     )

#     # ------------------------------------
#     # Exact Surface
#     # ------------------------------------

#     ax2.plot_surface(
#         X,
#         Tmesh,
#         v.T,
#         cmap='coolwarm',
#         edgecolor='k',
#         alpha=0.8
#     )

#     ax2.set_title(
#         "Exact Solution",
#         fontsize=14,
#         fontweight='bold'
#     )

#     ax2.set_xlabel('x')
#     ax2.set_ylabel('t')
#     ax2.set_zlabel('u(x,t)')

#     ax2.set_zlim(zmin,zmax)

#     ax2.view_init(
#         elev=50,
#         azim=100
#     )

#     # ------------------------------------
#     # Order Text
#     # ------------------------------------

#     if frame == 0:
#         order_text = "---"
#     else:
#         order_text = f"{orders[frame]:.4f}"

#     main_title.set_text(
#         f"Mesh Size h = {h:.6f}   |   "
#         f"Maximum Error = {error:.3e}   |   "
#         f"Order = {order_text}"
#     )

#     return []

# # =====================================================
# # Animation
# # =====================================================

# ani = FuncAnimation(
#     fig,
#     update,
#     frames=len(h_values),
#     interval=1800,
#     repeat=True,
#     blit=False
# )

# # =====================================================
# # Layout
# # =====================================================

# plt.tight_layout(
#     rect=[0,0,1,0.90]
# )

# # =====================================================
# # Save GIF
# # =====================================================

# ani.save(
#     "p21_3D_Numerical_vs_Exact.gif",
#     writer=PillowWriter(fps=1)
# )

# print("GIF SAVED SUCCESSFULLY!")

# plt.show()







# # ======================================================
# # GIF 1 : Solution Convergence
# # ======================================================

# fig, ax = plt.subplots(figsize=(8,6))

# num_line, = ax.plot(
#     [], [],
#     'ro-',
#     linewidth=2,
#     markersize=7,
#     label='Numerical'
# )

# exact_line, = ax.plot(
#     [], [],
#     'b^--',
#     linewidth=2,
#     markersize=5,
#     label='Exact'
# )

# info = ax.text(
#     0.02,
#     0.95,
#     '',
#     transform=ax.transAxes,
#     verticalalignment='top',
#     bbox=dict(facecolor='white')
# )

# ax.set_xlim(0,1)

# # fixed y-range
# all_y = []

# for h in h_values:
#     x,t,u,v,error = results[h]
#     all_y.extend(u[:,-1])
#     all_y.extend(v[:,-1])

# ax.set_ylim(
#     min(all_y)*0.95,
#     max(all_y)*1.05
# )

# ax.grid(True)
# ax.legend()

# ax.set_xlabel('x')
# ax.set_ylabel('u(x,T)')

# def update_solution(frame):

#     h = h_values[frame]

#     x,t,u,v,error = results[h]

#     num_line.set_data(
#         x,
#         u[:,-1]
#     )

#     exact_line.set_data(
#         x,
#         v[:,-1]
#     )

#     if frame == 0:
#         order = "---"
#     else:
#         order = f"{orders[frame]:.4f}"

#     info.set_text(
#         f"h = 1/{int(round(1/h))}\n"
#         f"Grid Points = {len(x)}\n"
#         f"Error = {error:.3e}\n"
#         f"Order = {order}"
#     )

#     return num_line, exact_line, info

# ani1 = FuncAnimation(
#     fig,
#     update_solution,
#     frames=len(h_values),
#     interval=1800
# )

# ani1.save(
#     "p21-solution_convergence.gif",
#     writer=PillowWriter(fps=1)
# )

# plt.close()



# # ======================================================
# # GIF 2 : Order Convergence
# # ======================================================

# fig, ax = plt.subplots(figsize=(8,6))

# ax.set_xscale('log')
# ax.set_yscale('log')

# ax.invert_xaxis()

# ax.grid(True, which='both')

# ax.set_xlabel('h')
# ax.set_ylabel('Maximum Error')

# line, = ax.plot(
#     [],
#     [],
#     'ro-',
#     linewidth=2,
#     markersize=8
# )

# h_array = np.array(h_values)

# reference = errors[0]*(h_array/h_array[0])**2

# ax.plot(
#     h_array,
#     reference,
#     'k--',
#     linewidth=2,
#     label=r'$O(h^2)$'
# )

# ax.legend()

# txt = ax.text(
#     0.05,
#     0.95,
#     '',
#     transform=ax.transAxes,
#     verticalalignment='top',
#     bbox=dict(facecolor='white')
# )

# def update_order(frame):

#     line.set_data(
#         h_array[:frame+1],
#         errors[:frame+1]
#     )

#     if frame == 0:
#         txt.set_text("Initial mesh")
#     else:
#         txt.set_text(
#             f"Order ≈ {orders[frame]:.4f}"
#         )

#     return line, txt

# ani2 = FuncAnimation(
#     fig,
#     update_order,
#     frames=len(h_values),
#     interval=1800
# )

# ani2.save(
#     "p22-order_convergence.gif",
#     writer=PillowWriter(fps=1)
# )

# plt.close()

