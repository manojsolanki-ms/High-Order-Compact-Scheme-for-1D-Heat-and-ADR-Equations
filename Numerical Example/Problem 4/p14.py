import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from matplotlib.animation import FuncAnimation, PillowWriter

# ==========================================================
# PDE SOLVER
# ==========================================================

def solve_pde(h, Pe):

    k = h**2

    x0 = 0
    xm = 1
    Tfinal = 1

    beta = -1
    alpha = abs(beta/Pe)

    a = 1
    B = 0.1

    m = int((xm-x0)/h)
    n = int(Tfinal/k)

    # ------------------------------------------------------
    # Grid
    # ------------------------------------------------------

    x = np.linspace(x0, xm, m+1)
    t = np.linspace(0, n*k, n+1)

    # ------------------------------------------------------
    # Constant
    # ------------------------------------------------------

    cb = (
        -beta +
        np.sqrt(beta**2 + 4*alpha*B)
    )/(2*alpha)

    # ------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------

    S1 = -(
        (alpha/h**2)
        + (beta**2)/(12*alpha)
    )

    S2 = beta/(2*h)

    S3 = 1/12

    S4 = -(beta*h)/(24*alpha)

    P = -((S1+S2)/2) - ((S3+S4)/k)

    Pb = ((S1+S2)/2) - ((S3+S4)/k)

    Q = S1 - ((1-2*S3)/k)

    Qb = -S1 - ((1-2*S3)/k)

    R = -(S1-S2)/2 - ((S3-S4)/k)

    Rb = (S1-S2)/2 - ((S3-S4)/k)

    # ------------------------------------------------------
    # Numerical Solution
    # ------------------------------------------------------

    u = np.zeros((m+1,n+1))

    # Initial condition

    for i in range(m+1):

        u[i,0] = a*np.exp(-cb*x[i])

    # Boundary conditions

    for j in range(1,n+1):

        u[0,j] = a*np.exp(B*t[j])

        u[m,j] = a*np.exp(
            B*t[j] - cb
        )

    # ------------------------------------------------------
    # Matrix A
    # ------------------------------------------------------

    li = np.full(m-2,R)
    di = np.full(m-1,Q)
    ui = np.full(m-2,P)

    A = diags(
        [li,di,ui],
        [-1,0,1],
        shape=(m-1,m-1)
    ).toarray()

    b = np.zeros(m-1)
    C = np.zeros(m-1)

    # ------------------------------------------------------
    # Time Marching
    # ------------------------------------------------------

    for j in range(n):

        C[:] = 0

        C[0] = R*u[0,j+1]

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
    # Exact Solution
    # ------------------------------------------------------

    v = np.zeros((m+1,n+1))

    for i in range(m+1):

        for j in range(n+1):

            v[i,j] = a*np.exp(
                B*t[j]
                - cb*x[i]
            )

    # ------------------------------------------------------
    # Error
    # ------------------------------------------------------

    err = np.abs(u-v)

    error = np.max(err)

    return x,t,u,v,error


# ==========================================================
# PARAMETERS
# ==========================================================

Pe_values = [10,20,100,1000]

h_values = [
    1/4,
    1/8,
    1/16,
    1/32,
    1/64,
    1/128
]

# ==========================================================
# SOLVE ALL CASES
# ==========================================================

results = {}

print("\nRunning simulations...\n")

for Pe in Pe_values:

    results[Pe] = {}

    for h in h_values:

        x,t,u,v,error = solve_pde(h,Pe)

        results[Pe][h] = (
            x,t,u,v,error
        )

        print(
            f"Pe = {Pe:<5} "
            f"h = {h:<10.6f} "
            f"Error = {error:.3e}"
        )
# ==========================================================
# CONVERGENCE DATA
# ==========================================================

conv_data = {}

for Pe in Pe_values:

    errors = []

    for h in h_values:

        errors.append(
            results[Pe][h][4]
        )

    orders = [np.nan]

    for i in range(1,len(errors)):

        p = np.log(
            errors[i-1]/errors[i]
        )/np.log(2)

        orders.append(p)

    conv_data[Pe] = {
        "errors": errors,
        "orders": orders
    }
# ==========================================================
# GLOBAL Y LIMITS
# ==========================================================

ymin = 1e100
ymax = -1e100

for Pe in Pe_values:

    for h in h_values:

        x,t,u,v,error = results[Pe][h]

        ymin = min(
            ymin,
            np.min(u[:,-1]),
            np.min(v[:,-1])
        )

        ymax = max(
            ymax,
            np.max(u[:,-1]),
            np.max(v[:,-1])
        )


# ==========================================================
# FIGURE LAYOUT
# ==========================================================

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15,20))

gs = gridspec.GridSpec(
    3,
    2,
    height_ratios=[1,1,0.9]
)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])

ax5 = fig.add_subplot(gs[2,:])

axs = [ax1,ax2,ax3,ax4]

main_title = fig.suptitle(
    "",
    fontsize=18,
    fontweight='bold'
)

# ==========================================================
# UPDATE FUNCTION
# ==========================================================

def update(frame):

    h = h_values[frame]

    for ax in [ax1,ax2,ax3,ax4,ax5]:
        ax.clear()

    # ---------------------------------------
    # Numerical vs Exact Panels
    # ---------------------------------------

    for idx,Pe in enumerate(Pe_values):

        ax = axs[idx]

        x,t,u,v,error = results[Pe][h]

        final_step = len(t)-1

        ax.plot(
            x,
            u[:,final_step],
            'ro-',
            linewidth=2,
            markersize=5,
            label='Numerical'
        )

        ax.plot(
            x,
            v[:,final_step],
            'bs--',
            linewidth=2,
            markersize=4,
            label='Exact'
        )

        avg_order = np.nanmean(
            conv_data[Pe]["orders"][1:]
        )

        ax.set_title(
            f"Pe={Pe}\n"
            f"Error={error:.3e}\n"
            f"Avg Order={avg_order:.3f}",
            fontsize=11,
            fontweight='bold'
        )

        ax.set_xlabel("x")
        ax.set_ylabel("u(x,T)")

        ax.grid(True)

        ax.legend()

    # ---------------------------------------
    # Convergence History
    # ---------------------------------------

    styles = [
        'ro-',
        'bs-',
        'g^-',
        'md-'
    ]

    for i,Pe in enumerate(Pe_values):

        ax5.loglog(
            h_values[:frame+1],
            conv_data[Pe]["errors"][:frame+1],
            styles[i],
            linewidth=2,
            markersize=8,
            label=f'Pe={Pe}'
        )

    # ---------------------------------------
    # Reference O(h^4)
    # ---------------------------------------

    ref0 = conv_data[10]["errors"][0]

    C = ref0/(h_values[0]**4)

    ref = [
        C*(hh**4)
        for hh in h_values[:frame+1]
    ]

    ax5.loglog(
        h_values[:frame+1],
        ref,
        'k--',
        linewidth=2,
        label=r'$O(h^4)$'
    )

    ax5.set_title(
        "Convergence History",
        fontsize=13,
        fontweight='bold'
    )

    ax5.set_xlabel("h")

    ax5.set_ylabel(
        "Maximum Error"
    )

    ax5.grid(
        True,
        which='both',
        linestyle='--'
    )

    ax5.legend(
        ncol=5
    )

    ax5.invert_xaxis()

    # ---------------------------------------
    # Orders Box
    # ---------------------------------------

    txt = ""

    for Pe in Pe_values:

        if frame >= 1:

            ord_now = np.nanmean(
                conv_data[Pe]["orders"][1:frame+1]
            )

            txt += (
                f"Pe={Pe}: "
                f"{ord_now:.3f}   "
            )

    ax5.text(
        0.02,
        0.04,
        txt,
        transform=ax5.transAxes,
        fontsize=10,
        bbox=dict(
            facecolor='white',
            edgecolor='black'
        )
    )

    main_title.set_text(
        f"Mesh Size h = {h:.6f}"
    )

    plt.tight_layout(
        rect=[0,0,1,0.95]
    )

    return []

# ==========================================================
# ANIMATION
# ==========================================================

ani = FuncAnimation(
    fig,
    update,
    frames=len(h_values),
    interval=1500,
    repeat=True,
    blit=False
)

# ==========================================================
# SAVE GIF
# ==========================================================

ani.save(
    "Pe_Solution_Convergence.gif",
    writer=PillowWriter(fps=1)
)

print(
    "\nGIF SAVED SUCCESSFULLY!"
)

plt.show()



# # ==========================================================
# # FIGURE
# # ==========================================================

# fig, axs = plt.subplots(
#     2,
#     2,
#     figsize=(14,10)
# )

# axs = axs.flatten()

# # ==========================================================
# # MAIN TITLE
# # ==========================================================

# main_title = fig.suptitle(
#     "",
#     fontsize=18,
#     fontweight='bold'
# )

# # ==========================================================
# # UPDATE FUNCTION
# # ==========================================================

# def update(frame):

#     h = h_values[frame]

#     for ax in axs:
#         ax.clear()

#     for idx, Pe in enumerate(Pe_values):

#         ax = axs[idx]

#         x,t,u,v,error = results[Pe][h]

#         final_step = len(t)-1

#         # Numerical

#         ax.plot(
#             x,
#             u[:,final_step],
#             'ro-',
#             linewidth=2,
#             markersize=5,
#             label='Numerical'
#         )

#         # Exact

#         ax.plot(
#             x,
#             v[:,final_step],
#             'bs--',
#             linewidth=2,
#             markersize=4,
#             label='Exact'
#         )

#         ax.set_title(
#             f"Pe = {Pe}\n"
#             f"Error = {error:.3e}",
#             fontsize=12,
#             fontweight='bold'
#         )

#         ax.set_xlabel("x")
#         ax.set_ylabel("u(x,T)")

#         ax.grid(True)

#         ax.legend()

#         ax.set_ylim(
#             ymin*0.95,
#             ymax*1.05
#         )

#     main_title.set_text(
#         f"Mesh Size h = {h:.6f}"
#     )

#     plt.tight_layout(
#         rect=[0,0,1,0.95]
#     )

#     return []

# # ==========================================================
# # ANIMATION
# # ==========================================================

# ani = FuncAnimation(
#     fig,
#     update,
#     frames=len(h_values),
#     interval=1500,
#     repeat=True,
#     blit=False
# )

# # ==========================================================
# # SAVE GIF
# # ==========================================================

# ani.save(
#     "p4-1-Pe_Comparison_Animation.gif",
#     writer=PillowWriter(fps=1)
# )

# print("\nGIF SAVED SUCCESSFULLY!")

# plt.show()