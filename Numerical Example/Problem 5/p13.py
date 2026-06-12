import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from matplotlib.animation import FuncAnimation, PillowWriter

# ==========================================================
# PDE SOLVER
# ==========================================================

def solve_pde(h):

    k = h**2

    x0 = 0
    xm = 2*np.pi
    Tfinal = 0.5

    alpha = 1
    beta = 1
    gamma = -3

    m = int((xm-x0)/h)
    n = int(Tfinal/k)

    # ------------------------------------------------------
    # Grid
    # ------------------------------------------------------

    x = np.zeros(m+1)

    for i in range(m+1):
        x[i] = x0 + i*h

    t = np.zeros(n+1)

    for j in range(n+1):
        t[j] = j*k

    # ------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------

    S1 = -((alpha/h**2)
           + (beta**2/(12*alpha))
           - (gamma/12))

    S2 = beta/(2*h) - (gamma*beta*h)/(24*alpha)

    S3 = 1/12

    S4 = -(beta*h)/(24*alpha)

    S5 = 2*S1 - gamma

    P = -((S1+S2)/2) - ((S3+S4)/k)

    Pb = ((S1+S2)/2) - ((S3+S4)/k)

    Q = (S5)/2 - ((1-2*S3)/k)

    Qb = -(S5)/2 - ((1-2*S3)/k)

    R = -(S1-S2)/2 - ((S3-S4)/k)

    Rb = (S1-S2)/2 - ((S3-S4)/k)

    # ------------------------------------------------------
    # Numerical Solution
    # ------------------------------------------------------

    u = np.zeros((m+1, n+1))

    # Initial Condition

    for i in range(m+1):
        u[i,0] = np.sin(x[i])

    # Boundary Conditions

    for j in range(1,n+1):

        u[0,j] = np.exp(2*t[j])*np.sin(-t[j])

        u[m,j] = np.exp(2*t[j])*np.sin(
            x[m]-t[j]
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

            v[i,j] = (
                np.exp(2*t[j])
                * np.sin(x[i]-t[j])
            )

    # ------------------------------------------------------
    # Error
    # ------------------------------------------------------

    err = np.abs(u-v)

    max_error = np.max(err)

    return x,t,u,v,max_error


# ==========================================================
# Mesh Sizes
# ==========================================================

h_values = [
    1/4,
    1/8,
    1/16,
    1/32,
    1/64
]

results = {}
errors = []

# ==========================================================
# Solve PDE For Each h
# ==========================================================

for h in h_values:

    x,t,u,v,error = solve_pde(h)

    results[h] = (x,t,u,v,error)

    errors.append(error)

    print(
        f"h = {h:.6f}   "
        f"Error = {error:.10e}"
    )

# ==========================================================
# Orders
# ==========================================================

orders = [np.nan]

for i in range(1,len(errors)):

    p = np.log(
        errors[i-1]/errors[i]
    ) / np.log(2)

    orders.append(p)

# ==========================================================
# Professional 3D Animation
# ==========================================================

fig = plt.figure(figsize=(15,7))

ax1 = fig.add_subplot(
    121,
    projection='3d'
)

ax2 = fig.add_subplot(
    122,
    projection='3d'
)

# ==========================================================
# Global Z Limits
# ==========================================================

all_values = []

for h in h_values:

    x,t,u,v,error = results[h]

    all_values.extend(u.flatten())
    all_values.extend(v.flatten())

zmin = min(all_values)
zmax = max(all_values)

# ==========================================================
# Main Title
# ==========================================================

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

# ==========================================================
# Update Function
# ==========================================================

def update(frame):

    ax1.clear()
    ax2.clear()

    # Change mesh every 15 frames

    mesh_frame = frame // 15

    if mesh_frame >= len(h_values):
        mesh_frame = len(h_values)-1

    h = h_values[mesh_frame]

    x,t,u,v,error = results[h]

    X,Tmesh = np.meshgrid(x,t)

    angle = 120 + frame*2

    # ------------------------------------------------------
    # Numerical Solution
    # ------------------------------------------------------

    ax1.plot_surface(
        X,
        Tmesh,
        u.T,
        cmap='hot',
        edgecolor='k',
        linewidth=0.3,
        alpha=0.85
    )

    ax1.set_title(
        "Numerical Solution",
        fontsize=13,
        fontweight='bold'
    )

    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_zlabel("u(x,t)")

    ax1.set_zlim(zmin,zmax)

    ax1.view_init(
        elev=25,
        azim=angle
    )

    # ------------------------------------------------------
    # Exact Solution
    # ------------------------------------------------------

    ax2.plot_surface(
        X,
        Tmesh,
        v.T,
        cmap='coolwarm',
        edgecolor='k',
        linewidth=0.3,
        alpha=0.85
    )

    ax2.set_title(
        "Exact Solution",
        fontsize=13,
        fontweight='bold'
    )

    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_zlabel("u(x,t)")

    ax2.set_zlim(zmin,zmax)

    ax2.view_init(
        elev=25,
        azim=angle
    )

    # ------------------------------------------------------
    # Order Text
    # ------------------------------------------------------

    if mesh_frame == 0:
        order_text = "---"
    else:
        order_text = f"{orders[mesh_frame]:.4f}"

    main_title.set_text(
        f"h = {h:.6f}   |   "
        f"Maximum Error = {error:.3e}   |   "
        f"Order = {order_text}"
    )

    return []

# ==========================================================
# Animation
# ==========================================================

ani = FuncAnimation(
    fig,
    update,
    frames=90,
    interval=50,
    repeat=True,
    blit=False
)

# ==========================================================
# Layout
# ==========================================================

plt.tight_layout(
    rect=[0,0,1,0.92]
)

# ==========================================================
# Save GIF
# ==========================================================

ani.save(
    "3-1-PDE_3D_Professional.gif",
    writer=PillowWriter(fps=8)
)

print("\nGIF SAVED SUCCESSFULLY!")

plt.show()