import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt('out_30.txt')
time = data[:, 0]
lift = data[:, 1]
drag = data[:, 2]

# Create figure with 2 subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# ---- Left subplot: Lift ----
axes[0].plot(time, lift, color='blue')
axes[0].set_title('Lift Coefficient')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Coefficient Value')
axes[0].grid(True)

# ---- Right subplot: Drag ----
axes[1].plot(time, drag, color='red')
axes[1].set_title('Drag Coefficient')
axes[1].set_xlabel('Time')
axes[1].grid(True)

# Global title (optional)
fig.suptitle('Lift and Drag Coefficients over Time (Re = 30)', fontsize=14)

# Save with SAME filename as before
plt.tight_layout()
plt.savefig('lift_drag_RE_30.png', dpi=300)
plt.show()
