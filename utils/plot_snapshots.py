from environment.env import Env
import config
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np

# Set scientific plotting style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':'
})

def _draw_sphere_wireframe(ax, center: np.ndarray, radius: float, color: str, alpha: float = 0.1) -> None:
    """Draw a sparse wireframe sphere to represent 3D coverage area."""
    # Reduced density for cleaner look
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 7)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.5)


def _draw_beam_cone(ax, uav_pos: np.ndarray, beam_dir: tuple[float, float], 
                    length: float = 80.0, color: str = 'tab:orange') -> None:
    """Draw a simplified line and cone representing the beam direction."""
    theta_rad = np.radians(beam_dir[0])
    phi_rad = np.radians(beam_dir[1])
    
    dir_x = np.sin(theta_rad) * np.cos(phi_rad)
    dir_y = np.sin(theta_rad) * np.sin(phi_rad)
    dir_z = np.cos(theta_rad)
    
    end_point = uav_pos + length * np.array([dir_x, dir_y, dir_z])
    
    # Draw beam center line (thinner, cleaner)
    ax.plot([uav_pos[0], end_point[0]], [uav_pos[1], end_point[1]], [uav_pos[2], end_point[2]], 
            color=color, linewidth=1.5, alpha=0.8, linestyle='-')


def plot_snapshot(env: Env, progress_step: int, step: int, save_dir: str, name: str, timestamp: str, initial: bool = False) -> None:
    """Generates and saves a publication-quality 3D plot of the environment."""
    save_path = f"{save_dir}/state_images_{timestamp}/{name}_{progress_step:04d}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Clean 3D style: remove gray panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Set axis limits
    ax.set_xlim(0, config.AREA_WIDTH)
    ax.set_ylim(0, config.AREA_HEIGHT)
    ax.set_zlim(0, config.UE_MAX_ALT)
    
    ax.set_title(f"Network Topology Snapshot\n{name.title()} - Episode: {progress_step}, Step: {step}")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Altitude (m)")

    # Separate ground and aerial UEs
    ground_ues = [ue for ue in env.ues if ue.pos[2] < 1.0]
    aerial_ues = [ue for ue in env.ues if ue.pos[2] >= 1.0]
    
    # Plot ground UEs (Black dots)
    if ground_ues:
        ground_pos = np.array([ue.pos for ue in ground_ues])
        # Base color black
        colors = ["black"] * len(ground_ues)
        # Mark served UEs as green
        for i, ue in enumerate(ground_ues):
            if ue.assigned:
                colors[i] = "tab:green"
        
        ax.scatter(ground_pos[:, 0], ground_pos[:, 1], ground_pos[:, 2], 
                   c=colors, marker=".", s=15, label="Ground UEs", alpha=0.6, edgecolors='none')
    
    # Plot aerial UEs (Blue dots)
    if aerial_ues:
        aerial_pos = np.array([ue.pos for ue in aerial_ues])
        # Base color blue
        colors = ["tab:blue"] * len(aerial_ues)
        # Mark served UEs as green
        for i, ue in enumerate(aerial_ues):
            if ue.assigned:
                colors[i] = "tab:green"

        ax.scatter(aerial_pos[:, 0], aerial_pos[:, 1], aerial_pos[:, 2], 
                   c=colors, marker=".", s=25, label="Aerial UEs", alpha=0.7, edgecolors='none')

    # Plot UAVs and connections
    uav_plotted = False
    for uav in env.uavs:
        label = "UAV (ABS)" if not uav_plotted else ""
        # UAV position
        ax.scatter(uav.pos[0], uav.pos[1], uav.pos[2], 
                   c="tab:red", marker="s", s=80, label=label, edgecolors='white', linewidths=0.5, alpha=1.0)
        uav_plotted = True
        
        # Coverage area (faint)
        _draw_sphere_wireframe(ax, uav.pos, config.UAV_COVERAGE_RADIUS, "tab:red", alpha=0.1)
        
        # Beam direction
        beam_dir = uav.get_final_beam_direction()
        _draw_beam_cone(ax, uav.pos, beam_dir, length=config.UAV_COVERAGE_RADIUS * 0.7, color='tab:orange')

        # Association Links (very faint)
        for ue in uav.current_covered_ues:
            ax.plot([uav.pos[0], ue.pos[0]], [uav.pos[1], ue.pos[1]], [uav.pos[2], ue.pos[2]], 
                    color="gray", linestyle="-", linewidth=0.3, alpha=0.3)

        # Collaboration Links
        if uav.current_collaborator:
            ax.plot([uav.pos[0], uav.current_collaborator.pos[0]], 
                    [uav.pos[1], uav.current_collaborator.pos[1]], 
                    [uav.pos[2], uav.current_collaborator.pos[2]], 
                    color="tab:purple", linestyle="--", linewidth=1.0, alpha=0.7)

    # Plot MBS
    ax.scatter(config.MBS_POS[0], config.MBS_POS[1], config.MBS_POS[2], 
               c="black", marker="D", s=100, label="MBS", edgecolors='white', linewidths=0.5)

    # Ground plane reference
    xx, yy = np.meshgrid([0, config.AREA_WIDTH], [0, config.AREA_HEIGHT])
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.03, color='gray', shade=False)

    # View angle
    ax.view_init(elev=25, azim=45)
    
    # Legend
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor='gray', fontsize=9)

    # Save
    filename = "initial.png" if initial else f"step_{step:04d}.png"
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
