import matplotlib.pyplot as plt
import numpy as np

def plot_beam_data(C_1, angle, z, grid_shape, z_label, z_lims=None, train_pts=None, test_pts=None, num_levels=1000, num_ticks=6, ticks_dp=2):
    
    if z_lims is None:
        z_lims = (np.around(np.min(z), decimals=ticks_dp), 
                  np.around(np.max(z), decimals=ticks_dp))
    levels = np.linspace(z_lims[0], z_lims[1], num_levels)
    
    # Reshape inputs:
    C_1, angle, z = C_1.reshape(grid_shape), angle.reshape(grid_shape), z.reshape(grid_shape)
    
    # Create surface plot:
    fig, ax = plt.subplots()
    contour_fig = ax.contourf(C_1, angle, z, levels=levels, cmap='coolwarm')
    ticks = np.linspace(z_lims[0], z_lims[1], num_ticks)
    cbar = fig.colorbar(contour_fig, ticks=ticks)
    cbar.set_label(z_label, rotation=270, labelpad=15)
    ax.set_xlabel("Normalised Stiffness")
    ax.set_ylabel('Normalised Beam Angle')
    fig.patch.set_facecolor('white')
    
    if train_pts is not None:
        # Plots as 'x's
        plt.plot(train_pts['x'][:,0], train_pts['x'][:,1], 'x', color='black', markersize=5)
    
    if test_pts is not None:
        # Plots as 'x's
        plt.plot(test_pts['x'][:,0], test_pts['x'][:,1], 'x', color='green', markersize=5)
    
    return fig

def create_gp_prediction_grid(C_1_lims, angle_lims, num_C_1_pts, num_angle_pts):
    C_1_pts = np.linspace(np.atleast_1d(C_1_lims[0]), np.atleast_1d(C_1_lims[1]), num_C_1_pts)
    angle_pts = np.linspace(np.atleast_1d(angle_lims[0]), np.atleast_1d(angle_lims[1]), num_angle_pts)
    C_1_grid, angle_grid = np.meshgrid(C_1_pts.squeeze(), angle_pts.squeeze())
    pred_pts = np.stack([C_1_grid.flatten(), angle_grid.flatten()], axis=1)
    return C_1_grid, angle_grid, pred_pts 
