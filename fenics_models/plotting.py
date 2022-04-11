def plot_convergence_contours(data, y_axis, num_levels=1000, z_lims=None):
    
    disp, vols, kappa = np.array(data['disp']), np.array(data['volume']), np.array(data['bulk_modulus'])
    
    y = data[y_axis]
    y = np.array(y)
        
    set_zlims = True if z_lims is None else False
    
    # Compute volume change:
    dvol = 100*(vols[:,0]-vols[:,1])/vols[:,0]
    
    # Reshape arrays:
    grid_shape = [np.unique(x).size for x in (kappa, y)]
    
    # Create surface plot:
    for i, z in enumerate([disp, dvol]):
        fig, ax = plt.subplots()
        
        #Set z limits + color levels:
        if set_zlims:
            z_lims = (z.min(), z.max()) 
        levels = np.linspace(z_lims[0], z_lims[1], num_levels)
        
        contour_fig = ax.contourf(kappa.reshape(grid_shape), y.reshape(grid_shape), z.reshape(grid_shape), 
                                  levels=levels, cmap=cm.coolwarm) 
        
        # Adjust ticks + color bar:
        ticks = np.linspace(z_lims[0], z_lims[1], 10)
        cbar = fig.colorbar(contour_fig, ticks=ticks) 
        cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in ticks])
        
        # Set labels:
        y_label = 'Number of Elements' if y_axis=='num_elem' else "Young's Modulus (kPa)"
        z_label = 'End Displacement (mm)' if i==0 else 'Percentage Volume Change'    
        ax.set_xlabel('Bulk Modulus (kPa)')
        ax.set_ylabel(y_label)
        cbar.set_label(z_label, rotation=270, labelpad=15)
        
        # Add data points:
        plt.plot(kappa, y, 'x', color='black', markersize=6)
        fig.patch.set_facecolor('white')
        plt.show()
        
def plot_convergence_lines_kappa_and_mesh(data, elem_plot_val=300, kappa_plot_val=1000, labelpad=8):
    
    disp, vols, num_elem, kappa = np.array(data['disp']), np.array(data['volume']), \
                                  np.array(data['num_elem']), np.array(data['bulk_modulus'])
    
    dvol = 100*(vols[:,0]-vols[:,1])/vols[:,0]
    
    # Plot volume %:
    plot_idx = num_elem==elem_plot_val
    fig, ax = plt.subplots()
    sns.lineplot(x=kappa[plot_idx], y=dvol[plot_idx])
    plt.plot(kappa[plot_idx], dvol[plot_idx], 'x', color='black', markersize=10)
    ax.set_xlabel('Bulk Modulus (kPa)', labelpad=labelpad)
    ax.set_ylabel('Percentage Volume Change', labelpad=labelpad)
    plt.show()
    
    # Plot displacement:
    plot_idx = kappa==kappa_plot_val
    fig, ax = plt.subplots()
    sns.lineplot(x=num_elem[plot_idx], y=disp[plot_idx])
    plt.plot(num_elem[plot_idx], disp[plot_idx], 'x', color='black', markersize=10)
    ax.set_xlabel('Number of Elements', labelpad=labelpad)
    ax.set_ylabel('End Displacement (mm)', labelpad=labelpad)
    plt.show()

def plot_convergence_lines_kappa_and_E(data, E_plot_val=40, labelpad=8):
    
    vols, kappa, E = np.array(data['volume']), np.array(data['bulk_modulus']), np.array(data['E'])
    
    dvol = 100*(vols[:,0]-vols[:,1])/vols[:,0]
    
    # Plot volume %:
    plot_idx = E==E_plot_val
    fig, ax = plt.subplots()
    sns.lineplot(x=kappa[plot_idx], y=dvol[plot_idx])
    plt.plot(kappa[plot_idx], dvol[plot_idx], 'x', color='black', markersize=10)
    ax.set_xlabel('Bulk Modulus (kPa)', labelpad=labelpad)
    ax.set_ylabel('Percentage Volume Change', labelpad=labelpad)
    plt.show()