import numpy as np
import matplotlib.pyplot as plt
# -------------------
# Constants
# -------------------

m = 1  # Arbitrary magnetic dipole moment. It should be irrelevant due to normalization of H-field module.

# -------------------
# Inputs
# -------------------

# emitter
x_t = 10 # Position in x-axis [m] of emitter (wrt Inertial reference system [i_h, j_h, k_h]
y_t = 10 # Position in y-axis [m] of emitter (wrt Inertial reference system [i_h, j_h, k_h]
z_t = -1 # Position in z-axis [m] of emitter (wrt Inertial reference system [i_h, j_h, k_h]

alpha = 45 # Inclination of magnetic dipole [deg] wrt to z-axis where 0 is a magnetic dipole aligned with positive z-axis
beta = 0 # Inclination of magnetic dipole [deg] wrt to x-axis where for alpha=90deg & beta=0deg is a magnetic dipole aligned with positive x-axis

# receiver
z_r = 2 # Position in z-axis [m] of emitter (Inertial reference system [i_h, j_h, k_h]

# options
plot_flag = 1 # Flag to ask for plots
plot_2d_flag = 0 # Flag to plot H-field magnitude and inclination [just theta], useful to compare with univ. Zaragoza
plot_colormap_flag = 0 # Flag to plot 2D map (contour) of H-field magnitude normalized with max. magnitude [in dB]
plot_3d_flag = 1 # Flag to plot 3D map of H-field magnitude normalized with max. magnitude [in dB]

spatial_resolution = 2 # Distance in [m] between consecutive readings in x and y axis

# -------------------
# Auxiliary Functions
# -------------------

# Change from deg to rad & from rad to deg
def deg_to_rad (ang_deg):
    return (ang_deg * np.pi / 180)

def rad_to_deg (ang_rad):
    return (ang_rad * 180 / np.pi)

# Define the magnetic field components

def Hx(x, y, z, alpha, beta):
    r = np.sqrt(x**2 + y**2 + z**2)
    return (m / (4 * np.pi * r**5)) * (
        3 * (np.sin(alpha) * np.cos(beta) * x**2 + np.sin(alpha) * np.sin(beta) * x * y + np.cos(alpha) * z * x) -
        r**2 * np.sin(alpha) * np.cos(beta)   
    )

def Hy(x, y, z, alpha, beta):
    r = np.sqrt(x**2 + y**2 + z**2)
    return (m / (4 * np.pi * r**5)) * (
        3 * (np.sin(alpha) * np.cos(beta) * x * y + np.sin(alpha) * np.sin(beta) * y**2 + np.cos(alpha) * y * z) -
        r**2 * np.sin(alpha) * np.sin(beta)
    )

def Hz(x, y, z, alpha, beta):
    r = np.sqrt(x**2 + y**2 + z**2)
    return (m / (4 * np.pi * r**5)) * (
        3 * (np.sin(alpha) * np.cos(beta) * x * z + np.sin(alpha) * np.sin(beta) * y * z + np.cos(alpha) * z**2) -
        r**2 * np.cos(alpha)
    )

# Define the module of the H field
def H_field(x, y, z, alpha, beta):
    #retrive individual components
    H_field_x = Hx(x, y, z, alpha, beta)
    H_field_y = Hy(x, y, z, alpha, beta)
    H_field_z = Hz(x, y, z, alpha, beta)

    H_module = np.sqrt(H_field_x ** 2 + H_field_y ** 2 + H_field_z ** 2)
    H_theta = rad_to_deg(np.arccos(H_field_z / H_module))
    #H_theta = rad_to_deg(np.arcsin(H_field_z / H_module)) #Implied Definition for Zaragoza Univ. Results comparison
    H_phi = rad_to_deg(np.arctan(H_field_y / H_field_x))

    return [H_field_x, H_field_y, H_field_z, H_module, H_theta, H_phi]



# Plotting function

def plot_function(x_range, y_range, H_field_theta_map, H_field_module_normalized_db_map, H_field_module_normalized_map):

    if plot_2d_flag:
        fig1, (ax1, ax2) = plt.subplots(1, 2)  #
        fig1.set_size_inches(15, 8)
        #ax1.plot(x_range, H_field_module_normalized_db_map[:, 0],
                 #label='H-field measured at z = ' + str(z_r) + 'm over surface')
        ax1.plot(x_range, H_field_module_normalized_map[:, 0],
                 label='H-field measured at z = ' + str(z_r) + 'm over surface')
        ax2.plot(x_range, H_field_theta_map[:, 0],
                 label='H-field inclination measured at z = ' + str(z_r) + 'm over surface')
        
        ax1.set_yscale('log')
        ax1.set_xlabel('Distance in x-axis [m] - Inertial Reference System')
        ax1.set_ylabel('Normalized H-field magnitude [dB]')
        ax1.set_title('H-field Magnitude vs Distance in X-axis')
        ax1.grid(True, which="both", ls="--")
        ax1.set_xlim([-150, 150]) #0, 20
        #ax1.set_ylim([-120, 20])
        ax1.set_ylim([10**-6,10**0])
        ax1.legend()

        ax2.set_xlabel('Distance in x-axis [m] - Inertial Reference System')
        ax2.set_ylabel('Magnetic Inclination H-field z-x plane [deg]')
        ax2.set_title('Magnetic Inclination vs Distance in X-axis')
        ax2.grid(True, which="both", ls="--")
        ax2.set_xlim([-150, 150])
        ax2.set_ylim([-90, 90])
        ax2.legend()

    if plot_colormap_flag:
        fig2, ax3 = plt.subplots(1, 1)  #
        fig2.set_size_inches(12, 10)

        CS = ax3.contourf(x_range, y_range, H_field_module_normalized_db_map, 15, cmap='winter')
        # CS2 = ax3.contour(CS, levels=CS.levels, colors='r')

        ax3.set_title('Normalized H-field magnitude [dB]')
        ax3.set_xlabel('Distance in x-axis [m] - Inertial Reference System')
        ax3.set_ylabel('Distance in y-axis [m] - Inertial Reference System')

        # Make a colorbar for the ContourSet returned by the contourf call.
        cbar = fig2.colorbar(CS)
        cbar.ax.set_ylabel('Normalized H-field magnitude [dB]')

        # Add the contour line levels to the colorbar
        # cbar.add_lines(CS2)

        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()

    if plot_3d_flag:
        x_meshgrid, y_meshgrid = np.meshgrid(x_range, y_range)
        ax4 = plt.figure(figsize=(12, 10)).add_subplot(projection='3d')

        # Plot the 3D surface
        surface_res = np.ceil(spatial_resolution / 2)
        ax4.plot_surface(x_meshgrid, y_meshgrid, H_field_module_normalized_db_map, edgecolor='royalblue', lw=0.5,
                         rstride=1, cstride=1, alpha=0.3)

        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph
        CS2 = ax4.contourf(x_meshgrid, y_meshgrid, H_field_module_normalized_db_map, 15, zdir='z', offset=-60,
                     cmap='winter')  # cmap coolwarm
        CS3 = ax4.contourf(x_meshgrid, y_meshgrid, H_field_module_normalized_db_map, 15, zdir='x', offset=-5,
                     cmap='winter')
        CS4 = ax4.contourf(x_meshgrid, y_meshgrid, H_field_module_normalized_db_map, 15, zdir='y', offset=25,
                     cmap='winter')

        ax4.set(xlim=(-5, 25), ylim=(-5, 25), zlim=(-60, 10))

        ax4.set_title('3D view - Emitter @ [' + str(x_t) + ',' + str(y_t) + ',' + str(
            z_t) + ']m - Receiver @ z-plane = ' + str(z_r) + 'm')
        ax4.set_xlabel('Distance in x-axis [m] - Inertial Ref. System')
        ax4.set_ylabel('Distance in y-axis [m] - Inertial Ref. System')
        ax4.set_zlabel('Normalized H-field magnitude [dB]')

    plt.show()

# -------------------
# Main Function
# -------------------
def main():
    # Define the x rnge and y range to compute H field the receiving z-plane
    x_range = np.arange(0 + spatial_resolution / 2, 20 + spatial_resolution / 2, spatial_resolution)
    y_range = np.arange(0 + spatial_resolution / 2, 20 + spatial_resolution / 2, spatial_resolution)
    # x,y ranges for comparison with Univ. Zaragoza
    # x_range = np.arange(-150, 150+spatial_resolution, spatial_resolution)
    # y_range = [0]

    # Variables Initialization
    H_field_x_map = np.empty([len(x_range), len(y_range)])
    H_field_y_map = np.empty([len(x_range), len(y_range)])
    H_field_z_map = np.empty([len(x_range), len(y_range)])

    H_field_module_map = np.empty([len(x_range), len(y_range)])
    H_field_module_normalized_map = np.empty([len(x_range), len(y_range)])

    H_field_theta_map = np.empty([len(x_range), len(y_range)])
    H_field_phi_map = np.empty([len(x_range), len(y_range)])

    for [i,x_r] in enumerate(x_range):
        for [j,y_r] in enumerate(y_range):

            [H_field_x, H_field_y, H_field_z, H_field_module, H_field_theta, H_field_phi] = H_field(x_r - x_t, y_r - y_t, z_r - z_t, deg_to_rad(alpha), deg_to_rad(beta))

            H_field_x_map[i, j] = H_field_x
            H_field_y_map[i, j] = H_field_y
            H_field_z_map[i, j] = H_field_z

            H_field_module_map[i, j] = H_field_module
            H_field_theta_map[i, j] = H_field_theta
            H_field_phi_map[i, j] = H_field_phi

    # Normalize the H field module values

    H_max = H_field_module_map.max()
    H_field_module_normalized_map = np.divide(H_field_module_map, H_max)
    H_field_module_normalized_db_map = 20*np.log10(H_field_module_normalized_map)

    # Plot the results
    if plot_flag:
        plot_function(x_range, y_range, H_field_theta_map, H_field_module_normalized_db_map, H_field_module_normalized_map)

if __name__ == '__main__':
    main()