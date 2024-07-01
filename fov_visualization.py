import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from icosphere import icosphere

def dcm_from_q(q):
    norm = np.linalg.norm(q)
    q0, q1, q2, q3 = q / norm if norm != 0 else q
    # DCM
    Q = np.array(
        [
            [
                2 * q1**2 + 2 * q0**2 - 1,
                2 * (q1 * q2 - q3 * q0),
                2 * (q1 * q3 + q2 * q0),
            ],
            [
                2 * (q1 * q2 + q3 * q0),
                2 * q2**2 + 2 * q0**2 - 1,
                2 * (q2 * q3 - q1 * q0),
            ],
            [
                2 * (q1 * q3 - q2 * q0),
                2 * (q2 * q3 + q1 * q0),
                2 * q3**2 + 2 * q0**2 - 1,
            ],
        ]
    )
    return Q


def rot_from_two_vectors(Pc, Pt=np.array([0, 0, 1])):
    c = np.cross(Pt, Pc)

    if np.linalg.norm(c) < 1e-10:
        if np.dot(Pc,Pt) > 0:
            q0 =  np.array([1,0,0.0,0])
        else:
            q0 =  np.array([0,1,0.0,0])
    else:
        n = c / np.linalg.norm(c)
        θ = np.arctan2(np.linalg.norm(c),np.dot(Pc,Pt))

        q0 = np.zeros(4)
        q0[0] = np.cos(θ/2)
        q0[1:] = n*np.sin(θ/2)

    return dcm_from_q(q0)



"""def plot_fovs(sensor_directions, fovs_deg):
    # Unit sphere
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the unit sphere
    ax.plot_surface(x, y, z, color='b', alpha=0.1, rstride=5, cstride=5)


    N = len(sensor_directions)


    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for i in range(N):
        sensor_direction = sensor_directions[i]
        fov_deg = fovs_deg[i]

        # Plot the sensor direction
        ax.quiver(0, 0, 0, sensor_direction[0], sensor_direction[1], sensor_direction[2], color=colors[i], length=0.5)

        # Calculate the FOV boundary on the sphere
        fov_rad = np.radians(90 - fov_deg)
        u = np.linspace(0, 2 * np.pi, 100)
        fov_x = np.cos(fov_rad) * np.cos(u)
        fov_y = np.cos(fov_rad) * np.sin(u)
        fov_z = np.sin(fov_rad) * np.ones_like(u)

        # Rotation matrix to align FOV with sensor direction
        rot_matrix = rot_from_two_vectors(sensor_direction)  
        fov_boundary = np.dot(rot_matrix, np.array([fov_x, fov_y, fov_z]))

        # Plot FOV boundary
        #ax.plot(fov_boundary[0], fov_boundary[1], fov_boundary[2], color=color[i])
        vertices = np.array([fov_boundary[0], fov_boundary[1], fov_boundary[2]]).T
        poly = art3d.Poly3DCollection([vertices], alpha=0.3, facecolor=colors[i])
        ax.add_collection3d(poly)

    # Labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()"""

def plot_fovs(sensor_directions, fovs_deg):
    # Unit sphere
    phi, theta = np.mgrid[0.0:np.pi:150j, 0.0:2.0*np.pi:150j]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    print(x.shape, y.shape, z.shape)

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the unit sphere
    ax.plot_surface(x, y, z, color='b', alpha=0.05, rstride=5, cstride=5)

    N = len(sensor_directions)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # Generate points on the unit sphere
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    for i in range(N):
        sensor_direction = sensor_directions[i]
        fov_deg = fovs_deg[i]
        fov_rad = np.radians(fov_deg)

        # Calculate angles between the sensor direction and points on the sphere
        angles = np.arccos(np.dot(points, sensor_direction) / np.linalg.norm(sensor_direction))
        
        # Mask points within the FOV
        mask = angles < fov_rad
        x_fov = points[mask, 0]
        y_fov = points[mask, 1]
        z_fov = points[mask, 2]


        # Plot FOV on the unit sphere
        ax.scatter(x_fov, y_fov, z_fov, color=colors[i], alpha=0.3, s=0.5)

        # Plot the sensor direction
        ax.quiver(0, 0, 0, sensor_direction[0], sensor_direction[1], sensor_direction[2], color=colors[i], length=0.5)

    # Labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    plt.show()

def plot_fovs_ico(sensor_directions, fovs_deg, density_icosphere=30):
    # Unit sphere
    phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the unit sphere
    ax.plot_surface(x, y, z, color='b', alpha=0.05, rstride=5, cstride=5)

    N = len(sensor_directions)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    nu = density_icosphere  # or any other integer
    vertices, faces = icosphere(nu)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    points = np.vstack((x, y, z)).T

    for i in range(N):
        sensor_direction = sensor_directions[i]
        fov_deg = fovs_deg[i]
        fov_rad = np.radians(fov_deg)

        # Calculate angles between the sensor direction and points on the sphere
        angles = np.arccos(np.dot(points, sensor_direction) / np.linalg.norm(sensor_direction))
        
        # Mask points within the FOV
        mask = angles < fov_rad
        x_fov = points[mask, 0]
        y_fov = points[mask, 1]
        z_fov = points[mask, 2]


        # Plot FOV on the unit sphere
        ax.scatter(x_fov, y_fov, z_fov, color=colors[i], alpha=0.4, s=5)

        # Plot the sensor direction
        ax.quiver(0, 0, 0, sensor_direction[0], sensor_direction[1], sensor_direction[2], color=colors[i], length=0.5)

    # Labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    plt.show()





def plot_fovs_hemisphere(sensor_directions, fovs_deg, density_icosphere=30):
    # Unit sphere
    phi, theta = np.mgrid[0.0:np.pi/2:50j, 0.0:2.0*np.pi:50j]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the unit sphere
    ax.plot_surface(x, y, z, color='b', alpha=0.05, rstride=5, cstride=5)

    N = len(sensor_directions)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # Generate points on the unit sphere
    nu = density_icosphere  # or any other integer
    vertices, faces = icosphere(nu)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    points = np.vstack((x, y, z)).T

    for i in range(N):
        sensor_direction = sensor_directions[i]
        fov_deg = fovs_deg[i]
        fov_rad = np.radians(fov_deg)

        # Calculate angles between the sensor direction and points on the sphere
        angles = np.arccos(np.dot(points, sensor_direction) / np.linalg.norm(sensor_direction))
        
        # Mask points within the FOV
        mask = angles < fov_rad 
        x_fov = points[mask, 0]
        y_fov = points[mask, 1]
        z_fov = points[mask, 2]

        # Mask points with negative z-coordinate
        mask_negative_z = z_fov > 0
        x_fov = x_fov[mask_negative_z]
        y_fov = y_fov[mask_negative_z]
        z_fov = z_fov[mask_negative_z]
        

        # Plot FOV on the unit sphere
        ax.scatter(x_fov, y_fov, z_fov, color=colors[i], alpha=0.3, s=5)

        # Plot the sensor direction
        ax.quiver(0, 0, 0, sensor_direction[0], sensor_direction[1], sensor_direction[2], color=colors[i], length=0.5)

    # Labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    plt.show()


def plot_fovs_2d(sensor_directions, fovs_deg, density_icosphere=50, hemisphere=False):
    # Generate points on the unit sphere
    nu = density_icosphere  # or any other integer
    vertices, faces = icosphere(nu)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    points = np.vstack((x, y, z)).T

    # Convert points to azimuth and elevation
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    N = len(sensor_directions)

    for i in range(N):
        sensor_direction = sensor_directions[i]
        fov_deg = fovs_deg[i]
        fov_rad = np.radians(fov_deg)

        # Calculate angles between the sensor direction and points on the sphere
        angles = np.arccos(np.dot(points, sensor_direction) / np.linalg.norm(sensor_direction))
        
        # Mask points within the FOV
        mask = angles < fov_rad 
        azimuth_fov = azimuth[mask]
        elevation_fov = elevation[mask]


        if hemisphere:
            mask_negative_elevation = elevation_fov > 0
            azimuth_fov = azimuth_fov[mask_negative_elevation]
            elevation_fov = elevation_fov[mask_negative_elevation]


        # Plot FOV on 2D plot
        ax.scatter(np.rad2deg(azimuth_fov), np.rad2deg(elevation_fov), color=colors[i], alpha=0.3, s=5, label=f'Sensor {i+1}')

        # Plot the sensor direction
        sensor_azimuth = np.rad2deg(np.arctan2(sensor_direction[1], sensor_direction[0]))
        sensor_elevation = np.rad2deg(np.arcsin(sensor_direction[2]))
 
        ax.plot(sensor_azimuth, sensor_elevation, 'o', color=colors[i])
            

    # Labels and show plot
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Elevation (deg)')
    ax.legend()
    plt.show()

def plot_fovs_sensor_coverage(sensor_directions, fovs_deg, density_icosphere=50, hemisphere=False):
    # Generate points on the unit sphere
    nu = density_icosphere  # or any other integer
    vertices, faces = icosphere(nu)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    points = np.vstack((x, y, z)).T

    # Initialize coverage count array
    coverage_count = np.zeros(len(points), dtype=int)

    N = len(sensor_directions)

    for i in range(N):
        sensor_direction = sensor_directions[i]
        fov_deg = fovs_deg[i]
        fov_rad = np.radians(fov_deg)

        # Calculate angles between the sensor direction and points on the sphere
        angles = np.arccos(np.dot(points, sensor_direction) / np.linalg.norm(sensor_direction))
        
        # Mask points within the FOV
        mask = angles < fov_rad 
        coverage_count[mask] += 1

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    max_coverage = np.max(coverage_count)
    colors = plt.cm.viridis(np.linspace(0, 1, max_coverage))

    for i in range(1, max_coverage + 1):
        mask = coverage_count == i
        x_fov = points[mask, 0]
        y_fov = points[mask, 1]
        z_fov = points[mask, 2]

        if hemisphere:
            mask_negative_z = z_fov > 0
            x_fov = x_fov[mask_negative_z]
            y_fov = y_fov[mask_negative_z]
            z_fov = z_fov[mask_negative_z]

        # Plot FOV on the unit sphere
        ax.scatter(x_fov, y_fov, z_fov, color=colors[i-1], alpha=0.3, s=5, label=f'Covered by {i} sensor(s)')

    # Labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()



def plot_fovs_sensor_coverage_2d(sensor_directions, fovs_deg, density_icosphere=50, hemisphere=False):
    # Generate points on the unit sphere
    nu = density_icosphere  # or any other integer
    vertices, faces = icosphere(nu)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    points = np.vstack((x, y, z)).T

    # Convert points to azimuth and elevation
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z)

    # Initialize coverage count array
    coverage_count = np.zeros(len(points), dtype=int)

    N = len(sensor_directions)

    for i in range(N):
        sensor_direction = sensor_directions[i]
        fov_deg = fovs_deg[i]
        fov_rad = np.radians(fov_deg)

        # Calculate angles between the sensor direction and points on the sphere
        angles = np.arccos(np.dot(points, sensor_direction) / np.linalg.norm(sensor_direction))
        
        # Mask points within the FOV
        mask = angles < fov_rad 
        coverage_count[mask] += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    
    max_coverage = np.max(coverage_count)
    colors = plt.cm.viridis(np.linspace(0, 1, max_coverage))

    for i in range(1, max_coverage + 1):
        mask = coverage_count == i
        azimuth_fov = azimuth[mask]
        elevation_fov = elevation[mask]

        if hemisphere:
            mask_negative_elevation = elevation_fov > 0
            azimuth_fov = azimuth_fov[mask_negative_elevation]
            elevation_fov = elevation_fov[mask_negative_elevation]

        ax.scatter(np.rad2deg(azimuth_fov), np.rad2deg(elevation_fov), color=colors[i-1], alpha=0.3, s=5, label=f'Covered by {i} sensor(s)')

    # Labels and show plot
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Elevation (deg)')
    ax.legend()
    plt.show()

def set_axes_equal(ax):
    """
    https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z 

    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == '__main__':


    # Sensor parameters
    #sensor_directions = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([-1, 0, 0]), np.array([0, -1, 0])]
    #fovs_deg = 5 * [75]


    sensor_directions = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([-1, 0, 0]), np.array([0, -1, 0])]
    fovs_deg = 5 * [75]

    plot_fovs_ico(sensor_directions, fovs_deg)








