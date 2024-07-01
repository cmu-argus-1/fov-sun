
import numpy as np
import math


def unit(v):
    """
    Returns unit vector of v
    """
    return v / np.linalg.norm(v)




class Pyramid:
    def __init__(self, azimuths, elevation):

        self.azimuths = azimuths
        self.elevation = elevation

        self.n0 = np.array([np.sin(self.azimuths[0])*np.cos(self.elevation), 
                            np.cos(self.azimuths[0])*np.cos(self.elevation), 
                            np.sin(self.elevation)])
        

        self.n1 = np.array([np.sin(self.azimuths[1])*np.cos(self.elevation), 
                            np.cos(self.azimuths[1])*np.cos(self.elevation), 
                            np.sin(self.elevation)])

        self.n2 = np.array([np.sin(self.azimuths[2])*np.cos(self.elevation), 
                            np.cos(self.azimuths[2])*np.cos(self.elevation), 
                            np.sin(self.elevation)])
                            
        self.n3 = np.array([np.sin(self.azimuths[3])*np.cos(self.elevation), 
                            np.cos(self.azimuths[3])*np.cos(self.elevation), 
                            np.sin(self.elevation)])
        
        self.B = np.array([self.n0, self.n1, self.n2, self.n3])

        self.sun_position = np.array([0, 0, 1])

    def rotation(self, axis, angle):
        """
        Matrix representation for a coordinates system rotation depending on axis
        """
        angle_rad = angle * math.pi/180
        cos = math.cos(angle_rad)
        sin = math.sin(angle_rad)

        M = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

        if axis == 'x' or axis == 'X':
            M[1][1] = M[2][2] = cos
            M[2][1] = sin
            M[1][2] = -sin
        elif axis == 'y' or axis == 'Y':
            M[0][0] = M[2][2] = cos
            M[0][2] = sin
            M[2][0] = -sin
        elif axis == 'z' or axis == 'Z':
            M[0][0] = M[1][1] = cos
            M[1][0] = sin
            M[0][1] = -sin
        
        return M
    
    def true_sun_orientation(self, pitch, yaw, roll):
        """
        Calculates the true sun orientation based on Euler angles rotations 
        of the spacecraft considering a stationary light source with respect 
        to a fixed coordinate system
        """
        rot_X = self.rotation('X', pitch)
        rot_Y = self.rotation('Y', yaw)
        rot_Z = self.rotation('Z', roll)
        rot = np.dot(np.dot(rot_Z, rot_Y), rot_X) 
        sun = np.dot(rot, self.sun_position)
        return np.array([sun[0], sun[1], sun[2]])
    


    def measured_sun_orientation(self, lux):
        """
        Calculates the measured sun orientation based on
        Section 2 of "A Sun Sensor Based on Regular Pyramid Sensor Arrays"
        by Wang et al. (https://doi.org/10.1088/1742-6596/1207/1/012010)
        """

        e = np.array([lux[0], lux[1], lux[2], lux[3]])
        B_T = self.B.transpose()
        B_inv = np.linalg.inv(np.dot(B_T, self.B))
        B = np.dot(B_inv,B_T)
        s = np.dot(B, e)

        return unit(s)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    azimuths = [0.0, np.deg2rad(90), np.deg2rad(180), np.deg2rad(270)]
    elevation = np.deg2rad(45)


    pyramid = Pyramid(azimuths, elevation)

    from fov_visualization import plot_fovs, plot_fovs_hemisphere
    sensor_directions = [pyramid.n0, pyramid.n1, pyramid.n2, pyramid.n3]
    fovs_deg = 4 * [75]
    plot_fovs_hemisphere(sensor_directions, fovs_deg)

    """lux = [1000,1000,0,0]

    measured_vec = pyramid.measured_sun_orientation(lux)

    np.set_printoptions(precision=5, suppress=True, floatmode='fixed')
    print(measured_vec)


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



    ax.quiver(0, 0, 0, measured_vec[0], measured_vec[1], measured_vec[2], color='r', length=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()"""
                
            
