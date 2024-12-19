import numpy as np

from configs import *


class Orbit:

    def __init__(self, radius, right_ascension, inclination, initial_true_anomaly):

        self.radius = radius
        self.right_ascension = np.deg2rad(right_ascension)
        self.inclination = np.deg2rad(inclination)
        self.initial_true_anomaly = np.deg2rad(initial_true_anomaly)

    def _get_curr_true_anomaly_t(self, t):

        """
        Calculate the true anomaly after a certain time for a circular orbit.

        Parameters:
        a       : Semi-major axis (orbital radius for circular orbit) [m]
        mu      : Standard gravitational parameter of the central body [m^3/s^2]
        nu_0    : Initial true anomaly [radians]
        delta_t : Time elapsed since initial position [seconds]

        Returns:
        nu      : True anomaly after delta_t [radians]
        """
        # Orbital period
        T = 2 * np.pi * np.sqrt(self.radius**3 / MARS_G_CONST)

        # Mean motion
        n = 2 * np.pi / T

        # Compute true anomaly
        self.curr_true_anomaly = (self.initial_true_anomaly + n * t) % (2 * np.pi)

    def _get_rotation_matrix(self):

        """
        Compute the rotation matrix to transform coordinates from the perifocal
        (PQW) frame to the inertial (IJK) frame based on orbital Euler angles.
        """
        # Extract Euler angles
        RAAN = self.right_ascension  # Right Ascension of the Ascending Node
        inclination = self.inclination
        arg_of_periapsis = 0  # Argument of periapsis (set to 0 for circular orbits)

        # Rotation matrices
        R_RAAN = np.array([
            [np.cos(RAAN), -np.sin(RAAN), 0],
            [np.sin(RAAN),  np.cos(RAAN), 0],
            [0,             0,            1]
        ])

        R_inclination = np.array([
            [1, 0,                  0],
            [0, np.cos(inclination), -np.sin(inclination)],
            [0, np.sin(inclination),  np.cos(inclination)]
        ])

        R_arg_of_periapsis = np.array([
            [np.cos(arg_of_periapsis), -np.sin(arg_of_periapsis), 0],
            [np.sin(arg_of_periapsis),  np.cos(arg_of_periapsis), 0],
            [0,                        0,                        1]
        ])

        # Combined rotation matrix
        self.rotation_matrix = R_RAAN @ R_inclination @ R_arg_of_periapsis

    def find_inertial_r_and_v_t(self, t):

        """
        Compute the position and velocity vectors in the inertial frame.
        
        Parameters:
            radius: Orbital radius (assumed circular orbit).
            orbit_euler_angles: Tuple of (right ascension, inclination, true anomaly).
        """
        self._get_curr_true_anomaly_t(t)

        # Compute position and velocity in the perifocal frame
        self.r_o = np.array([
            self.radius * np.cos(self.curr_true_anomaly), 
            self.radius * np.sin(self.curr_true_anomaly),
            0
        ])

        self.v_o = np.array([
            np.sqrt(MARS_G_CONST / self.radius) * -np.sin(self.curr_true_anomaly), 
            np.sqrt(MARS_G_CONST / self.radius) * np.cos(self.curr_true_anomaly),
            0
        ])

        # Compute rotation matrix
        self._get_rotation_matrix()

        # Transform to inertial frame
        self.r_inertial = self.rotation_matrix @ self.r_o
        self.v_inertial = self.rotation_matrix @ self.v_o

    def _get_h_t(self, t):

        self.find_inertial_r_and_v_t(t)

        i_r = self.r_inertial / np.linalg.norm(self.r_inertial)
        i_h = np.cross(self.r_inertial, self.v_inertial) / np.linalg.norm(np.cross(self.r_inertial, self.v_inertial))
        i_theta = np.cross(i_h, i_r)

        self.curr_hn = np.column_stack([i_r, i_theta, i_h])

    def find_dcm_hn_t(self, t):

        self._get_h_t(t)

if __name__ == "__main__":

    nano_orbit = Orbit(5000, 30, 30, 30)
    mother_orbit = Orbit(5000, 30, 30, 30)

    ############### Task 1 ###############
    nano_orbit.find_inertial_r_and_v_t(450)
    mother_orbit.find_inertial_r_and_v_t(1150)

    ############### Task 2 ###############
    nano_orbit._get_h_t(500)
