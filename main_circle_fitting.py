from dataclasses import dataclass
from shapely.geometry import Point
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
from ransac import Circle_Model, Ransac
import numpy as np
from math import cos, sin, radians
import pandas as pd


global R
R = 6371 * 1000  # Earth radius in in km


@dataclass
class PointData:
    """Data class for point data"""

    latitude: float
    longitude: float
    altitude: float

    def to_cartesian(self):
        lat_rad = radians(self.latitude)
        lon_rad = radians(self.longitude)
        x = R * cos(lat_rad) * cos(lon_rad)
        y = R * cos(lat_rad) * sin(lon_rad)
        z = R * sin(lat_rad)
        return (x, y, z)

    def to_point(self):
        return Point(self.longitude, self.latitude)

    def coordinates(self):
        return (self.longitude, self.latitude, self.altitude)


class PlaneFitting:
    """Class to fit a plane to a set of 3D points and project the points to the plane"""

    data = None
    normal = None
    d = None
    centroid = None

    def fit_plane(self, data):
        # fit the model
        self.data = data
        # calculate the centroid and subtract it from the data
        self.centroid = np.mean(self.data, axis=0)
        self.data = self.data - self.centroid
        # calculate the covariance matrix
        cov = np.cov(self.data, rowvar=False)
        # calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # sort the eigenvalues in descending order
        self.normal = eigenvectors[:, np.argmin(eigenvalues)]
        # calculate the d value
        self.d = -np.dot(self.normal, self.centroid)
        return self.normal, self.d

    def project_points(self, data):
        # project a point to the plane
        self.fit_plane(data)
        if self.normal[0] != 0 or self.normal[1] != 0:
            u = np.array([-self.normal[1], self.normal[0], 0])
        else:
            u = np.array([1, 0, 0])
        # Normalize u
        u = u / np.linalg.norm(u)
        # Create the second basis vector using the cross product
        v = np.cross(self.normal, u)
        # Normalize v
        v = v / np.linalg.norm(v)
        # Form the 3x2 projection matrix using the basis vectors u and v
        projection_matrix = np.array([u, v]).T
        # Project onto the 2D plane
        projected_2d = self.data @ projection_matrix
        # self.plot()
        return projected_2d

    # draw the plane and the normal vector in 3D

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            [x[0] for x in self.data],
            [x[1] for x in self.data],
            [x[2] for x in self.data],
        )
        # ax + by + cz = 0 => z = (-ax - by - d) / c
        xx, yy = np.meshgrid(np.linspace(-400, 400, 2), np.linspace(-400, 400, 2))
        zz = (-self.normal[0] * xx - self.normal[1] * yy) * 1.0 / self.normal[2]
        ax.plot_surface(xx, yy, zz, alpha=0.5)
        # plot the normal vector
        ax.quiver(
            0,
            0,
            0,
            -self.normal[0] * 10,
            -self.normal[1] * 10,
            -self.normal[2] * 10,
            color="r",
        )

        plt.title("Fitted plane")
        plt.show()


# 3d plot of coordinate
def plot(data, title):
    # 2d plot of coordinate
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([x[0] for x in data], [x[1] for x in data])
    plt.axis("equal")
    plt.title(title)
    plt.show()

    plt.show()


def save_xyz(points, file_path):
    with open(file_path, "w") as f:
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


if __name__ == "__main__":
    # read kml file
    file_path = "fordulokor.kml"
    data = json.loads(kml_to_json(file_path))
    gps_data = data["boundary_data"]
    gps_locations = [
        PointData(float(x["latitude"]), float(x["longitude"]), float(x["altitude"]))
        for x in gps_data[0]["coordinates"]
    ]
    # read csv file
    file_path = "fordulokor.csv"
    df = pd.read_csv(file_path, delimiter=";")
    gps_locations = [
        PointData(row["Lat"], row["Lon"], row["Alt"]) for _, row in df.iterrows()
    ]

    cartesian_location = [gps_location.to_cartesian() for gps_location in gps_locations]
    cartesian_location = np.array(cartesian_location)

    plane = PlaneFitting()
    projected_2d = plane.project_points(cartesian_location)
    plot(projected_2d, "fitted points in 2D")
    model = Circle_Model()
    ransac = Ransac(model)
    inliers = ransac.fit_model(projected_2d)
    # Save inliers to XYZ file
    inliers_xyz = np.hstack((inliers, np.zeros((inliers.shape[0], 1))))
    # Add z=0 for 2D points
    save_xyz(inliers_xyz, "inliers.xyz")

    print(
        "Inliers saved to inliers.xyz. You can now open this file in Meshlab for visualization."
    )
