import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import DBSCAN


@dataclass
class Circle_Model:
    """Circle model class"""

    x0: float = 0
    y0: float = 0
    radius: float = 0

    def fit(self, data):
        """Fit the model to the data"""
        # fit the model
        A = np.zeros((len(data), 3))
        b = np.zeros(len(data))
        for i in range(len(data)):
            x = data[i][0]
            y = data[i][1]
            A[i] = [2 * x, 2 * y, 1]
            b[i] = x**2 + y**2
        X = np.linalg.lstsq(A, b, rcond=None)
        x0, y0 = X[0][0], X[0][1]
        radius = np.sqrt(X[0][2] + np.dot(X[0][0:2], X[0][0:2]))
        self.x0, self.y0, self.radius = x0, y0, radius
        # print("X0", x0, y0, radius)
        return self

    def predict(self, data, threshold):
        """Predict the inliers"""
        predictions = []
        for i in range(data.shape[0]):
            x, y = data[i]
            error = self.error([x, y])
            if error < threshold:
                predictions.append([x, y])

        return predictions

    def error(self, point):
        """Calculate the error for a point"""
        distance = np.sqrt((point[0] - self.x0) ** 2 + (point[1] - self.y0) ** 2)
        error = abs(distance - self.radius)
        return error

    def mean_square_error(self, data):
        """Calculate the mean square error' for the whole data"""
        loss = np.sum(
            (
                np.sqrt((data[:, 0] - self.x0) ** 2 + (data[:, 1] - self.y0) ** 2)
                - self.radius
            )
            ** 2
        ) / (2 * len(data))
        return loss

    def plot(self, data, inliers, random_sample=None):
        """Plot the data and the inliers"""
        center = [self.x0, self.y0]
        radius = self.radius
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter([x[0] for x in data], [x[1] for x in data])
        ax.scatter([x[0] for x in inliers], [x[1] for x in inliers], color="r")
        circle = plt.Circle(center, radius, color="r", fill=False)
        ax.add_artist(circle)
        plt.axis("equal")
        plt.legend(["Data", "Inliers"])
        plt.show()


@dataclass
class Ransac:

    model: Circle_Model  # model to
    data: np.ndarray = None
    filtered_data: np.ndarray = None
    threshold: float = 0.1  # threshold for inlier
    max_iterations: int = 1000  # max number of iterations
    min_sample_size: int = 3  # number of samples to select
    threshold_inlier_count: int = 100  # threshold for inlier points
    best_inlier_points: np.ndarray = None  # inliers
    best_error: float = 0  # loss
    best_model: Circle_Model = None  # best model
    filtered_data: np.ndarray = None

    # original_data  # original data
    def random_sample(self, data):
        # select random sample
        indices = np.random.choice(data.shape[0], self.min_sample_size, replace=False)
        return data[indices]

    def filter_dense_clusters(self, eps=0.1, min_samples=5):
        # Use DBSCAN to filter out dense clusters
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.data)
        labels = db.labels_
        # Identify noise points (label == -1)
        noise_mask = labels == -1
        filtered_data = self.data[noise_mask]
        return filtered_data

    def fit_model(self, data):
        self.data = data
        self.filtered_data = self.filter_dense_clusters(eps=0.1, min_samples=5)
        # iterate over the max number of iterations
        for i in range(self.max_iterations):
            # select random sample
            random_sample = self.random_sample(self.filtered_data)
            # fit the model
            model = self.model.fit(random_sample)
            maybe_inliers = model.predict(self.filtered_data, self.threshold)
            if i == 0:
                self.best_inlier_points = maybe_inliers
                self.best_model = copy.deepcopy(model)
                self.best_error = model.mean_square_error(np.array(maybe_inliers))
            if len(maybe_inliers) > self.threshold_inlier_count:
                model = self.model.fit(np.array(maybe_inliers))
                inliers = model.predict(self.filtered_data, self.threshold)
                error = model.mean_square_error(np.array(inliers))
                if len(inliers) > len(self.best_inlier_points):
                    self.best_error = error
                    self.best_inlier_points = inliers
                    self.best_model = copy.deepcopy(model)

                elif len(inliers) == len(self.best_inlier_points):
                    if error < self.best_error:
                        self.best_error = error
                        self.best_inlier_points = inliers
                        self.best_model = copy.deepcopy(model)

        self.best_inlier_points = self.best_model.predict(self.data, self.threshold)
        self.best_model = self.best_model.fit(self.best_inlier_points)
        self.best_inlier_points = self.best_model.predict(self.data, self.threshold)

        self.best_model.plot(np.array(self.filtered_data), self.best_inlier_points)
        print("Best Model", self.best_model)
        return np.array(self.best_inlier_points)


if __name__ == "__main__":
    # generate sample data like create a circle
    radius = 3
    n_samples = 100
    theta = np.linspace(0, 2 * np.pi, n_samples)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    # add noise to the data
    noise = 0.5
    x += np.random.normal(0, noise, n_samples)
    y += np.random.normal(0, noise, n_samples)
    data = np.column_stack([x, y])
    # generate random data
    data = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [-1, -1],
            [-2, -2],
            [-1, 0],
            [1, 0],
            [0, 1],
            [0, -1],
            [0, -2],
            [0, 2],
            [2, 0],
            [-2, 0],
        ]
    )
    model = Circle_Model()
    ransac = Ransac(model)
    inliers = ransac.fit_model(data)
