import numpy as np
import cv2
from sklearn.cluster import KMeans, DBSCAN

class Boids:
    def __init__(self, N, width, height):
        self.N = N
        self.position = np.column_stack([np.random.randint(0, width, N), np.random.randint(0, height, N)])
        self.position = self.position.astype(np.float32)
        self.velocity = (np.random.rand(N, 2) * 2 - 1) * 5
        self.speed = 2.0
        self.perception = 50
        self.max_force_align = 0.5
        self.max_force_cohesion = 0.3
        self.max_force_separation = 0.6
        self.edge_force = 0.1

    def edges(self, width, height):
        """Force to prevent boids from escaping the screen."""
        force = np.zeros((self.N, 2))

        # Left Edge
        left_mask = self.position[:, 0] < 50
        force[left_mask, 0] += self.edge_force * (50 - self.position[left_mask, 0])

        # Right Edge
        right_mask = self.position[:, 0] > width - 50
        force[right_mask, 0] -= self.edge_force * (self.position[right_mask, 0] - (width - 50))

        # Top Edge
        top_mask = self.position[:, 1] < 50
        force[top_mask, 1] += self.edge_force * (50 - self.position[top_mask, 1])

        # Bottom Edge
        bottom_mask = self.position[:, 1] > height - 50
        force[bottom_mask, 1] -= self.edge_force * (self.position[bottom_mask, 1] - (height - 50))

        return force

    def limit_force(self, force, limit):
        """Limit the magnitude of the force vector."""
        if len(force.shape) == 1:  # 1D array
            magnitude = np.linalg.norm(force)
            if magnitude > limit:
                return (force / magnitude) * limit
            return force
        else:  # 2D array
            magnitude = np.linalg.norm(force, axis=1).reshape(-1, 1)
            mask = magnitude > limit
            force[mask] = (force[mask] / magnitude[mask]) * limit
            return force

    def align(self):
        # This is the most complex operation, due to pairwise operations
        steering = np.zeros((self.N, 2))
        for i in range(self.N):
            dist = np.linalg.norm(self.position - self.position[i], axis=1)
            mask = (dist < self.perception) & (dist != 0)
            if mask.any():
                avg_velocity = np.mean(self.velocity[mask], axis=0)
                steering[i] = self.limit_force(avg_velocity - self.velocity[i], self.max_force_align)
        return steering

    def cohesion(self):
        steering = np.zeros((self.N, 2))
        for i in range(self.N):
            dist = np.linalg.norm(self.position - self.position[i], axis=1)
            mask = (dist < self.perception) & (dist != 0)
            if mask.any():
                avg_position = np.mean(self.position[mask], axis=0)
                steering_dir = avg_position - self.position[i]
                steering[i] = self.limit_force(steering_dir - self.velocity[i], self.max_force_cohesion)
        return steering

    def separation(self):
        steering = np.zeros((self.N, 2))
        for i in range(self.N):
            diff = self.position[i] - self.position
            dist = np.linalg.norm(diff, axis=1)
            mask = (dist < self.perception) & (dist != 0)
            if mask.any():
                avg_diff = np.mean(diff[mask], axis=0)
                steering[i] = self.limit_force(avg_diff - self.velocity[i], self.max_force_separation)
        return steering

    def update(self, width=800, height=600):
        alignment = self.align()
        cohesion = self.cohesion()
        separation = self.separation()

        self.velocity += alignment + cohesion + separation
        self.velocity += self.edges(width, height)

        norm = np.linalg.norm(self.velocity, axis=1).reshape(-1, 1)
        self.velocity = np.divide(self.velocity, norm, where=norm != 0) * self.speed

        self.position += self.velocity

        # Clip the positions to ensure boids remain within the screen
        self.position[:, 0] = np.clip(self.position[:, 0], 0, width-1)
        self.position[:, 1] = np.clip(self.position[:, 1], 0, height-1)
    
    def cluster(self):
        """ use DBscan to cluser the boids and return a cluster mask"""
        db = DBSCAN(eps=self.perception, min_samples=3).fit(self.position)
        labels = db.labels_
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # print("number of clusters: ", n_clusters_)
        return labels

def main():
    width, height = 800, 600
    screen = np.zeros((height, width, 3), dtype=np.uint8)
    boids = Boids(100, width, height)
    # colors is a dict that maps cluster labels to colors
    colors = {i: np.random.randint(0, 255, 3) for i in range(100)}

    while True:
        screen.fill(0)
        boids.update(width, height)
        labels = boids.cluster()
        non_nan_positions = boids.position[~np.isnan(boids.position).any(axis=1)]
        non_nan_labels = labels[~np.isnan(boids.position).any(axis=1)]
        for position, label in zip(non_nan_positions, labels):
            # cv2.circle(screen, (int(position[0]), int(position[1])), 1, (255, 255, 255), -1)
            #  colorize according to label
            # if label != -1:
            label = label % 100
            cv2.circle(screen, (int(position[0]), int(position[1])), 3, colors[label].tolist(), -1)

        cv2.imshow("Boids Murmuration Simulation", screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
