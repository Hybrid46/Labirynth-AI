import numpy as np
import time
import os
import json
from collections import deque
import signal
import sys


class Neuron:
    def __init__(self, input_count):
        self.weights = np.random.uniform(-1, 1, input_count)
        self.bias = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feed_forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(weighted_sum)

    def update_weights(self, inputs, error, learning_rate):
        self.weights += learning_rate * error * inputs
        self.bias += learning_rate * error


class NeuralNetwork:
    def __init__(self, input_count, neuron_count, learning_rate=0.1):
        self.neurons = [Neuron(input_count) for _ in range(neuron_count)]
        self.learning_rate = learning_rate

    def feed_forward(self, inputs):
        return np.array([neuron.feed_forward(inputs) for neuron in self.neurons])

    def train(self, inputs, target_outputs):
        outputs = self.feed_forward(inputs)
        for i, neuron in enumerate(self.neurons):
            error = target_outputs[i] - outputs[i]
            neuron.update_weights(inputs, error, self.learning_rate)

    def save(self, filename):
        try:
            data = {
                "neurons": [
                    {"weights": neuron.weights.tolist(), "bias": neuron.bias}
                    for neuron in self.neurons
                ]
            }
            with open(filename, "w") as file:
                json.dump(data, file)
            print(f"Successfully saved network data to '{filename}'.")
        except Exception as e:
            print(f"Error saving network data: {e}")

    def load(self, filename):
        try:
            with open(filename, "r") as file:
                data = json.load(file)
            for i, neuron_data in enumerate(data["neurons"]):
                self.neurons[i].weights = np.array(neuron_data["weights"])
                self.neurons[i].bias = neuron_data["bias"]
            print(f"Successfully loaded network data from '{filename}'.")
        except FileNotFoundError:
            print(f"No saved network data found at '{filename}'. Starting with random weights.")
        except Exception as e:
            print(f"Error loading network data: {e}")


def print_labyrinth(labyrinth, agent_x, agent_y):
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console
    for i in range(labyrinth.shape[0]):
        for j in range(labyrinth.shape[1]):
            if i == agent_x and j == agent_y:
                print("A ", end="")  # Agent's position
            elif labyrinth[i, j]:
                print(". ", end="")  # Path
            else:
                print("# ", end="")  # Wall
        print()
    print()


def print_neuron_data(network):
    print("--- Neuron Data ---")
    for i, neuron in enumerate(network.neurons):
        print(f"Neuron {i + 1}:")
        print(f"  Weights: {neuron.weights}")
        print(f"  Bias: {neuron.bias}")
    print()


def print_progress(episode, max_episodes, steps, best_steps):
    print(f"Episode: {episode + 1}/{max_episodes}")
    print(f"Current Steps: {steps}")
    print(f"Best Steps: {best_steps}")
    print()


def find_shortest_path(labyrinth, start, goal):
    # Breadth-First Search (BFS) to find the shortest path
    queue = deque()
    queue.append((start, [start]))
    visited = set()

    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_x, new_y = x + dx, y + dy
            if (
                0 <= new_x < labyrinth.shape[0]
                and 0 <= new_y < labyrinth.shape[1]
                and labyrinth[new_x, new_y]
                and (new_x, new_y) not in visited
            ):
                visited.add((new_x, new_y))
                queue.append(((new_x, new_y), path + [(new_x, new_y)]))

    return None


# Global variable for the network
network = None


def signal_handler(sig, frame):
    global network
    print("\nCTRL+C detected. Saving network data and exiting...")
    if network:
        network.save("network.json")
    sys.exit(0)


def learning_loop(network, labyrinth, episodes, exploration_rate, exploration_decay, min_exploration_rate, discount_factor):
    best_steps = float('inf')  # Track the best (shortest) steps to the goal
    best_path = []  # Track the best path

    for episode in range(episodes):
        x, y = 0, 0  # Start position
        reached_goal = False
        steps = 0  # Track steps in the current episode
        current_path = []  # Track the current path

        while not reached_goal:
            # Get the state of surrounding cells (8 directions)
            inputs = np.array([
                y > 0 and labyrinth[x, y - 1],  # Up
                y < labyrinth.shape[1] - 1 and labyrinth[x, y + 1],  # Down
                x > 0 and labyrinth[x - 1, y],  # Left
                x < labyrinth.shape[0] - 1 and labyrinth[x + 1, y],  # Right
                x > 0 and y > 0 and labyrinth[x - 1, y - 1],  # Up-Left
                x > 0 and y < labyrinth.shape[1] - 1 and labyrinth[x - 1, y + 1],  # Down-Left
                x < labyrinth.shape[0] - 1 and y > 0 and labyrinth[x + 1, y - 1],  # Up-Right
                x < labyrinth.shape[0] - 1 and y < labyrinth.shape[1] - 1 and labyrinth[x + 1, y + 1]  # Down-Right
            ], dtype=np.float64)

            # Choose an action (exploration vs exploitation)
            if np.random.rand() < exploration_rate:
                action = np.random.randint(8)  # Random action
            else:
                outputs = network.feed_forward(inputs)
                action = np.argmax(outputs)  # Best action

            # Perform the action
            new_x, new_y = x, y
            if action == 0:
                new_y -= 1  # Up
            elif action == 1:
                new_y += 1  # Down
            elif action == 2:
                new_x -= 1  # Left
            elif action == 3:
                new_x += 1  # Right
            elif action == 4:
                new_x -= 1
                new_y -= 1  # Up-Left
            elif action == 5:
                new_x -= 1
                new_y += 1  # Down-Left
            elif action == 6:
                new_x += 1
                new_y -= 1  # Up-Right
            elif action == 7:
                new_x += 1
                new_y += 1  # Down-Right

            # Check if the new position is valid
            if 0 <= new_x < labyrinth.shape[0] and 0 <= new_y < labyrinth.shape[1] and labyrinth[new_x, new_y]:
                x, y = new_x, new_y
                steps += 1
                current_path.append((x, y))  # Record the current position

            # Calculate reward
            reward = -0.1  # Small penalty for each step
            if x == labyrinth.shape[0] - 1 and y == labyrinth.shape[1] - 1:
                reward = 1.0  # Large reward for reaching the goal
                reached_goal = True

                # Update best steps and path if the current path is better
                if steps < best_steps:
                    best_steps = steps
                    best_path = current_path.copy()

            # Get the Q-values for the new state
            new_inputs = np.array([
                y > 0 and labyrinth[x, y - 1],  # Up
                y < labyrinth.shape[1] - 1 and labyrinth[x, y + 1],  # Down
                x > 0 and labyrinth[x - 1, y],  # Left
                x < labyrinth.shape[0] - 1 and labyrinth[x + 1, y],  # Right
                x > 0 and y > 0 and labyrinth[x - 1, y - 1],  # Up-Left
                x > 0 and y < labyrinth.shape[1] - 1 and labyrinth[x - 1, y + 1],  # Down-Left
                x < labyrinth.shape[0] - 1 and y > 0 and labyrinth[x + 1, y - 1],  # Up-Right
                x < labyrinth.shape[0] - 1 and y < labyrinth.shape[1] - 1 and labyrinth[x + 1, y + 1]  # Down-Right
            ], dtype=np.float64)
            new_outputs = network.feed_forward(new_inputs)
            max_new_q = np.max(new_outputs)

            # Calculate target Q-value
            target_outputs = network.feed_forward(inputs)
            target_outputs[action] = reward + discount_factor * max_new_q

            # Train the network
            network.train(inputs, target_outputs)

            # Print the labyrinth with the agent's position
            print_labyrinth(labyrinth, x, y)
            print_neuron_data(network)  # Display neuron data
            print_progress(episode, episodes, steps, best_steps)  # Display progress
            time.sleep(0.0001)  # Adjusted delay for smoother visualization

        # Decay exploration rate
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

    # Save the trained network
    network.save("network.json")


def testing_loop(network, labyrinth):
    print("Testing the trained network...")
    test_x, test_y = 0, 0
    steps = 0
    wall_hits = 0
    total_reward = 0
    last_path = []  # Track the last path taken

    while test_x != labyrinth.shape[0] - 1 or test_y != labyrinth.shape[1] - 1:
        time.sleep(0.0001)  # Adjusted delay for smoother visualization
        last_path.append((test_x, test_y))  # Record the current position
        print_labyrinth(labyrinth, test_x, test_y)
        print_neuron_data(network)  # Display neuron data
        print_progress(0, 1, steps, 0)  # Display progress

        # Get the state of surrounding cells (8 directions)
        inputs = np.array([
            test_y > 0 and labyrinth[test_x, test_y - 1],  # Up
            test_y < labyrinth.shape[1] - 1 and labyrinth[test_x, test_y + 1],  # Down
            test_x > 0 and labyrinth[test_x - 1, test_y],  # Left
            test_x < labyrinth.shape[0] - 1 and labyrinth[test_x + 1, test_y],  # Right
            test_x > 0 and test_y > 0 and labyrinth[test_x - 1, test_y - 1],  # Up-Left
            test_x > 0 and test_y < labyrinth.shape[1] - 1 and labyrinth[test_x - 1, test_y + 1],  # Down-Left
            test_x < labyrinth.shape[0] - 1 and test_y > 0 and labyrinth[test_x + 1, test_y - 1],  # Up-Right
            test_x < labyrinth.shape[0] - 1 and test_y < labyrinth.shape[1] - 1 and labyrinth[test_x + 1, test_y + 1]  # Down-Right
        ], dtype=np.float64)

        # Get the neural network's decision
        outputs = network.feed_forward(inputs)
        action = np.argmax(outputs)

        # Debug: Print inputs, outputs, and chosen action
        print(f"Inputs: {inputs}")
        print(f"Outputs: {outputs}")
        print(f"Chosen Action: {action}")

        # Perform the action
        new_x, new_y = test_x, test_y
        if action == 0:
            new_y -= 1  # Up
        elif action == 1:
            new_y += 1  # Down
        elif action == 2:
            new_x -= 1  # Left
        elif action == 3:
            new_x += 1  # Right
        elif action == 4:
            new_x -= 1
            new_y -= 1  # Up-Left
        elif action == 5:
            new_x -= 1
            new_y += 1  # Down-Left
        elif action == 6:
            new_x += 1
            new_y -= 1  # Up-Right
        elif action == 7:
            new_x += 1
            new_y += 1  # Down-Right

        # Check if the new position is valid
        if 0 <= new_x < labyrinth.shape[0] and 0 <= new_y < labyrinth.shape[1] and labyrinth[new_x, new_y]:
            test_x, test_y = new_x, new_y
            steps += 1
            total_reward -= 0.1  # Small penalty for each step
            print(f"Moved to ({test_x}, {test_y})")  # Debug: Print new position
        else:
            wall_hits += 1
            print(f"Wall hit at ({new_x}, {new_y})")  # Debug: Print wall hit

        # Check if the goal is reached
        if test_x == labyrinth.shape[0] - 1 and test_y == labyrinth.shape[1] - 1:
            last_path.append((test_x, test_y))  # Record the final position
            total_reward += 1.0  # Large reward for reaching the goal

    print_labyrinth(labyrinth, test_x, test_y)
    print("Reached the end of the labyrinth!")
    print("\n--- Statistics ---")
    print(f"Total Steps: {steps}")
    print(f"Wall Hits: {wall_hits}")
    print(f"Total Reward: {total_reward}")

    # Print the last path
    print("--- Last Path ---")
    for step, (x, y) in enumerate(last_path):
        print(f"Step {step + 1}: ({x}, {y})")
        print_labyrinth(labyrinth, x, y)
        time.sleep(0.1)  # Pause for visualization


def main():
    global network

    # Define the labyrinth
    labyrinth = np.array([
        [True,  False, True,  True,  True,  False],
        [True,  True,  False, True,  False, True],
        [False, True,  True,  True,  True,  True],
        [True,  False, True,  False, True,  True],
        [True,  True,  True,  True,  False, True],
        [False, True,  False, True,  True,  True]
    ])

    # Create a neural network with 8 inputs (surrounding cells) and 8 outputs (possible moves)
    network = NeuralNetwork(input_count=8, neuron_count=8, learning_rate=0.1)

    # Load saved weights and biases if the file exists
    network.load("network.json")

    print("Network loaded. Weights and biases:")
    print_neuron_data(network)

    # Set up signal handler for CTRL+C
    signal.signal(signal.SIGINT, signal_handler)

    # Ask the user to choose between learning and testing
    mode = input("Choose mode (Learn/Test): ").strip().lower()
    if mode == "learn":
        # Training parameters
        episodes = 1000
        exploration_rate = 1.0
        exploration_decay = 0.995
        min_exploration_rate = 0.01
        discount_factor = 0.9

        # Run the learning loop
        learning_loop(network, labyrinth, episodes, exploration_rate, exploration_decay, min_exploration_rate, discount_factor)
    elif mode == "test":
        # Run the testing loop
        testing_loop(network, labyrinth)
    else:
        print("Invalid mode selected. Exiting...")


if __name__ == "__main__":
    main()