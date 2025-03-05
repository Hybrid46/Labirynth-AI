import numpy as np
import time
import os
import json
from collections import deque
import signal
import sys

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros(1)
        self.inputs = None
        self.output = None
        self.delta = None

    def sigmoid(self, x):
        # Numerically stable sigmoid
        return np.where(x >= 0, 
                        1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feed_forward(self, inputs):
        self.inputs = inputs.flatten()  # Ensure inputs are 1D
        z = np.dot(self.inputs, self.weights) + self.bias
        self.output = self.sigmoid(z)
        return self.output

    # ADD THIS MISSING METHOD
    def compute_delta(self, error):
        self.delta = error * self.sigmoid_derivative(self.output)

class Layer:
    def __init__(self, input_size, num_neurons):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]
        
    def feed_forward(self, inputs):
        return np.array([neuron.feed_forward(inputs) for neuron in self.neurons])

class NeuralNetwork:
    def __init__(self, layers_config, learning_rate=0.1):
        self.layers = []
        self.learning_rate = learning_rate
        
        for i in range(1, len(layers_config)):
            self.layers.append(Layer(layers_config[i-1], layers_config[i]))
            
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
            
    def feed_forward(self, inputs):
        inputs = np.array(inputs).flatten()
        for i, layer in enumerate(self.layers):
            # For all layers except the last, use sigmoid activation
            if i != len(self.layers) - 1:
                inputs = layer.feed_forward(inputs)
            else:
                # Output layer: linear activation (z = Wx + b, no sigmoid)
                outputs = []
                for neuron in layer.neurons:
                    neuron.inputs = inputs.flatten()
                    z = np.dot(neuron.inputs, neuron.weights) + neuron.bias
                    neuron.output = z  # Store z directly (no activation)
                    outputs.append(z)
                inputs = np.array(outputs)
        return inputs
    
    def backpropagate(self, target):
        # Output layer (linear activation)
        output_layer = self.layers[-1]
        for i, neuron in enumerate(output_layer.neurons):
            error = target[i] - neuron.output
            neuron.delta = error * 1  # Derivative of linear activation is 1

        # Hidden layers (sigmoid activation)
        for layer_idx in reversed(range(len(self.layers)-1)):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx+1]
            
            for i, neuron in enumerate(current_layer.neurons):
                error = sum(n.weights[i] * n.delta for n in next_layer.neurons)
                neuron.compute_delta(error)  # Uses sigmoid derivative
                
    def update_weights(self, clip_value=5.0):
        for layer in self.layers:
            for neuron in layer.neurons:
                # Clip gradients to prevent exploding gradients
                clipped_delta = np.clip(neuron.delta, -clip_value, clip_value)
                neuron.weights += self.learning_rate * clipped_delta * neuron.inputs
                neuron.bias += self.learning_rate * clipped_delta

    def save(self, filename):
        try:
            data = {"layers": []}
            for layer in self.layers:
                layer_data = []
                for neuron in layer.neurons:
                    layer_data.append({
                        "weights": neuron.weights.tolist(),
                        "bias": neuron.bias.tolist()
                    })
                data["layers"].append(layer_data)
            
            with open(filename, "w") as f:
                json.dump(data, f)
            print(f"Network saved to {filename}")
        except Exception as e:
            print(f"Error saving network: {e}")

    def load(self, filename):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    
                for layer_idx, layer_data in enumerate(data["layers"]):
                    for neuron_idx, neuron_data in enumerate(layer_data):
                        self.layers[layer_idx].neurons[neuron_idx].weights = np.array(neuron_data["weights"])
                        self.layers[layer_idx].neurons[neuron_idx].bias = np.array(neuron_data["bias"])
                print(f"Network loaded from {filename}")
            except Exception as e:
                print(f"Error loading network: {e}")

def create_labyrinth(size=10, obstacle_density=0.2):
    labyrinth = np.ones((size, size), dtype=bool)
    # Add random obstacles
    for i in range(size):
        for j in range(size):
            if np.random.random() < obstacle_density:
                labyrinth[i][j] = False
    # Ensure start and goal can be set
    labyrinth[0][0] = True
    labyrinth[-1][-1] = True
    return labyrinth

def get_random_position(labyrinth):
    while True:
        x, y = np.random.randint(0, labyrinth.shape[0]), np.random.randint(0, labyrinth.shape[1])
        if labyrinth[x][y]:
            return (x, y)

def print_labyrinth(labyrinth, agent_pos, goal_pos):
    os.system('cls' if os.name == 'nt' else 'clear')
    size = labyrinth.shape[0]
    for i in range(size):
        for j in range(size):
            if (i, j) == agent_pos:
                print(" A ", end="")
            elif (i, j) == goal_pos:
                print(" O ", end="")
            else:
                print(" . " if labyrinth[i][j] else " # ", end="")
        print("\n")
    print()

def learning_loop(network, labyrinth, episodes=1000, 
                 exploration_rate=1.0, exploration_decay=0.995,
                 min_exploration=0.01, discount_factor=0.9, verbose=True,
                 initial_learning_rate=0.5, learning_rate_decay=0.995, min_learning_rate=0.01):
    size = labyrinth.shape[0]
    
    for episode in range(episodes):
        start_pos = get_random_position(labyrinth)
        goal_pos = get_random_position(labyrinth)
        while goal_pos == start_pos:
            goal_pos = get_random_position(labyrinth)
            
        current_pos = start_pos
        steps = 0
        episode_reward = 0
        
        while current_pos != goal_pos and steps < 100:
            x, y = current_pos
            gx, gy = goal_pos
            
            # Create inputs
            inputs = []
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1),
                          (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = x+dx, y+dy
                inputs.append(1 if (0 <= nx < size and 0 <= ny < size and labyrinth[nx][ny]) else 0)
            
            # Add normalized goal direction
            inputs.extend([(gx - x)/size, (gy - y)/size])
            inputs = np.array(inputs).flatten()  # Ensure 1D array
            
            # Exploration vs exploitation
            if np.random.random() < exploration_rate:
                action = np.random.randint(8)
            else:
                q_values = network.feed_forward(inputs)
                action = np.argmax(q_values)
                
            # Move agent
            move_map = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
            dx, dy = move_map[action]
            new_x = x + dx
            new_y = y + dy
            
            # Check valid move
            reward = -0.1
            if 0 <= new_x < size and 0 <= new_y < size and labyrinth[new_x][new_y]:
                current_pos = (new_x, new_y)
                steps += 1
                
                # Calculate distance reward
                old_dist = abs(x - gx) + abs(y - gy)
                new_dist = abs(new_x - gx) + abs(new_y - gy)
                reward += 0.1 * (old_dist - new_dist)
                
                if current_pos == goal_pos:
                    reward = 1.0
            else:
                reward = -0.5
                
            # Calculate target Q-value
            if current_pos == goal_pos:
                target_q = reward
            else:
                next_inputs = []
                for dx, dy in move_map:
                    nx, ny = new_x+dx, new_y+dy
                    if 0 <= nx < size and 0 <= ny < size and labyrinth[nx][ny]:
                        next_inputs.append(1)
                    else:
                        next_inputs.append(0)
                next_inputs.extend([(gx - new_x)/size, (gy - new_y)/size])
                next_q = network.feed_forward(next_inputs)
                target_q = reward + discount_factor * np.max(next_q)
                
            # Train network
            target = network.feed_forward(inputs)
            target[action] = target_q
            
            network.backpropagate(target)
            network.update_weights()
            
            episode_reward += reward
            
            # Print current state only if verbose is True
            if verbose:
                print_labyrinth(labyrinth, current_pos, goal_pos)
                print(f"Episode: {episode+1}/{episodes}")
                print(f"Steps: {steps}  Reward: {episode_reward:.2f}")
                print(f"Exploration Rate: {exploration_rate:.8f}")
                print(f"Learning Rate: {network.learning_rate:.8f}")
                time.sleep(0.0001)
                
        # Decay exploration rate
        exploration_rate = max(min_exploration, exploration_rate * exploration_decay)
        
        # Decay learning rate
        network.learning_rate = max(min_learning_rate, network.learning_rate * learning_rate_decay)
        
        # Print episode summary even in non-verbose mode
        if not verbose:
            print(f"Episode: {episode+1}/{episodes}, Steps: {steps}, Reward: {episode_reward:.2f}, Exploration Rate: {exploration_rate:.8f}, Learning Rate: {network.learning_rate:.8f}")
        
    network.save("trained_network.json")

def testing_loop(network, labyrinth):
    size = labyrinth.shape[0]
    start_pos = get_random_position(labyrinth)
    goal_pos = get_random_position(labyrinth)
    
    current_pos = start_pos
    steps = 0
    
    while current_pos != goal_pos and steps < 100:
        x, y = current_pos
        gx, gy = goal_pos
        
        inputs = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1),
                      (-1,-1), (-1,1), (1,-1), (1,1)]:
            nx, ny = x+dx, y+dy
            inputs.append(1 if (0 <= nx < size and 0 <= ny < size and labyrinth[nx][ny]) else 0)
        inputs.extend([(gx - x)/size, (gy - y)/size])
        inputs = np.array(inputs).flatten()  # Ensure 1D array
        
        q_values = network.feed_forward(inputs)
        action = np.argmax(q_values)
        
        move_map = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        dx, dy = move_map[action]
        new_x = x + dx
        new_y = y + dy
        
        if 0 <= new_x < size and 0 <= new_y < size and labyrinth[new_x][new_y]:
            current_pos = (new_x, new_y)
            steps += 1
            
        print_labyrinth(labyrinth, current_pos, goal_pos)
        print(f"Steps: {steps}")
        time.sleep(0.1)
        
    if current_pos == goal_pos:
        print("Goal reached!")
    else:
        print("Failed to reach goal")

def signal_handler(sig, frame):
    global network
    print("\nSaving network before exit...")
    network.save("autosave_network.json")
    sys.exit(0)

def main():
    global network
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create 10x10 labyrinth
    labyrinth = create_labyrinth(10, obstacle_density=0.2)
    
    # Network architecture: [input(10), hidden(16), output(8)]
    network = NeuralNetwork([10, 16, 8], learning_rate=0.5)  # Start with a higher learning rate
    
    # Try to load existing network
    try:
        network.load("trained_network.json")
    except:
        print("No existing network found, starting fresh")
    
    mode = input("Choose mode (learn/test): ").lower()
    
    if mode == "learn":
        # Ask for verbose mode
        verbose = input("Enable verbose mode? (y/n): ").lower() == 'y'
        
        # Ask for number of episodes (default: 1000)
        episodes = input(f"Enter number of episodes (default: 1000): ").strip()
        episodes = int(episodes) if episodes.isdigit() else 1000  # Use default if invalid input
        
        # Ask for minimum exploration rate (default: 0.01)
        min_exploration = input(f"Enter minimum exploration rate (default: 0.01): ").strip()
        min_exploration = float(min_exploration) if min_exploration.replace('.', '', 1).isdigit() else 0.01  # Use default if invalid input
        
        # Ask for initial learning rate (default: 0.5)
        initial_learning_rate = input(f"Enter initial learning rate (default: 0.5): ").strip()
        initial_learning_rate = float(initial_learning_rate) if initial_learning_rate.replace('.', '', 1).isdigit() else 0.5  # Use default if invalid input
        
        # Ask for learning rate decay (default: 0.995)
        learning_rate_decay = input(f"Enter learning rate decay (default: 0.995): ").strip()
        learning_rate_decay = float(learning_rate_decay) if learning_rate_decay.replace('.', '', 1).isdigit() else 0.995  # Use default if invalid input
        
        # Ask for minimum learning rate (default: 0.01)
        min_learning_rate = input(f"Enter minimum learning rate (default: 0.01): ").strip()
        min_learning_rate = float(min_learning_rate) if min_learning_rate.replace('.', '', 1).isdigit() else 0.01  # Use default if invalid input
        
        # Start learning loop with user-defined parameters
        learning_loop(
            network, 
            labyrinth, 
            episodes=episodes, 
            min_exploration=min_exploration, 
            verbose=verbose,
            initial_learning_rate=initial_learning_rate,
            learning_rate_decay=learning_rate_decay,
            min_learning_rate=min_learning_rate
        )
    elif mode == "test":
        testing_loop(network, labyrinth)
    else:
        print("Invalid mode selected")

if __name__ == "__main__":
    main()