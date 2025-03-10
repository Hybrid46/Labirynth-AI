import numpy as np
import time
import os
import json
from collections import deque
import signal
import sys
import heapq

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * np.sqrt(1. / input_size)  # Xavier initialization
        self.bias = np.zeros(1)
        self.inputs = None
        self.output = None
        self.delta = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feed_forward(self, inputs):
        self.inputs = inputs.flatten()
        z = np.dot(self.inputs, self.weights) + self.bias
        self.output = self.sigmoid(z)
        return self.output

    def compute_delta(self, error):
        self.delta = error * self.sigmoid_derivative(self.output)

class Layer:
    def __init__(self, input_size, num_neurons):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]
        
    def feed_forward(self, inputs):
        return np.array([neuron.feed_forward(inputs) for neuron in self.neurons])

class NeuralNetwork:
    def __init__(self, layers_config, learning_rate=0.1):
        self.layers_config = layers_config
        self.layers = []
        self.learning_rate = learning_rate
        for i in range(1, len(layers_config)):
            self.layers.append(Layer(layers_config[i-1], layers_config[i]))
            
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
            
    def feed_forward(self, inputs):
        inputs = np.array(inputs).flatten()
        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
                inputs = layer.feed_forward(inputs)
            else:
                # Keep output layer linear
                outputs = []
                for neuron in layer.neurons:
                    neuron.inputs = inputs.flatten()
                    z = np.dot(neuron.inputs, neuron.weights) + neuron.bias
                    neuron.output = z  # Linear activation for output
                    outputs.append(z)
                inputs = np.array(outputs)
        return inputs
    
    def backpropagate(self, target):
        output_layer = self.layers[-1]
        for i, neuron in enumerate(output_layer.neurons):
            error = target[i] - neuron.output
            neuron.delta = error * 1  # Keep linear derivative for output

        for layer_idx in reversed(range(len(self.layers)-1)):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx+1]
            
            for i, neuron in enumerate(current_layer.neurons):
                error = sum(n.weights[i] * n.delta for n in next_layer.neurons)
                neuron.compute_delta(error)  # Uses sigmoid derivative
                
    def update_weights(self, clip_value=5.0):
        for layer in self.layers:
            for neuron in layer.neurons:
                clipped_delta = np.clip(neuron.delta, -clip_value, clip_value)
                neuron.weights += self.learning_rate * clipped_delta * neuron.inputs
                neuron.bias += self.learning_rate * clipped_delta

    def save(self, filename):
        try:
            data = {
                "layers_config": self.layers_config,
                "layers": []
            }
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
                if data["layers_config"] != self.layers_config:
                    raise ValueError("Network configuration mismatch")
                
                for layer_idx, layer_data in enumerate(data["layers"]):
                    for neuron_idx, neuron_data in enumerate(layer_data):
                        self.layers[layer_idx].neurons[neuron_idx].weights = np.array(neuron_data["weights"])
                        self.layers[layer_idx].neurons[neuron_idx].bias = np.array(neuron_data["bias"])
            print(f"Network loaded from {filename}")
        except Exception as e:
            print(f"Error loading network: {e}")

def calculate_optimal_path(labyrinth, start, goal):
    """A* pathfinding algorithm with 8-direction movement"""
    size = labyrinth.shape[0]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            return reconstruct_path_length(came_from, current)
        
        # Check all 8 possible directions
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1),
                    (-1,-1), (-1,1), (1,-1), (1,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check boundaries and walkability
            if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size and labyrinth[neighbor[0]][neighbor[1]]:
                # Movement cost is 1 for all directions
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

def heuristic(a, b):
    """Chebyshev distance for 8-direction movement"""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy)

def reconstruct_path_length(came_from, current):
    """Calculate path length from reconstruction"""
    path_length = 0
    while current in came_from:
        current = came_from[current]
        path_length += 1
    return path_length

def create_labyrinth(size=6, obstacle_density=0.2):
    labyrinth = np.ones((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            if np.random.random() < obstacle_density:
                labyrinth[i][j] = False
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
                 exploration_rate=0.1, exploration_decay=0.995, min_exploration=0.01,
                 discount_factor=0.95, verbose=True,
                 learning_rate=0.1):  # Removed vision_range parameter
    size = labyrinth.shape[0]
    steps_history = deque(maxlen=100)
    efficiency_history = deque(maxlen=100)
    reward_history = deque(maxlen=100)
    
    network.set_learning_rate(learning_rate)
    current_exploration = exploration_rate
    
    for episode in range(episodes):
        while True:
            start_pos = get_random_position(labyrinth)
            goal_pos = get_random_position(labyrinth)
            while goal_pos == start_pos:
                goal_pos = get_random_position(labyrinth)
            optimal_steps = calculate_optimal_path(labyrinth, start_pos, goal_pos)
            if optimal_steps is not None:
                break
                
        current_pos = start_pos
        steps = 0
        episode_reward = 0
        last_action = None
        
        while current_pos != goal_pos and steps < 500:
            x, y = current_pos
            gx, gy = goal_pos

            inputs = []
            
            # Check 8 surrounding cells (NEW)
            for dx, dy in [(-1,-1), (-1,0), (-1,1),
                          (0,-1),         (0,1),
                          (1,-1),  (1,0), (1,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size and labyrinth[nx][ny]:
                    inputs.append(1)
                else:
                    inputs.append(0)
            
            # Goal direction
            inputs.extend([(gx - x)/size, (gy - y)/size])

            # Last movement
            if last_action is not None:
                action_history = [0]*8
                action_history[last_action] = 1
            else:
                action_history = [0]*8
            inputs.extend(action_history)

            inputs = np.array(inputs).flatten()
            
            # Exploration vs exploitation
            if np.random.random() < current_exploration:
                action = np.random.randint(8)
            else:
                q_values = network.feed_forward(inputs)
                action = np.argmax(q_values)

            last_action = action
                
            # Move agent
            move_map = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
            dx, dy = move_map[action]
            new_x = x + dx
            new_y = y + dy
            
            # Calculate reward
            reward = 0
            if 0 <= new_x < size and 0 <= new_y < size and labyrinth[new_x][new_y]:
                old_dist = np.sqrt((x - gx)**2 + (y - gy)**2)
                new_dist = np.sqrt((new_x - gx)**2 + (new_y - gy)**2)
                
                current_pos = (new_x, new_y)
                steps += 1

                if current_pos == goal_pos:
                    efficiency = optimal_steps / steps
                    reward += 500.0 * efficiency
                    if efficiency >= 0.9:
                        reward += 500.0

                # Direction alignment
                dx_move = new_x - x
                dy_move = new_y - y
                move_dir = np.array([dx_move, dy_move])
                goal_dir = np.array([gx - x, gy - y])

                if np.linalg.norm(move_dir) > 0 and np.linalg.norm(goal_dir) > 0:
                    move_dir_normalized = move_dir / np.linalg.norm(move_dir)
                    goal_dir_normalized = goal_dir / np.linalg.norm(goal_dir)
                    alignment = np.dot(move_dir_normalized, goal_dir_normalized)
                    reward += 10 * alignment
            else:
                reward = -10.0
                
            # Calculate target Q-value
            if current_pos == goal_pos:
                target_q = reward
            else:
                # Generate next state inputs (UPDATED)
                next_inputs = []
                nx, ny = current_pos
                
                # Check 8 surrounding cells for new position
                for dx, dy in [(-1,-1), (-1,0), (-1,1),
                              (0,-1),         (0,1),
                              (1,-1),  (1,0), (1,1)]:
                    cnx, cny = nx + dx, ny + dy
                    if 0 <= cnx < size and 0 <= cny < size and labyrinth[cnx][cny]:
                        next_inputs.append(1)
                    else:
                        next_inputs.append(0)
                
                next_inputs.extend([(gx - nx)/size, (gy - ny)/size])
                action_history = [0]*8
                action_history[action] = 1
                next_inputs.extend(action_history)
                next_inputs = np.array(next_inputs).flatten()
                
                next_q = network.feed_forward(next_inputs)
                target_q = reward + discount_factor * np.max(next_q)
                
            # Train network
            target = network.feed_forward(inputs)
            target[action] = target_q
            
            network.backpropagate(target)
            network.update_weights()
            
            episode_reward += reward

        current_exploration = max(min_exploration, current_exploration * exploration_decay)
            
        # Calculate efficiency metrics
        if current_pos == goal_pos:
            efficiency = (optimal_steps / steps) * 100
        else:
            efficiency = 0.0
            
        efficiency_history.append(efficiency)
        avg_efficiency = sum(efficiency_history) / len(efficiency_history) if efficiency_history else 0.0
        
        # Track rewards
        reward_history.append(episode_reward)
        avg_reward = sum(reward_history) / len(reward_history) if reward_history else 0.0
        
        # Modified reporting with both efficiency and reward
        if verbose:
            print(f"\nEpisode {episode+1} Summary:")
            print(f"Steps: {steps} (Optimal: {optimal_steps})")
            print(f"Efficiency: {efficiency:.2f}%")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Avg Efficiency (Last 100): {avg_efficiency:.2f}%")
            print(f"Avg Reward (Last 100): {avg_reward:.2f}")
            print(f"Exploration: {current_exploration:.2f}")
            time.sleep(0.01)
        else:
            print(f"Episode: {episode+1}/{episodes} | "
                  f"Steps: {steps} (Opt: {optimal_steps}) | "
                  f"Eff: {efficiency:.2f}% | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Eff: {avg_efficiency:.2f}% | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Exploration: {current_exploration:.2f}")

    network.save("trained_network.json")

def testing_loop(network, labyrinth):  # Removed vision_range parameter
    size = labyrinth.shape[0]
    start_pos = get_random_position(labyrinth)
    goal_pos = get_random_position(labyrinth)
    
    current_pos = start_pos
    steps = 0
    last_action = None
    
    while current_pos != goal_pos and steps < 500:
        x, y = current_pos
        gx, gy = goal_pos
        
        inputs = []

        # 8 surrounding cells
        for dx, dy in [(-1,-1), (-1,0), (-1,1),
                      (0,-1),         (0,1),
                      (1,-1),  (1,0), (1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and labyrinth[nx][ny]:
                inputs.append(1)
            else:
                inputs.append(0)
        
        inputs.extend([(gx - x)/size, (gy - y)/size])
        
        if last_action is not None:
            action_history = [0]*8
            action_history[last_action] = 1
        else:
            action_history = [0]*8
        inputs.extend(action_history)

        inputs = np.array(inputs).flatten()
        
        q_values = network.feed_forward(inputs)
        action = np.argmax(q_values)
        
        last_action = action
        
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
        time.sleep(0.001)
        
    if current_pos == goal_pos:
        print("Goal reached!")
    else:
        print("Failed to reach goal")

def signal_handler(sig, frame):
    global network
    print("\nSaving network before exit...")
    network.save("trained_network.json")
    sys.exit(0)

def main():
    global network
    signal.signal(signal.SIGINT, signal_handler)
    
    labyrinth = create_labyrinth(6, obstacle_density=0.2)
    
    mode = input("Choose mode (learn/test): ").lower()
    
    if mode == "learn":
        # Fixed input size: 8 (surroundings) + 2 (goal) + 8 (last action) = 18
        network = NeuralNetwork([18, 72, 36, 8], learning_rate=0.1)

        # Get parameters (removed vision_range questions)
        fixed_learning_rate = float(input("Enter learning rate (e.g. 0.1): ") or "0.1")
        initial_exploration = float(input("Enter initial exploration rate (e.g. 0.5): ") or "0.5")
        exploration_decay = float(input("Enter exploration decay rate (e.g. 0.995): ") or "0.995")
        min_exploration = float(input("Enter minimum exploration (e.g. 0.01): ") or "0.01")

        try:
            network.load("trained_network.json")
            print("Continuing training...")
        except:
            print("New training session")
        
        verbose = input("Enable verbose mode? (y/n): ").lower() == 'y'
        episodes = int(input(f"Enter number of episodes (default: 1000): ") or 1000)
        
        learning_loop(
            network, 
            labyrinth, 
            episodes=episodes, 
            verbose=verbose,
            learning_rate=fixed_learning_rate,
            exploration_rate=initial_exploration,
            exploration_decay=exploration_decay,
            min_exploration=min_exploration
        )
    elif mode == "test":
        network = NeuralNetwork([18, 72, 36, 8], learning_rate=0.1)
        try:
            network.load("trained_network.json")
        except:
            print("No saved network")
        testing_loop(network, labyrinth)
    else:
        print("Invalid mode")

if __name__ == "__main__":
    main()