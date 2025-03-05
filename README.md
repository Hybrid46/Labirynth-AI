Labyrinth AI - Neural Network Pathfinding

Labyrinth Example <!-- Add an image if you have one -->

This project implements a neural network-based AI to solve labyrinth pathfinding problems. The AI learns to navigate through randomly generated labyrinths, avoiding obstacles and finding the shortest path to the goal. The neural network is trained using a combination of Q-learning and backpropagation.

The project consists of two main scripts:

    LabirynthAI_v2.py: The main script for training and testing the AI.

    visualize_network.py: A script to visualize the weights and biases of the trained neural network.

Features

    Dynamic Labyrinth Generation: Create labyrinths of any size with customizable obstacle density.

    Neural Network AI: A fully connected neural network learns to navigate the labyrinth.

    Training and Testing Modes: Train the AI or test its performance on pre-trained models.

    Exploration vs Exploitation: Implements an exploration rate that decays over time for balanced learning.

    Learning Rate Decay: Adjusts the learning rate dynamically during training for stable convergence.

    Save and Load Models: Save trained models to a file and load them for future use.

    Visualization: Visualize the weights and biases of the neural network using heatmaps and bar charts.

Installation

    Clone the Repository:
    bash
    Copy

    git clone https://github.com/your-username/labyrinth-ai.git
    cd labyrinth-ai

    Install Dependencies:
    Ensure you have Python 3.7+ installed. Install the required libraries using:
    bash
    Copy

    pip install numpy matplotlib

    Run the Program:
    Execute the main script to start the program:
    bash
    Copy

    python LabirynthAI_v2.py

Usage
Training the AI

    Choose the learn mode when prompted.

    Set the number of episodes, exploration rate, and learning rate parameters.

    The AI will train on randomly generated labyrinths and save the trained model to trained_network.json.

Testing the AI

    Choose the test mode when prompted.

    The AI will load the pre-trained model from trained_network.json and demonstrate its performance on a new labyrinth.

Visualizing the Network

Run the visualization script to see the weights and biases of the trained network:
bash
Copy

python visualize_network.py

Code Structure
LabirynthAI_v2.py

    Neuron Class: Represents a single neuron in the neural network.

    Layer Class: Represents a layer of neurons.

    NeuralNetwork Class: Implements the neural network, including feedforward, backpropagation, and weight updates.

    create_labyrinth Function: Generates a random labyrinth.

    learning_loop Function: Handles the training process.

    testing_loop Function: Handles the testing process.

    signal_handler Function: Saves the network when the program is interrupted (e.g., with CTRL+C).

visualize_network.py

    load_network Function: Loads the saved network data from a JSON file.

    visualize_neuron_weights Function: Visualizes the weights of a single neuron as a heatmap.

    visualize_layer_weights Function: Visualizes the weights of an entire layer as a heatmap.

    visualize_layer_biases Function: Visualizes the biases of a layer as a bar chart.

    visualize_network Function: Visualizes the weights and biases of the entire network.

Example
Training
bash
Copy

Choose mode (learn/test): learn
Enable verbose mode? (y/n): n
Enter number of episodes (default: 1000): 500
Enter minimum exploration rate (default: 0.01): 0.01
Enter initial learning rate (default: 0.5): 0.5
Enter learning rate decay (default: 0.995): 0.995
Enter minimum learning rate (default: 0.01): 0.01

Testing
bash
Copy

Choose mode (learn/test): test

Visualization
bash
Copy

python visualize_network.py

Labyrinth Representation

The labyrinth is represented as a grid where:

    . represents a path.

    # represents a wall.

    A represents the agent.

    O represents the goal.

Example labyrinth at the start of training:
Copy

A . . # . .
. . # . # .
# . . . . .
. # . # . .
. . . . # .
# . # . . .

Example labyrinth at the end of training (goal reached):
Copy

. . . # . .
. . # . # .
# . . . . .
. # . # . .
. . . . # .
# . # . A O

Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

    Fork the repository.

    Create a new branch for your feature or bugfix.

    Commit your changes and push to your branch.

    Submit a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

    Inspired by reinforcement learning and neural network tutorials.

    Special thanks to the open-source community for providing valuable resources.