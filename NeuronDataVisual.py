import json
import numpy as np
import matplotlib.pyplot as plt

def load_network(filename):
    """Load the saved network data from a JSON file."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        print(f"Network data loaded from {filename}")
        return data
    except Exception as e:
        print(f"Error loading network data: {e}")
        return None

def visualize_neuron_weights(weights, layer_idx, neuron_idx):
    """Visualize the weights of a single neuron as a heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow([weights], cmap='viridis', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.title(f"Neuron Weights\nLayer {layer_idx + 1}, Neuron {neuron_idx + 1}")
    plt.xlabel("Input Index")
    plt.ylabel("Neuron")
    plt.show()

def visualize_layer_weights(layer_weights, layer_idx):
    """Visualize the weights of an entire layer as a heatmap."""
    plt.figure(figsize=(10, 8))
    plt.imshow(layer_weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.title(f"Layer {layer_idx + 1} Weights")
    plt.xlabel("Input Index")
    plt.ylabel("Neuron Index")
    plt.show()

def visualize_layer_biases(layer_biases, layer_idx):
    """Visualize the biases of a layer as a bar chart."""
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(layer_biases)), layer_biases, color='skyblue')
    plt.title(f"Layer {layer_idx + 1} Biases")
    plt.xlabel("Neuron Index")
    plt.ylabel("Bias Value")
    plt.show()

def visualize_network(network_data):
    """Visualize the weights and biases of the entire network."""
    for layer_idx, layer in enumerate(network_data["layers"]):
        # Extract weights and biases for the current layer
        weights = np.array([neuron["weights"] for neuron in layer])
        biases = np.array([neuron["bias"] for neuron in layer])
        
        # Visualize the weights of the entire layer
        visualize_layer_weights(weights, layer_idx)
        
        # Visualize the biases of the layer
        visualize_layer_biases(biases, layer_idx)
        
        # Optionally, visualize weights of individual neurons
        for neuron_idx, neuron in enumerate(layer):
            visualize_neuron_weights(neuron["weights"], layer_idx, neuron_idx)

def main():
    # Load the saved network data
    network_data = load_network("trained_network.json")
    
    if network_data:
        # Visualize the network's weights and biases
        visualize_network(network_data)

if __name__ == "__main__":
    main()