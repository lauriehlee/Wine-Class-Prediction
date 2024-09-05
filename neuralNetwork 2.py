import math
import copy

def backpropagation(neurons_in_layers, theta_values, training_set, alpha, mini_batch_size, regularization_parameter):
    # theta_values: [[LAYER 1: the weights of each neuron, including the bias weight, are stored in the rows)], [LAYER 2: the weights of each neuron, including the bias weight, are stored in the rows], etc.]
    # Initialize all the weights of the network with non-zero random values

    total_cost = (float)(0)
    summed_gradients = []
    batch_iteration = 1
    original_theta_values = copy.deepcopy(theta_values)
    # For each instance (x,y) in the training set:
    for instance in training_set:
        # Propagate the instance through the network, computing its output/prediction
        activation_matrix, output_activations = forward_propagation(neurons_in_layers, theta_values, instance)
        print("Output activation f(x): ", output_activations)
        print("Expected output: ", instance["y"])

        cost = find_cost(instance, output_activations)
        total_cost += cost
        print("Cost, J associated with instance: ", cost)

        # Compute the delta values for all neurons in the output layer
        delta_values = []
        delta_output(delta_values, output_activations, instance)

        # Compute the delta values for all neurons in the hidden layers (NOT needed for bias neurons)
        # The ordering corresponds in a backwards manner (e.g., index 0 contains delta values for output layer, index 1 contains delta values for last hidden layer, etc.) 
        delta_hidden(delta_values, (len(neurons_in_layers) - 1), theta_values, activation_matrix)

        # Compute the gradients of all weights in the network (Different for bias and non-bias neurons)
        gradients = compute_gradients(theta_values, activation_matrix, delta_values)
        
        # If you are starting a new batch of instances, initialize the summed gradients
        if batch_iteration == 1:
            summed_gradients = gradients
        elif batch_iteration < mini_batch_size:
            # Instances between first and last in batch: add gradients calculated so that they can be averaged after the whole batch is processed
            add_gradients(gradients, summed_gradients)
        else:
            # Last instance in batch: add gradient, then calculate the average gradient, then update the weights and reset the batch.
            add_gradients(gradients, summed_gradients)
            gradients = average_gradients(summed_gradients, mini_batch_size, theta_values, regularization_parameter)
            print("Final regularized gradients: ", gradients) 
            # Adjust all weights of the network according to the gradient descent update rule
            update_weights(gradients, theta_values, alpha)
            print("Updated weights: ", theta_values)
            batch_iteration = 0

        batch_iteration += 1
    
    # Evaluate the performance, J, of the network; if improvement too small, stop
    total_cost /= (float)(len(training_set))
    total_cost = penalization(total_cost, original_theta_values, regularization_parameter, len(training_set))

    print("Final regularized cost, J, based on the complete training set: ", total_cost)


    return #TODO fix this!

def penalization(total_cost, theta_values, regularization_parameter, n):
    penalization = 0
    for layer in range(len(theta_values)):
        for neuron in range(len(theta_values[layer])):
            for weight in range(len(theta_values[layer][neuron])):
                # Exclude bias terms
                if weight == 0:
                    continue
                penalization += (math.pow(theta_values[layer][neuron][weight], 2))

    penalization *= (regularization_parameter / (2 * n))
    
    return total_cost + penalization

def average_gradients(summed_gradients, mini_batch_size, theta_values, regularization_parameter):
    for layer in range(len(summed_gradients)):
        for neuron in range(len(summed_gradients[layer])):
            for weight in range(len(summed_gradients[layer][neuron])):
                if weight != 0:
                    # Add penalization for non-bias neurons
                    summed_gradients[layer][neuron][weight] += (theta_values[layer][neuron][weight] * regularization_parameter)
                
                summed_gradients[layer][neuron][weight] /= (float)(mini_batch_size)
    

    return summed_gradients

def add_gradients(gradients, summed_gradients):
    for layer in range(len(summed_gradients)):
        for neuron in range(len(summed_gradients[layer])):
            for weight in range(len(summed_gradients[layer][neuron])):
                summed_gradients[layer][neuron][weight] += gradients[layer][neuron][weight]
    
    return 

def find_cost(instance, output_activations):
    # Sum the nonregularized cost for each output of the network
    total_cost = 0
    idx = 0
    for output in instance["y"]:
        total_cost += (((-1 * output) * math.log(output_activations[idx])) - ((1 - output) * math.log(1 - output_activations[idx])))
        idx += 1

    return total_cost 

def update_weights(gradients, theta_values, alpha):

    # Update the weights using the gradients calculated, which are held in exactly the same structure as the theta_values
    for layer in range(len(theta_values)):
        for neuron in range(len(theta_values[layer])):
            for weight_idx in range(len(theta_values[layer][neuron])):
                theta_values[layer][neuron][weight_idx] -= (alpha * gradients[layer][neuron][weight_idx])
    
    return 
    

def compute_gradients(theta_values, activation_matrix, delta_values):
    gradients = []

    activation_layer_idx = 0
    delta_layer_idx = len(delta_values) - 2
    theta_layer_idx = 0
    i = 1
    for layer in theta_values:
        non_regularized_layer_gradients = []
        delta_idx = 0
        for neuron_incoming_weights in layer:
            # Create an array for incoming weights to the neuron
            neuron_incoming_gradients = []
            neuron_idx = 0
            activation_idx = 0
            weight_idx = 0
            for weight in neuron_incoming_weights:
                gradient = (activation_matrix[activation_layer_idx][activation_idx] * delta_values[delta_layer_idx][delta_idx])
                neuron_incoming_gradients.append(gradient)
                activation_idx += 1
                weight_idx += 1
            
            # Append to layer gradients
            non_regularized_layer_gradients.append(neuron_incoming_gradients)
            delta_idx += 1
            neuron_idx += 1
        print("Gradients for theta ", i, non_regularized_layer_gradients)

        gradients.append(non_regularized_layer_gradients)
        activation_layer_idx += 1
        delta_layer_idx -= 1
        theta_layer_idx += 1
        i += 1


    return gradients

def delta_hidden(delta_values, num_considered_layers, theta_values, activation_matrix):
    layer = num_considered_layers
    theta_layer_idx = len(theta_values) - 1
    activation_layer_idx = len(activation_matrix) - 2
    delta_layer_idx = 0

    while layer >= 1:
        # Evaluate the delta values for neurons in the current layer
        activation_idx = 1 # Ignore bias neuron at index 0
        delta_values_curr_layer = []
        
        # The number of iterations depends on how many weights contribute to each neuron in the following layer 
        # e.g., since the first neuron in the following layer has three input weights, that means the previous layer 
        # contains 3 neurons for which 2 of them you need to find delta values for (excludes bias neuron).
        for i in range(len(theta_values[theta_layer_idx][0])):

            delta_idx = 0
            # Disregard bias neurons
            if i == 0:
                continue

            delta = 0
            for neuron in theta_values[theta_layer_idx]:
                delta += (neuron[i] * delta_values[delta_layer_idx][delta_idx])
                delta_idx += 1
            
            delta *= activation_matrix[activation_layer_idx][activation_idx]
            delta *= (1 - activation_matrix[activation_layer_idx][activation_idx])
            activation_idx += 1

            delta_values_curr_layer.append(delta)
        
        
        delta_values.append(delta_values_curr_layer)
        if layer != 1:
            print("Delta values for layer ", layer, delta_values_curr_layer)
        
        delta_layer_idx += 1
        theta_layer_idx -= 1
        activation_layer_idx -= 1
        layer -= 1


    return

def delta_output(delta_values, output_activations, instance):
    delta_output_values = []
    i = 0
    for activation in output_activations:
        delta = activation - instance["y"][i]
        delta_output_values.append(delta)
        i += 1
    
    print("Delta values for output layer: ", delta_output_values)

    delta_values.append(delta_output_values)

    return


def forward_propagation(neurons_in_layers, theta_values, instance):

    z_matrix = []          # Represents the value of z of each neuron in each layer (excluding the first layer)
    activation_matrix = [] # Represents the activation of each neuron in each layer

    # Append all the feature values of the input layer
    feature_activation = []
    feature_activation.append((float)(1)) # Bias neuron activation
    for feature_value in instance["x"]:
        feature_activation.append(feature_value)
        
    activation_matrix.append(feature_activation)


    print("Activation values for layer 1: ", feature_activation)
    # Calculate the activation of each neuron in the layers after the input layer & Find the indices of the relevant info 
    layer = 2
    current_layer_idx = 0
    parent_neuron_idx = 0
    while layer <= len(neurons_in_layers):

        # Add bias activation, unless curr layer is the output layer.
        curr_activation_values = []
        if layer < len(neurons_in_layers):
            curr_activation_values.append((float)(1))

        # Calculate the z value of each neuron in the current layer
        curr_z_values = []
        for neuron_weights in theta_values[current_layer_idx]:
            z_value = 0 
            weight_idx = 0
            parent_neuron_activation_idx = 0
            for i in range(len(neuron_weights)):
                z_value += (neuron_weights[weight_idx] * activation_matrix[parent_neuron_idx][parent_neuron_activation_idx])

                weight_idx += 1
                parent_neuron_activation_idx += 1

            curr_z_values.append(z_value)
            curr_activation_values.append(sigmoid(z_value)) 
        
        z_matrix.append(curr_z_values)
        activation_matrix.append(curr_activation_values)

        print("Z values for layer ", layer, curr_z_values)
        print("Activation values for layer ", layer, curr_activation_values)

        
        current_layer_idx += 1
        parent_neuron_idx += 1
        layer += 1

    # Return the activation matrix & the output activation(s)
    return activation_matrix, activation_matrix[parent_neuron_idx]

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

if __name__ == "__main__":
    neurons_in_layers = [2, 4, 3, 2]

    theta_values = [[[0.42000, 0.15000, 0.40000],  
                    [0.72000, 0.10000, 0.54000],  
                    [0.01000, 0.19000, 0.42000],  
                    [0.30000, 0.35000, 0.68000]],
                    [[0.21000, 0.67000, 0.14000, 0.96000, 0.87000],  
                    [0.87000, 0.42000, 0.20000, 0.32000, 0.89000],  
                    [0.03000, 0.56000, 0.80000, 0.69000, 0.09000]],
                    [[0.04000, 0.87000, 0.42000, 0.53000],  
                    [0.17000, 0.10000, 0.95000, 0.69000]]]

    training_set = [{"x": [0.32000, 0.68000], "y": [0.75000, 0.98000]}, {"x": [0.83000, 0.02000], "y": [0.75000, 0.28000]}]
    alpha = 1
    mini_batch_size = 2
    regularization_parameter = 0.250
    backpropagation(neurons_in_layers, theta_values, training_set, alpha, mini_batch_size, regularization_parameter)

    """neurons_in_layers = [1, 2, 1]

    theta_values = [[[0.4, 0.1], [0.3, 0.2]], [[0.7, 0.5, 0.6]]]

    training_set = [{"x": [0.13], "y": [0.9000]}, {"x": [0.42000], "y": [0.23000]}]
    alpha = 1
    mini_batch_size = 2
    regularization_parameter = 0
    backpropagation(neurons_in_layers, theta_values, training_set, alpha, mini_batch_size, regularization_parameter)"""



