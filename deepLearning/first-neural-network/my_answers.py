import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        #self.activation_function = lambda x : 0  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        def sigmoid(x):
            return 1/(1 + np.exp(-x))  # Replace 0 with your sigmoid calculation here
        self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0] 
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)  #features[1] x hidden_nodes = features[1] x 2
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape) #hidden_nodes x output_nodes = 2 x 1
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        ### Forward pass ##
        
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) 
        # n_records x n_features * n_features x n_hidden_nodes = n_records x n_hidden_nodes
        
        hidden_outputs = self.activation_function(hidden_inputs) 
        # n_records x n_hidden_nodes
        
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer --> n_records
        # n_records x n_hidden_nodes * n_hidden_nodes x n_output_nodes = n_records x n_output_nodes
        
        final_outputs = final_inputs # signals from final output layer  --> n_records
        # n_records x 1
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
    
        ### Backward pass ###

        error = y - final_outputs
        # n_records x 1
        
        output_error_term = error * 1  
        # n_records x 1
      
        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)
        # n_records x 1 * (n_hidden_nodes x 1).T = n_records x n_hidden_nodes 
        
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs) 
        # n_records x n_hidden_nodes * n_records x n_hidden_nodes * n_records x n_hidden_nodes = n_records x n_hidden_nodes
        
        delta_weights_i_h += np.dot(X[:,None], (hidden_error_term[:,None]).T)
        # n_features x n_hidden_nodes + n_features x n_records * n_records x n_hidden_nodes = n_features x n_hidden_nodes
        
        delta_weights_h_o += np.dot(hidden_outputs[:,None], output_error_term[:,None])  
        # n_hidden_nodes x n_output_nodes + n_hidden_nodes x n_records * n_records x 1 
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 5000
learning_rate = 0.7 #initial 0.1
hidden_nodes = 15 #initial 2
output_nodes = 1
