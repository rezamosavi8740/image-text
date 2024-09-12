import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons, act_func, dropout_nums, dropout_prob=0.3, weight_init='random'):
        super().__init__()
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        self.act_func = act_func
        self.dropout_nums = dropout_nums
        self.dropout_prob = dropout_prob
        self.weight_init = weight_init

        self._init_model()

    def _init_model(self):
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_neurons, self.hidden_neurons[0]))
        self._init_weights(self.layers[0])

        for i in range(len(self.hidden_neurons) - 1):
            self.layers.append(nn.Linear(self.hidden_neurons[i], self.hidden_neurons[i + 1]))
            self._init_weights(self.layers[-1])

        self.layers.append(nn.Linear(self.hidden_neurons[-1], self.output_neurons))
        self._init_weights(self.layers[-1])

    def _init_weights(self, layer):
        if self.weight_init == 'random':
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        elif self.weight_init == 'He':
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        elif self.weight_init == 'Xavier':
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            if i in self.dropout_nums:
                x = F.dropout(self.act_func(layer(x)), self.dropout_prob, training=self.training)
            else:
                x = self.act_func(layer(x))

        y = self.layers[-1](x)

        return y

class AutoEncoders(nn.Module):
    def __init__(self, input_shape_image, latent_shape_image, output_shape_image, input_shape_text, latent_shape_text, output_shape_text):
        super().__init__()
        # Image AE
        self.image_encoder = MLP(input_shape_image, [1024, 512, 256], latent_shape_image, F.relu, [1, 2], weight_init='He')
        self.image_decoder = MLP(latent_shape_image, [256, 512, 1024], output_shape_image, F.relu, [1, 2], weight_init='He')

        # Text AE
        self.text_encoder = MLP(input_shape_text, [1024, 512, 256], latent_shape_text, F.relu, [1, 2], weight_init='He')
        self.text_decoder = MLP(latent_shape_text, [256, 512, 1024], output_shape_text, F.relu, [1, 2], weight_init='He')

    def forward(self, x1, x2):
        encoded_image = self.image_encoder(x1)
        encoded_text = self.text_encoder(x2)

        decoded_image = self.image_decoder(encoded_text)
        decoded_text = self.text_decoder(encoded_image)

        return encoded_image, encoded_text, decoded_image, decoded_text