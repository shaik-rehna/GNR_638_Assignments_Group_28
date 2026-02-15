import my_framework as mf
import json


class CNN:

    def __init__(self, in_channels, num_classes):

        self.in_channels = in_channels
        self.num_classes = num_classes

        # --------------------------------------------------
        # Lightweight Dynamic Architecture
        # --------------------------------------------------

        if num_classes <= 10:
            conv1_out = 4
            conv2_out = 8
            fc_hidden = 32
        else:
            conv1_out = 8
            conv2_out = 16
            fc_hidden = 64

        # --------------------------------------------------
        # Convolution Layers
        # --------------------------------------------------

        self.conv1 = mf.Conv2D(in_channels, conv1_out, 3)
        self.conv2 = mf.Conv2D(conv1_out, conv2_out, 3)
        self.pool = mf.MaxPool2D(2, 2)

        # 32 → 30 → 28 → pool → 14
        self.flatten_dim = conv2_out * 14 * 14

        # --------------------------------------------------
        # Fully Connected Layers
        # --------------------------------------------------

        self.fc1 = mf.Linear(self.flatten_dim, fc_hidden)
        self.fc2 = mf.Linear(fc_hidden, num_classes)

        print("\n--- Model Configuration ---")
        print(f"Conv1: {in_channels} → {conv1_out}")
        print(f"Conv2: {conv1_out} → {conv2_out}")
        print(f"FC1: {self.flatten_dim} → {fc_hidden}")
        print(f"FC2: {fc_hidden} → {num_classes}")
        print("----------------------------")

    def forward(self, x):

        x = self.conv1.forward(x)
        x = mf.relu(x)

        x = self.conv2.forward(x)
        x = mf.relu(x)

        x = self.pool.forward(x)

        x = mf.flatten(x)

        x = self.fc1.forward(x)
        x = mf.relu(x)

        x = self.fc2.forward(x)

        return x

    def parameters(self):
        return (
            self.conv1.parameters() +
            self.conv2.parameters() +
            self.fc1.parameters() +
            self.fc2.parameters()
        )

    def load(self, path):

        with open(path, "r") as f:
            weights = json.load(f)

        params = self.parameters()

        if len(weights) != len(params):
            raise ValueError("Mismatch between saved weights and model parameters")

        for idx, param in enumerate(params):
            param.data = weights[f"param_{idx}"]

        print(f"Model loaded from {path}")

################################################

# import my_framework as mf
# import json

# class CNN:

#     def __init__(self, in_channels, num_classes):

#         # Convolution layers
#         self.conv1 = mf.Conv2D(in_channels, 4, 3)
#         self.conv2 = mf.Conv2D(4, 8, 3)
#         self.pool = mf.MaxPool2D(2, 2)

#         # 32 → 30 → 28 → pool → 14
#         self.flatten_dim = 8 * 14 * 14

#         # Fully connected layers
#         self.fc1 = mf.Linear(self.flatten_dim, 32)
#         self.fc2 = mf.Linear(32, num_classes)

#     def forward(self, x):

#         x = self.conv1.forward(x)
#         x = mf.relu(x)

#         x = self.conv2.forward(x)
#         x = mf.relu(x)

#         x = self.pool.forward(x)

#         x = mf.flatten(x)

#         x = self.fc1.forward(x)
#         x = mf.relu(x)

#         x = self.fc2.forward(x)

#         return x

#     def parameters(self):
#         return (
#             self.conv1.parameters() +
#             self.conv2.parameters() +
#             self.fc1.parameters() +
#             self.fc2.parameters()
#         )

#     # --------------------------------------------------
#     # Load Model Weights
#     # --------------------------------------------------
#     def load(self, path):

#         with open(path, "r") as f:
#             weights = json.load(f)

#         params = self.parameters()

#         if len(weights) != len(params):
#             raise ValueError("Mismatch between saved weights and model parameters")

#         for idx, param in enumerate(params):
#             key = f"param_{idx}"

#             if key not in weights:
#                 raise KeyError(f"Missing {key} in saved weights")

#             param.data = weights[key]

#         print(f"Model loaded from {path}")