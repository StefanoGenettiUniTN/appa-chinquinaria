import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from chinquinaria.config import CONFIG
from chinquinaria.utils.logger import get_logger

logger = get_logger(__name__)

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_layers=[128, 64, 32]):
        super(MLPModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class TorchMLPModel:
    def __init__(self, input_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = MLPModel(input_dim=input_dim, hidden_layers=[128,64,32]).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = 64
        self.epochs = 50
    
    def train(self, X_train, y_train):
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            if CONFIG["debug"]:
                logger.debug(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(loader):.4f}")

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy().flatten()
        return y_pred

    def __call__(self, X):
        return self.predict(X)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path, input_dim):
        self.model = MLPModel(input_dim).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))