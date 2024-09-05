import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import requests

class CoordinateTransformNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=2):
        super(CoordinateTransformNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        return self.model(x)


class CoordinateDataHandler:
    def __init__(self, data_file="coordinate_data.json"):
        self.data_file = data_file
        self.game_coords = []
        self.map_coords = []
        self.game_coords_mean = None
        self.game_coords_std = None
        self.map_coords_min = None
        self.map_coords_max = None
        self.model_ok = False
        self.load_data()

    def set_data(self, game_coord, map_coord):
        """
        添加单个坐标对到数据集中
        :param game_coord: 游戏坐标元组 (x, y)
        :param map_coord: 地图坐标元组 (x, y)
        """
        if not isinstance(game_coord, tuple) or not isinstance(map_coord, tuple):
            raise ValueError("Both game_coord and map_coord must be tuples")

        if len(game_coord) != 2 or len(map_coord) != 2:
            raise ValueError("Both coordinates must be 2D (x, y)")

        self.game_coords.append(game_coord)
        self.map_coords.append(map_coord)

    def save_data(self):
        """保存坐标数据到文件"""
        data = {"game_coords": self.game_coords, "map_coords": self.map_coords}
        with open(self.data_file, "w") as f:
            json.dump(data, f)
        print(f"Data saved to {self.data_file}")

    def load_data(self):
        """从文件加载坐标数据"""
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as f:
                data = json.load(f)
            self.game_coords = [tuple(coord) for coord in data["game_coords"]]
            self.map_coords = [tuple(coord) for coord in data["map_coords"]]
            print(f"Data loaded from {self.data_file}")
        else:
            print(f"No existing data file found at {self.data_file}")

    def set_data(self, game_coord, map_coord):
        """
        添加单个坐标对到数据集中
        :param game_coord: 游戏坐标元组 (x, y)
        :param map_coord: 地图坐标元组 (x, y)
        """
        if not isinstance(game_coord, tuple) or not isinstance(map_coord, tuple):
            raise ValueError("Both game_coord and map_coord must be tuples")

        if len(game_coord) != 2 or len(map_coord) != 2:
            raise ValueError("Both coordinates must be 2D (x, y)")

        self.game_coords.append(game_coord)
        self.map_coords.append(map_coord)

    def preprocess_data(self):
        if not self.game_coords or not self.map_coords:
            raise ValueError(
                "No data available. Add data using set_data() before preprocessing."
            )

        game_coords_array = np.array(self.game_coords)
        map_coords_array = np.array(self.map_coords)

        # 移除重复的数据点
        unique_data = np.unique(
            np.concatenate((game_coords_array, map_coords_array), axis=1), axis=0
        )
        game_coords_array = unique_data[:, :2]
        map_coords_array = unique_data[:, 2:]

        self.game_coords_mean = game_coords_array.mean(axis=0)
        self.game_coords_std = game_coords_array.std(axis=0)
        self.map_coords_min = map_coords_array.min(axis=0)
        self.map_coords_max = map_coords_array.max(axis=0)

        # 添加 epsilon 以避免除以零
        epsilon = 1e-8
        game_coords_normalized = (game_coords_array - self.game_coords_mean) / (
            self.game_coords_std + epsilon
        )
        map_coords_normalized = (map_coords_array - self.map_coords_min) / (
            self.map_coords_max - self.map_coords_min + epsilon
        )

        return game_coords_normalized, map_coords_normalized

    def clean_data(self):
        """清除所有坐标数据"""
        self.game_coords = []
        self.map_coords = []
        self.game_coords_mean = None
        self.game_coords_std = None
        self.map_coords_min = None
        self.map_coords_max = None
        self.model_ok = False
        self.save_data()
        print("All coordinate data has been cleared.")


class CoordinateTransformer(CoordinateDataHandler):
    def __init__(self, data_file="coordinate_data.json", input_dim=2, hidden_dim=16, output_dim=2):
        super().__init__(data_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CoordinateTransformNet(input_dim, hidden_dim, output_dim).to(self.device)

    def train_model(self, epochs=5000, batch_size=4, lr=0.001, early_stop_threshold=1e-6, patience=100):
        print(f"Using device: {self.device}")
        
        game_coords_normalized, map_coords_normalized = self.preprocess_data()
        
        X_train, X_val, y_train, y_val = train_test_split(game_coords_normalized, map_coords_normalized, test_size=0.2, random_state=42)
        
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5, verbose=True)
        
        best_val_loss = float('inf')
        no_improve_count = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for batch_game, batch_map in train_loader:
                batch_game, batch_map = batch_game.to(self.device), batch_map.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_game)
                loss = criterion(outputs, batch_map)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            self.model.eval()
            with torch.no_grad():
                val_game, val_map = next(iter(val_loader))
                val_game, val_map = val_game.to(self.device), val_map.to(self.device)
                val_outputs = self.model(val_game)
                val_loss = criterion(val_outputs, val_map).item()
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.8f}, Val Loss: {val_loss:.8f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if val_loss < early_stop_threshold:
                print(f"Validation loss {val_loss:.8f} is below threshold {early_stop_threshold}. Stopping training.")
                break
            
            if no_improve_count >= patience:
                print(f"No improvement for {patience} epochs. Stopping training.")
                break
        
        self.model_ok = True

    def predict(self, game_coords):
        if not self.model_ok:
            raise ValueError("Model has not been trained yet. Call train_model() first.")

        self.model.eval()
        with torch.no_grad():
            game_coords_normalized = (np.array(game_coords) - self.game_coords_mean) / (self.game_coords_std + 1e-8)
            game_coords_tensor = torch.FloatTensor(game_coords_normalized).to(self.device)
            map_coords_normalized = self.model(game_coords_tensor)
            map_coords = map_coords_normalized.cpu().numpy() * (self.map_coords_max - self.map_coords_min) + self.map_coords_min
            return map_coords

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'game_coords_mean': self.game_coords_mean,
            'game_coords_std': self.game_coords_std,
            'map_coords_min': self.map_coords_min,
            'map_coords_max': self.map_coords_max
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.game_coords_mean = checkpoint['game_coords_mean']
        self.game_coords_std = checkpoint['game_coords_std']
        self.map_coords_min = checkpoint['map_coords_min']
        self.map_coords_max = checkpoint['map_coords_max']
        self.model_ok = True

class CoordinateGet:
    def __init__(self, api_url="http://192.168.2.60:8001/coordinate"):
        self.api_url = api_url

    def get_coordinate(self):
        try:
            response = requests.get(self.api_url)
            if response.status_code == 200:
                data = response.json()
                return data['x'], data['y']
            else:
                print(f"Failed to fetch coordinate. Status code: {response.status_code}")
                return 0, 0
        except Exception as e:
            print(f"Error fetching coordinate: {e}")
            return 0, 0
