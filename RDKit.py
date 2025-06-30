import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

train_data = pd.read_csv('neurips-open-polymer-prediction-2025/train.csv')
test_data = pd.read_csv('neurips-open-polymer-prediction-2025/test.csv')

# 特征提取：Morgan Fingerprint
def get_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 1024
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return list(map(int, fp.ToBitString()))

# 先提取指纹
fingerprints = train_data['SMILES'].apply(get_morgan_fingerprint)

# 过滤掉无效指纹（None）
valid_idx = fingerprints.notnull()
fingerprints = fingerprints[valid_idx]
X = pd.DataFrame(fingerprints.tolist())

# 同步更新 y
y = train_data.loc[valid_idx][['Tg', 'FFV', 'Tc', 'Density', 'Rg']].values

# 数据预处理
X = pd.DataFrame(X)
scaler = StandardScaler()
y = scaler.fit_transform(y)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 数据格式
train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型
class MultiOutputRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MultiOutputRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output_layer(x)

model = MultiOutputRegressor(input_dim=1024)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    predictions = []
    for batch_features, _ in test_loader:
        outputs = model(batch_features)
        predictions.extend(outputs.numpy())
    predictions = scaler.inverse_transform(predictions)
    print(predictions)