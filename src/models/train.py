import torch
from torch.utils.data import DataLoader
from src.data.loader import DummyDataset
from src.models.model import SimpleModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=4)

    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(2):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            preds = model(x)
            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} done")

if __name__ == "__main__":
    main()
