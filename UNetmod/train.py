import torch
from torch.utils.data import DataLoader
from lion_pytorch import Lion
from huggingface_hub import list_repo_files
from configs.settings import Config
from src.models.unet_cnn import UNetmod
from src.data.loader import BubbleFastDataset
from src.utils.helpers import set_deterministic

from dotenv import load_dotenv
load_dotenv()

def main():
    Config.setup_env()
    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetmod().to(device)
    optimizer = Lion(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.MSELoss()

    # Load data file list
    all_files = list_repo_files("hpcforge/BubbleML_2", repo_type="dataset")
    train_files = [f for f in all_files if "Saturated" in f and f.endswith(".hdf5")][:12]

    global_step = 0
    print(f"Starting training on {len(train_files)} files...")

    for epoch in range(Config.EPOCHS):
        model.train()
        dataset = BubbleFastDataset(train_files)
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE)

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred = model(x)
                loss = criterion(pred, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % 100 == 0:
                print(f"Epoch {epoch} | Step {global_step} | Loss {loss.item():.6f}")

            if global_step % Config.SAVE_EVERY == 0:
                torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
                print(f"Saved checkpoint: {Config.CHECKPOINT_PATH}")

    torch.save(model.state_dict(), "final_model.pt")

if __name__ == "__main__":
    main()