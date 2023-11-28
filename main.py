import torch
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from dataset import TID2013
from modules import NoisePredictor, UNet
from models import CBDNetwork
from losses import TotalLoss
from trainer import BaselineTrainer
from google.colab import files

def main():

    # Define any additional transformations if needed
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Example transformation
        transforms.ToTensor(),
    ])

    # Create an instance of the TID2013 dataset
    crop_size = (128, 128)
    dataset = TID2013(transform=transform, crop_size=crop_size)

    # Split the dataset into train and test
    train_size = int(1 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for train set
    train_batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # # Create DataLoader for test set
    # test_batch_size = 32
    # test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    noise_predictor = NoisePredictor()
    reconstruction_network = UNet()

    cbd_network = CBDNetwork(noise_predictor = noise_predictor, reconstruction_network=reconstruction_network)
    loss = TotalLoss()
    optimizer = torch.optim.Adam(cbd_network.parameters())
    trainer = BaselineTrainer(
        model=cbd_network,
        loss=loss,
        optimizer=optimizer,
        use_cuda=False
    )

    trainer.fit(train_data_loader=train_dataloader,epoch=25)

        # Save the trained model weights
    model_save_path = "cbd_network_weights.pth"
    torch.save(cbd_network.state_dict(), model_save_path)

    # Download the saved model weights
    files.download(model_save_path)



if __name__ == "__main__":
    main()