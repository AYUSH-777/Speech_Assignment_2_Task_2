import torch
import torchmetrics
from torch.utils.data import DataLoader, Subset
from your_sepformer_module import SepFormerModel  # Import the pre-trained SepFormer model
from your_librimix_dataset_module import LibriMixDataset  # Import LibriMix dataset class

# Load the pre-trained SepFormer model
model = SepFormerModel()  # Initialize or load your pre-trained model

# Define the SISNRi and SDRi metrics
sisnri_metric = torchmetrics.SignalToInterferenceNoiseRatioImproved()
sdr_metric = torchmetrics.SignalToDistortionRatioImproved()

# Load the LibriMix dataset
librimix_dataset = LibriMixDataset('/drive/content/librimix_dataset')  # Replace with your LibriMix dataset path

# Split the dataset into training and testing subsets
num_samples = len(librimix_dataset)
train_indices = torch.randperm(num_samples)[:int(0.7 * num_samples)]
test_indices = torch.randperm(num_samples)[int(0.7 * num_samples):]

train_dataset = Subset(librimix_dataset, train_indices)
test_dataset = Subset(librimix_dataset, test_indices)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluation loop
model.eval()
with torch.no_grad():
    for batch in test_loader:
        mixture, sources = batch['mixture'], batch['sources']
        estimated_sources = model(mixture)

        # Compute SISNRi and SDRi metrics
        sisnri = sisnri_metric(estimated_sources, sources)
        sdr = sdr_metric(estimated_sources, sources)

# Compute overall metrics
overall_sisnri = sisnri_metric.compute()
overall_sdr = sdr_metric.compute()

print("Overall SISNRi:", overall_sisnri)
print("Overall SDRi:", overall_sdr)
