import torch
import phymlq


phymlq.data.tagging.TopTaggingDataset.download_files()
train_dataset = phymlq.data.tagging.TopTaggingGraphDataset('train_1').get_loader()
val_dataset = phymlq.data.tagging.TopTaggingGraphDataset('val_1').get_loader()

model = phymlq.models.particlenet.ParticleNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# phymlq.trainer.train(model, criterion, optimizer, train_dataset, val_dataset)
