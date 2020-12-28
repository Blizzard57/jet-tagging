import phymlq
import torch


phymlq.data.tagging.TopTaggingDataset.download_files()

# Just to test shapes of arrays
df_train = phymlq.data.tagging.TopTaggingDataset('train_1.npz')
df_val = phymlq.data.tagging.TopTaggingDataset('val_1.npz')
print(df_train.points.shape, df_train.features.shape, df_train.mask.shape, df_train.y.shape)

train_dataset = phymlq.data.tagging.TopTaggingGraphDataset('train_1').get_loader()
val_dataset = phymlq.data.tagging.TopTaggingGraphDataset('val_1').get_loader()

model = phymlq.models.lienet.LieNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

phymlq.trainer.train(model, criterion, optimizer, train_dataset, val_dataset)
