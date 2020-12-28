import phymlq

df_train = phymlq.data.tagging.TopTaggingDataset('train_1.npz')
print(df_train.points.shape, df_train.features.shape, df_train.mask.shape, df_train.y.shape)
