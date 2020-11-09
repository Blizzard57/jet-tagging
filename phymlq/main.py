import phymlq

phymlq.hep.particle_net.datasets.TopTaggingPreparer().download().prepare(100000)
train_df = phymlq.hep.particle_net.datasets.TopTaggingDataset('data/train_file_0.awkd')
val_df = phymlq.hep.particle_net.datasets.TopTaggingDataset('data/val_file_0.awkd')
test_df = phymlq.hep.particle_net.datasets.TopTaggingDataset('data/test_file_0.awkd')
model = phymlq.hep.particle_net.models.ParticleNet()
