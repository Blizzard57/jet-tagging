import phymlq


def test_arrays():
    phymlq.data.tagging.TopTaggingDataset.download_files()
    phymlq.data.tagging.TopTaggingDataset.transform_datafiles("train", 318, 73, 1, True)
    df = phymlq.data.tagging.TopTaggingDataset("train_1.npz")
    assert df.points.shape == (318, 73, 2)
    assert df.features.shape == (318, 73, 4)
    assert df.mask.shape == (318, 73, 1)
