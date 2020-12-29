import phymlq


def test_num_params():
    model = phymlq.models.lienet.LieNet()
    assert phymlq.utils.summary.count_params(model) == 51670
    model = phymlq.models.particlenet.ParticleNet()
    assert phymlq.utils.summary.count_params(model) == 366029
    model = phymlq.models.particlenet.ParticleNetLite()
    assert phymlq.utils.summary.count_params(model) == 26220
