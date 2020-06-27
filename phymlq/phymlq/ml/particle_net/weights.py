import urllib


def load_weights(model):
    if model.name == 'particlenet-lite':
        model.load_weights(
            download_weights('phymlq/phymlq/ml/particle_net/weights-particlenet-lite.h5')
        )
    else:
        # TODO: Train on ADA, Colab runs out of memory with the full model
        raise NotImplementedError('ParticleNet Full has not been trained yet.')


def download_weights(path_url: str):
    host_url = 'https://github.com/Jai2500/particle-tagging/blob/master/'
    urllib.request.urlretrieve(
        host_url + path_url + '?raw=true',
        path_url.split('/')[-1]
    )
    return path_url.split('/')[-1]
