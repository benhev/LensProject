from lensing_simulation import simulation
from LensCNN import create_cnn

if __name__ == '__main__':
    training_dir = simulation(npix=150, deltapix=0.1, stacks=10, stack_size=10000, action='save')
    create_cnn(model_name='Alpha', batch_size=100, epochs=3, training_dir='test training', callback=['mbst'],
               comments='automated run')
