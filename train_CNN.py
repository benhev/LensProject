from lensing_simulation import simulation
from LensCNN import create_cnn, metrics, losses, optimizers

if __name__ == '__main__':
    training_dir = 'training data'  # simulation(npix=152, deltapix=0.1, stacks=10, stack_size=10000, action='save')
    create_cnn(metric=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()], model_name='Beta', batch_size=100,
               epochs=5, training_dir=training_dir,
               callback=['tb', 'mbst'], comments='automated run')
