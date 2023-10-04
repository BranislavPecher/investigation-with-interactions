import math
from tqdm import tqdm
import numpy as np


class MetaLearner(object):
    """Generic class for wrapping meta-learning approaches.
    Implements the meta-training and meta-evaluation.
    Parameters
    ----------
    meta_learning_model: Object of the meta-learning approach
    """

    def __init__(self, meta_learning_model):
        self.meta_learner = meta_learning_model

    def train(self, train_data_loader, num_batches, epochs, verbose=True):
        epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(epochs)))
        for epoch in range(epochs):
            print(f'Running epoch {epoch+1}/{epochs}\n')
            self.train_iter(train_data_loader,
                            max_batches=num_batches,
                            verbose=verbose,
                            epoch=epoch,
                            num_epochs=epochs,
                            desc='Training',
                            leave=False)

    def train_iter(self, dataloader, max_batches, verbose, epoch, num_epochs, **kwargs):
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.meta_learner.meta_iteration(dataloader, max_batches=max_batches, epoch=epoch,
                                                            num_epochs=num_epochs):
                pbar.update(1)
                postfix = {'loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    postfix['accuracy'] = '{0:.4f}'.format(
                        np.mean(results['accuracies_after']))
                pbar.set_postfix(**postfix)

    def evaluate(self, dataloader, max_batches=500, verbose=True, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.meta_learner.evaluate(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                                      - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['accuracies_after'] = mean_accuracy

        return mean_results

    def evaluate_in_batch(self, dataloader, batch=64):
        return self.meta_learner.evaluate_in_batch(dataloader, batch)
    
    def serialize(self, path):
        print(f'Serializing model: {self.meta_learner.__class__.__name__}!')
        self.meta_learner.serialize(path)
    
    def load_serialization(self, path):
        print(f'Loading serialization of {self.meta_learner.__class__.__name__}!')
        self.meta_learner.load_serialization(path)
    
    def serialize_or_load(self, path):
        self.meta_learner.serialize_or_load(path)
    
    def serialization_exists(self, path):
        return self.meta_learner.serialization_exists(path)
