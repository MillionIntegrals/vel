import torch
import torch.optim
import torch.nn.functional as F


class SimpleTrainCommand:
    """ Very simple training command - just run the supplied generators """

    def __init__(self, epochs, callbacks):
        self.epochs = epochs
        self.callbacks = callbacks
        self.device = None

    def run(self, model, optimizer, source, model_config):
        """ Run the command with supplied configuration """
        # train_source = source.train_source
        # val_source = source.val_source
        # print("Running training:", model, source, model_config)

        self.device = torch.device(model_config.device)
        model = model.to(self.device)

        for i in range(1, self.epochs+1):
            print("Epoch", i)
            self.run_epoch(i, model, source, model_config, optimizer)

    def run_epoch(self, epoch_idx, model, source, model_config, optimizer):
        """ Run single epoch of training """

        model.train()

        # First run the training
        for batch_idx, (data, target) in enumerate(source.train_source):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch_idx, batch_idx * len(data), len(source.train_source.dataset), 100. * batch_idx / len(source.train_source), loss.item()))

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in source.val_source:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(source.val_source.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(source.val_source.dataset),
            100. * correct / len(source.val_source.dataset)))


def create(epochs, callbacks=None):
    """ Simply train the model """
    callbacks = callbacks or []

    return SimpleTrainCommand(epochs=epochs, callbacks=callbacks)
