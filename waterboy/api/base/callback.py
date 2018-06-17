class Callback:
    '''
    An abstract class that all callback classes extends from.
    Must be extended before usage.
    '''
    def on_train_begin(self): pass

    def on_train_end(self): pass

    def on_phase_begin(self): pass

    def on_phase_end(self): pass

    def on_epoch_begin(self, epoch_idx): pass

    def on_epoch_end(self, epoch_idx, metrics): pass

    def on_batch_begin(self, progress_idx): pass

    def on_batch_end(self, progress_idx,  metrics): pass

    def on_batch_end_optimizer(self, progress_idx,  metrics, optimizer): pass

    def on_validation_begin(self, epoch_idx): pass

    def on_validation_end(self, epoch_idx,  metrics): pass

    def write_state_dict(self, hidden_state_dict): pass

    def load_state_dict(self, hidden_state_dict): pass
