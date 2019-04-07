import numpy as np
import tqdm

import torch
import torch.nn.functional as F
import torch.distributions as dist

from vel.api import TrainingInfo


class GenerateTextCommand:
    """ Generate text using RNN model """

    def __init__(self, model_config, model_factory, source, storage, start_letter, length, temperature):
        self.model_config = model_config
        self.model_factory = model_factory
        self.source = source
        self.storage = storage
        self.start_letter = start_letter
        self.length = length
        self.temperature = temperature

    @torch.no_grad()
    def run(self):
        device = self.model_config.torch_device()
        model = self.model_factory.instantiate().to(device)

        start_epoch = self.storage.last_epoch_idx()

        training_info = TrainingInfo(
            start_epoch_idx=start_epoch,
            run_name=self.model_config.run_name,
        )

        model_state, hidden_state = self.storage.load(training_info)
        model.load_state_dict(model_state)

        model.eval()

        current_char = self.start_letter
        current_char_encoded = self.source.encode_character(self.start_letter)

        generated_text = [current_char]

        state = model.zero_state(1).to(device)

        char_tensor = torch.from_numpy(np.array([current_char_encoded])).view(1, 1).to(device)

        for _ in tqdm.trange(self.length):
            prob_logits, state = model.forward_state(char_tensor, state)

            # Apply temperature to the logits
            prob_logits = F.log_softmax(prob_logits.view(-1).div(self.temperature), dim=0)

            distribution = dist.Categorical(logits=prob_logits)

            char_tensor = distribution.sample().view(1, 1)
            current_char_encoded = char_tensor.item()

            if current_char_encoded == 0:
                # End of sequence marker
                break

            current_char = self.source.decode_character(current_char_encoded)

            generated_text.append(current_char)

        print("============================ START GENERATED TEXT ================================================")
        print(''.join(generated_text))
        print("============================ END GENERATED TEXT ================================================")


def create(model_config, model, source, storage, start_letter, length, temperature):
    """ Vel factory function """
    return GenerateTextCommand(model_config, model, source, storage, start_letter, length, temperature)
