import torch


def discount_bootstrap(rewards_buffer, dones_buffer, final_values, discount_factor, number_of_steps):
    """ Calculate state values bootstrapping off the following state values """
    true_value_buffer = torch.zeros_like(rewards_buffer)

    # discount/bootstrap off value fn
    current_value = final_values

    for i in reversed(range(number_of_steps)):
        current_value = rewards_buffer[i] + discount_factor * current_value * (1.0 - dones_buffer[i])
        true_value_buffer[i] = current_value

    return true_value_buffer


def discount_bootstrap_gae(rewards_buffer, dones_buffer, values_buffer, final_values, discount_factor, gae_lambda,
                           number_of_steps):
    """
    Calculate state values bootstrapping off the following state values - Generalized Advantage Estimation
    https://arxiv.org/abs/1506.02438
    """
    advantage_buffer = torch.zeros_like(rewards_buffer)

    # Accmulate sums
    sum_accumulator = 0

    for i in reversed(range(number_of_steps)):
        if i == number_of_steps - 1:
            next_value = final_values
        else:
            next_value = values_buffer[i + 1]

        bellman_delta = (
                rewards_buffer[i] + discount_factor * next_value * (1.0 - dones_buffer[i]) - values_buffer[i]
        )

        advantage_buffer[i] = sum_accumulator = (
                bellman_delta + discount_factor * gae_lambda * sum_accumulator * (1.0 - dones_buffer[i])
        )

    return advantage_buffer
