import os.path


from vel.api import ModelConfig

project_dir = ModelConfig.find_project_directory(os.getcwd())

model_config = ModelConfig.from_file(
    filename=os.path.join(
        project_dir, 'examples-configs/generative-likelihood/mnist/mnist_cnn_iwae.yaml',
    ),
    run_number=2,
)

model_config.set_seed()
model_config.banner('train')
model_config.run_command('train')
