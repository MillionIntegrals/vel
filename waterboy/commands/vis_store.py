import visdom


from waterboy.util.visdom import visdom_push_metrics


class VisdomCommand:
    def __init__(self, model_config, storage):
        self.model_config = model_config
        self.storage = storage
        self.vis = visdom.Visdom(env=self.model_config.run_name)

    def run(self):
        metrics = self.storage.get_frame().drop('run_name', axis=1)
        visdom_push_metrics(self.vis, metrics)


def create(model_config, storage):
    return VisdomCommand(model_config, storage)
