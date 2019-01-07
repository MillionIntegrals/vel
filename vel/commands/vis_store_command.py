import visdom


from vel.util.visdom import visdom_push_metrics, VisdomSettings


class VisdomCommand:
    """ Send metrics from database into VISDOM """
    def __init__(self, model_config, storage, visdom_settings: VisdomSettings):
        self.model_config = model_config
        self.storage = storage
        self.vis = visdom.Visdom(
            server=visdom_settings.server,
            endpoint=visdom_settings.endpoint,
            port=visdom_settings.port,
            env=self.model_config.run_name.replace('/', '_')
        )

    def run(self):
        metrics = self.storage.get_metrics_frame().drop('run_name', axis=1)
        visdom_push_metrics(self.vis, metrics)


def create(model_config, storage, visdom_settings):
    """ Vel factory function """
    return VisdomCommand(model_config, storage, VisdomSettings(**visdom_settings))
