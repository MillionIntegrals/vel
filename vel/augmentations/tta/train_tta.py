# import torch
# import torch.utils.data as tdata
#
# import vel.api.data as wdata
#
#
# class TrainTTALoader:
#     def __init__(self, n_augmentations, batch_size, data_source, augmentations, num_workers):
#         self.n_augmentations = n_augmentations
#         self.data_source = data_source
#         self.augmentations = augmentations
#
#         self.val_ds = wdata.DataFlow(self.data_source, augmentations, tag='val')
#         self.train_ds = wdata.DataFlow(self.data_source, augmentations, tag='train')
#
#         self.val_loader = tdata.DataLoader(
#             self.val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
#         )
#
#         self.train_loader = tdata.DataLoader(
#             self.train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
#         )
#
#     def __len__(self):
#         return (1 + self.n_augmentations) * len(self.val_loader)
#
#     def __iter__(self):
#         iterlist = [iter(self.val_loader)]
#
#         for _ in range(self.n_augmentations):
#             iterlist.append(iter(self.train_loader))
#
#         for _ in range(len(self.val_loader)):
#             for iterator in iterlist:
#                 yield next(iterator)
#
#
# class TrainTTAAccumulator:
#     def __init__(self, metric_accumulator, n_augmentations, data_source):
#         self.metric_accumulator = metric_accumulator
#
#         self.source_elements = len(data_source)
#         self.n_augmentations = n_augmentations
#
#         self.data = None
#         self.target = None
#
#         self.accumulated_output = []
#         self.accumulated_context = []
#
#         self.index = 0
#
#     # def calculate(self, data, target, output, context):
#     def calculate(self, data_dict):
#         """ Accumulate results """
#         data = data_dict['data']
#         target = data_dict['target']
#         output = data_dict['output']
#
#         if self.index == 0:
#             self.data = data
#
#         self.target = target
#
#         self.accumulated_output.append(output)
#         self.accumulated_context.append(context)
#
#         self.index += 1
#
#         if self.index == (1 + self.n_augmentations):
#             new_output = torch.mean(torch.stack(self.accumulated_output, dim=-1), dim=-1)
#             new_context = {
#                 k: torch.mean(torch.stack([c[k] for c in self.accumulated_context], dim=-1), dim=-1) for k in context.keys()
#             }
#
#             self.metric_accumulator.calculate(self.data, self.target, new_output, new_context)
#
#             self.index = 0
#             self.data = None
#             self.target = None
#             self.accumulated_output = []
#             self.accumulated_context = []
#
#
# class TrainTTA:
#     """ Test time augmentation that generates additional samples according to the training set augmentations """
#     def __init__(self, n_augmentations):
#         self.n_augmentations = n_augmentations
#
#     def loader(self, data_source, augmentations, batch_size, num_workers):
#         """ Return loader for the test-time-augmentation set """
#         return TrainTTALoader(
#             n_augmentations=self.n_augmentations,
#             batch_size=batch_size,
#             data_source=data_source,
#             augmentations=augmentations,
#             num_workers=num_workers
#         )
#
#     def accumulator(self, metric_accumulator, val_source):
#         """ Reset internal state """
#         return TrainTTAAccumulator(metric_accumulator, self.n_augmentations, val_source)
#
#
# def create(n_augmentations):
#     return TrainTTA(n_augmentations)
