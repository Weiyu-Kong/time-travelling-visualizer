from abc import ABC, abstractmethod

import os
import torch
import json
import transformers

class SummaryWriterAbstractClass(ABC):
    """Writes entries directly to event files in the log_dir to be
    consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """
    def __init__(self, log_dir):
        """Creates a `SummaryWriter` that will write out events and summaries
        to the event file.

        Args:
            log_dir (string): Save directory location.
        """
        log_dir = str(log_dir)
        self.log_dir = log_dir
    
    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self.log_dir

    # TODO
    # @abstractmethod
    # def add_source(self, ):
    #     pass
    # sprite images, text,...


class SummaryWriter(SummaryWriterAbstractClass):

    def __init__(self, log_dir, batch_size=1000, num_worker=2):
        super().__init__(log_dir)
        self.batch_size = batch_size
        self.num_worker = num_worker

    # at start
    def write_meta_data(self, train_dataloader, test_dataloader, index = None):
        # xx_data.pth, xx_label.pth
        self.train_num = self._write_training_data(train_dataloader)
        self.test_num = self._write_testing_data(test_dataloader)
        
        # sprites
        # Note that we should follow the order of dataloader instead of dataset because of shuffling
        self._write_sprites(train_dataloader, test_dataloader)

        # index.json
        if index == None: 
            index = list(range(self.train_num))
        self._write_index_file(index)

    # every epoch
    def write_checkpoint(self, state_dict, epoch, prev_epoch):
        self._write_checkpoint_data(state_dict,epoch)
        self._write_iteration_structure_data(epoch,prev_epoch)


    # ==================================================================
    
    def _write_training_data(self, dataloader):
        trainset_data = None
        trainset_label = None
        for batch in dataloader:
            inputs, targets = batch
            if trainset_data != None:
                trainset_data = torch.cat((trainset_data, inputs), 0)
                trainset_label = torch.cat((trainset_label, targets), 0)
            else:
                trainset_data = inputs
                trainset_label = targets

        training_path = os.path.join(self.log_dir, "Training_data")
        os.makedirs(training_path, exist_ok=True)
        torch.save(trainset_data, os.path.join(training_path, "training_dataset_data.pth"))
        torch.save(trainset_label, os.path.join(training_path, "training_dataset_label.pth"))
        print('Finish writing training data into training dynamics!')
        return len(trainset_data)

    def _write_testing_data(self, dataloader):
        testset_data = None
        testset_label = None
        for batch in dataloader:
            inputs, targets = batch
            if testset_data is not None:
                testset_data = torch.cat((testset_data, inputs), 0)
                testset_label = torch.cat((testset_label, targets), 0)
            else:
                testset_data = inputs
                testset_label = targets
        testing_path = os.path.join(self.log_dir, "Testing_data")
        os.makedirs(testing_path, exist_ok=True)
        torch.save(testset_data, os.path.join(testing_path, "testing_dataset_data.pth"))
        torch.save(testset_label, os.path.join(testing_path, "testing_dataset_label.pth"))
        print('Finish writing testing data into testing dynamics!')
        return len(testset_data)

    
    def _write_checkpoint_data(self, state_dict, epoch):
        checkpoints_path = os.path.join(self.log_dir, "Model")
        checkpoint_path = os.path.join(checkpoints_path, "Epoch_{}".format(epoch))
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(state_dict, os.path.join(checkpoint_path, "subject_model.pth"))

    def _write_iteration_structure_data(self, epoch, prev_epoch):
        iteration_structure_path = os.path.join(self.log_dir, "iteration_structure.json")
        if prev_epoch < 1:
            iter_s = [{"value": epoch, "name": "Epoch", "pid": ""}]
            with open(iteration_structure_path, "w") as f:
                json.dump(iter_s, f)
                f.close()
        else:
            with open(iteration_structure_path,encoding='utf8')as fp:
                json_data = json.load(fp)
                json_data.append({'value': epoch, 'name': 'Epoch', 'pid': "{}".format(prev_epoch)})
                fp.close()
            with open(iteration_structure_path,'w') as f:
                json.dump(json_data, f)
                f.close()

    def _write_index_file(self, index):
        checkpoints_path = os.path.join(self.log_dir, "Model")
        os.makedirs(checkpoints_path, exist_ok=True)
        with open(os.path.join(checkpoints_path, "index.json"), "w") as f:
            json.dump(index, f)
            f.close()

    def _write_sprites(self, train_dataloader, test_dataloader):
        for batch_idx, (images, _) in enumerate(train_dataloader):
            for i in range(images.size(0)):
                img = transformers.to_pil_image(images[i])
                img_path = os.path.join(self.log_dir,'sprites',f'{batch_idx*train_dataloader.batch_size + i}.png')
                img.save(img_path)
        
        for batch_idx, (images, _) in enumerate(test_dataloader):
            for i in range(images.size(0)):
                img = transformers.to_pil_image(images[i])
                img_path = os.path.join(self.log_dir,'sprites',f'{self.train_num + batch_idx*test_dataloader.batch_size + i}.png')
                img.save(img_path)
        print("Finish writing sprites!")
    
    # def _write_sprites_image(self, train_dataset, test_dataset):
    #     sprites_path = os.path.join(self.log_dir,'sprites')
    #     os.makedirs(sprites_path,exist_ok=True)
    #     train_num = len(train_dataset)

    #     def transform_to_tensor(img):
    #         if isinstance(img, torch.Tensor):
    #             return img
    #         elif isinstance(img, Image.Image):
    #             to_tensor = transforms.ToTensor()
    #             img_tensor = to_tensor(img)
    #             return img_tensor
    #         else:
    #             raise TypeError("Not a Tensor or Image in the dataset")

    #     for idx, (img, label) in enumerate(train_dataset):
    #         img = transform_to_tensor(img)
    #         img = img.permute(1, 2, 0)  # 将张量维度从 (C, H, W) 转换为 (H, W, C)
    #         img = Image.fromarray((img.numpy() * 255).astype('uint8'))  # 将值缩放到[0, 255]范围
    #         img.save(os.path.join(self.log_dir,'sprites', f'{idx}.png'))

    #     for idx, (img, label) in enumerate(test_dataset):
    #         img = transform_to_tensor(img)
    #         img = img.permute(1, 2, 0)  # 将张量维度从 (C, H, W) 转换为 (H, W, C)
    #         img = Image.fromarray((img.numpy() * 255).astype('uint8'))  # 将值缩放到[0, 255]范围
    #         img.save(os.path.join(self.log_dir,'sprites', f'{idx+train_num}.png'))

                

            



