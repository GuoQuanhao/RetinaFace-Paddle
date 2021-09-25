from paddle.io import Dataset
import cv2
import numpy as np
from paddle.io import BatchSampler, DistributedBatchSampler, RandomSampler, SequenceSampler, DataLoader

class WiderFaceDetection(Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return img, target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0].astype('float32'))
        targets.append(sample[1].astype('float32'))
    return (np.stack(imgs, 0), targets)
    '''
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if len(tup.shape) == 3:
                imgs.append(tup.astype('float32'))
            elif len(tup.shape) == 2:
                annos = tup.astype('float32')
                targets.append(annos)
    '''
    return (np.stack(imgs, 0), targets)


def make_dataloader(dataset, shuffle=True, batchsize=12, distributed=False, num_workers=0, num_iters=None, start_iter=0, collate_fn=None):
    if distributed:
        data_sampler=DistributedBatchSampler(dataset, batch_size=batchsize, shuffle=True, drop_last=True)
        dataloader = DataLoader(dataset, batch_sampler=data_sampler, num_workers=num_workers, collate_fn=collate_fn)

    if not distributed and shuffle:
        sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler=sampler, batch_size=batchsize, drop_last=True)
        if num_iters is not None:
            batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
        dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn)
    else:
        sampler = SequenceSampler(dataset)
        batch_sampler = BatchSampler(sampler=sampler, batch_size=batchsize, drop_last=True)
        if num_iters is not None:
            batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
        dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn)

    return dataloader


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations
