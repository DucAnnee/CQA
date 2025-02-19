import torch
import os
import gc
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from PIL import Image

root_dir = '/kaggle/working/'
images_dir = '/kaggle/input/docbank-50k-split/splitted_images/splitted_images'
labels_dir = '/kaggle/input/docbank-50k-split/splitted_labels/splitted_labels'
train_idx_file_path = '/kaggle/input/docbank-50k-split/40k_train.txt'
val_idx_file_path = '/kaggle/input/docbank-50k-split/5k_val.txt'
test_idx_file_path = '/kaggle/input/docbank-50k-split/5k_test.txt'

TRAIN_LIMIT = 2000
VAL_LIMIT = 200
TEST_LIMIT = 200
LR = 0.001 
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 10
BATCH_SIZE = 4

class DocBankDataset(Dataset):
    def __init__(
        self,
        idx_file_path,
        limit,
        images_dir,
        labels_dir,
        transform=None,
    ):
        self.transform = transform
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.limit = limit

        with open(idx_file_path, "r") as f:
            self.file_list = [line.strip() for line in f.readlines()[:limit]]

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img_path = os.path.join(self.images_dir, file_name[:-4] + "_ori.jpg")
        ann_path = os.path.join(self.labels_dir, file_name[:-4] + ".txt")

        # Load image
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Load annotation
        target = self.process_ann_path(ann_path)

        return img, target

    def __len__(self):
        return self.limit 

    def process_ann_path(self, ann_path):
        target = {}
        boxes = []
        labels = []
        with open(ann_path, "r") as f:
            for line in f:
                content = line.strip().split("\t")
                token, x0, y0, x1, y1, R, G, B, font, label = content

                if (
                    (int(x0) < 0)
                    or (int(y0) < 0)
                    or (int(x1) < 0)
                    or (int(y1) < 0)
                    or (x1 <= x0 or y1 <= y0)
                ):
                    continue

                boxes.append([float(x0), float(y0), float(x1), float(y1)])
                labels.append(1 if label == "figure" else 0)

                del token, x0, y0, x1, y1, R, G, B, font, label
                gc.collect()

        target["boxes"] = torch.FloatTensor(boxes)
        target["labels"] = torch.tensor(labels)
        return target
    
class RetinaTrainer:
    def __init__(self, model, optimizer, train_dataset, val_dataset, device):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device

    def _run_batch(self, batch):
        images, targets = batch
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        return losses.item()

    def _run_epoch(self, dataloader, is_training=True):
        epoch_loss = 0
        self.model.train(is_training)

        for batch in dataloader:
            batch_loss = self._run_batch(batch)
            epoch_loss += batch_loss

        return epoch_loss / len(dataloader)

    def save_checkpoint(self, epoch, train_loss, val_loss):
        if dist.get_rank() == 0:  
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, f'{root_dir}checkpoint_epoch_{epoch}.pth')
            print(f"Checkpoint saved for epoch {epoch}")

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file {checkpoint_path} does not exist. Starting from scratch.")
            return 0  # Start from epoch 0

        map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'] + 1  # Start from the next epoch

    def train(self, num_epochs, checkpoint_interval=5, resume_from=None):
        train_sampler = DistributedSampler(self.train_dataset)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=4,
            collate_fn=lambda x: tuple(zip(*x))
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            collate_fn=lambda x: tuple(zip(*x))
        )

        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)

        for epoch in range(start_epoch, num_epochs):
            train_sampler.set_epoch(epoch)
            train_loss = self._run_epoch(train_loader, is_training=True)
            val_loss = self._run_epoch(val_loader, is_training=False)

            if dist.get_rank() == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1, train_loss, val_loss)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}') 

    model = retinanet_resnet50_fpn_v2(num_classes=2)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    train_dataset = DocBankDataset(train_idx_file_path, TRAIN_LIMIT, images_dir, labels_dir)
    val_dataset = DocBankDataset(val_idx_file_path, VAL_LIMIT, images_dir, labels_dir)
    
    trainer = RetinaTrainer(model, optimizer, train_dataset, val_dataset, device)

    try:
        trainer.train(num_epochs=NUM_EPOCHS, checkpoint_interval=2)
    finally:
        cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()