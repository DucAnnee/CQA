import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
from torchvision.models.detection import (
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights,
)
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torchvision.ops import box_iou
from torch.optim.lr_scheduler import OneCycleLR
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

#######################################################################################################################

images_dir = '/kaggle/input/docbank-50k-split/splitted_images/splitted_images'
    labels_dir = '/kaggle/input/docbank-50k-split/splitted_labels/splitted_labels'
train_idx_file_path = '/kaggle/input/docbank-50k-split/40k_train.txt'
val_idx_file_path = '/kaggle/input/docbank-50k-split/5k_val.txt'
test_idx_file_path = '/kaggle/input/docbank-50k-split/5k_test.txt'

TRAIN_LIMIT = 2000
VAL_LIMIT = 200
TEST_LIMIT = 200

num_gpus = torch.cuda.device_count()

LR = 0.001 
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

BATCH_SIZE = 4 * num_gpus
NUM_EPOCHS = 10
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = retinanet_resnet50_fpn(num_classes=2)

#######################################################################################################################

class DocBankDataset(torch.utils.data.Dataset):
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
        return len(self.file_list)

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

        target["boxes"] = torch.FloatTensor(boxes)
        target["labels"] = torch.tensor(labels)
        return target

#######################################################################################################################

def maintain_aspect_ratio_resize(image, target_size):
    w, h = image.size
    aspect_ratio = w / h
    if w > h:
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    else:
        new_h = target_size
        new_w = int(target_size * aspect_ratio)
    return image.resize((new_w, new_h), Image.LANCZOS)

target_size = 500

transform = transforms.Compose(
    [
#         transforms.Lambda(lambda img: maintain_aspect_ratio_resize(img, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ]
)

train_dataset = DocBankDataset(
    train_idx_file_path, TRAIN_LIMIT, images_dir, labels_dir, transform
)
val_dataset = DocBankDataset(
    val_idx_file_path, VAL_LIMIT, images_dir, labels_dir, transform
)
test_dataset = DocBankDataset(
    test_idx_file_path, TEST_LIMIT, images_dir, labels_dir, transform
)

def collate_fn(batch):
    images, targets = zip(*batch)
    return images, targets


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
)

#######################################################################################################################

optimizer = torch.optim.SGD(
    model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)

# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer, milestones=[4, 8], gamma=0.1
# )

lr_scheduler = OneCycleLR(
    optimizer, max_lr=0.01, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader)
)

#######################################################################################################################

def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class DDPTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        optimizer,
        lr_scheduler,
        device,
        num_epochs,
        batch_size,
        accumulation_steps=4,
        patience=5,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.patience = patience
        
        self.best_f1_score = -float("inf")
        self.patience_counter = 0
        self.scaler = GradScaler()

    def setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        
        self.model = DDP(self.model.to(rank), device_ids=[rank])
        
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        return images, targets

    def _run_batch(self, images, targets):
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        if any(len(t["boxes"]) == 0 for t in targets):
            print("Skipping batch with empty target boxes.")
            return 0

        with autocast(device_type="cuda"):
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss = losses / self.accumulation_steps

        self.scaler.scale(loss).backward()
        
        return losses.item()

    def _run_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        
        for i, (images, targets) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)):
            loss = self._run_batch(images, targets)
            train_loss += loss

            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            del images, targets
            gc.collect()
            torch.cuda.empty_cache()

        self.lr_scheduler.step()
        return train_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        TP, FP, FN = 0, 0, 0
        iou_threshold = 0.5

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating", leave=False):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                if any(len(t["boxes"]) == 0 for t in targets):
                    print("Skipping validation batch with empty target boxes.")
                    continue

                predictions = self.model(images)

                for target, prediction in zip(targets, predictions):
                    target_boxes = target["boxes"].cpu()
                    predicted_boxes = prediction["boxes"].cpu()

                    if target_boxes.ndim == 1:
                        target_boxes = target_boxes.unsqueeze(0)
                    if predicted_boxes.ndim == 1 and predicted_boxes.numel() > 0:
                        predicted_boxes = predicted_boxes.unsqueeze(0)
                    elif predicted_boxes.numel() == 0:
                        continue

                    iou = box_iou(predicted_boxes, target_boxes)
                    for i in range(predicted_boxes.shape[0]):
                        if (iou[i] > iou_threshold).any():
                            TP += 1
                        else:
                            FP += 1
                    FN += target_boxes.shape[0] - sum((iou > iou_threshold).any(dim=0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1_score

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self._run_epoch(epoch)
            precision, recall, f1_score = self._validate()

            if self.device.index == 0:  # Only print on main process
                print(f"Epoch {epoch+1}/{self.num_epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            if f1_score > self.best_f1_score:
                self.best_f1_score = f1_score
                if self.device.index == 0:  # Save only on main process
                    torch.save(self.model.module.state_dict(), "best_retina.pth")
                    print("Saved new best model")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        print("Training completed.")

def setup_logging():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='ddp_training.log',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging.getLogger(__name__)

logger = setup_logging()
        

def ddp_main(rank, world_size, model, train_dataset, val_dataset, optimizer_params, lr_scheduler_params, num_epochs, batch_size):
    try:
        logger.info(f"Initializing process {rank}")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        
        logger.info(f"Process {rank}: Moving model to device")
        model = model.to(device)
        model = DDP(model, device_ids=[rank])
        
        logger.info(f"Process {rank}: Setting up optimizer and scheduler")
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
        lr_scheduler = OneCycleLR(optimizer, **lr_scheduler_params)
        
        logger.info(f"Process {rank}: Initializing DDPTrainer")
        trainer = DDPTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        
        logger.info(f"Process {rank}: Setting up trainer")
        trainer.setup(rank, world_size)
        
        logger.info(f"Process {rank}: Starting training")
        trainer.train()
        
        logger.info(f"Process {rank}: Training completed, cleaning up")
        destroy_process_group()
    except Exception as e:
        logger.error(f"Error in process {rank}: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

if __name__ == "__main__":
    try:
        logger.info("Starting main process")
        
        # Define your model, datasets, and other parameters here
        logger.info("Initializing model and datasets")
        model = retinanet_resnet50_fpn(num_classes=2)
        train_dataset = DocBankDataset(train_idx_file_path, TRAIN_LIMIT, images_dir, labels_dir, transform)
        val_dataset = DocBankDataset(val_idx_file_path, VAL_LIMIT, images_dir, labels_dir, transform)
        
        optimizer_params = {
            "lr": LR,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY
        }
        
        lr_scheduler_params = {
            "max_lr": 0.01,
            "epochs": NUM_EPOCHS,
            "steps_per_epoch": len(train_dataset) // BATCH_SIZE
        }
        
        world_size = torch.cuda.device_count()
        logger.info(f"Detected {world_size} GPUs")
        
        logger.info("Spawning processes")
        mp.spawn(
            ddp_main,
            args=(world_size, model, train_dataset, val_dataset, optimizer_params, lr_scheduler_params, NUM_EPOCHS, BATCH_SIZE),
            nprocs=world_size
        )
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        logger.error(traceback.format_exc())
        raise e