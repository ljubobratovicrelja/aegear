# %%
import os
import random
import time

import zipfile
from google.cloud import storage

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from tqdm import tqdm

from aegear.nn.model import EfficientUNet, SiameseTracker
from aegear.nn.datasets import CachedTrackingDataset
from aegear.utils import get_latest_model_path
from aegear.nn.training import denormalize, SiameseTrackingVisualizer, SiameseLoss, get_centroids_per_sample

# %% [markdown]
# # Download the training data
# 
# We download the training data cached from tracking data (ROIs cropped and saved as images for faster loading).

# %%

dataset_dir = "../data/training/cache"
bucket_name = "aegear-training-data"
blob_path = "cache/tracking.zip"

if os.path.exists(os.path.join(dataset_dir, "tracking")):
    print("Dataset already exists. Skipping download.")
else:
    print("Dataset not found. Downloading...")
    zip_path = os.path.join(dataset_dir, "tracking.zip")
    os.makedirs(dataset_dir, exist_ok=True)

    # Download the zip file if it doesn't exist
    if not os.path.exists(zip_path):
        print(f"Downloading gs://{bucket_name}/{blob_path} to {zip_path}...")
        
        # Initialize anonymous GCS client for public data
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(zip_path)
        
        print("Download complete.")

    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
        print(f"Extracted to {dataset_dir}")

# %% [markdown]
# # Dataset construction
# 
# Next we construct the dataset structure and the batch loader for training.

# %%
cache_dir = "../data/training/cache/tracking"

# List the cache directory to find available datasets
datasets = os.listdir(cache_dir)

gaussian_sigma = 12.0
batch_size = 128

train_dataset = ConcatDataset([CachedTrackingDataset(os.path.join(cache_dir, name, "train"), gaussian_sigma=gaussian_sigma) for name in datasets])
val_dataset = ConcatDataset([CachedTrackingDataset(os.path.join(cache_dir, name, "val"), gaussian_sigma=gaussian_sigma) for name in datasets])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# %%
N = 10  # Number of samples to visualize

fig, axes = plt.subplots(N, 3, figsize=(8, 2 * N))

for i in range(N):
    random.seed(i)
    idx = random.randint(0, len(train_dataset) - 1)
    template, search, heatmap = train_dataset[idx]

    template_img = TF.to_pil_image(denormalize(template))
    search_img = TF.to_pil_image(denormalize(search))
    heatmap_np = heatmap.squeeze().numpy()

    # === Resize heatmap
    search_w, search_h = search_img.size  # PIL: size = (W, H)
    heatmap_np = cv2.resize(heatmap_np, (search_w, search_h), interpolation=cv2.INTER_LINEAR)

    # === Normalize heatmap for display
    heatmap_norm = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)

    # === Blend
    search_np = TF.to_tensor(search_img).permute(1, 2, 0).numpy()
    heatmap_rgb = plt.cm.jet(heatmap_norm)[..., :3]
    overlay = 0.6 * search_np + 0.4 * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)

    # === Plot
    axes[i, 0].imshow(template_img)
    axes[i, 0].set_title("Template")

    axes[i, 1].imshow(search_img)
    axes[i, 1].set_title("Search")

    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title("Overlay")

    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# # Prepare the model and folder structure
# 
# We designate the directories we use to store the training telemetry and model checkpoints.

# %%
# Root model path
pretrained_model_dir = '../models/'

assert os.path.exists(pretrained_model_dir), f"Pretrained model directory {pretrained_model_dir} does not exist."

model_dir = '../data/training/models/siamese'
log_dir = f'{model_dir}/runs'
checkpoint_dir = f'{model_dir}/checkpoints'
epoch_vis = f'{model_dir}/epoch_vis'

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(epoch_vis, exist_ok=True)

# %%
# Device definition.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Model loaded and moved to device:", device)

def str2bool(val):
    return str(val).lower() in ("1", "true", "yes", "on")

# Model settings for training
continue_training = str2bool(os.environ.get("CONTINUE_TRAINING", "false"))  # Continue training from the latest production model
use_best_model = str2bool(os.environ.get("USE_BEST_MODEL", "false"))  # If contnue_training is True, uses the cached best_model.pth, else uses the latest production model.

if continue_training:
   if use_best_model:
      # Load the best model
      best_model_path = os.path.join(model_dir, "best_model.pth")
      assert os.path.exists(best_model_path)

   else:     
      siamese_model_filename = "model_siamese"
      best_model_path = get_latest_model_path(pretrained_model_dir, siamese_model_filename)

   print("Continuing training of the Siamese model from:", best_model_path)

   model = SiameseTracker()
   model.load_state_dict(torch.load(best_model_path, map_location=device))

else:
   efficient_unet_filename = "model_efficient_unet"
   latest_model_stats = get_latest_model_path(pretrained_model_dir, efficient_unet_filename)

   unet = EfficientUNet()
   unet.load_state_dict(torch.load(latest_model_stats, map_location=device))
   unet.eval()

   model = SiameseTracker(unet)

   print("Training a new Siamese model with EfficientNet encoder from:", latest_model_stats)

model.to(device);

# Prepare the loss function.
siamese_loss = SiameseLoss(
   limit=0.5,
   pos_weight=10.0,
   centroid_weight=2.5e-3,  # weight for centroid distance loss
   rgb_weight=5e-3,  # weight for color consistency loss
   sparsity_weight=1e-3,  # weight for sparsity loss
)

# %%
training_stages = [
   {
       "freeze_layers": [
            model.enc1,
            model.enc2,
            model.enc3,
            model.enc4,
            model.enc5,
       ],
       "epochs": 10,
       "lr": 5e-4,
   },
   {
       "freeze_layers": [
            model.enc1,
            model.enc2,
            model.enc3,
            model.enc4,
       ],
       "epochs": 5,
       "lr": 1e-4,
   },
]

# %% [markdown]
# # Training Loop
# 
# We run the training on the configured stages.

# %%
# Training loop
best_val_loss = float('inf')
losses = []

epoch_save_interval = 1

for stage, training_stage in enumerate(training_stages):

    freeze_layers = training_stage["freeze_layers"]
    epochs = training_stage["epochs"]
    lr = training_stage["lr"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=3,
    )

    for param in model.parameters():
        param.requires_grad = False

    for layer in model.children():
        if layer not in freeze_layers:
            for param in layer.parameters():
                param.requires_grad = True

    for epoch in range(epochs):
        model.train()

        for layer in freeze_layers:
            layer.eval()

        train_loss = 0.0

        random.seed(time.time())
        train_bar = tqdm(train_loader, desc=f"Stage {stage + 1}, Training {epoch + 1}", leave=False)
        for template, search, heatmap in train_bar:
            template = template.to(device)
            search = search.to(device)
            target = heatmap.to(device)

            output = model(template, search)

            loss = siamese_loss(output, target, template, search)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        val_results = []  # NEW

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Stage {stage + 1}, Validation {epoch + 1}", leave=False)

            for template, search, heatmap in val_bar:
                template = template.to(device)
                search = search.to(device)
                target = heatmap.to(device)

                output = model(template, search)

                loss = siamese_loss(output, target, template, search)

                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

                # --------- COLLECT SAMPLE DATA ---------
                pred = torch.sigmoid(output)

                # Get centroids and confidence per sample
                centroids_pred = get_centroids_per_sample(pred)
                centroids_gt = get_centroids_per_sample(target)

                for i in range(template.size(0)):
                    p = centroids_pred[i]
                    t = centroids_gt[i]

                    if p is None or t is None:
                        continue  # skip bad samples

                    x_pred, y_pred, confidence = p
                    x_gt, y_gt, _ = t

                    xp, yp = x_pred.item(), y_pred.item()
                    xg, yg = x_gt.item(), y_gt.item()

                    dist = np.sqrt((xp - xg) ** 2 + (yp - yg) ** 2)

                    val_results.append({
                        'template': template[i].cpu(),
                        'search': search[i].cpu(),
                        'gt_heatmap': target[i, 0].cpu(),
                        'pred_heatmap': pred[i, 0].cpu(),
                        'gt_centroid': (xg, yg),
                        'pred_centroid': (xp, yp),
                        'confidence': confidence.item(),
                        'distance': dist,
                    })

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        losses.append((train_loss, val_loss))

        scheduler.step(val_loss)

        # Visualize results
        visualizer = SiameseTrackingVisualizer(model, device, val_results, stage, epoch, output_dir=epoch_vis)

        visualizer.performance(num_samples=10)
        visualizer.activations(num_samples=3)

        print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("New best model, saving.")
            torch.save(model.state_dict(), f'{model_dir}/best_model.pth')

        if (epoch + 1) % epoch_save_interval == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth'))


# %%
# Plotting the losses
train_losses, val_losses = zip(*losses)

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Load the best model for final evaluation.
model.load_state_dict(torch.load(f'{model_dir}/best_model.pth'))
model.eval()
model.to(device);

# %%
results = []
total_distances = []
total_confidences = []
within_radius = {r: 0 for r in [3, 5, 10]}
n_samples = 0

model.eval()

with torch.no_grad():
    for templates, searches, heatmaps in tqdm(val_loader, desc="Evaluating"):
        templates = templates.to(device)
        searches = searches.to(device)
        heatmaps = heatmaps.to(device)

        preds = torch.sigmoid(model(templates, searches))

        heatmaps = F.interpolate(heatmaps, size=templates.shape[2:], mode='bilinear', align_corners=False)
        preds = F.interpolate(preds, size=templates.shape[2:], mode='bilinear', align_corners=False)

        centroids_pred = get_centroids_per_sample(preds)
        centroids_gt = get_centroids_per_sample(heatmaps)

        for i in range(len(templates)):
            p = centroids_pred[i]
            t = centroids_gt[i]

            if p is None or t is None:
                continue

            x_pred, y_pred, confidence = p
            x_gt, y_gt, _ = t

            xp, yp = x_pred.item(), y_pred.item()
            xg, yg = x_gt.item(), y_gt.item()

            confidence = confidence.item()

            dist = np.sqrt((xp - xg) ** 2 + (yp - yg) ** 2)
            total_distances.append(dist)
            total_confidences.append(confidence)

            for r in within_radius:
                if dist <= r:
                    within_radius[r] += 1
            n_samples += 1

            img = searches[i].cpu()
            pred_hm = preds[i, 0].cpu()
            gt_hm = heatmaps[i, 0].cpu()

            results.append({
                'image': img,
                'gt_heatmap': gt_hm,
                'pred_heatmap': pred_hm,
                'gt_centroid': (xg, yg),
                'pred_centroid': (xp, yp),
                'confidence': confidence,
                'distance': dist,
            })

avg_dist = np.mean(total_distances)
avg_conf = np.mean(total_confidences)
print(f"\nAverage centroid distance: {avg_dist:.2f} px, Average confidence: {avg_conf:.2f}")

for r in sorted(within_radius):
    print(f"Within {r}px: {within_radius[r] / n_samples:.2%}")


# %%
results.sort(key=lambda x: -x['distance'])  # descending


def show_sample(result, index=None):
    img = result['image']
    gt = result['gt_heatmap']
    pred = result['pred_heatmap']
    xg, yg = result['gt_centroid']
    xp, yp = result['pred_centroid']
    dist = result['distance']
    conf = result['confidence']

    # Denormalize image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_disp = img * std + mean
    img_disp = img_disp.clamp(0, 1).permute(1, 2, 0).numpy()

    xc, yc = img_disp.shape[1] // 2, img_disp.shape[0] // 2

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(img_disp)
    axs[0].scatter([xg], [yg], c='green', label='GT', marker='o')
    axs[0].scatter([xp], [yp], c='red', label='Pred', marker='o')
    axs[0].scatter([xc], [yc], c='green', label='Template', marker='x')
    
    axs[0].set_title(f'Image (Err: {dist:.1f}px, Conf: {conf:.2f})')
    axs[0].legend()

    axs[1].imshow(gt.numpy(), cmap='hot')
    axs[1].set_title('GT Heatmap')

    axs[2].imshow(pred.numpy(), cmap='hot')
    axs[2].set_title('Predicted Heatmap')

    if index is not None:
        fig.suptitle(f"Sample #{index}", fontsize=16)

    plt.tight_layout()
    plt.show()

# %%
# Show some of the worst predictions
#
# The worst ones are expected to be the outliers in the
# dataset, as we know we have some bad labeling.
for i in range(10):
    show_sample(results[i], index=i+1)

# %%
result_size = len(results)
# Show the middle ones
for i in range(result_size//2-5, result_size//2+5):
    show_sample(results[i], index=i+1)

# %%
# Show some of the best predictions
for i in range(result_size-10, result_size):
    show_sample(results[i], index=i+1)


