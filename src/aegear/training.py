"""
Module containing various training-related utilities and functions.
"""

import os

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF
import torch.nn.functional as F


def get_confidence(heatmap):
    """
    Get confidence score from a heatmap by finding the maximum value.
    """
    b, _, _, w = heatmap.shape
    flat_idx = torch.argmax(heatmap.view(b, -1), dim=1)
    y = flat_idx // w
    x = flat_idx % w
    return heatmap[0, 0, y, x].item()


def overlay_heatmap_on_rgb(rgb_tensor, heatmap, alpha=0.5, centroid_color=(0, 1, 0)):
    """
    Overlay heatmap onto RGB image and draw a circle at the predicted centroid.

    Args:
        rgb_tensor: [3, H, W] tensor
        heatmap: [H, W] numpy array
        alpha: blending weight
        centroid_color: (R, G, B) tuple in range 0–1
    Returns:
        overlay: [H, W, 3] numpy image
    """
    rgb = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    rgb = rgb * 0.229 + 0.485
    rgb = rgb.clip(0, 1)

    heatmap_color = plt.cm.hot(heatmap)[..., :3]
    overlay = (1 - alpha) * rgb + alpha * heatmap_color

    # Find centroid
    flat_idx = heatmap.reshape(-1).argmax()
    h, w = heatmap.shape
    cy = flat_idx // w
    cx = flat_idx % w

    # Draw circle
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    cx_int, cy_int = int(cx), int(cy)
    color_bgr = tuple(int(c * 255) for c in reversed(centroid_color))
    cv2.circle(overlay_uint8, (cx_int, cy_int), 4, color_bgr, thickness=1)

    return overlay_uint8 / 255.0


def denormalize(img_tensor, clamp=True):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                       device=img_tensor.device).view(3, 1, 1)
    out = img_tensor * std + mean
    return out.clamp(0, 1) if clamp else out


def get_centroids_per_sample(heatmap):
    """
    Get centroids from a batch of heatmaps."""
    b, _, _, w = heatmap.shape
    heatmaps = heatmap.squeeze(1)
    centroids = []

    for i in range(b):
        hm = heatmaps[i]
        hm_sum = hm.mean().item()

        if hm_sum < 1e-8:
            centroids.append(None)
        else:
            flat_idx = torch.argmax(hm)
            y = flat_idx // w
            x = flat_idx % w
            conf = hm[y, x]
            centroids.append((x.float(), y.float(), conf.float()))

    return centroids


class WeightedBCEWithLogitsLoss:
    """
    Custom loss function that applies weighted binary cross-entropy with logits.
    It emphasizes the center of the Gaussian heatmap.
    """

    def __init__(self, limit=0.5, pos_weight=10.0):
        self.limit = limit
        self.pos_weight = pos_weight

    def __call__(self, pred, target):
        weights = torch.ones_like(target)
        # emphasize center of Gaussian
        weights[target > self.limit] = self.pos_weight

        bce = F.binary_cross_entropy_with_logits(
            pred, target, weight=weights, reduction='mean')
        return bce


class EfficientUNetLoss(WeightedBCEWithLogitsLoss):
    def __init__(self, limit=0.5, pos_weight=10.0, centroid_weight=2.5e-3, sparsity_weight=1e-3):
        """
        Initialize the loss with weights for BCE and centroid distance.
        """
        super().__init__(limit, pos_weight)
        self.centroid_weight = centroid_weight
        self.sparsity_weight = sparsity_weight

    def __call__(self, pred, target):
        bce_loss = super().__call__(pred, target)
        cdist_loss = self.centroid_distance_loss(pred, target)
        sparsity_loss = self.sparsity_weight * pred.pow(2).mean()
        return bce_loss + self.centroid_weight * cdist_loss + sparsity_loss

    @staticmethod
    def centroid_distance_loss(pred, target):
        preds = get_centroids_per_sample(torch.sigmoid(pred))
        targets = get_centroids_per_sample(target)

        distances = []

        for p, t in zip(preds, targets):
            if p is not None and t is not None:
                x_p, y_p, _ = p
                x_t, y_t, _ = t
                dist = torch.sqrt((x_p - x_t) ** 2 + (y_p - y_t) ** 2 + 1e-8)
                distances.append(dist)

        if not distances:
            return torch.tensor(0.0).to(pred.device)

        return torch.stack(distances).mean()


class SiameseLoss(EfficientUNetLoss):
    """
    Siamese loss function that combines the EfficientUNetLoss with an RGB consistency loss.
    """

    def __init__(
        self,
        limit=0.5,
        pos_weight=10.0,
        centroid_weight=2.5e-3,
        sparsity_weight=1e-3,
        rgb_weight=5e-3,
        rgb_sigma=2.0,
        rgb_threshold=0.5
    ):
        """
        Initialize the SiameseLoss with weights for different components.
        """
        super().__init__(limit, pos_weight, centroid_weight, sparsity_weight)

        self.rgb_weight = rgb_weight
        self.rgb_sigma = rgb_sigma
        self.rgb_threshold = rgb_threshold

    def __call__(self, output, target, template, search):
        """
        Compute the total loss given predictions and targets.
        """
        main_loss = super().__call__(output, target)
        rgb_loss = self.rgb_consistency_loss(template, search, output)

        return main_loss + self.rgb_weight * rgb_loss

    def rgb_consistency_loss(self, template_img, search_img, pred_heatmap):
        """
        Compute the RGB consistency loss between template and search images
        based on the predicted heatmap.
        """
        B, _, H, W = template_img.shape
        device = template_img.device

        # === Create fixed centered Gaussian for all batch
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H - 1, H, device=device),
            torch.linspace(0, W - 1, W, device=device),
            indexing='ij'
        )
        center_y = (H - 1) / 2
        center_x = (W - 1) / 2
        gaussian = torch.exp(-((grid_x - center_x)**2 +
                             (grid_y - center_y)**2) / (2 * self.rgb_sigma**2))
        gaussian /= gaussian.sum() + 1e-8
        gaussian = gaussian[None, None, :, :]  # shape (1, 1, H, W)

        loss = 0.0
        for i in range(B):
            # === Mask and normalize predicted heatmap
            mask = (pred_heatmap[i] > self.rgb_threshold).float()
            weighted_mask = pred_heatmap[i] * mask
            weighted_mask /= weighted_mask.sum() + 1e-8  # (1, H, W)

            # === Compute mean RGB in search
            rgb_search = (search_img[i] * weighted_mask).view(3, -1).sum(dim=1)

            # === Compute mean RGB in template using Gaussian
            rgb_template = (template_img[i] *
                            gaussian[0]).view(3, -1).sum(dim=1)

            loss += F.mse_loss(rgb_search, rgb_template)

        return loss / B


def _sort_samples(val_results, num_samples):
    sorted_results = sorted(
        val_results, key=lambda r: r['confidence'], reverse=True)
    worst = sorted_results[-num_samples:][::-1]
    best = sorted_results[:num_samples]
    mid_start = len(sorted_results) // 2 - num_samples // 2
    mid_end = mid_start + num_samples
    middle = sorted_results[mid_start:mid_end]
    return worst + middle + best


class BaseVisualizer:
    def __init__(self, model, device, val_results, stage, epoch, output_dir="vis_epochs"):
        self.model = model
        self.device = device
        self.val_results = val_results
        self.stage = stage
        self.epoch = epoch
        self.output_dir = output_dir

    def _save_fig(self, fig, subdir, prefix):
        os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        path = os.path.join(
            self.output_dir,
            subdir,
            f"{prefix}_stage_{self.stage:03d}_epoch_{self.epoch:03d}.png"
        )
        fig.savefig(path, dpi=200)
        plt.close(fig)


class SiameseTrackingVisualizer(BaseVisualizer):
    def performance(self, num_samples=5):
        samples = _sort_samples(self.val_results, num_samples)
        fig, axes = plt.subplots(
            len(samples), 3, figsize=(9, 3 * len(samples)))

        for i, result in enumerate(samples):
            template_img = TF.to_pil_image(denormalize(result['template']))
            search_img = TF.to_pil_image(denormalize(result['search']))
            search_np = TF.to_tensor(search_img).permute(1, 2, 0).numpy()

            pred, gt = result['pred_heatmap'], result['gt_heatmap']
            pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
            diff_norm = np.abs(pred_norm - gt_norm)

            overlay = np.clip(0.6 * search_np + 0.4 *
                              plt.cm.jet(pred_norm)[..., :3], 0, 1)
            diff_rgb = plt.cm.magma(diff_norm)[..., :3]

            xg, yg = result['gt_centroid']
            xp, yp = result['pred_centroid']
            confidence = result['confidence']

            axes[i, 0].imshow(template_img)
            axes[i, 0].set_title(f"Template idx {i}")

            axes[i, 1].imshow(overlay)
            axes[i, 1].scatter([xp], [yp], c='red', marker='x', label='Pred')
            axes[i, 1].scatter([xg], [yg], c='green', marker='o', label='GT')
            axes[i, 1].set_title(f"Search | Conf: {confidence:.2f}")
            axes[i, 1].legend()

            axes[i, 2].imshow(diff_rgb)
            axes[i, 2].set_title("Abs Diff")

            for ax in axes[i]:
                ax.axis("off")

        plt.tight_layout()
        self._save_fig(plt.gcf(), "performance", "epoch")

    def activations(self, num_samples=3):
        output_dir = os.path.join(self.output_dir, "activations")
        os.makedirs(output_dir, exist_ok=True)

        stages = ['enc3', 'enc4', 'enc5', 'up4',
                  'up3', 'up2', 'up1', 'up0', 'out']
        channels_per_stage = 3

        activations = {}
        for name in stages:
            layer = getattr(self.model, name)
            layer.register_forward_hook(
                lambda m, i, o, n=name: activations.update({n: o.detach().cpu()}))

        samples = _sort_samples(self.val_results, num_samples)
        n_cols = 1 + channels_per_stage * len(stages)
        fig, axs = plt.subplots(len(samples), n_cols, figsize=(
            n_cols * 2.5, len(samples) * 3))
        axs = axs if len(samples) > 1 else axs[None, :]

        self.model.eval()
        for row, sample in enumerate(samples):
            template = sample['template'].unsqueeze(0).to(self.device)
            search = sample['search'].unsqueeze(0).to(self.device)
            heatmap = sample['pred_heatmap'].numpy()

            with torch.no_grad():
                _ = self.model(template, search)

            overlay = denormalize(search[0]).permute(1, 2, 0).cpu().numpy()
            overlay[..., 0] = np.clip(overlay[..., 0] + 0.5 * heatmap, 0, 1)

            axs[row, 0].imshow(overlay)
            axs[row, 0].scatter([sample['gt_centroid'][0]], [
                                sample['gt_centroid'][1]], c='green', marker='o')
            axs[row, 0].scatter([sample['pred_centroid'][0]], [
                                sample['pred_centroid'][1]], c='red', marker='x')
            axs[row, 0].set_title(f"Conf: {sample['confidence']:.2f}")
            axs[row, 0].axis('off')

            col = 1
            for stage in stages:
                act = activations[stage][0]
                for ch in range(channels_per_stage):
                    if ch < act.shape[0]:
                        axs[row, col].imshow(act[ch], cmap='viridis')
                        axs[row, col].set_title(f'{stage} | Ch {ch}')
                    axs[row, col].axis('off')
                    col += 1

        plt.tight_layout()
        self._save_fig(plt.gcf(), "activations", "activation")


class EfficientUNetVisualizer(BaseVisualizer):
    def performance(self, num_samples=5):
        samples = _sort_samples(self.val_results, num_samples)
        fig, axes = plt.subplots(
            len(samples), 3, figsize=(9, 3 * len(samples)))

        for i, result in enumerate(samples):
            search_img = TF.to_pil_image(denormalize(result['search']))
            search_np = TF.to_tensor(search_img).permute(1, 2, 0).numpy()

            pred, gt = result['pred_heatmap'], result['gt_heatmap']
            pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
            diff_norm = np.abs(pred_norm - gt_norm)

            overlay = np.clip(0.6 * search_np + 0.4 *
                              plt.cm.jet(pred_norm)[..., :3], 0, 1)
            diff_rgb = plt.cm.magma(diff_norm)[..., :3]

            xg, yg = result['gt_centroid']
            xp, yp = result['pred_centroid']
            confidence = result['confidence']

            axes[i, 0].imshow(overlay)
            axes[i, 0].scatter([xp], [yp], c='red', marker='x', label='Pred')
            axes[i, 0].scatter([xg], [yg], c='green', marker='o', label='GT')
            axes[i, 0].set_title(f"Search | Conf: {confidence:.2f}")
            axes[i, 0].legend()

            axes[i, 1].imshow(diff_rgb)
            axes[i, 1].set_title("Abs Diff")

            for ax in axes[i]:
                ax.axis("off")

        plt.tight_layout()
        self._save_fig(plt.gcf(), "performance", "stage")

    def activations(self, num_samples=3):
        output_dir = os.path.join(self.output_dir, "activations")
        os.makedirs(output_dir, exist_ok=True)

        stages = ['enc1', 'enc2', 'enc3', 'enc4', 'enc5',
                  'up4', 'up3', 'up2', 'up1', 'up0', 'out']
        channels_per_stage = 3

        activations = {}
        for name in stages:
            layer = getattr(self.model, name)
            layer.register_forward_hook(
                lambda m, i, o, n=name: activations.update({n: o.detach().cpu()}))

        samples = _sort_samples(self.val_results, num_samples)
        n_cols = 1 + channels_per_stage * len(stages)
        fig, axs = plt.subplots(len(samples), n_cols, figsize=(
            n_cols * 2.5, len(samples) * 3))
        axs = axs if len(samples) > 1 else axs[None, :]

        self.model.eval()
        for row, sample in enumerate(samples):
            search = sample['search'].unsqueeze(0).to(self.device)
            heatmap = sample['pred_heatmap'].numpy()

            with torch.no_grad():
                _ = self.model(search)

            overlay = denormalize(search[0]).permute(1, 2, 0).cpu().numpy()
            overlay[..., 0] = np.clip(overlay[..., 0] + 0.5 * heatmap, 0, 1)

            axs[row, 0].imshow(overlay)
            axs[row, 0].scatter([sample['gt_centroid'][0]], [
                                sample['gt_centroid'][1]], c='green', marker='o')
            axs[row, 0].scatter([sample['pred_centroid'][0]], [
                                sample['pred_centroid'][1]], c='red', marker='x')
            axs[row, 0].set_title(f"Conf: {sample['confidence']:.2f}")
            axs[row, 0].axis('off')
            axs[row, 0].legend()

            col = 1
            for stage in stages:
                act = activations[stage][0]
                for ch in range(channels_per_stage):
                    if ch < act.shape[0]:
                        axs[row, col].imshow(act[ch], cmap='viridis')
                        axs[row, col].set_title(f'{stage} | Ch {ch}')
                    axs[row, col].axis('off')
                    col += 1

        plt.tight_layout()
        self._save_fig(plt.gcf(), "activations", "activation")
