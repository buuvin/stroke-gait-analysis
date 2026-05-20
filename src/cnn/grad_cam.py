import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: shape (1, 1, H, W)
        returns:
            cam: np.ndarray shape (H, W)
            pred_class: int
            probs: np.ndarray shape (2,)
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)   # (1, 2)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = int(torch.argmax(probs).item())

        if class_idx is None:
            class_idx = pred_class

        score = logits[:, class_idx].sum()
        score.backward()

        # activations: (1, C, h, w)
        # gradients:   (1, C, h, w)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), pred_class, probs.detach().cpu().numpy()


def show_gradcam(img_tensor, cam, true_label=None, pred_class=None, probs=None):
    """
    img_tensor: shape (1, H, W) or (H, W)
    cam: (H, W)
    """
    if img_tensor.ndim == 3:
        img_np = img_tensor.squeeze(0).cpu().numpy()
    else:
        img_np = img_tensor.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(8, 8))

    # axes[0].imshow(img_np, cmap="gray")
    # axes[0].set_title("Original Plot")
    # axes[0].axis("off")

    # axes[1].imshow(cam, cmap="jet")
    # axes[1].set_title("Grad-CAM")
    # axes[1].axis("off")

    axes[2].imshow(img_np, cmap="gray")
    axes[2].imshow(cam, cmap="jet", alpha=0.4)
    title = "Overlay"
    if true_label is not None and pred_class is not None:
        title += f"\nTrue={true_label}, Pred={pred_class}"
    if probs is not None:
        title += f"\nP(healthy)={probs[0]:.3f}, P(stroke)={probs[1]:.3f}"
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def show_gradcam_overlay_only(img_tensor, cam, true_label=None, pred_class=None, probs=None, outfile=None):
    """
    Shows only the Grad-CAM overlay on the original image.
    
    img_tensor: shape (1, H, W) or (H, W)
    cam: (H, W)
    """
    if img_tensor.ndim == 3:
        img_np = img_tensor.squeeze(0).cpu().numpy()
    else:
        img_np = img_tensor.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.imshow(img_np, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.4)
    
    title = "Grad-CAM Overlay"
    if true_label is not None and pred_class is not None:
        title += f"\nTrue={true_label}, Pred={pred_class}"
    if probs is not None:
        title += f"\nP(healthy)={probs[0]:.3f}, P(stroke)={probs[1]:.3f}"
    
    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    if outfile:
        plt.savefig(str(outfile), dpi=150)
    plt.show()


def show_gradcam_overlay_only(img_tensor, cam, true_label=None, pred_class=None, probs=None, outfile=None):
    """
    Shows only the Grad-CAM overlay on the original image.
    
    img_tensor: shape (1, H, W) or (H, W)
    cam: (H, W)
    """
    if img_tensor.ndim == 3:
        img_np = img_tensor.squeeze(0).cpu().numpy()
    else:
        img_np = img_tensor.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.imshow(img_np, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.4)
    
    title = "Grad-CAM Overlay"
    if true_label is not None and pred_class is not None:
        title += f"\nTrue={true_label}, Pred={pred_class}"
    if probs is not None:
        title += f"\nP(healthy)={probs[0]:.3f}, P(stroke)={probs[1]:.3f}"
    
    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    if outfile:
        plt.savefig(str(outfile), dpi=150)
    plt.show()

def disable_inplace_relu(module):
    for child in module.children():
        if isinstance(child, torch.nn.ReLU):
            child.inplace = False
        else:
            disable_inplace_relu(child)