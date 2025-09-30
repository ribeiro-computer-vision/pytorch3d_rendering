## unproject_3d_from_depth_tools.py

import numpy as np
import torch
import cv2
from typing import Literal, Tuple
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras, FoVPerspectiveCameras

import math

from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, SoftPhongShader,
    RasterizationSettings, PointLights, PerspectiveCameras, look_at_view_transform
)

# ---------------------------- IMPORTS -----------------------------------------
# Stdlib
import os
import sys
import math
import shutil
from pathlib import Path
from typing import Optional, Tuple, Literal, Dict, Any

# Third-party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import imageio
import requests
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm.notebook import tqdm
from skimage import img_as_ubyte

# PyTorch3D — IO & data structures
from pytorch3d.io import load_obj, load_ply, load_objs_as_meshes
from pytorch3d.structures import Meshes

# PyTorch3D — transforms
from pytorch3d.transforms import Rotate, Translate

# PyTorch3D — rendering
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
    look_at_view_transform,
    look_at_rotation,
    camera_position_from_spherical_angles,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    SoftSilhouetteShader,
    SoftPhongShader,
    HardPhongShader,
    PointLights,
    DirectionalLights,
    Materials,
    TexturesUV,
    TexturesVertex,
)
from pytorch3d.renderer.cameras import CamerasBase

# PyTorch3D — visualization helpers (optional)
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib

# Project utils path (adjust as needed)
sys.path.append(os.path.abspath(''))
# ------------------------------------------------------------------------------


#----------------------------------- Util ---------------------------------------
class Util:
    @staticmethod
    def clear_cuda_cache():
        import gc
        gc.collect()                 # clear Python refs
        torch.cuda.empty_cache()     # release cached blocks to the driver
        torch.cuda.ipc_collect()     # (optional) clean up inter-proc handles


#----------------------------------- Image ---------------------------------------
class ImageProcessor:
    @staticmethod
    def alpha_over_rgba(
        background_rgba: np.ndarray,
        overlay_rgba: np.ndarray,
        resize_overlay: bool = False,
        premultiplied: bool = False
    ) -> np.ndarray:
        """
        Composite overlay_rgba OVER background_rgba (both uint8 RGBA).
        Returns uint8 RGBA.

        Args:
            background_rgba: (H,W,4) uint8
            overlay_rgba:    (h,w,4) uint8
            resize_overlay:  if True, resize overlay to background size
            premultiplied:   set True if images are premultiplied alpha; defaults to straight alpha
        """
        if background_rgba.ndim != 3 or background_rgba.shape[-1] != 4:
            raise ValueError("background_rgba must be (H,W,4)")
        if overlay_rgba.ndim != 3 or overlay_rgba.shape[-1] != 4:
            raise ValueError("overlay_rgba must be (h,w,4)")

        H, W = background_rgba.shape[:2]
        h, w = overlay_rgba.shape[:2]

        if (h, w) != (H, W):
            if not resize_overlay:
                raise ValueError("Size mismatch; pass resize_overlay=True to auto-resize overlay.")
            overlay_rgba = cv2.resize(overlay_rgba, (W, H), interpolation=cv2.INTER_AREA)

        # Convert to float32 in [0,1]
        bg = background_rgba.astype(np.float32) / 255.0
        fg = overlay_rgba.astype(np.float32) / 255.0

        a_b = bg[..., 3:4]  # (H,W,1)
        a_f = fg[..., 3:4]

        if premultiplied:
            # If inputs are premultiplied: colors already multiplied by alpha
            # out.rgb = fg.rgb + (1 - a_f) * bg.rgb
            # out.a   = a_f + (1 - a_f) * a_b
            out_rgb = fg[..., :3] + (1.0 - a_f) * bg[..., :3]
        else:
            # Straight alpha
            # out.rgb = (fg.rgb*a_f + bg.rgb*a_b*(1 - a_f)) / out.a   (but we usually keep straight result)
            out_rgb = fg[..., :3] * a_f + bg[..., :3] * (1.0 - a_f)

        out_a = a_f + (1.0 - a_f) * a_b

        out = np.concatenate([out_rgb, out_a], axis=-1)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return out


    @staticmethod
    def mesh_wireframe_image(
        mesh: Meshes,
        cameras: CamerasBase,
        image_size: Tuple[int, int] = (480, 640),         # (H, W)
        edge_mode: Literal["all", "boundary", "feature", "boundary+feature"] = "boundary+feature",
        feature_deg: float = 30.0,
        line_rgb: Tuple[int, int, int] = (0, 255, 0),     # RGB color for lines
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Render a wireframe-only RGB image from a PyTorch3D mesh by projecting edges.

        Args:
            mesh: PyTorch3D Meshes (batch=1).
            cameras: PyTorch3D camera (PerspectiveCameras/FoV...); must support transform_points_screen.
            image_size: (H, W) in pixels.
            edge_mode:
                "all"               -> all unique triangle edges
                "boundary"          -> only boundary (silhouette/topology) edges
                "feature"           -> only edges with dihedral angle > feature_deg
                "boundary+feature"  -> union of the above (default)
            feature_deg: dihedral angle threshold for feature edges.
            line_rgb: RGB color for wireframe.
            line_thickness: line thickness in pixels.

        Returns:
            img_rgb: np.uint8 array of shape (H, W, 3) in RGB (ready for plt.imshow).
        """
        # assert mesh.num_meshes() == 1, "Provide a single mesh (batch=1)."
        device = mesh.device
        H, W = image_size

        verts = mesh.verts_packed()   # (V,3)
        faces = mesh.faces_packed()   # (F,3)

        # --- Build unique edge list ---
        e01 = faces[:, [0, 1]]
        e12 = faces[:, [1, 2]]
        e20 = faces[:, [2, 0]]
        edges = torch.cat([e01, e12, e20], dim=0)           # (3F,2)
        edges = torch.sort(edges, dim=1).values             # canonicalize (i<j)
        edges = torch.unique(edges, dim=0)                  # (E,2)

        def edge_face_adjacency(faces_t: torch.Tensor):
            e2f = {}
            F = faces_t.shape[0]
            for fid in range(F):
                f = faces_t[fid].tolist()
                for (a, b) in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                    i, j = (a, b) if a < b else (b, a)
                    e2f.setdefault((i, j), []).append(fid)
            return e2f  # map (i,j) -> [face_ids...]

        # --- Select edges (boundary/feature) if requested ---
        if edge_mode != "all":
            e2f = edge_face_adjacency(faces)
            select = []

            want_boundary = ("boundary" in edge_mode)
            want_feature  = ("feature"  in edge_mode)
            if want_feature:
                v0 = verts[faces[:, 0]]
                v1 = verts[faces[:, 1]]
                v2 = verts[faces[:, 2]]
                fn = torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)  # (F,3)
                cos_thresh = float(np.cos(np.deg2rad(feature_deg)))

            for (i, j), fids in e2f.items():
                add = False
                if want_boundary and len(fids) == 1:
                    add = True
                if want_feature and len(fids) == 2:
                    n0, n1 = fn[fids[0]], fn[fids[1]]
                    cosang = torch.dot(n0, n1).clamp(-1, 1).item()
                    if cosang < cos_thresh:
                        add = True
                if add:
                    select.append((i, j))
            if not select:  # fallback
                select = [tuple(e.tolist()) for e in edges]
            edges = torch.tensor(select, dtype=torch.int64, device=device)

        # --- Project vertices to pixels ---
        # transform_points_screen returns (B,N,3) with xy in pixels when in_ndc=False & image_size is given
        verts_batched = verts.unsqueeze(0)  # (1,V,3)
        verts_scr = cameras.transform_points_screen(verts_batched, image_size=((H, W),))[0, :, :2]  # (V,2)
        # Also get camera-space Z to cull behind-camera points
        to_view = cameras.get_world_to_view_transform()
        verts_cam = to_view.transform_points(verts_batched)[0]  # (V,3)

        # --- Draw on white canvas using OpenCV ---
        img_bgr = np.full((H, W, 3), 255, dtype=np.uint8)  # white background (BGR)
        # OpenCV expects BGR; convert our RGB color:
        bgr = (int(line_rgb[2]), int(line_rgb[1]), int(line_rgb[0]))

        v2d = verts_scr.detach().cpu().numpy()
        zc  = verts_cam[:, 2].detach().cpu().numpy()

        for i0, i1 in edges.detach().cpu().numpy():
            # simple visibility: both endpoints in front of camera
            if zc[i0] <= 0 or zc[i1] <= 0:
                continue
            u0, v0 = v2d[i0]
            u1, v1 = v2d[i1]
            # clip to image bounds (optional: skip if far off-screen)
            if not (np.isfinite([u0, v0, u1, v1]).all()):
                continue
            p0 = (int(round(u0)), int(round(v0)))
            p1 = (int(round(u1)), int(round(v1)))
            # Draw if at least partially in the image
            if (0 <= p0[0] < W or 0 <= p1[0] < W) and (0 <= p0[1] < H or 0 <= p1[1] < H):
                cv2.line(img_bgr, p0, p1, color=bgr, thickness=line_thickness, lineType=cv2.LINE_AA)

        # Convert BGR -> RGB for matplotlib
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb


    @staticmethod
    def contour_from_nonwhite_rgb(
        rgb: np.ndarray,
        *,
        draw_color: Tuple[int,int,int] = (0,255,0),   # RGB for contour lines
        thickness: int = 2,
        mode: Literal["largest","all"] = "all",   # external: largest or all
        white_tol: int = 0,                          # tolerance for "white" (per channel)
        post_close: int = 3,                          # morph close kernel (0=skip)
        fill_holes: bool = False,                      # fill interior holes to avoid inner contours
        return_masks: bool = False
    ):
        """
        Create contour overlay from a PyTorch3D RGB render by masking non-white pixels.

        Args:
            rgb: (H,W,3) float [0,1] or uint8 [0,255]
            draw_color: RGB contour color
            thickness: line thickness
            mode: "largest" or "all" (external contours only)
            white_tol: pixels with all channels >= 255-white_tol -> treated as background
            post_close: morphology close kernel size (e.g., 3 or 5). 0 disables.
            fill_holes: flood-fill background then invert to remove interior holes
            return_masks: also return (mask, contour_only_img)

        Returns:
            contour_rgb (H,W,3) uint8
            [optional] mask (H,W) uint8 in {0,255}
            [optional] contour_only (H,W,3) uint8 on white bg
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be (H,W,3)")

        # Normalize to uint8
        if np.issubdtype(rgb.dtype, np.floating):
            img = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = np.clip(rgb, 0, 255).astype(np.uint8)

        H, W = img.shape[:2]

        # 1) Foreground mask = non-white pixels
        # nonwhite if ANY channel < 255 - tol
        nonwhite = (img < (255 - white_tol)).any(axis=2)
        mask = (nonwhite.astype(np.uint8) * 255)  # (H,W) {0,255}

        # 2) Optional cleanup: close small gaps on edges
        if post_close and post_close > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (post_close, post_close))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # 3) Optional: fill holes so internal contours disappear
        if fill_holes:
            # flood fill from border to get background, invert to keep filled foreground
            ff = mask.copy()
            h, w = ff.shape
            ff_pad = np.pad(ff, ((1,1),(1,1)), mode='constant', constant_values=0)
            mask_filled = ff_pad.copy()
            cv2.floodFill(mask_filled, None, (0,0), 255)             # fill background outside object
            mask_bg = (mask_filled == 255)[1:-1,1:-1]                # remove pad
            # foreground = original OR NOT background-fill
            mask = np.where(mask_bg, 0, 255).astype(np.uint8)

        # 4) External contours only (prevents inner contours)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            contour_rgb = img.copy()
            if return_masks:
                return contour_rgb, mask, np.full_like(img, 255)
            return contour_rgb

        if mode == "largest":
            contours = [max(contours, key=cv2.contourArea)]

        # 5) Draw contours on original (OpenCV uses BGR)
        contour_rgb_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bgr = (draw_color[2], draw_color[1], draw_color[0])
        cv2.drawContours(contour_rgb_bgr, contours, -1, bgr, thickness)
        contour_rgb = cv2.cvtColor(contour_rgb_bgr, cv2.COLOR_BGR2RGB)

        if return_masks:
            contour_only = np.full_like(img, 255)
            c_bgr = cv2.cvtColor(contour_only, cv2.COLOR_RGB2BGR)
            cv2.drawContours(c_bgr, contours, -1, bgr, thickness)
            contour_only = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
            return contour_rgb, mask, contour_only

        return contour_rgb


    @staticmethod
    def contour_from_nonwhite_rgb_semi_transparent(
        rgb: np.ndarray,
        *,
        draw_color: Tuple[int,int,int] = (0,255,0),   # RGB for contour lines
        thickness: int = 2,
        mode: Literal["largest","all"] = "all",       # external: largest or all
        white_tol: int = 0,                           # tolerance for "white" (per channel)
        post_close: int = 3,                          # morph close kernel (0=skip)
        fill_holes: bool = False,                     # fill interior holes to avoid inner contours
        return_masks: bool = False,
        return_rgba: bool = False,                    # NEW: if True, return RGBA image instead of RGB
        alpha_value: float = 0.5                      # alpha value for the non-contour region
    ):
        """
        Create contour overlay from a PyTorch3D RGB render by masking non-white pixels.

        Args:
            rgb: (H,W,3) float [0,1] or uint8 [0,255]
            draw_color: RGB contour color
            thickness: line thickness
            mode: "largest" or "all" (external contours only)
            white_tol: pixels with all channels >= 255-white_tol -> treated as background
            post_close: morphology close kernel size (e.g., 3 or 5). 0 disables.
            fill_holes: flood-fill background then invert to remove interior holes
            return_masks: also return (mask, contour_only_img)
            return_rgba: if True, output is (H,W,4) with alpha channel
            alpha_value: alpha assigned to non-contour pixels [0..1]

        Returns:
            contour_img (H,W,3) or (H,W,4) uint8
            [optional] mask (H,W) uint8 in {0,255}
            [optional] contour_only (H,W,3) uint8
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be (H,W,3)")

        # Normalize to uint8
        if np.issubdtype(rgb.dtype, np.floating):
            img = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = np.clip(rgb, 0, 255).astype(np.uint8)

        H, W = img.shape[:2]

        # 1) Foreground mask = non-white pixels
        nonwhite = (img < (255 - white_tol)).any(axis=2)
        mask = (nonwhite.astype(np.uint8) * 255)  # (H,W)

        # 2) Cleanup
        if post_close and post_close > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (post_close, post_close))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # 3) Fill holes (optional)
        if fill_holes:
            ff = mask.copy()
            ff_pad = np.pad(ff, ((1,1),(1,1)), mode='constant', constant_values=0)
            cv2.floodFill(ff_pad, None, (0,0), 255)
            mask_bg = (ff_pad == 255)[1:-1,1:-1]
            mask = np.where(mask_bg, 0, 255).astype(np.uint8)

        # 4) Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if return_rgba:
                rgba = np.dstack([img, np.full((H,W), int(alpha_value*255), np.uint8)])
                return rgba
            if return_masks:
                return img, mask, np.full_like(img, 255)
            return img

        if mode == "largest":
            contours = [max(contours, key=cv2.contourArea)]

        # 5) Prepare output
        contour_rgb_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bgr = (draw_color[2], draw_color[1], draw_color[0])
        cv2.drawContours(contour_rgb_bgr, contours, -1, bgr, thickness)
        contour_rgb = cv2.cvtColor(contour_rgb_bgr, cv2.COLOR_BGR2RGB)

        if return_rgba:
            # Build alpha: everything gets alpha_value, contour pixels forced to 1
            alpha = np.full((H, W), int(alpha_value*255), dtype=np.uint8)
            contour_mask = np.zeros((H,W), dtype=np.uint8)
            cv2.drawContours(contour_mask, contours, -1, 255, thickness)
            alpha[contour_mask > 0] = 255  # contours fully opaque
            contour_rgba = np.dstack([contour_rgb, alpha])
            return contour_rgba

        if return_masks:
            contour_only = np.full_like(img, 255)
            c_bgr = cv2.cvtColor(contour_only, cv2.COLOR_RGB2BGR)
            cv2.drawContours(c_bgr, contours, -1, bgr, thickness)
            contour_only = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
            return contour_rgb, mask, contour_only

        return contour_rgb


    @staticmethod
    def depth_to_rgb(
        depth: torch.Tensor,
        cmap: str = "viridis",
        bg_mode: Literal["black","white","transparent"] = "black"
    ) -> np.ndarray:
        """
        Convert a PyTorch3D depth map into a visualization image.
        Automatically treats depth = -1 as invalid background.

        Args:
            depth: (H,W) torch.Tensor of depth values (float).
            cmap:  Matplotlib colormap name (default: "viridis").
            bg_mode:
                "black"       -> background stays black (default)
                "white"       -> background set to white
                "transparent" -> return RGBA image with alpha=0 for background

        Returns:
            img: (H,W,3) uint8 RGB or (H,W,4) uint8 RGBA if bg_mode="transparent"
        """
        if not isinstance(depth, torch.Tensor):
            raise ValueError("depth must be a torch.Tensor")

        depth_np = depth.detach().cpu().numpy().astype(np.float32)

        # Mask out invalid regions (zbuf = -1 means background in PyTorch3D)
        valid_mask = depth_np > 0

        if np.any(valid_mask):
            dmin, dmax = depth_np[valid_mask].min(), depth_np[valid_mask].max()
            depth_norm = np.zeros_like(depth_np, dtype=np.float32)
            depth_norm[valid_mask] = (depth_np[valid_mask] - dmin) / (dmax - dmin + 1e-8)
        else:
            depth_norm = np.zeros_like(depth_np, dtype=np.float32)

        cmap_func = cm.get_cmap(cmap)
        rgba = cmap_func(depth_norm)  # (H,W,4) floats in [0,1]

        if bg_mode == "transparent":
            rgba[..., 3] = 0.0          # alpha=0 where background
            rgba[valid_mask, 3] = 1.0   # alpha=1 where valid
            img = (rgba * 255).astype(np.uint8)  # RGBA
        else:
            rgb = (rgba[..., :3] * 255).astype(np.uint8)
            if bg_mode == "white":
                rgb[~valid_mask] = [255,255,255]
            elif bg_mode == "black":
                rgb[~valid_mask] = [0,0,0]
            img = rgb

        return img



    @staticmethod
    def make_binary_mask(img: np.ndarray, white_tol: int = 5, black_tol: int = 5) -> np.ndarray:
        """
        Create a binary mask (1=foreground, 0=background) from an RGB image
        where the background can be either black or white.

        Args:
            img: (H,W,3) uint8 or float image.
            white_tol: tolerance for detecting white background (0–255).
            black_tol: tolerance for detecting black background (0–255).

        Returns:
            mask: (H,W) np.uint8 with {0,1}
        """
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (H,W,3).")

        # Normalize to uint8 if in float
        if np.issubdtype(img.dtype, np.floating):
            if img.max() <= 1.0:
                img_u8 = (img * 255).astype(np.uint8)
            else:
                img_u8 = img.astype(np.uint8)
        else:
            img_u8 = img.copy()

        H, W, _ = img_u8.shape

        # Look at border pixels to decide background type
        border = np.concatenate([img_u8[0,:,:], img_u8[-1,:,:], img_u8[:,0,:], img_u8[:,-1,:]], axis=0)
        border_mean = border.mean()

        if border_mean < 127:
            # Background is black
            mask = (np.any(img_u8 > black_tol, axis=-1)).astype(np.uint8)
        else:
            # Background is white
            mask = (np.any(img_u8 < 255 - white_tol, axis=-1)).astype(np.uint8)

        return mask









