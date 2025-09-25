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




#--------------------------------- Unprojector ---------------------------------
class Unprojector:
    @staticmethod
    def bilinear_sample_depth(depth: torch.Tensor, uv):
        """Bilinear sample depth map at pixel (u,v)."""
        H, W = depth.shape
        u, v = uv
        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)

        u0, v0 = int(np.floor(u)), int(np.floor(v))
        u1, v1 = min(u0+1, W-1), min(v0+1, H-1)
        du, dv = u - u0, v - v0

        d00 = depth[v0, u0]
        d10 = depth[v0, u1]
        d01 = depth[v1, u0]
        d11 = depth[v1, u1]

        d0 = d00 * (1 - du) + d10 * du
        d1 = d01 * (1 - du) + d11 * du
        return float((d0 * (1 - dv) + d1 * dv).item())


    @staticmethod
    def unproject_points(pts_uv, depths, fx, fy, cx, cy, R, T):
        """Convert pixels+depth to 3D in camera and world coords."""
        Kinv = np.linalg.inv(np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64))
        X_cam_list = []
        for (u,v), z in zip(pts_uv, depths):
            if z <= 0:
                X_cam_list.append([np.nan]*3)
                continue
            pix = np.array([u, v, 1.0], dtype=np.float64)
            ray = Kinv @ pix
            X_cam_list.append((ray * z).tolist())
        X_cam = np.array(X_cam_list)
        R_np = R.detach().cpu().numpy()[0]
        T_np = T.detach().cpu().numpy()[0]
        X_world = (X_cam - T_np) @ R_np
        return X_cam, X_world



    @staticmethod
    def recover_3D_points(points_uv, depth, fx, fy, cx, cy, R, T):

        # Get the interpolated depths for the list of (u,v) points
        depths = [Unprojector.bilinear_sample_depth(depth, uv) for uv in points_uv]

        # Unproject the interpolated (u,v) points to obtain 3-D coordinates
        X_cam, X_world = Unprojector.unproject_points(points_uv, depths, fx, fy, cx, cy, R, T)

        return X_cam, X_world, depths

    @staticmethod
    def list_recovered_3D_points(points_uv, depth, fx, fy, cx, cy, R, T):

        # depths = [Unprojector.bilinear_sample_depth(depth, uv) for uv in points_uv]

        # X_cam, X_world = Unprojector.unproject_points(points_uv, depths, fx, fy, cx, cy, R, T)

        X_cam, X_world, depths = Unprojector.recover_3D_points(points_uv, depth, fx, fy, cx, cy, R, T)

        for (u,v), z, xc, xw in zip(points_uv, depths, X_cam, X_world):
            print(f"Pixel ({u},{v}) -> Z={z:.4f}, "
                  f"Cam=({xc[0]:.4f},{xc[1]:.4f},{xc[2]:.4f}), "
                  f"World=({xw[0]:.4f},{xw[1]:.4f},{xw[2]:.4f})")


#----------------------------------- CAM ---------------------------------------
class Cam:

    @staticmethod
    def add_camera_roll_to_RT(R, T, roll_deg, *, device=None, mode="camera"):
        """
        Compose a Z-axis roll into (R,T), keeping the same camera center C.
        Grad-safe: roll_deg can be a Tensor/Parameter.
        mode: "camera" -> R' = Rz @ R ; "world" -> R' = R @ Rz
        """
        if not torch.is_tensor(R): R = torch.as_tensor(R)
        if not torch.is_tensor(T): T = torch.as_tensor(T)
        dev   = device or R.device
        dtype = torch.float32
        R = R.to(dev, dtype)
        T = T.to(dev, dtype)

        unbatched = (R.ndim == 2)
        if unbatched:
            R = R[None, ...]
            T = T[None, ...]

        # C from T = -R^T C  =>  C = -R T
        C = -torch.matmul(R, T[..., None]).squeeze(-1)

        theta = torch.as_tensor(roll_deg, dtype=dtype, device=dev).reshape(1)  # keep grad
        c, s = torch.cos(torch.deg2rad(theta)), torch.sin(torch.deg2rad(theta))
        z = torch.zeros_like(c); o = torch.ones_like(c)
        Rz = torch.stack([
            torch.stack([ c, -s, z], dim=-1),
            torch.stack([ s,  c, z], dim=-1),
            torch.stack([ z,  z,  o], dim=-1),
        ], dim=1)  # (1,3,3)

        R_new = torch.matmul(Rz, R) if mode == "camera" else torch.matmul(R, Rz)
        T_new = -torch.matmul(R_new.transpose(1, 2), C[..., None]).squeeze(-1)

        if unbatched:
            R_new, T_new = R_new[0], T_new[0]
        return R_new, T_new

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



class RenderWithPytorch3D:


    @staticmethod
    def render_rgb_depth_from_view_from_RT(
        mesh: Meshes,
        *,
        fx: float, fy: float, cx: float, cy: float,
        width: int, height: int,
        # distance: float, elev: float, azim: float, roll_deg: float = 0.0,
        # roll_mode: str = "world",                  # "camera" or "world"
        R: Optional[torch.Tensor]  | None = None,          # (1,3,3)
        T: Optional[torch.Tensor]  | None = None,          # (1,3)
        raster_settings: RasterizationSettings | None = None,
        lights: PointLights | None = None,
        device: torch.device | None = None
    ):
        """
        Render an RGB image and a depth map from a PyTorch3D mesh at a given camera view + roll.

        Returns:
            rgb_np   : (H,W,3) float32 in [0,1]
            depth_t  : (H,W)   torch.float32, metric Z in camera coords; invalid pixels == -1
            cameras  : the PerspectiveCameras used (in case you want to reuse)
        """
        device = device or mesh.device


        # # 1) Base view (look-at origin) from spherical params
        # R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)

        # # 2) Add in-plane roll
        # R, T = add_camera_roll_to_RT(R, T, roll_deg=roll_deg, device=device, mode=roll_mode)

        print("Rotation inside function: \n", R)
        print("Translation inside T: \n", T)



        # 3) Camera with pixel intrinsics (OpenCV-like) and in_ndc=False
        cameras = PerspectiveCameras(
            focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
            principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
            R=R, T=T,
            image_size=torch.tensor([[height, width]], dtype=torch.float32, device=device),
            in_ndc=False, device=device
        )

        # 4) Default raster/shader if none provided
        if raster_settings is None:
            raster_settings = RasterizationSettings(
                image_size=(height, width),
                blur_radius=0.0,
                faces_per_pixel=1
            )
        if lights is None:
            lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
        )

        # 5) Render RGB
        # images = renderer(mesh, cameras=cameras, lights=lights)          # (1,H,W,4)
        images = renderer(mesh)          # (1,H,W,4)
        rgb_np = images[0, ..., :3].detach().cpu().numpy()               # (H,W,3) float in [0,1]

        # 6) Depth (metric Z; background == -1)
        # fragments = renderer.rasterizer(mesh, cameras=cameras)
        fragments = renderer.rasterizer(mesh)
        depth_t = fragments.zbuf[0, ..., 0].detach()                     # (H,W) torch.float32

        return rgb_np, depth_t, cameras


    # --- Helper: add an in-plane roll to a PyTorch3D camera (R, T) ---
    @staticmethod
    def add_camera_roll_to_RT(R, T, roll_deg: float, device=None, mode: str = "world"):
        """
        Add a roll (rotation about the camera's viewing axis) to (R, T).

        Args:
            R, T: PyTorch3D extrinsics (as from look_at_view_transform), shapes (1,3,3), (1,3)
            roll_deg: roll angle in degrees, positive = CCW in image plane
            device: torch device
            mode:
              - "camera": pre-multiply in camera space  (R' = Rroll @ R)
              - "world" : post-multiply in world space  (R' = R @ Rroll)
                For PyTorch3D’s convention (X_cam = X_world @ R^T + T), "world" is often intuitive.

        Returns:
            (R_rolled, T)  (T unchanged)
        """
        device = device or R.device
        th = math.radians(roll_deg)
        c, s = math.cos(th), math.sin(th)
        Rroll = torch.tensor([[ c, -s, 0.0],
                              [ s,  c, 0.0],
                              [0.0, 0.0, 1.0]], dtype=R.dtype, device=device).unsqueeze(0)  # (1,3,3)

        if mode == "camera":
            R_new = torch.bmm(Rroll, R)     # (1,3,3)
        elif mode == "world":
            R_new = torch.bmm(R, Rroll)     # (1,3,3)
        else:
            raise ValueError("mode must be 'camera' or 'world'")
        return R_new, T


    @staticmethod
    def render_rgb_depth_from_view(
        mesh: Meshes,
        *,
        fx: float, fy: float, cx: float, cy: float,
        width: int, height: int,
        distance: float, elev: float, azim: float, roll_deg: float = 0.0,
        roll_mode: str = "world",                  # "camera" or "world"
        raster_settings: RasterizationSettings | None = None,
        lights: PointLights | None = None,
        device: torch.device | None = None
    ):
        """
        Render an RGB image and a depth map from a PyTorch3D mesh at a given camera view + roll.

        Returns:
            rgb_np   : (H,W,3) float32 in [0,1]
            depth_t  : (H,W)   torch.float32, metric Z in camera coords; invalid pixels == -1
            cameras  : the PerspectiveCameras used (in case you want to reuse)
        """
        device = device or mesh.device

        # 1) Base view (look-at origin) from spherical params
        R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim, device=device)

        # 2) Add in-plane roll
        R, T = RenderWithPytorch3D.add_camera_roll_to_RT(R, T, roll_deg=roll_deg, device=device, mode=roll_mode)

        print("Rotation inside function: \n", R)
        print("Translation inside T: \n", T)


        # 3) Camera with pixel intrinsics (OpenCV-like) and in_ndc=False
        cameras = PerspectiveCameras(
            focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
            principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
            R=R, T=T,
            image_size=torch.tensor([[height, width]], dtype=torch.float32, device=device),
            in_ndc=False, device=device
        )

        # 4) Default raster/shader if none provided
        if raster_settings is None:
            raster_settings = RasterizationSettings(
                image_size=(height, width),
                blur_radius=0.0,
                faces_per_pixel=1
            )
        if lights is None:
            lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
        )

        # Sanity: how does the camera map two simple world points?
        H, W = height, width
        test = torch.tensor([  # world points
            [ 0.0, 0.0, 0.0],   # origin
            [ 0.1, 0.0, 0.0],   # +x_world
            [-0.1, 0.0, 0.0],   # -x_world
            [ 0.0, 0.1, 0.0],   # +y_world
            [ 0.0,-0.1, 0.0],   # -y_world
        ], device=device).unsqueeze(0)  # (1,5,3)

        uvz = cameras.transform_points_screen(
            test, image_size=torch.tensor([[H, W]], device=device)
        )[0]  # (5,3)

        print("u(+x_world) =", float(uvz[1,0]), "  u(-x_world) =", float(uvz[2,0]), "  cx ~", float(cameras.principal_point[0,0]))
        print("v(+y_world) =", float(uvz[3,1]), "  v(-y_world) =", float(uvz[4,1]), "  cy ~", float(cameras.principal_point[0,1]))

        print("focal_length=", cameras.focal_length[0])  # <-- check sign of fx, fy




        # 5) Render RGB
        # images = renderer(mesh, cameras=cameras, lights=lights)          # (1,H,W,4)
        images = renderer(mesh)          # (1,H,W,4)
        rgb_np = images[0, ..., :3].detach().cpu().numpy()               # (H,W,3) float in [0,1]

        # 6) Depth (metric Z; background == -1)
        # fragments = renderer.rasterizer(mesh, cameras=cameras)
        fragments = renderer.rasterizer(mesh)
        depth_t = fragments.zbuf[0, ..., 0].detach()                     # (H,W) torch.float32

        return rgb_np, depth_t, cameras
















##---------------------------------------------------
#         Gradio interface as a function
##---------------------------------------------------

def launch_point_picker(my_image):

    # --- Single-image point picker using notebook variable `my_image` ---
    import gradio as gr
    from PIL import Image, ImageDraw
    import numpy as np
    import threading, json, os, io

    # Data
    points_store = []
    app = None
    SELECTED_POINTS = None

    def _to_pil_from_numpy(arr: np.ndarray) -> Image.Image:
        arr = np.asarray(arr)
        # channel-first -> channel-last
        if arr.ndim == 3 and arr.shape[0] in (1,3,4) and arr.shape[-1] not in (1,3,4):
            arr = np.transpose(arr, (1,2,0))
        if np.issubdtype(arr.dtype, np.floating):
            # scale floats in [0,1] to [0,255]
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # Choose mode
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.ndim == 3 and arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        if arr.ndim == 3 and arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA")
        if arr.ndim == 3 and arr.shape[2] == 1:
            return Image.fromarray(arr[:,:,0], mode="L")
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    def _to_pil(img):
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            return _to_pil_from_numpy(img)
        raise gr.Error("Set `my_image` to a PIL image or NumPy array before launching.")

    def _draw_points(base_img: Image.Image, pts, radius=5):
        img = base_img.copy().convert("RGB")
        d = ImageDraw.Draw(img)
        for (x, y) in pts:
            d.ellipse([x-radius, y-radius, x+radius, y+radius], outline=(255,0,0), width=2)
        return img

    # Prepare base image from notebook variable
    # if 'my_image' not in globals():
        # raise RuntimeError("Please define `my_image` (PIL image or NumPy array) before running this cell.")
    # base_pil = _to_pil(globals()['my_image'])
    base_pil = _to_pil(my_image)

    def _refresh_numpy():
        """Return current preview (base + points) as numpy for Gradio."""
        return np.array(_draw_points(base_pil, points_store))

    def on_click(evt: gr.SelectData):
        # Get coordinates robustly
        x = y = None
        if hasattr(evt, "index") and evt.index is not None:
            try: x, y = evt.index
            except: pass
        if (x is None or y is None) and hasattr(evt, "x") and hasattr(evt, "y"):
            x, y = evt.x, evt.y
        if x is None or y is None:
            return gr.update(), json.dumps(points_store)

        # Clamp to image bounds
        w, h = base_pil.size
        x = int(max(0, min(w-1, x)))
        y = int(max(0, min(h-1, y)))

        points_store.append([x, y])
        return _refresh_numpy(), json.dumps(points_store)

    def undo_last():
        if points_store:
            points_store.pop()
        return _refresh_numpy(), json.dumps(points_store)

    def clear_points():
        points_store.clear()
        return np.array(base_pil), "[]"

    def done_btn_click():
        """Save to notebook var `selected_points` and close the app."""
        global SELECTED_POINTS
        SELECTED_POINTS = [list(p) for p in points_store]
        try:
            ip = get_ipython()
            if ip is not None:
                ip.user_ns['selected_points'] = SELECTED_POINTS
        except Exception:
            pass
        threading.Thread(target=lambda: app.close(), daemon=True).start()
        return f"✅ Saved {len(SELECTED_POINTS)} points to `selected_points`. Closing…"

    with gr.Blocks(title="Point Picker (single image)") as demo:
        gr.Markdown("**Click on the image to add points.** Use Undo / Clear as needed, then press **Done**.")
        img = gr.Image(
            value=np.array(base_pil), label="Image (click to add points)",
            type="numpy", interactive=True, sources=[]  # sources=[] disables uploads
        )
        with gr.Row():
            undo_btn = gr.Button("↩️ Undo")
            clear_btn = gr.Button("🧹 Clear")
            done_btn = gr.Button("✅ Done", variant="primary")
        pts_text = gr.Textbox(label="Points (JSON)", value="[]", interactive=False)
        status = gr.Markdown("")

        # One image used for both input and output
        img.select(on_click, inputs=None, outputs=[img, pts_text])
        undo_btn.click(lambda: undo_last(), outputs=[img, pts_text])
        clear_btn.click(lambda: clear_points(), outputs=[img, pts_text])
        done_btn.click(done_btn_click, outputs=[status])


        app = demo.launch(inline=True, prevent_thread_lock=True)
        return app


# ##---------------------------------------------------
# #         Gradio interface as a function
# ##---------------------------------------------------

# def launch_point_matcher(image_left, image_right, gap=40):
#     """
#     Gradio app: pick corresponding points on two images and visualize connections.
#     - image_left, image_right: PIL.Image or numpy arrays (H,W,[C])
#     - gap: pixels between images on the stitched correspondence view
#     """
#     import gradio as gr
#     from PIL import Image, ImageDraw
#     import numpy as np
#     import threading, json

#     # ---------- helpers ----------
#     def _to_pil_from_numpy(arr: np.ndarray) -> Image.Image:
#         arr = np.asarray(arr)
#         # CHW -> HWC
#         if arr.ndim == 3 and arr.shape[0] in (1,3,4) and arr.shape[-1] not in (1,3,4):
#             arr = np.transpose(arr, (1,2,0))
#         if np.issubdtype(arr.dtype, np.floating):
#             arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
#         elif arr.dtype != np.uint8:
#             arr = np.clip(arr, 0, 255).astype(np.uint8)
#         if arr.ndim == 2:
#             return Image.fromarray(arr, mode="L")
#         if arr.ndim == 3 and arr.shape[2] == 3:
#             return Image.fromarray(arr, mode="RGB")
#         if arr.ndim == 3 and arr.shape[2] == 4:
#             return Image.fromarray(arr, mode="RGBA")
#         if arr.ndim == 3 and arr.shape[2] == 1:
#             return Image.fromarray(arr[:,:,0], mode="L")
#         raise ValueError(f"Unsupported array shape: {arr.shape}")

#     def _to_pil(img):
#         if isinstance(img, Image.Image):
#             return img.convert("RGB")
#         if isinstance(img, np.ndarray):
#             return _to_pil_from_numpy(img).convert("RGB")
#         raise gr.Error("Please pass PIL images or NumPy arrays.")

#     def _draw_points(base_img: Image.Image, pts, radius=5, color=(255,0,0)):
#         img = base_img.copy().convert("RGB")
#         d = ImageDraw.Draw(img)
#         for (x, y) in pts:
#             d.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, width=2)
#             d.line([(x-8,y), (x+8,y)], fill=color, width=2)
#             d.line([(x,y-8), (x,y+8)], fill=color, width=2)
#         return img

#     def _stitch_and_draw_lines(imgL: Image.Image, imgR: Image.Image, ptsL, ptsR, gap: int):
#         """Return a stitched (H, W_L + gap + W_R) image with lines between min(lenL, lenR) pairs."""
#         Wl, Hl = imgL.size
#         Wr, Hr = imgR.size
#         H = max(Hl, Hr)
#         W = Wl + gap + Wr
#         canvas = Image.new("RGB", (W, H), (255,255,255))
#         canvas.paste(imgL, (0, 0))
#         canvas.paste(imgR, (Wl + gap, 0))
#         d = ImageDraw.Draw(canvas)

#         n = min(len(ptsL), len(ptsR))
#         for i in range(n):
#             (xl, yl) = ptsL[i]
#             (xr, yr) = ptsR[i]
#             # draw endpoints
#             d.ellipse([xl-4, yl-4, xl+4, yl+4], outline=(255,0,0), width=2)
#             d.ellipse([Wl+gap + xr-4, yr-4, Wl+gap + xr+4, yr+4], outline=(0,0,255), width=2)
#             # line
#             d.line([(xl, yl), (Wl+gap + xr, yr)], fill=(0,128,255), width=2)
#             # small index label
#             d.text((xl+6, yl-14), str(i), fill=(0,0,0))
#             d.text((Wl+gap + xr+6, yr-14), str(i), fill=(0,0,0))
#         return canvas

#     # ---------- state ----------
#     pts_left = []
#     pts_right = []
#     app = None

#     baseL = _to_pil(image_left)
#     baseR = _to_pil(image_right)

#     def _refresh_left():
#         return np.array(_draw_points(baseL, pts_left, color=(255,0,0)))

#     def _refresh_right():
#         return np.array(_draw_points(baseR, pts_right, color=(0,0,255)))

#     def _refresh_pairs():
#         stitched = _stitch_and_draw_lines(baseL, baseR, pts_left, pts_right, gap=gap)
#         return np.array(stitched)

#     # Click handlers
#     def on_click_left(evt: gr.SelectData):
#         x, y = None, None
#         if hasattr(evt, "index") and evt.index is not None:
#             try: x, y = evt.index
#             except: pass
#         if (x is None or y is None) and hasattr(evt, "x") and hasattr(evt, "y"):
#             x, y = evt.x, evt.y
#         if x is None or y is None:
#             return gr.update(), gr.update(), json.dumps(pts_left), json.dumps(pts_right)
#         W, H = baseL.size
#         x = int(max(0, min(W-1, x))); y = int(max(0, min(H-1, y)))
#         pts_left.append([x, y])
#         return _refresh_left(), _refresh_pairs(), json.dumps(pts_left), json.dumps(pts_right)

#     def on_click_right(evt: gr.SelectData):
#         x, y = None, None
#         if hasattr(evt, "index") and evt.index is not None:
#             try: x, y = evt.index
#             except: pass
#         if (x is None or y is None) and hasattr(evt, "x") and hasattr(evt, "y"):
#             x, y = evt.x, evt.y
#         if x is None or y is None:
#             return gr.update(), gr.update(), json.dumps(pts_left), json.dumps(pts_right)
#         W, H = baseR.size
#         x = int(max(0, min(W-1, x))); y = int(max(0, min(H-1, y)))
#         pts_right.append([x, y])
#         return _refresh_right(), _refresh_pairs(), json.dumps(pts_left), json.dumps(pts_right)

#     # Buttons
#     def undo_left():
#         if pts_left: pts_left.pop()
#         return _refresh_left(), _refresh_pairs(), json.dumps(pts_left), json.dumps(pts_right)

#     def undo_right():
#         if pts_right: pts_right.pop()
#         return _refresh_right(), _refresh_pairs(), json.dumps(pts_left), json.dumps(pts_right)

#     def clear_left():
#         pts_left.clear()
#         return _refresh_left(), _refresh_pairs(), json.dumps(pts_left), json.dumps(pts_right)

#     def clear_right():
#         pts_right.clear()
#         return _refresh_right(), _refresh_pairs(), json.dumps(pts_left), json.dumps(pts_right)

#     def clear_both():
#         pts_left.clear(); pts_right.clear()
#         return _refresh_left(), _refresh_right(), _refresh_pairs(), "[]", "[]"

#     def done_btn_click():
#         # expose to notebook namespace
#         try:
#             ip = get_ipython()
#             if ip is not None:
#                 ip.user_ns['selected_points_A'] = [list(p) for p in pts_left]
#                 ip.user_ns['selected_points_B'] = [list(p) for p in pts_right]
#         except Exception:
#             pass
#         msg = f"✅ Saved A:{len(pts_left)} and B:{len(pts_right)} points to `selected_points_A/B`. Closing…"
#         threading.Thread(target=lambda: app.close(), daemon=True).start()
#         return msg

#     # ---------- UI ----------
#     with gr.Blocks(title="Two-Image Correspondence Picker") as demo:
#         gr.Markdown("**Click points on each image.** Lines connect pairs by index: 0↔0, 1↔1, … (min length).")
#         with gr.Row():
#             with gr.Column():
#                 imgL = gr.Image(value=np.array(baseL), label="Image A (click to add)", type="numpy", interactive=True, sources=[])
#                 with gr.Row():
#                     btn_undo_L = gr.Button("↩️ Undo A")
#                     btn_clear_L = gr.Button("🧹 Clear A")
#                 txtA = gr.Textbox(label="Points A (JSON)", value="[]", interactive=False)
#             with gr.Column():
#                 imgR = gr.Image(value=np.array(baseR), label="Image B (click to add)", type="numpy", interactive=True, sources=[])
#                 with gr.Row():
#                     btn_undo_R = gr.Button("↩️ Undo B")
#                     btn_clear_R = gr.Button("🧹 Clear B")
#                 txtB = gr.Textbox(label="Points B (JSON)", value="[]", interactive=False)

#         gr.Markdown("### Correspondences")
#         pair_view = gr.Image(value=_refresh_pairs(), label="Pairs (A | B with lines)", interactive=False)

#         with gr.Row():
#             btn_clear_both = gr.Button("🧼 Clear Both")
#             btn_done = gr.Button("✅ Done", variant="primary")
#         status = gr.Markdown()

#         # events
#         imgL.select(on_click_left, inputs=None, outputs=[imgL, pair_view, txtA, txtB])
#         imgR.select(on_click_right, inputs=None, outputs=[imgR, pair_view, txtA, txtB])

#         btn_undo_L.click(undo_left, outputs=[imgL, pair_view, txtA, txtB])
#         btn_undo_R.click(undo_right, outputs=[imgR, pair_view, txtA, txtB])

#         btn_clear_L.click(clear_left, outputs=[imgL, pair_view, txtA, txtB])
#         btn_clear_R.click(clear_right, outputs=[imgR, pair_view, txtA, txtB])

#         btn_clear_both.click(clear_both, outputs=[imgL, imgR, pair_view, txtA, txtB])
#         btn_done.click(done_btn_click, outputs=[status])

#         app = demo.launch(inline=True, prevent_thread_lock=True)
#         return app

def launch_point_matcher(image_left, image_right, gap=40, on_done=None, inline=True):
    """
    Gradio app: pick corresponding points on two images and visualize connections.
    - image_left, image_right: PIL.Image or numpy arrays (H,W,[C])
    - gap: pixels between images on the stitched correspondence view
    - on_done: optional callback `on_done(points_A, points_B)` called when user clicks "Done"
    - inline: show Gradio inline (True for notebooks)
    """
    import gradio as gr
    from PIL import Image, ImageDraw
    import numpy as np
    import threading, json

    try:
        # Optional: for notebook fallback write-back
        from IPython import get_ipython
    except Exception:  # running outside IPython
        get_ipython = lambda: None

    # ---------- helpers ----------
    def _to_pil_from_numpy(arr: np.ndarray) -> Image.Image:
        arr = np.asarray(arr)
        # CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.ndim == 3 and arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        if arr.ndim == 3 and arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA")
        if arr.ndim == 3 and arr.shape[2] == 1:
            return Image.fromarray(arr[:, :, 0], mode="L")
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    def _to_pil(img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, np.ndarray):
            return _to_pil_from_numpy(img).convert("RGB")
        raise gr.Error("Please pass PIL images or NumPy arrays.")

    def _draw_points(base_img: Image.Image, pts, radius=5, color=(255, 0, 0)):
        img = base_img.copy().convert("RGB")
        d = ImageDraw.Draw(img)
        for (x, y) in pts:
            d.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=2)
            d.line([(x - 8, y), (x + 8, y)], fill=color, width=2)
            d.line([(x, y - 8), (x, y + 8)], fill=color, width=2)
        return img

    def _stitch_and_draw_lines(imgL: Image.Image, imgR: Image.Image, ptsL, ptsR, gap: int):
        """Return a stitched (H, W_L + gap + W_R) image with lines between min(lenL, lenR) pairs."""
        Wl, Hl = imgL.size
        Wr, Hr = imgR.size
        H = max(Hl, Hr)
        W = Wl + gap + Wr
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        canvas.paste(imgL, (0, 0))
        canvas.paste(imgR, (Wl + gap, 0))
        d = ImageDraw.Draw(canvas)

        n = min(len(ptsL), len(ptsR))
        for i in range(n):
            (xl, yl) = ptsL[i]
            (xr, yr) = ptsR[i]
            # draw endpoints
            d.ellipse([xl - 4, yl - 4, xl + 4, yl + 4], outline=(255, 0, 0), width=2)
            d.ellipse([Wl + gap + xr - 4, yr - 4, Wl + gap + xr + 4, yr + 4], outline=(0, 0, 255), width=2)
            # line
            d.line([(xl, yl), (Wl + gap + xr, yr)], fill=(0, 128, 255), width=3)
            # index labels
            d.text((xl + 6, yl - 14), str(i), fill=(0, 0, 0))
            d.text((Wl + gap + xr + 6, yr - 14), str(i), fill=(0, 0, 0))
        return canvas

    # ---------- state ----------
    pts_left = []
    pts_right = []
    app = None

    baseL = _to_pil(image_left)
    baseR = _to_pil(image_right)

    def _refresh_left():
        return np.array(_draw_points(baseL, pts_left, color=(255, 0, 0)))

    def _refresh_right():
        return np.array(_draw_points(baseR, pts_right, color=(0, 0, 255)))

    def _refresh_pairs():
        stitched = _stitch_and_draw_lines(baseL, baseR, pts_left, pts_right, gap=gap)
        return np.array(stitched)

    # ---------- click handlers ----------
    def _parse_xy(evt, base_img):
        x, y = None, None
        if hasattr(evt, "index") and evt.index is not None:
            try:
                x, y = evt.index
            except Exception:
                pass
        if (x is None or y is None) and hasattr(evt, "x") and hasattr(evt, "y"):
            x, y = evt.x, evt.y
        if x is None or y is None:
            return None, None
        W, H = base_img.size
        x = int(max(0, min(W - 1, x))); y = int(max(0, min(H - 1, y)))
        return x, y

    def on_click_left(evt: gr.SelectData):
        x, y = _parse_xy(evt, baseL)
        if x is None:
            return gr.update(), gr.update(), json.dumps(pts_left), json.dumps(pts_right)
        pts_left.append([x, y])
        return _refresh_left(), _refresh_pairs(), json.dumps(pts_left), json.dumps(pts_right)

    def on_click_right(evt: gr.SelectData):
        x, y = _parse_xy(evt, baseR)
        if x is None:
            return gr.update(), gr.update(), json.dumps(pts_left), json.dumps(pts_right)
        pts_right.append([x, y])
        return _refresh_right(), _refresh_pairs(), json.dumps(pts_left), json.dumps(pts_right)

    # ---------- buttons ----------
    def undo_left():
        if pts_left: pts_left.pop()
        return _refresh_left(), _refresh_pairs(), json.dumps(pts_left), json.dumps(pts_right)

    def undo_right():
        if pts_right: pts_right.pop()
        return _refresh_right(), _refresh_pairs(), json.dumps(pts_left), json.dumps(pts_right)

    def clear_left():
        pts_left.clear()
        return _refresh_left(), _refresh_pairs(), json.dumps(pts_left), json.dumps(pts_right)

    def clear_right():
        pts_right.clear()
        return _refresh_right(), _refresh_pairs(), json.dumps(pts_left), json.dumps(pts_right)

    def clear_both():
        pts_left.clear(); pts_right.clear()
        return _refresh_left(), _refresh_right(), _refresh_pairs(), "[]", "[]"

    def done_btn_click():
        # 1) Send to caller via callback (preferred)
        if callable(on_done):
            try:
                on_done([list(p) for p in pts_left], [list(p) for p in pts_right])
            except Exception as e:
                print(f"[launch_point_matcher] on_done callback error: {e}")

        # 2) Also write to IPython namespace (backward compatibility in notebooks)
        try:
            ip = get_ipython()
            if ip is not None:
                ip.user_ns['selected_points_A'] = [list(p) for p in pts_left]
                ip.user_ns['selected_points_B'] = [list(p) for p in pts_right]
        except Exception:
            pass

        msg = f"✅ Saved A:{len(pts_left)} and B:{len(pts_right)} points. Closing…"
        threading.Thread(target=lambda: app.close(), daemon=True).start()
        return msg

    # ---------- UI ----------
    with gr.Blocks(title="Two-Image Correspondence Picker") as demo:
        gr.Markdown("**Click points on each image.** Lines connect pairs by index: 0↔0, 1↔1, … (min length).")
        with gr.Row():
            with gr.Column():
                imgL = gr.Image(value=np.array(baseL), label="Image A (click to add)", type="numpy", interactive=True, sources=[])
                with gr.Row():
                    btn_undo_L = gr.Button("↩️ Undo A")
                    btn_clear_L = gr.Button("🧹 Clear A")
                txtA = gr.Textbox(label="Points A (JSON)", value="[]", interactive=False)
            with gr.Column():
                imgR = gr.Image(value=np.array(baseR), label="Image B (click to add)", type="numpy", interactive=True, sources=[])
                with gr.Row():
                    btn_undo_R = gr.Button("↩️ Undo B")
                    btn_clear_R = gr.Button("🧹 Clear B")
                txtB = gr.Textbox(label="Points B (JSON)", value="[]", interactive=False)

        gr.Markdown("### Correspondences")
        pair_view = gr.Image(value=_refresh_pairs(), label="Pairs (A | B with lines)", interactive=False)

        with gr.Row():
            btn_clear_both = gr.Button("🧼 Clear Both")
            btn_done = gr.Button("✅ Done", variant="primary")
        status = gr.Markdown()

        # events
        imgL.select(on_click_left, inputs=None, outputs=[imgL, pair_view, txtA, txtB])
        imgR.select(on_click_right, inputs=None, outputs=[imgR, pair_view, txtA, txtB])

        btn_undo_L.click(undo_left, outputs=[imgL, pair_view, txtA, txtB])
        btn_undo_R.click(undo_right, outputs=[imgR, pair_view, txtA, txtB])

        btn_clear_L.click(clear_left, outputs=[imgL, pair_view, txtA, txtB])
        btn_clear_R.click(clear_right, outputs=[imgR, pair_view, txtA, txtB])

        btn_clear_both.click(clear_both, outputs=[imgL, imgR, pair_view, txtA, txtB])
        btn_done.click(done_btn_click, outputs=[status])

        app = demo.launch(inline=inline, prevent_thread_lock=True)
        return app



