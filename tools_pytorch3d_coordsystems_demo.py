## tools_pytorch3d_coordsystems.py

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



# ---------- pretty print helpers ----------
RESET="\033[0m"; BOLD="\033[1m"
C={"ok":"\033[1;32m","info":"\033[1;36m","step":"\033[1;35m","warn":"\033[1;33m"}
CYAN  = "\033[1;36m"; GREEN = "\033[1;32m"; YELLOW = "\033[1;33m"


def say(kind,msg): print(f"{C[kind]}{msg}{RESET}")
torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


# Overlay world & camera axes on a rendered image (PyTorch3D)
import torch
import numpy as np
import matplotlib.pyplot as plt

def overlay_axes_p3d(
    rgb: np.ndarray,
    cameras,
    height: int,
    width: int,
    world_origin=(0.0, 0.0, 0.0),
    axis_len=0.5,
    draw_world_axes=True,
    draw_camera_axes=True,
    cam_axis_len=0.5,
    title=None,
):
    """
    Overlay +X (red), +Y (green), +Z (blue) axes on top of an image.

    Args:
      rgb: (H,W,3) float image in [0,1] you rendered with `cameras`.
      cameras: a PyTorch3D Cameras object (PerspectiveCameras or FoVPerspectiveCameras).
      height, width: image size used for rendering.
      world_origin: 3-tuple; where to anchor the world axes in world coordinates.
      axis_len: length (in world units) of each world-axis arm.
      draw_world_axes: if True, draw axes at `world_origin`.
      draw_camera_axes: if True, draw axes at the camera center (projected).
      cam_axis_len: length (in world units) for camera axes.
      title: optional figure title.

    Notes:
      - Uses `transform_points_screen` so (u, v) are image pixels (v goes down).
      - Works for both in_ndc=True (FoV) and in_ndc=False (pixel intrinsics).
    """
    device = cameras.R.device
    dtype  = cameras.R.dtype

    H, W = int(height), int(width)
    imgsz = torch.tensor([[H, W]], device=device)

    plt.figure(figsize=(6,6))
    plt.imshow(rgb)
    plt.axis("off")

    # --- World axes at `world_origin` ---
    if draw_world_axes:
        ox, oy, oz = map(float, world_origin)
        L = float(axis_len)
        pts_world = torch.tensor([
            [ox, oy, oz], [ox+L, oy,   oz],   # +X (red)
            [ox, oy, oz], [ox,   oy+L, oz],   # +Y (green)
            [ox, oy, oz], [ox,   oy,   oz+L], # +Z (blue)
        ], device=device, dtype=dtype)[None]  # (1,6,3)

        uvz = cameras.transform_points_screen(pts_world, image_size=imgsz)[0].detach().cpu().numpy()
        (u0,v0),(ux,vx),(_,_),(uy,vy),(__,__),(uz,vz) = uvz[:6,0:2]

        plt.plot([u0, ux], [v0, vx], '-', lw=3, color='red',   label='+X_world')
        plt.plot([u0, uy], [v0, vy], '-', lw=3, color='green', label='+Y_world')
        plt.plot([u0, uz], [v0, vz], '-', lw=3, color='blue',  label='+Z_world')

    # --- Camera axes (at camera center, directions expressed in WORLD coords) ---
    if draw_camera_axes:
        # PyTorch3D row-vector convention:
        # Camera center in world: Cw = -T @ R    (with T as row vector)
        R = cameras.R[0]            # (3,3)
        T = cameras.T[0]            # (3,)
        Cw = -T @ R                 # (3,)

        Lc = float(cam_axis_len)
        # Camera axes in WORLD coords are the COLUMNS of R
        x_cam_w = R[:, 0]   # direction of +X_cam in world
        y_cam_w = R[:, 1]   # direction of +Y_cam in world
        z_cam_w = R[:, 2]   # direction of +Z_cam in world

        cam_pts = torch.stack([
            Cw, Cw + Lc * x_cam_w,   # +X_cam (red)
            Cw, Cw + Lc * y_cam_w,   # +Y_cam (green)
            Cw, Cw + Lc * z_cam_w,   # +Z_cam (blue)
        ], dim=0).to(device=device, dtype=dtype)[None]  # (1,6,3)

        uvz_cam = cameras.transform_points_screen(cam_pts, image_size=imgsz)[0].detach().cpu().numpy()
        (u0,v0),(ux,vx),(_,_),(uy,vy),(__,__),(uz,vz) = uvz_cam[:6,0:2]

        # Use dashed lines to distinguish camera axes from world axes
        plt.plot([u0, ux], [v0, vx], '--', lw=2, color='red',   label='+X_cam')
        plt.plot([u0, uy], [v0, vy], '--', lw=2, color='green', label='+Y_cam')
        plt.plot([u0, uz], [v0, vz], '--', lw=2, color='blue',  label='+Z_cam')
        # camera center marker
        plt.scatter([u0],[v0], c='white', s=20, edgecolor='k', zorder=5)

    if draw_world_axes or draw_camera_axes:
        plt.legend(loc='lower right', frameon=True)

    if title:
        plt.title(title)
    plt.show()


# ---------------------- Example usage ----------------------
# Assumes you already have:
#   - `rgb` : (H,W,3) numpy array in [0,1]
#   - `cameras` : a PyTorch3D Cameras object used for the render
#   - `height`, `width` : image size used for the render
# Call:

# rgb = images[0, ..., :3].cpu().numpy()

# overlay_axes_p3d(rgb, cameras, 512, 512,
#                  world_origin=(0,0,0), axis_len=0.5,
#                  draw_world_axes=True, draw_camera_axes=False,
#                  cam_axis_len=0.5,
#                  title="Axes overlay")


# ---------- download cow mesh with feedback ----------
import os, subprocess

def download_cow_mesh():
    say("step", "Creating data/cow_mesh directory...")
    os.makedirs("data/cow_mesh", exist_ok=True)

    urls = [
        "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj",
        "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl",
        "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png",
    ]

    for url in urls:
        fname = os.path.basename(url)
        out = os.path.join("data/cow_mesh", fname)
        if not os.path.exists(out):
            say("info", f"Downloading {fname} ...")
            subprocess.run(["wget", "-q", "-O", out, url], check=True)
            say("ok", f"Saved → {out}")
        else:
            say("warn", f"{fname} already exists, skipping.")

    say("ok", "Cow mesh download complete!")


def print_spherical_coords(distance, elev, azim):
    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)

    print(
        "\n\033[95m" + "═"*38 + "\033[0m\n"
        "\033[95m\033[0m \033[1m🧭 Spherical camera pose\033[0m  \033[95m\033[0m\n"
        "\033[95m\033[0m                          \033[95m\033[0m\n"
        f"\033[95m\033[0m   dist = {distance:.3f} \033[95m\033[0m\n"
        f"\033[95m\033[0m   elev = {elev:.3f}°    \033[95m\033[0m\n"
        f"\033[95m\033[0m   azim = {azim:.3f}°    \033[95m\033[0m\n"
        "\033[95m" + "═"*38 + "\033[0m\n"
    )


def print_camera_pose_matrices(R, T, title=""):
    with np.printoptions(precision=6, suppress=True, floatmode="fixed"):
        # Let's print the camera pose matrices to check the view pose
        print(f"\n{BOLD}📷 {title}:{RESET}")
        print(f"\n{CYAN}Camera Extrinsics (world→view, row-vector):{RESET}")
        print(f"{BOLD}  R (1,3,3):{RESET}\n{R}")
        print(f"{BOLD}  T (1,3):  {RESET}{T}")

        # Camera axes in WORLD (columns of R)
        R0 = R[0]; T0 = T[0]
        x_cam_w, y_cam_w, z_cam_w = R0[:,0], R0[:,1], R0[:,2]
        print("\n")
        print(f"{BOLD}+X_cam in world:{RESET} {x_cam_w}")
        print(f"{BOLD}+Y_cam in world:{RESET} {y_cam_w}")
        print(f"{BOLD}+Z_cam in world:{RESET} {z_cam_w}")

        # # Camera center in world (row-vector convention): Cw = -T @ R.T
        # Cw = -T0 @ R0.T
        # print(f"\n{BOLD}📍 Camera center (world):{RESET} {np.asarray(Cw.detach().cpu().numpy())}")



def make_phong_renderer(W, H, device):
    raster_settings = RasterizationSettings(
        image_size=(W, H),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    # No cameras/lights bound here; we’ll pass them at render time.
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(device=device)
    )
    return renderer

def set_renderer_state(renderer, *, cameras=None, lights=None):
    if cameras is not None:
        renderer.rasterizer.cameras = cameras
        renderer.shader.cameras     = cameras
    if lights is not None:
        renderer.shader.lights      = lights



# A renderer in PyTorch3D is composed of a rasterizer and a shader which each 
# have a number of subcomponents such as a camera (orthographic/perspective). 
def create_phong_renderer(cameras, W, H, device, light_location=[[0.0, 0.0, -3.0]]):
    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong


    # Define the settings for rasterization and shading. As we are rendering 
    # images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=(W,H),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device, location=light_location)

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    return renderer


def render_image_with_opencv_lines_opencv_coordinates(mesh, R_cv, t_cv, K, W, H):

    # Simple Lambert per-face grayscale based on camera-space normals
    def face_normals_cam(Xc, faces):
        v0 = Xc[faces[:,0]]
        v1 = Xc[faces[:,1]]
        v2 = Xc[faces[:,2]]
        n  = np.cross(v1 - v0, v2 - v0)          # (F,3)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        return n / n_norm


    # This part extracts the vertices and faces to create a simple 
    # rendering using OpenCV and matplotlib

    # Extract verts/faces from mesh for OpenCV pass
    verts_t = mesh.verts_list()[0]        # (V,3) torch
    faces_t = mesh.faces_list()[0].long() # (F,3) torch
    verts_np = verts_t.detach().cpu().numpy()
    faces_np = faces_t.detach().cpu().numpy()


    # Convert from torch to numpy
    R_cv_np = R_cv.detach().cpu().numpy()[0]
    t_cv_np = t_cv.detach().cpu().numpy()[0]

    # Camera center in World coordinates
    C_world = -R_cv_np.T @ t_cv_np

    # Transform verts to camera (OpenCV) for z, normals, and sorting
    Xc = (R_cv_np @ verts_np.T + t_cv_np.reshape(3,1)).T  # (V,3)
    Z  = Xc[:, 2]

    # Back-face cull: keep triangles with all vertices in front (Z>0)
    faces_keep = []
    for f in faces_np:
        if np.all(Z[f] > 0):
            faces_keep.append(f)
    faces_keep = np.asarray(faces_keep, dtype=np.int64)
    if faces_keep.size == 0:  # no culling fallback
        faces_keep = faces_np

    # Depth sort (far -> near) by mean Z
    meanZ = Xc[faces_keep].mean(axis=1)[:, 2]
    order = np.argsort(meanZ)[::-1]
    faces_sorted = faces_keep[order]

    n_cam = face_normals_cam(Xc, faces_sorted)

    light_dir = np.array(C_world, dtype=np.float64)


    light_dir /= np.linalg.norm(light_dir) + 1e-12
    intensity = np.clip((n_cam @ light_dir), 0.0, 1.0)  # (F,)

    # Project vertices to pixels
    uv = project_cv(verts_np, R_cv_np, t_cv_np, K)  # (V,2)

    # Draw
    img_cv = np.ones((H, W, 3), dtype=np.uint8) * 255  # light gray bg
    for f, s in zip(faces_sorted, intensity):
        pts = np.round(uv[f]).astype(np.int32)
        col = int(60 + 180 * float(s))  # 60..240 grayscale
        cv2.fillConvexPoly(img_cv, pts, color=(0, col, col), lineType=cv2.LINE_AA)
        # Optional wireframe:
        cv2.polylines(img_cv, [pts], True, (0,150,150), 1, cv2.LINE_AA)

    # OpenCV image
    img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    return img_cv_rgb


# ------------------------- Helpers: P3D <-> OpenCV ----------------------------
def project_cv(pts3d, R_cv, t_cv, K, dist=None):
    """OpenCV projection (pts3d Nx3)."""
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R_cv.astype(np.float64))
    tvec = t_cv.reshape(3,1).astype(np.float64)
    uv, _ = cv2.projectPoints(pts3d.astype(np.float64), rvec, tvec, K, dist)
    return uv[:,0,:]  # (N,2)


# Color / marker by world-quadrant sign
def sign2(x): return "+" if x >= 0 else "−"

def color_for(x, y):
    # (+,+)=red, (−,+)=blue, (−,−)=green, (+,−)=orange, origin=white
    if x==0 and y==0: return "white"
    if x>0 and y>0:   return "red"
    if x<0 and y>0:   return "blue"
    if x<0 and y<0:   return "green"
    if x>0 and y<0:   return "orange"
    return "yellow"

def marker_for(x, y):
    if x==0 and y==0: return "o"
    return "o"



import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pytorch3d.renderer import look_at_view_transform

def plot_world_axes(ax, length=1.0):
    origin = torch.tensor([0.,0.,0.])
    dirs = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    colors=['r','g','b']
    names = ['+X_world','+Y_world','+Z_world']
    for d,c,n in zip(dirs,colors,names):
        ax.quiver(*origin, *d, length=length, color=c, linestyle='dashed')
        ax.text(*(origin + d*length*1.05), n, color=c)

def camera_center_from_RT_openCV(R_cv, T_cv):
    """
    R, T are (1,3,3) and (1,3) from look_at_view_transform.
    Returns camera center C in world coords: C = -R^T @ T
    """
    Rm = R_cv[0].cpu().numpy()
    tm = T_cv[0].cpu().numpy()
    C = -Rm.T @ tm  # (3,)
    return C

def camera_center_from_RT_PyTorch3D(R_p3d, T_p3d):
    """
    R, T are (1,3,3) and (1,3) from look_at_view_transform.
    Returns camera center C in world coords: C = -R^T @ T
    """
    Rm = R_p3d[0].cpu().numpy()
    tm = T_p3d[0].cpu().numpy()
    C = -tm @ Rm.T  # (3,)
    return C





def plot_camera_axes_and_ray(R, T, ax, label, axis_len=0.6, ray_style=dict(color='k', linestyle='--', linewidth=1.5)):
    """
    Plot camera axes at true camera center and the line-of-sight to origin.
    """
    import numpy as np
    Rm = R[0].cpu().numpy()
    tm = T[0].cpu().numpy()
    C  = -(Rm.T @ tm)   # camera center in world
    axes = Rm           # rows are +X_cam, +Y_cam, +Z_cam in world

    # Camera axes
    colors = ['r','g','b']
    names  = ['+X_cam','+Y_cam','+Z_cam']
    for i in range(3):
        ax.quiver(*C, *axes[i], length=axis_len, color=colors[i])
        ax.text(*(C + axes[i]*axis_len*1.15), names[i], color=colors[i])




    # Camera center
    ax.scatter(C[0], C[1], C[2], c='k', s=35, marker='o')
    ax.text(C[0], C[1], C[2], f'  C ({label})', color='k')

    # Line-of-sight from camera to origin
    O = np.array([0.0, 0.0, 0.0])
    ax.plot([C[0], O[0]], [C[1], O[1]], [C[2], O[2]], **ray_style)


import torch, numpy as np
import matplotlib.pyplot as plt
from pytorch3d.renderer import look_at_view_transform

def cam_center_from_RT(R, T):
    """Camera center in world: C = -R^T @ T  (batch size 1)"""
    Rm, tm = R[0].numpy(), T[0].numpy()
    return -(Rm.T @ tm)

def plot_orbit_with_cam_dirs(dist=3.0, elev=0.0, step_curve=5, step_arrows=45):
    azims_curve  = list(range(0, 361, step_curve))   # smooth curve
    azims_arrows = list(range(0, 361, step_arrows))  # arrow samples

    xs, zs = [], []
    for a in azims_curve:
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=float(a), device='cpu')
        C = cam_center_from_RT(R, T)
        xs.append(C[0]); zs.append(C[2])

    plt.figure(figsize=(6.5,6.5))
    # World axes
    plt.axhline(0, color='k', lw=0.6); plt.axvline(0, color='k', lw=0.6)
    # Orbit path
    plt.plot(xs, zs, '-', color='0.2', lw=1.5, label='Camera centers (elev=0)')

    # Arrows at coarse azims
    scale = dist * 0.25
    for a in azims_arrows:
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=float(a), device='cpu')
        C = cam_center_from_RT(R, T)
        Rm = R[0].numpy()

        # Image-right in world is -row0(R); +Z_cam forward is row2(R)
        v_right = -Rm[0]    # (x, y, z) world
        v_fwd   =  Rm[2]

        # Place arrows in X-Z plane (drop Y component)
        plt.arrow(C[0], C[2], scale*v_right[0], scale*v_right[2],
                  head_width=0.08*dist, length_includes_head=True, color='tab:orange')
        plt.arrow(C[0], C[2], scale*v_fwd[0],   scale*v_fwd[2],
                  head_width=0.08*dist, length_includes_head=True, color='tab:blue')
        # Label azimuth
        plt.text(C[0], C[2], f" {a}°", fontsize=8, va='center')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("+X_world →")
    plt.ylabel("+Z_world ↑")
    plt.title("Full 360° orbit — camera centers, image-right (orange), +Z_cam forward (blue)")
    # Simple legend patches
    import matplotlib.patches as mpatches
    plt.legend(handles=[
        mpatches.Patch(color='0.2', label='Orbit path'),
        mpatches.Patch(color='tab:orange', label='Image right (−X_cam)'),
        mpatches.Patch(color='tab:blue', label='+Z_cam (forward)')
    ], loc='upper right', frameon=False)

    plt.show()






