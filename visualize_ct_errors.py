import os
import json
import numpy as np
import pandas as pd
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = "/data/Predict-Pneumoperitoneum_LaB-GATr/ct_scans/pointclouds_iso"
FILENAME = "lab-gatr_results"


def add_markers(pts, size=2, opacity=.8, color="red", colorscale=None, showscale=False, cmin=None, cmax=None):
    return go.Scatter3d(
        x = pts[:, 0], y = pts[:, 1], z = -pts[:, 2], mode = "markers",
        marker = {
            "size": size,
            "opacity": opacity,
            "color": color,
            "colorscale": colorscale,
            "showscale": showscale,
            "cmin": cmin,
            "cmax": cmax
        })


def add_lines(pts_a, pts_b, width=2, color="black"):
    x, y, z = [], [], []
    for p, q in zip(pts_a, pts_b):
        x += [p[0], q[0], None]
        y += [p[1], q[1], None]
        z += [-p[2], -q[2], None]
    return go.Scatter3d(
        x = x, y = y, z = z, mode = "lines",
        line = {
            "color": color,
            "width": width
        })


def plots_ct_scans(key, depth_pts_a, depth_pts_b, ct_pts_a, ct_pts_b, depth_marks_a, depth_marks_b, ct_marks_a, ct_marks_b):
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]], horizontal_spacing=0.04)
    # Depth plot
    fig.add_trace(add_markers(depth_pts_a, size=2, opacity=1., color=-depth_pts_a[:, 2], colorscale="Reds"), row=1, col=1)
    fig.add_trace(add_markers(depth_pts_b, size=2, opacity=.5, color=-depth_pts_b[:, 2], colorscale="Greens"), row=1, col=1)
    # CT plot
    fig.add_trace(add_markers(ct_pts_a, size=2, opacity=1., color=-ct_pts_a[:, 2], colorscale="Reds"), row=1, col=2)
    fig.add_trace(add_markers(ct_pts_b, size=2, opacity=.5, color=-ct_pts_b[:, 2], colorscale="Greens"), row=1, col=2)
    # Landmarks plot
    fig.add_trace(add_markers(ct_pts_a, color=-ct_pts_a[:, 2], opacity=1., colorscale="Viridis"), row=1, col=3)
    fig.add_trace(add_lines(depth_marks_a, depth_marks_b, color="Blue"), row=1, col=3)
    fig.add_trace(add_lines(ct_marks_a, ct_marks_b, color="Red"), row=1, col=3)
    scene_axes = {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "zaxis": {"visible": False}}
    cam = {"eye": {"x": 1.2, "y": -1.6, "z": 1.2}}
    fig.update_layout(
        width=1200, height=400, template="plotly_white",
        scene=scene_axes, scene2=scene_axes, scene3=scene_axes,
        scene_camera=cam, scene2_camera=cam, scene3_camera=cam,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    # fig.show()
    fig.write_html(os.path.join(ROOT, "visualizations", f"{key}.html"))

if __name__ == "__main__":
    csv = pd.read_csv(os.path.join(ROOT, FILENAME + ".csv"))
    with open(os.path.join(ROOT, FILENAME + ".json"), "r") as f:
        jsn = json.load(f)
    
    print(f"rmse: {csv['rmse'].mean():.2f} ({csv['rmse'].std():.2f})")
    print(f"mae:  {csv['mae'].mean():.2f} ({csv['mae'].std():.2f})")
    print(f"magn: {csv['magn'].mean():.1f} ({csv['magn'].std():.1f})")
    print(f"ang:  {csv['ang'].mean():.1f} ({csv['ang'].std():.1f})")
    
    
    for key in jsn.keys():
        pt = jsn[key]
        print(f"({key}) rmse: {pt['rmse']:.2f}, mae: {pt['mae']:.2f}, " +
              f"magn: {pt['magn']:.1f}, ang: {pt['ang']:.1f}")
        
        ct_start = np.asarray(o3d.io.read_point_cloud(
            os.path.join(ROOT, key, "ct_abd_segm_pcd_clean.ply")
        ).points) * 100
        ct_pred = np.asarray(o3d.io.read_point_cloud(
            os.path.join(ROOT, key, "ct_pred_pcd.ply")
        ).points) * 100
        dpt_start = np.asarray(o3d.io.read_point_cloud(
            os.path.join(ROOT, key, "depth_pcd_start.ply")
        ).points) * 100
        dpt_end = np.asarray(o3d.io.read_point_cloud(
            os.path.join(ROOT, key, "depth_pcd_end.ply")
        ).points) * 100
        
        anns_start = np.array(pt["landmarks"]["anns_start"])
        anns_end = np.array(pt["landmarks"]["anns_end"])
        anns_pred = np.array(pt["landmarks"]["pred_pos"])
        anns_ct = anns_pred - np.array(pt["landmarks"]["pred_displ"])
    
        plots_ct_scans(
            key, dpt_start, dpt_end, ct_start, ct_pred,
            anns_start, anns_end, anns_ct, anns_pred
        )

