#!/usr/bin/env python3

# img_viewer.py

import PySimpleGUI as sg
import os.path
import natsort
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def draw_figure(canvas, figure):
    # if canvas.children:
    #     for child in canvas.winfo_children():
    #         child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


matplotlib.use("TkAgg")


# --- LAYOUT DEFINITION ---
# First the window layout in 2 columns
file_list_column = [
    [
        sg.Text("Data Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [sg.Listbox(values=[], enable_events=True, size=(40, 5), key="-FILE LIST-")],
]

image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Canvas(key="-CANVAS-")],
]

# Full layout
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Segmentation Explor-inator 3000", layout)
fig, axs = plt.subplots(figsize=(3, 3))
fig_agg = None


# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            folder_list = natsort.natsorted(os.listdir(folder))
        except:
            folder_list = []

        fnames = [
            f
            for f in folder_list
            if os.path.isdir(os.path.join(folder, f))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            foldername = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            image_name = os.path.join(foldername, "dirty.png")
            seg_name = os.path.join(foldername, "dirty_cv_seg.npy")
            im = Image.open(image_name)
            seg = np.load(seg_name)
            axs.cla()
            axs.imshow(np.array(im), cmap="gray")
            axs.contour(np.clip(seg, 0.5, 1), colors="red", linewidths=0.25)
            if fig_agg is None:
                fig_agg = draw_figure(window["-CANVAS-"].TKCanvas, fig)
            else:
                fig_agg.draw()
            window.refresh()

        except:
            pass

window.close()
