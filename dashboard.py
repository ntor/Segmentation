#!/usr/bin/env python3

import torch
import matplotlib
import numpy as np
import os.path
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ClassFiles.ChanVese import ChanVese
from ClassFiles.DeepSegmentation import DeepSegmentation
import ClassFiles.Networks as net

# import natsort


# Make matplotlib use the tk backend, so we can plot into the GUI
matplotlib.use("TkAgg")


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def digit_check(win, values, element_id):
    if not (values[element_id].isdigit() or values[element_id] == ""):
        win.Element(element_id).Update("")
        sg.popup_error("Invalid input. Enter an integer number of steps.")
    return values[element_id].isdigit()


# --- START LAYOUT DEFINITION ---

layout_image_selection = [
    [
        sg.In(
            size=(25, 1),
            default_text="Select image...",
            enable_events=True,
            readonly=True,
            key="_IMAGE_IN_",
        ),
        sg.FileBrowse(key="_IMAGE_BROWSE_"),
    ],
]

layout_segmentation_options = [
    [
        sg.Text("Threshold"),
        sg.Slider(
            range=(0.1, 0.9),
            size=(20, 10),
            default_value=0.5,
            resolution=0.1,
            tick_interval=0.2,
            orientation="h",
            enable_events=True,
            key="_THRESHOLD_SLIDER_",
        ),
    ],
    [
        sg.Text("Initialisation"),
        sg.Combo(
            ["Random", "Pixel Intensity"],
            size=(12, 1),
            default_value="Random",
            readonly=True,
        ),
        sg.Button("Initialise", size=(8, 1), font="Helvetica 12", key="_INIT_BUTTON_"),
    ],
]

layout_chan_vese = [
    [
        sg.Text("λ"),
        sg.Slider(
            range=(-3, 3),
            size=(15, 7),
            default_value=0.0,
            disable_number_display=True,
            enable_events=True,
            orientation="h",
            key="_CV_LAMBDA_SLIDER_",
        ),
        sg.Text("1", size=(10, 1), key="_CV_LAMBDA_OUT_"),
    ],
    [
        sg.Text("ε"),
        sg.Slider(
            range=(-4, 0),
            size=(15, 7),
            default_value=-1.0,
            disable_number_display=True,
            enable_events=True,
            orientation="h",
            key="_CV_EPSILON_SLIDER_",
        ),
        sg.Text("0.1", size=(10, 1), key="_CV_EPSILON_OUT_"),
    ],
    [
        sg.Text("Steps"),
        sg.In("40", size=(5, 1), enable_events=True, key="_CV_STEPS_"),
        sg.Button("Run", key="_CV_RUN_BUTTON_"),
    ],
]

layout_deep_segmentation = [
    [
        sg.Text("λ"),
        sg.Slider(
            range=(0, 30),
            size=(15, 7),
            default_value=14,
            disable_number_display=True,
            enable_events=True,
            orientation="h",
            key="_DS_LAMBDA_SLIDER_",
        ),
        sg.Text("14", size=(10, 1), key="_DS_LAMBDA_OUT_"),
    ],
    [
        sg.Text("ε"),
        sg.Slider(
            range=(-4.0, 0),
            size=(15, 7),
            default_value=-1.0,
            disable_number_display=True,
            enable_events=True,
            orientation="h",
            key="_DS_EPSILON_SLIDER_",
        ),
        sg.Text("0.1", size=(10, 1), key="_DS_EPSILON_OUT_"),
    ],
    [
        sg.Text("Steps"),
        sg.In("40", size=(5, 1), enable_events=True, key="_DS_STEPS_"),
        sg.Button("Run", key="_DS_RUN_BUTTON_"),
    ],
]

layout_controls = layout_image_selection + [
    [sg.Frame("Segmentation", layout_segmentation_options)],
    [sg.Frame("Chan-Vese", layout_chan_vese)],
    [sg.Frame("Deep Segmentation", layout_deep_segmentation)],
]

layout_plots = [
    [sg.Canvas(size=(30, 30), key="_PLOT_CANVAS_")],
    [
        sg.Text("Animation sleep time (ms)"),
        sg.In("100", size=(5, 1), enable_events=True, key="_ANIMATION_SLEEP_"),
    ],
]

# Full layout
layout = [
    [
        sg.Column(layout_controls),
        sg.VSeperator(),
        sg.Column(layout_plots),
    ]
]


# --- WINDOW INITIALISATION ---
window = sg.Window("Dashboard", layout, finalize=True, font="Typewriter 14")


# --- FIGURE INITIALISATION ---
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, aspect="equal")
ax.axis("off")

# add plot to the window
fig_agg = draw_figure(window["_PLOT_CANVAS_"].TKCanvas, fig)


# --- SEGMENTATION VARIABLES ---
seg_function = None  # the 'u' function
image = None  # Pillow image to be segmented
contour = None  # object for contour plot
seg_object = None  # ChanVese or DeepSegmentation instance
seg_threshold = 0.5
ani = None
cv_lambda = 1
cv_epsilon = 0.1
cv_steps = 40
ds_lambda = 14
ds_epsilon = 0.1
ds_steps = 40
animation_sleep = 100
steps_left = 0

# --- NETWORK INITIALISATION ---
NN = net.ConvNet8(1, 128, 128)
NN.load_state_dict(torch.load("./Neural_Networks_lunglike/ConvNet8_trained_v2", map_location=torch.device('cpu')))


def draw_contour(contour):
    if contour is not None:
        for c in contour.collections:
            c.remove()
    return ax.contour(seg_function, [seg_threshold], colors="red", linewidths=1)


def cv_init():
    print("Init.")
    global seg_function, contour, seg_object, steps_left
    seg_object = ChanVese(
        image, u_init=seg_function, segmentation_threshold=seg_threshold
    )
    if seg_function is None:
        seg_function = seg_object.u
    steps_left = cv_steps
    return (contour,)


def cv_animate(i):
    global contour, seg_function, steps_left
    print(f"Step {i}")
    seg_object.update_c()
    seg_object.single_step(cv_lambda, cv_epsilon)
    seg_function = seg_object.u
    steps_left -= 1
    contour = draw_contour(contour)
    return (contour,)


def ds_init():
    print("Init.")
    global seg_function, contour, seg_object, steps_left
    seg_object = DeepSegmentation(
        image, NN, u_init=seg_function, segmentation_threshold=seg_threshold
    )
    if seg_function is None:
        seg_function = seg_object.u.numpy()
    steps_left = ds_steps
    return (contour,)


def ds_animate(i):
    global contour, seg_function, steps_left
    print(f"Step {i}")
    seg_object.update_c()
    seg_object.single_step(ds_lambda, ds_epsilon)
    seg_function = seg_object.u.numpy()
    contour = draw_contour(contour)
    steps_left -= 1
    return (contour,)


# --- EVENT LOOP ---
while True:
    event, values = window.Read()
    if event in (None, "Exit"):
        break

    if event == "_IMAGE_IN_":
        filename = values["_IMAGE_IN_"]
        if os.path.isfile(filename) and filename.endswith((".png", ".jpg")):
            image = Image.open(filename).convert("L")
            im_arr = np.array(image, dtype=float) / 255
            ax.imshow(im_arr, cmap="gray", vmin=0, vmax=1)
            fig_agg.draw()
        else:
            sg.popup_error("Invalid file name. Please select a valid image file.")
    elif event == "_THRESHOLD_SLIDER_":
        seg_threshold = values["_THRESHOLD_SLIDER_"]
        if seg_object is not None:
            seg_object.segmentation_threshold = seg_threshold
        if contour is not None:
            contour = draw_contour(contour)
            fig_agg.draw()
    elif event == "_INIT_BUTTON_":
        if image is not None:
            seg_function = np.random.random(np.array(image).shape)
            contour = draw_contour(contour)
            fig_agg.draw()
    elif event == "_CV_LAMBDA_SLIDER_":
        cv_lambda = 10 ** int(values["_CV_LAMBDA_SLIDER_"])
        window.Element("_CV_LAMBDA_OUT_").Update(cv_lambda)
    elif event == "_CV_EPSILON_SLIDER_":
        cv_epsilon = 10 ** int(values["_CV_EPSILON_SLIDER_"])
        window.Element("_CV_EPSILON_OUT_").Update(cv_epsilon)
    elif event == "_CV_STEPS_":
        if digit_check(window, values, "_CV_STEPS_"):
            cv_steps = int(values["_CV_STEPS_"])
    elif event == "_DS_LAMBDA_SLIDER_":
        ds_lambda = int(values["_DS_LAMBDA_SLIDER_"])
        window.Element("_DS_LAMBDA_OUT_").Update(ds_lambda)
    elif event == "_DS_EPSILON_SLIDER_":
        ds_epsilon = 10 ** int(values["_DS_EPSILON_SLIDER_"])
        window.Element("_DS_EPSILON_OUT_").Update(ds_epsilon)
    elif event == "_DS_STEPS_":
        if digit_check(window, values, "_DS_STEPS_"):
            ds_steps = int(values["_DS_STEPS_"])
    elif event == "_ANIMATION_SLEEP_":
        if digit_check(window, values, "_ANIMATION_SLEEP_"):
            animation_sleep = int(values["_ANIMATION_SLEEP_"])
    elif event == "_CV_RUN_BUTTON_":
        if steps_left <= 0:
            ani = animation.FuncAnimation(
                fig,
                cv_animate,
                frames=cv_steps,
                interval=animation_sleep,
                repeat=False,
                blit=False,
                init_func=cv_init,
            )
            fig_agg.draw()
        else:
            sg.popup_error(f"Still running, {steps_left} steps left.")
    elif event == "_DS_RUN_BUTTON_":
        if steps_left <= 0:
            ani = animation.FuncAnimation(
                fig,
                ds_animate,
                frames=ds_steps,
                interval=animation_sleep,
                repeat=False,
                blit=False,
                init_func=ds_init,
            )
            fig_agg.draw()
        else:
            sg.popup_error(f"Still running, {steps_left} steps left.")


window.Close()
