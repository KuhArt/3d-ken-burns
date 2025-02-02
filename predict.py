""" Cog-based inference for animating still image (3D Ken Burns effect)"""
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import tempfile
import warnings

from cog import BasePredictor, Input, Path

warnings.filterwarnings("ignore")

#!/usr/bin/env python

import base64
import getopt
import glob
import io
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import urllib
import zipfile

import cupy
import cv2
import flask
import gevent
import gevent.pywsgi
import h5py
import moviepy
import moviepy.editor
import numpy
import scipy
import scipy.io
import torch
import torchvision

##########################################################

assert (
    int(str("").join(torch.__version__.split(".")[0:2])) >= 12
)  # requires at least pytorch version 1.2.0

torch.set_grad_enabled(
    False
)  # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = (
    True  # make sure to use cudnn for computational performance
)

##########################################################

objCommon = {}

exec(open("./common.py", "r").read())

exec(open("./models/disparity-estimation.py", "r").read())
exec(open("./models/disparity-adjustment.py", "r").read())
exec(open("./models/disparity-refinement.py", "r").read())
exec(open("./models/pointcloud-inpainting.py", "r").read())

##########################################################


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(
        self,
        image: Path = Input(description="Input image", default="default"),
        duration: str = Input(description="need be setted false", default="3.0"),
        direction: str = Input(description="need be setted false", default="zoom-in"),
        isWarmup: str = Input(description="need be setted false", default="false"),
    ) -> Path:
        """Run a single prediction on the model"""
        
        if isWarmup=="true":
          return

        npyImage = cv2.imread(filename=str(image), flags=cv2.IMREAD_COLOR)
        intWidth = npyImage.shape[1]
        intHeight = npyImage.shape[0]

        fltRatio = float(intWidth) / float(intHeight)

        intWidth = min(int(1024 * fltRatio), 1024)
        intHeight = min(int(1024 / fltRatio), 1024)

        npyImage = cv2.resize(
            src=npyImage,
            dsize=(intWidth, intHeight),
            fx=0.0,
            fy=0.0,
            interpolation=cv2.INTER_AREA,
        )

        process_load(npyImage, {})

        # objFrom = {
        #     "fltCenterU": intWidth / 2.0,
        #     "fltCenterV": intHeight / 2.0,
        #     "intCropWidth": int(math.floor(0.97 * intWidth)),
        #     "intCropHeight": int(math.floor(0.97 * intHeight)),
        # }
        # print(objFrom)


        # objTo = process_autozoom(
        #     {"fltShift": 100.0, "fltZoom": 1.25, "objFrom": objFrom}
        # )
        # print(objTo)

        shiftDiff = 32
        halfWidth = intWidth / 2
        halfHeight = intHeight / 2
        defaultCropDiff = 1
        defaultCroppedWidth = intWidth - defaultCropDiff
        defaultCroppedHeight = intHeight - defaultCropDiff

        initZoomCrop = 34 
        initZoomCroppedWidth = intWidth - initZoomCrop
        initZoomCroppedHeight = intHeight - initZoomCrop

        resZoomCrop = 384 
        resZoomCroppedWidth = intWidth - resZoomCrop
        resZoomCroppedHeight = intHeight - resZoomCrop

        directions = {
          'top-bottom': {
            'objFrom': {'fltCenterU': halfWidth, 'fltCenterV': halfHeight - shiftDiff, 'intCropWidth': defaultCroppedWidth, 'intCropHeight': defaultCroppedHeight},
            'objTo': {'fltCenterU': halfWidth, 'fltCenterV': halfHeight + shiftDiff, 'intCropWidth': defaultCroppedWidth, 'intCropHeight': defaultCroppedHeight}
          },
          'bottom-top': {
            'objFrom': {'fltCenterU': halfWidth, 'fltCenterV':  halfHeight + shiftDiff, 'intCropWidth': defaultCroppedWidth, 'intCropHeight': defaultCroppedHeight},
            'objTo': {'fltCenterU': halfWidth, 'fltCenterV': halfHeight - shiftDiff, 'intCropWidth': defaultCroppedWidth, 'intCropHeight': defaultCroppedHeight}
          },         
          'left-right': {
            'objFrom': {'fltCenterU': halfWidth - shiftDiff, 'fltCenterV': halfHeight, 'intCropWidth': defaultCroppedWidth, 'intCropHeight': defaultCroppedHeight},
            'objTo': {'fltCenterU': halfWidth + shiftDiff, 'fltCenterV': halfHeight, 'intCropWidth': defaultCroppedWidth, 'intCropHeight': defaultCroppedHeight}
          },
          'right-left': {
            'objFrom': {'fltCenterU': halfWidth + shiftDiff, 'fltCenterV': halfHeight, 'intCropWidth': defaultCroppedWidth, 'intCropHeight': defaultCroppedHeight},
            'objTo': {'fltCenterU': halfWidth - shiftDiff, 'fltCenterV': halfHeight, 'intCropWidth': defaultCroppedWidth, 'intCropHeight': defaultCroppedHeight}
          },
          'zoom-in': {
            'objFrom': {'fltCenterU': halfWidth, 'fltCenterV': halfHeight, 'intCropWidth': initZoomCroppedWidth, 'intCropHeight': initZoomCroppedHeight},
            'objTo': {'fltCenterU': halfWidth, 'fltCenterV': halfHeight, 'intCropWidth': resZoomCroppedWidth, 'intCropHeight': resZoomCroppedHeight}
          },
          'zoom-out': {
            'objFrom': {'fltCenterU': halfWidth, 'fltCenterV': halfHeight, 'intCropWidth': resZoomCroppedWidth, 'intCropHeight': resZoomCroppedHeight},
            'objTo': {'fltCenterU': halfWidth, 'fltCenterV': halfHeight, 'intCropWidth': initZoomCroppedWidth, 'intCropHeight': initZoomCroppedHeight}
          }
        }

        frame_rate = 25
        npyResult = process_kenburns(
            {
                "fltSteps": numpy.linspace(0.0, 1.0, int(float(duration) * frame_rate)).tolist(),
                "objFrom": directions[direction]['objFrom'],
                "objTo": directions[direction]['objTo'],
                "boolInpaint": True,
            }
        )

        output_path = "output.mp4"
        moviepy.editor.ImageSequenceClip(
            sequence=[
                npyFrame[:, :, ::-1]
                for npyFrame in npyResult
            ],
            fps=frame_rate,
        ).write_videofile(str(output_path))
        return Path("output.mp4")

    # end