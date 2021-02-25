#!/usr/bin/env python3

from ClassFiles.dataGenerator import ShapeGenerator

shapes = ShapeGenerator(512, 512)
shapes.add_polygon(times=4)
shapes.add_ellipse(times=5)

shapes.add_noise()
shapes.image.show()
