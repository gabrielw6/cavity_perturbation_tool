from vpython import *

##############################################################################
# Scene
##############################################################################

scene = canvas(
    title="Yee Cell (FDTD)",
    width=1000,
    height=700,
    background=color.white
)

scene.forward = vector(-1, -1, -1)
scene.up = vector(0, 0, 1)

##############################################################################
# Cube dimensions
##############################################################################

L = 2

##############################################################################
# Transparent Yee cell
##############################################################################

box(
    pos=vector(0,0,0),
    length=L,
    height=L,
    width=L,
    opacity=0.12,
    color=color.gray(0.8)
)

##############################################################################
# Coordinate axes
##############################################################################

arrow(
    pos=vector(-1.4,0,0),
    axis=vector(3,0,0),
    shaftwidth=0.03,
    color=color.red
)

label(
    pos=vector(1.8,0,0),
    text="x",
    box=False,
    color=color.red
)

arrow(
    pos=vector(0,-1.4,0),
    axis=vector(0,3,0),
    shaftwidth=0.03,
    color=color.green
)

label(
    pos=vector(0,1.8,0),
    text="y",
    box=False,
    color=color.green
)

arrow(
    pos=vector(0,0,-1.4),
    axis=vector(0,0,3),
    shaftwidth=0.03,
    color=color.blue
)

label(
    pos=vector(0,0,1.8),
    text="z",
    box=False,
    color=color.blue
)

##############################################################################
# Utility functions
##############################################################################

def e_arrow(pos, axis, text):
    arrow(
        pos=pos,
        axis=axis,
        shaftwidth=0.04,
        color=color.blue
    )

    label(
        pos=pos + axis*0.55,
        text=text,
        box=False,
        color=color.blue,
        height=12
    )


def h_arrow(pos, axis, text):
    arrow(
        pos=pos,
        axis=axis,
        shaftwidth=0.05,
        color=color.orange
    )

    label(
        pos=pos + axis*0.65,
        text=text,
        box=False,
        color=color.orange,
        height=12
    )

##############################################################################
# Electric field components
##############################################################################
#
# Ex lives on x-directed edges
#

for y in (-1,1):
    for z in (-1,1):

        e_arrow(
            vector(-1,y,z),
            vector(2,0,0),
            "Ex"
        )

##############################################################################
#
# Ey lives on y-directed edges
#

for x in (-1,1):
    for z in (-1,1):

        e_arrow(
            vector(x,-1,z),
            vector(0,2,0),
            "Ey"
        )

##############################################################################
#
# Ez lives on z-directed edges
#

for x in (-1,1):
    for y in (-1,1):

        e_arrow(
            vector(x,y,-1),
            vector(0,0,2),
            "Ez"
        )

##############################################################################
# Magnetic field components
##############################################################################
#
# Hx (center of yz faces)
#

h_arrow(
    vector(0,-0.35,-0.35),
    vector(0,0.7,0.7),
    "Hx"
)

##############################################################################

h_arrow(
    vector(0,0.35,0.35),
    vector(0,-0.7,-0.7),
    "Hx"
)

##############################################################################
#
# Hy (center of xz faces)
#

h_arrow(
    vector(-0.35,0,-0.35),
    vector(0.7,0,0.7),
    "Hy"
)

h_arrow(
    vector(0.35,0,0.35),
    vector(-0.7,0,-0.7),
    "Hy"
)

##############################################################################
#
# Hz (center of xy faces)
#

h_arrow(
    vector(-0.35,-0.35,0),
    vector(0.7,0.7,0),
    "Hz"
)

h_arrow(
    vector(0.35,0.35,0),
    vector(-0.7,-0.7,0),
    "Hz"
)

##############################################################################
# Caption
##############################################################################

scene.append_to_caption("""
Mouse controls

• Left button : rotate
• Right button: pan
• Mouse wheel : zoom

Blue arrows   : Electric field components (E)
Orange arrows : Magnetic field components (H)

""")

##############################################################################

while True:
    rate(60)