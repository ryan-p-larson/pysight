#! /usr/bin/env python3
# i3.py -- I3wm controller


##################################################################################
## Libraries
import sys
from math import sqrt
from subprocess import check_output

import i3ipc
import numpy as np
from rtree import index
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree


##################################################################################
## I3 Functions
def get_ws_windows(conn):
    """ Returns a list of I3 window objects for the focused workspace. """
    return conn.get_tree().find_focused().workspace().leaves()

def get_ws_dimensions(conn):
    """ Returns a tuple containing width, height for the currently focused I3 workspace. """
    for ws in conn.get_outputs():
        if ws['active']:
            return (ws['rect']['width'], ws['rect']['height'])
    return (None, None)

def get_window_info(windows):
    """ Formats I3 windows into smaller data dicts. """
    def info(window):
        return {
            'id'      : window.id,
            'name'    : window.name,
            'floating': window.floating.endswith('on'),
            'x'       : window.rect.x,
            'y'       : window.rect.y,
            'width'   : window.rect.width,
            'height'  : window.rect.height,
            'cx'      : window.rect.x + (window.rect.width // 2),
            'cy'      : window.rect.y + (window.rect.height // 2)
        }
    return [info(w) for w in windows]

def set_window_focus(window_id, conn):
    """ Switches the focus to a given window via i3-msg. """
    conn.command('[id="{}"] focus'.format(window_id))


##################################################################################
## Intersections
def get_intersecting_windows(x, y, windows):
    """ Filters window list to those who contain x, y. """
    return [w for w in windows if (
            w['x'] <= x <= (w['x'] + w['width']) and
            w['y'] <= y <= (w['y'] + w['height']))]

def get_closest_window(x, y, windows):
    """ Returns the closest window to x, y. """
    def distance(x, y, window):
        dx   = abs(window['cx'] - x) ** 2
        dy   = abs(window['cy'] - y) ** 2
        dist = int(sqrt(dx + dy))
        return dist

    if len(windows) == 0:
        raise Exception('No windows')
    elif len(windows) == 1:
        return windows[0]
    else:
        closest_so_far, dist = None, float('inf')
        for w in windows:
            dw = distance(x, y, w)
            if dw < dist:
                closest_so_far, dist = w, dw
        return closest_so_far

def get_intersecting_kdtree(x, y, windows):
    centerX = lambda window: window.rect.x + (window.rect.width  // 2)
    centerY = lambda window: window.rect.y + (window.rect.height // 2)
    kd      = KDTree(np.array([(centerX(w), centerY(w)) for w in windows]))

    dist, closest_idx = kd.query([x, y])
    return windows[closest_idx]

def get_intersecting_rtree(x, y, windows):
    """ Returns a list of I3 window objects who intersect witht he point x, y.

    Args:
        x (int): Query x. 0 <= x <= Screen width
        y (int): Query y. 0 <= y <= Screen height
        windows (list): Possibly empty list of i3ipc.window connections

    Returns:
        list: Possibly empty list of intersecting windows.
    """
    top    = lambda window: window.rect.y
    right  = lambda window: window.rect.x + window.rect.width
    bottom = lambda window: window.rect.y + window.rect.height
    left   = lambda window: window.rect.x

    rtree  = index.Index()
    for idx, w in enumerate(windows):
        rtree.insert(idx, (left(w), top(w), right(w), bottom(w)))

    return [windows[idx] for idx in rtree.intersection((x, y))]

def get_intersecting_voronoi(x, y, windows):
    pass


##################################################################################
## Script execution
def gaze_window_intersection(x, y):
    i3 = i3ipc.Connection()

    width, height = get_ws_dimensions(i3)
    assert width and height
    assert 0 <= x <= width
    assert 0 <= y <= height

    windows      = get_ws_windows(i3)
    intersecting = get_intersecting_rtree(x, y, windows)
    closest      = get_intersecting_kdtree(x, y, intersecting)

    set_window_focus(closest.window, i3)
    return

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('x', type=int, metavar='int', help="X position to query")
    parser.add_argument('y', type=int, metavar='int', help="Y position to query")
    args = parser.parse_args()

    return sys.exit(gaze_window_intersection(args.x, args.y))

if __name__ == "__main__":
    main()


