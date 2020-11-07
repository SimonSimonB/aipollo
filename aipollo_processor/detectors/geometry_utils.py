import collections
import numbers

class Point:
    def __init__(self, y, x):
        self.y = round(y)
        self.x = round(x)

def get_line_segment(point1, point2):
    if point1.x > point2.x:
        point1, point2 = point2, point1
    
    if point1.x == point2.x:
        return [(y, point1.x) for y in range(point1.y, point2.y+1)]
    else:
        slope = (point2.y - point1.y) / (point2.x - point1.x)
        return [(int(point1.y + slope * (x - point1.x)), x) for x in range(point1.x, point2.x + 1)]


def get_line(point1, point2, height, width):
    if point1[1] == point2[1]: 
        return [(y, point1[1]) for y in range(height)]

    if point2[1] < point1[1]:
        point1, point2 = point2, point1
    
    slope = (point2[0] - point1[0]) / (point2[1] - point1[1])
    intersect = (point1[0] - point1[1] * slope)

    line = []
    for x in range(width):
        y = round(intersect + slope * x)

        if 0 <= y < height:
            line.append((y, x))
    
    return line



