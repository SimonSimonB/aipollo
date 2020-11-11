import collections
import numbers

class Point:
    def __init__(self, y, x):
        self.y = int(round(y))
        self.x = int(round(x))

def get_line_segment(point1, point2):
    if point1.x > point2.x:
        point1, point2 = point2, point1
    
    if point1.x == point2.x:
        return [Point(y, point1.x) for y in range(point1.y, point2.y+1)]
    else:
        slope = (point2.y - point1.y) / (point2.x - point1.x)
        return [Point(point1.y + slope * (x - point1.x), x) for x in range(point1.x, point2.x + 1)]

def get_bounding_box(pixels):
    return (
        Point(min(point[0] for point in pixels), min(point[1] for point in pixels)),
        Point(max(point[0] for point in pixels), max(point[1] for point in pixels))
    )

def get_line(point1: Point, point2: Point, height, width):
    if point1.x == point2.x: 
        return [(y, point1.x) for y in range(height)]

    if point2.x < point1.x:
        point1, point2 = point2, point1
    
    slope = (point2.y - point1.y) / (point2.x - point1.x)
    intersect = (point1.y - point1.x * slope)

    line = []
    for x in range(width):
        y = round(intersect + slope * x)

        if 0 <= y < height:
            line.append(Point(y, x))
    
    return line



