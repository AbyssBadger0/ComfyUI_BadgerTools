from PIL import Image
from collections import deque

def draw_line(pixels, x0, y0, x1, y1):
    """Draw a white line from (x0, y0) to (x1, y1) on the provided pixels map."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        pixels[x0, y0] = 255
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def fill_white_segments(original_image, low_threshold, high_threshold):
    # Load the original image and convert it to grayscale
    original_image = original_image.convert('L')
    original_pixels = original_image.load()
    width, height = original_image.size

    low_threshold = int(width*low_threshold)
    high_threshold = int(width*high_threshold)

    # Create a new black image to draw the lines
    new_image = Image.new('L', (width, height), 0)
    new_pixels = new_image.load()

    # Scan horizontally
    for y in range(height):
        point_a = None
        for x in range(width):
            if original_pixels[x, y] == 255:
                if point_a is None:
                    point_a = (x, y)
                else:
                    if x - point_a[0] < high_threshold and x - point_a[0] > low_threshold :
                        draw_line(new_pixels, point_a[0], point_a[1], x, y)
                        point_a = (x, y)
                    else:
                        point_a = (x, y)


    # Scan vertically
    for x in range(width):
        point_a = None
        for y in range(height):
            if original_pixels[x, y] == 255:
                if point_a is None:
                    point_a = (x, y)
                else:
                    if y - point_a[1] < high_threshold and y - point_a[1] > low_threshold:
                        draw_line(new_pixels, point_a[0], point_a[1], x, y)
                        point_a = (x, y)
                    else:
                        point_a = (x, y)

    # Scan diagonally (top-left to bottom-right)
    for diag in range(-height + 1, width):
        point_a = None
        for y in range(max(-diag, 0), min(width - diag, height)):
            x = y + diag
            if original_pixels[x, y] == 255:
                if point_a is None:
                    point_a = (x, y)
                else:
                    if max(abs(x - point_a[0]), abs(y - point_a[1])) < high_threshold and max(abs(x - point_a[0]), abs(y - point_a[1])) > low_threshold:
                        draw_line(new_pixels, point_a[0], point_a[1], x, y)
                        point_a = (x, y)
                    else:
                        point_a = (x, y)

    # Scan diagonally (top-right to bottom-left)
    for diag in range(0, width + height):
        point_a = None
        for y in range(max(diag - width + 1, 0), min(diag + 1, height)):
            x = diag - y
            if original_pixels[x, y] == 255:
                if point_a is None:
                    point_a = (x, y)
                else:
                    if max(abs(x - point_a[0]), abs(y - point_a[1])) < high_threshold and max(abs(x - point_a[0]), abs(y - point_a[1])) > low_threshold:
                        draw_line(new_pixels, point_a[0], point_a[1], x, y)
                        point_a = (x, y)
                    else:
                        point_a = (x, y)

    # Save the new image with only the drawn lines
    return new_image

def find_largest_white_component(image):
    width, height = image.size
    visited = set()
    largest_component = []
    largest_size = 0

    def bfs(x, y):
        queue = deque([(x, y)])
        local_visited = set()
        while queue:
            x, y = queue.popleft()
            if (x, y) not in visited and 0 <= x < width and 0 <= y < height and image.getpixel((x, y)) == 255:
                visited.add((x, y))
                local_visited.add((x, y))
                queue.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        return local_visited

    for y in range(height):
        for x in range(width):
            if image.getpixel((x, y)) == 255 and (x, y) not in visited:
                component = bfs(x, y)
                if len(component) > largest_size:
                    largest_size = len(component)
                    largest_component = component

    # 创建一个新的图像来绘制最大的白色像素点整体
    new_image = Image.new('1', image.size)
    for x, y in largest_component:
        new_image.putpixel((x, y), 255)

    return new_image
