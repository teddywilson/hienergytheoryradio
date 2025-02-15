#!/usr/bin/env python
import random
import io
import math
import colorsys
import numpy as np
from flask import Flask, send_file, abort
from PIL import Image, ImageDraw, ImageFilter
import os

WIDTH, HEIGHT = 3000, 3000

def generate_palette(n=10):
    # Candidate colors: beige/light brown, earthy greens, blues, with traces of orange/red.
    candidates = [
        "#F5F5DC",  # beige
        "#D2B48C",  # tan
        "#F0E68C",  # khaki
        "#A0522D",  # sienna
        "#CD853F",  # peru
        "#556B2F",  # dark olive green
        "#6B8E23",  # olive drab
        "#4682B4",  # steel blue
        "#5F9EA0",  # cadet blue
        "#FFA500",  # orange
        "#FF4500"   # orange red
    ]
    weights = [15, 15, 10, 10, 10, 15, 15, 15, 15, 5, 5]
    return random.choices(candidates, weights=weights, k=n)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_gradient_image(w, h, color1, color2, direction):
    if direction == 'vertical':
        factor = np.linspace(0, 1, h).reshape(h, 1)
        factor = np.repeat(factor, w, axis=1)
    elif direction == 'horizontal':
        factor = np.linspace(0, 1, w).reshape(1, w)
        factor = np.repeat(factor, h, axis=0)
    else:
        Y, X = np.indices((h, w))
        factor = (X + Y) / float(w + h - 2)
    r = (color1[0] + (color2[0] - color1[0]) * factor).astype(np.uint8)
    g = (color1[1] + (color2[1] - color1[1]) * factor).astype(np.uint8)
    b = (color1[2] + (color2[2] - color1[2]) * factor).astype(np.uint8)
    gradient = np.dstack((r, g, b))
    return Image.fromarray(gradient, 'RGB')

def create_random_gradient_background(palette):
    direction = random.choice(['vertical', 'horizontal', 'diagonal'])
    color1_hex = random.choice(palette)
    color2_hex = random.choice(palette)
    while color2_hex == color1_hex:
        color2_hex = random.choice(palette)
    color1 = hex_to_rgb(color1_hex)
    color2 = hex_to_rgb(color2_hex)
    return create_gradient_image(WIDTH, HEIGHT, color1, color2, direction)

def create_aa_mask(draw_func, size, scale=2):
    w, h = size
    high_res_size = (w * scale, h * scale)
    mask_hr = Image.new("L", high_res_size, 0)
    draw_hr = ImageDraw.Draw(mask_hr)
    draw_func(draw_hr, (w * scale, h * scale), scale)
    mask = mask_hr.resize(size, Image.LANCZOS)
    return mask

def draw_rounded_rect(draw, size, scale, radii):
    w_hr, h_hr = size
    r_tl, r_tr, r_br, r_bl = [r * scale for r in radii]
    draw.rectangle([(r_tl, 0), (w_hr - r_tr, h_hr)], fill=255)
    draw.rectangle([(0, r_tl), (w_hr, h_hr - r_bl)], fill=255)
    if r_tl > 0:
        draw.pieslice([(0, 0), (2*r_tl, 2*r_tl)], 180, 270, fill=255)
    if r_tr > 0:
        draw.pieslice([(w_hr - 2*r_tr, 0), (w_hr, 2*r_tr)], 270, 360, fill=255)
    if r_br > 0:
        draw.pieslice([(w_hr - 2*r_br, h_hr - 2*r_br), (w_hr, h_hr)], 0, 90, fill=255)
    if r_bl > 0:
        draw.pieslice([(0, h_hr - 2*r_bl), (2*r_bl, h_hr)], 90, 180, fill=255)

def create_custom_rounded_rect_mask(w, h, radii):
    return create_aa_mask(lambda d, s, scale: draw_rounded_rect(d, s, scale, radii), (w, h), scale=2)

def create_ellipse_mask(w, h):
    return create_aa_mask(lambda d, s, scale: d.ellipse([0, 0, s[0], s[1]], fill=255), (w, h), scale=2)

def tessellate_rectangles(x0, y0, x1, y1, min_size=600, split_probability=0.8):
    rects = []
    width = x1 - x0
    height = y1 - y0
    if width < min_size or height < min_size or random.random() > split_probability:
        return [(x0, y0, x1, y1)]
    if random.random() < 0.5:
        split = random.uniform(0.3, 0.7)
        mid = int(x0 + width * split)
        rects += tessellate_rectangles(x0, y0, mid, y1, min_size, split_probability * 0.9)
        rects += tessellate_rectangles(mid, y0, x1, y1, min_size, split_probability * 0.9)
    else:
        split = random.uniform(0.3, 0.7)
        mid = int(y0 + height * split)
        rects += tessellate_rectangles(x0, y0, x1, mid, min_size, split_probability * 0.9)
        rects += tessellate_rectangles(x0, mid, x1, y1, min_size, split_probability * 0.9)
    return rects

def add_tessellation_overlay(base_image, palette):
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    rects = tessellate_rectangles(0, 0, WIDTH, HEIGHT, min_size=600, split_probability=0.8)
    for rect in rects:
        x0, y0, x1, y1 = rect
        w_rect = x1 - x0
        h_rect = y1 - y0
        color1_hex = random.choice(palette)
        color2_hex = random.choice(palette)
        while color2_hex == color1_hex:
            color2_hex = random.choice(palette)
        color1 = hex_to_rgb(color1_hex)
        color2 = hex_to_rgb(color2_hex)
        gradient_direction = random.choice(['vertical', 'horizontal', 'diagonal'])
        cell_gradient = create_gradient_image(w_rect, h_rect, color1, color2, gradient_direction)
        cell_gradient = cell_gradient.convert("RGBA")
        opacity = random.uniform(0.5, 1.0)
        alpha = cell_gradient.split()[3].point(lambda p: int(p * opacity))
        cell_gradient.putalpha(alpha)
        min_dim = min(w_rect, h_rect)
        num_rounded = random.randint(0, 2)
        indices = [0, 1, 2, 3]
        rounded_indices = random.sample(indices, num_rounded)
        radii = []
        for i in indices:
            if i in rounded_indices:
                r_val = random.uniform(10, 0.2 * min_dim)
                radii.append(r_val)
            else:
                radii.append(0)
        mask = create_custom_rounded_rect_mask(w_rect, h_rect, tuple(radii))
        overlay.paste(cell_gradient, (x0, y0), mask)
    return overlay

def add_extra_circles(base_image, palette):
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    for _ in range(random.randint(3, 7)):
        size = random.randint(150, 400)
        x = random.randint(0, WIDTH - size)
        y = random.randint(0, HEIGHT - size)
        color1_hex = random.choice(palette)
        color2_hex = random.choice(palette)
        while color2_hex == color1_hex:
            color2_hex = random.choice(palette)
        color1 = hex_to_rgb(color1_hex)
        color2 = hex_to_rgb(color2_hex)
        direction = random.choice(['vertical', 'horizontal', 'diagonal'])
        circle_gradient = create_gradient_image(size, size, color1, color2, direction)
        circle_gradient = circle_gradient.convert("RGBA")
        opacity = random.uniform(0.6, 1.0)
        alpha = circle_gradient.split()[3].point(lambda p: int(p * opacity))
        circle_gradient.putalpha(alpha)
        mask = create_ellipse_mask(size, size)
        if random.random() < 0.5:
            shadow_offset = (random.randint(2, 4), random.randint(2, 4))
            shadow = Image.new("RGBA", (size, size), (0, 0, 0, 80))
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=3))
            overlay.paste(shadow, (x+shadow_offset[0], y+shadow_offset[1]), mask)
        overlay.paste(circle_gradient, (x, y), mask)
    return overlay

def apply_global_effects(image):
    noise = Image.effect_noise((WIDTH, HEIGHT), 64).convert("L").convert("RGBA")
    image = Image.blend(image, noise, 0.1)
    image = image.filter(ImageFilter.SMOOTH)
    return image

def generate_artwork_to_stream():
    palette = generate_palette(10)
    background = create_random_gradient_background(palette)
    tess_overlay = add_tessellation_overlay(background, palette)
    circles_overlay = add_extra_circles(background, palette)
    final = Image.alpha_composite(background.convert("RGBA"), tess_overlay)
    final = Image.alpha_composite(final, circles_overlay)
    final = apply_global_effects(final)
    output = io.BytesIO()
    final.convert("RGB").save(output, "PNG")
    output.seek(0)
    return output

app = Flask(__name__)

@app.route("/")
def artwork():
    output = generate_artwork_to_stream()
    return send_file(output, mimetype="image/png")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
