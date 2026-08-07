# Printumo Print Resizer

Takes one high-resolution picture and produces print-ready files for every
[Printumo](https://printumo.com) fine art paper print size, correctly cropped
and tagged at the right print resolution.

## Using it

Open `index.html` in any modern browser — double-click the file, or serve the
folder. There is no build step, no install and no dependency; the whole tool is
one self-contained HTML file and it works offline.

1. **Drop in your picture.** It is read directly by the browser — nothing is
   uploaded anywhere.
2. **Set the crop.** Drag on the preview to place the focal point; every crop
   keeps that spot centred. Click a size in the list to see its exact crop
   outlined in orange.
3. **Pick the sizes** you want, or hit *Only sizes my image supports*.
4. **Generate**, then download files one at a time or all at once as a `.zip`.

## What it does to the image

- **Crop to fill** (default) takes the largest area of the source with the
  print's aspect ratio, centred on your focal point, so the print has no
  borders. **Fit whole image** instead scales the entire picture inside the
  print and pads the remainder with a colour you choose.
- **Downscaling is done in halving steps** rather than one big reduction, which
  is what keeps fine detail from aliasing into jaggies.
- **Output is tagged with its print resolution** — JFIF density for JPEG, a
  `pHYs` chunk for PNG — so Printumo and desktop software read each file at its
  intended physical size rather than guessing 72 DPI.
- **Upscaling is off by default.** When your file cannot supply enough pixels
  for a size, the tool writes the largest file the source genuinely supports at
  that aspect ratio. It still orders the same print; it just lands below the
  target DPI, which the badge tells you. Tick *Allow upscaling* to pad the
  pixel count up to the nominal size — it does not add real detail.

## Reading the badges

Badges rate the real resolution available after cropping:

| Badge | Meaning |
|---|---|
| **Excellent** | 150 DPI or better — Printumo's recommended print resolution |
| **Acceptable** | 100–150 DPI — usable, sharpness starts to soften |
| **Too small** | Under 100 DPI — visibly soft in print |

As a rule of thumb, covering Printumo's full range up to 100 × 140 cm at
150 DPI needs roughly a 50 MP source. A 24 MP camera file comfortably covers
everything up to about A1.

## Sizes

The catalogue follows Printumo's fine art paper prints, each rectangular size
offered in both portrait and landscape, plus three squares:

| Size (cm) | Also known as | Size (cm) | |
|---|---|---|---|
| 14 × 21 | A5 | 70 × 50 | |
| 21 × 29 | A4 | 59 × 84 | A1 |
| 30 × 40 | | 70 × 100 | |
| 29 × 42 | A3 | 100 × 140 | |
| 40 × 50 | | 50 × 50 | square |
| 42 × 59 | A2 | 70 × 70 | square |
| 50 × 70 | | 100 × 100 | square |

*Add custom size…* handles anything not in the list. If Printumo changes its
catalogue, edit the `PRINTUMO_SIZES` and `PRINTUMO_SQUARES` arrays at the top
of the `<script>` block in `index.html` — everything else derives from them.

## Browser support

Needs a browser with `createImageBitmap`, `canvas.toBlob` and pointer events —
current Chrome, Edge, Firefox and Safari all qualify. Very large outputs are
bounded by the browser's canvas limit of 16384 px per edge; at 300 DPI the
biggest sizes exceed it, and the tool flags those rather than producing a
broken file.
