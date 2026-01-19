# Interactive SAM Segmentation Tool

A GUI tool for manually placing landmarks on images and running SAM segmentation.

## Usage

```bash
python interactive_sam.py <image_directory> [options]
```

### Examples

```bash
# Basic usage
python interactive_sam.py ./surfboard_frames/

# Save masks to specific directory
python interactive_sam.py ./surfboard_frames/ --output ./masks/

# Load previous annotations
python interactive_sam.py ./surfboard_frames/ --load annotations.json
```

## Controls

| Key | Action |
|-----|--------|
| `←` / `→` | Navigate between images |
| `p` | Switch to point mode |
| `b` | Switch to bbox mode |
| `c` | Clear current image annotations |
| `r` / `Enter` | Run SAM segmentation |
| `s` | Save annotations to JSON |
| `l` | Load annotations from JSON |
| `m` | Save masks to output directory |
| `q` / `Esc` | Quit |

## Mouse Actions

**Point mode:**
- Left-click: Add foreground point
- Right-click: Remove nearest point (or clear bbox)

**Bbox mode:**
- Click + drag: Draw bounding box
- Right-click: Clear bbox (or remove nearest point)

## Workflow

1. Open images: `python interactive_sam.py ./my_images/`
2. Press `b` and draw a bounding box around the object
3. Press `p` and click to add foreground points on the object
4. Use arrow keys to annotate additional images
5. Press `r` to run SAM segmentation
6. Press `s` to save annotations, `m` to save masks
