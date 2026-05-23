# img2plot
Turn images into drawings for a pen plotter!

## What is this?
img2plot is a script that attempts to make artistic line drawings out of input images, for drawing on a pen plotter.  The script can read most image formats and saves outputs as an SVG file.  

Unlike other image-to-pen-plotter programs, img2plot tries to mimic human drawing styles with lines - instead of drawing densities with stippling, circles, or waves, the output will be lines following edges and sketchy representations of gradients.  The intention is to mimic the appearance of a quick notebook pen sketch.  

This was originally written with a Silhouette Cameo in mind, but many other plotters, laser cutters, or other devices that take vector input ought to work fine.

## How do I use it?
img2plot runs on Python 3 and requires `numpy`, `scipy`, `scikit-image`, `imageio`, and `svgwrite`. All can be installed via pip:

```
pip install numpy scipy scikit-image imageio svgwrite
```

(If you're on Windows you may need to install scikit-image via a binary Python wheel — see instructions [here](http://scikit-image.org/docs/dev/install.html).)

Once dependencies are set up:

1. Clone or download this repository.
2. Run the script, passing input and output paths on the command line:
   ```
   python img2plot.py --input path/to/photo.png --output path/to/result.svg
   ```
   Short forms `-i` and `-o` work too. Any image format readable by `imageio` is accepted; the output is always SVG.
3. To tune the look, adjust the dataclasses near the top of `img2plot.py`:
   - `PreprocessConfig` — CLAHE and Gaussian-blur kernel sizes (set to `None` to disable a step).
   - `ExtractionConfig` — line density, length, curvature tolerance.
   - `Config` — bundles the two; what `main` uses.
4. Load the SVG into a plotter program of your choice! (Or anything else, really.)

### Running the tests

```
pip install pytest
pytest test_img2plot.py
```

## Results

![betta](readme-imgs/betta.jpg)
![revali](readme-imgs/revali.jpg)
![dunwall](readme-imgs/dunwall.jpg)
![printed](readme-imgs/img2plot.jpg)

## Future Plans
* Draw smooth Bezier curves instead of lines - would change the art style a little, but would look more "human"
* Port to C++ / OpenCV for speed on large images
