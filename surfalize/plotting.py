import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy import ndimage

def _create_colorbar(vmin, vmax, cmap, label='z (µm)', height=0.5):
    """
    Creates a PIL image of a colorbar using matplotlib.

    Parameters
    ----------
    vmin : float
        Minimum value of the colorbar.
    vmax : float
        Maximum value of the colorbar.
    cmap : str
        Matplotlib colormap name.
    label : str
        Label of the colorbar. Defaults to z (µm).
    height : float
        Height of the colorbar as a fraction of the image height.

    Returns
    -------
    PIL.Image
    """
    fig, ax = plt.subplots(figsize=(0.5, 6))
    fig.patch.set_facecolor('white')
    cax = fig.add_axes([0, (1 - height) / 2, 1, height])  # Smaller subplot
    ax.axis('off')
    cax.set_axis_off()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.colorbar(sm, ax=cax, orientation='vertical', fraction=1, pad=0, label=label)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='PNG', dpi=200, bbox_inches='tight', transparent=False)
    plt.close()
    return Image.open(buffer)

def plot_3d(surface, vertical_angle=50, horizontal_angle=0, zoom=1, cmap='jet', colorbar=True, show_grid=True,
            light=0.3, light_position=None, crop_white=True, cbar_pad=50, cbar_height=0.5, scale=1,
            level_of_detail=100, interactive=False, window_title='surfalize', perspective_projection=False):
    """
    Renders a surface object in 3d using pyvista.

    Parameters
    ----------
    surface: surfalize.Surface
        Surface object.
    vertical_angle : float
        Angle of the camera in the vertical plane in degree. Defaults to 50.
    horizontal_angle : float
        Angle of the camera in the horizontal plane in degree. Defaults to 0.
    zoom : float
        Zoom factor of the surface render. Defaults to 1. Decreasing the value will zoom out the render.
    cmap : str
        Matplotlib colormap name. Defaults to jet.
    colorbar : bool
        Whether to show a colorbar. Defaults to True.
    show_grid : bool
        Whether to show a grid. Defaults to True.
    light : float
        Intensity of the light from 0 to 1. Defaults to 1.
    light_position : tuple[float, float, float]
        Position of the light source. Defaults to the position of the camera.
    crop_white : bool
        Whether to crop out white image borders in the horizontal axis. Defaults to True. Only valid for static mode.
    cbar_pad : int
        Additional padding of the colorbar from the 3d render in pixels. Defaults to 50. Only valid for static mode.
    cbar_height : float
        Height of the colorbar as a fraction of the image height. Only valid for static mode.
    scale : float
        Vertical scaling factor of the topography. Defaults to 1. Currently, there are issues with the grid rendering
        for scale values other than 1 due to the current pyvista implementation.
    level_of_detail : float
        Level of detail in % by which the topography is downsampled for the 3d plot. A value of 50 will downsample the
        number of points in each axis by a factor of 2. Defaults to 100.
    interactive : bool
        Specifies whether the plot should be shown in an interactive window. Does not currently work for jupyter.
        Defaults to False.
    window_title : str
        The window title to show in interactive mode. Defaults to 'surfalize'.
    perspective_projection : bool
        Whether to use perspective or parallel projection. Default is True.

    Returns
    -------
    PIL.Image
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError('3d-plotting requires several optional dependencies. To enable 3d-plotting, install surfalize '
                          'with optional dependencies: pip install surfalize[3d]') from None

    if level_of_detail < 100:
        factor = level_of_detail / 100
        surface = surface.__class__(ndimage.zoom(surface.data, factor),
                                    surface.step_x / factor,
                                    surface.step_y / factor)

    # Generate a grid of x, y values
    x = np.linspace(0, surface.width_um, surface.size.x)
    y = np.linspace(0, surface.height_um, surface.size.y)
    x, y = np.meshgrid(x, y)

    # Define a mathematical function for z values
    z = surface.data

    # Create a PyVista grid
    grid = pv.StructuredGrid(x, y, z)
    grid.point_data["height"] = z.T.ravel()

    # Initialize the PyVista plotter
    plotter = pv.Plotter(off_screen=not interactive, window_size=None if interactive else (1920, 1080))
    if not perspective_projection:
        plotter.enable_parallel_projection()

    # Add the surface plot to the plotter
    plotter.add_mesh(grid, cmap=cmap, scalars="height", show_edges=False, show_scalar_bar=False)

    plotter.set_scale(zscale=scale)

    target_position = (surface.width_um / 2, surface.height_um / 2, 0)
    distance = surface.width_um * 2 / zoom

    h_dist = np.cos(np.deg2rad(vertical_angle)) * distance
    x = target_position[0] + np.sin(np.deg2rad(horizontal_angle)) * h_dist
    y = target_position[1] + np.cos(np.deg2rad(horizontal_angle)) * h_dist
    z = target_position[2] + np.sin(np.deg2rad(vertical_angle)) * distance

    camera_position = (x, y, z)
    camera_normal = (0, 0, 1)
    plotter.camera_position = [camera_position, target_position, camera_normal]

    if light_position is None:
        light_position = camera_position

    light = pv.Light(position=light_position, focal_point=(0, 0, 0), color='white', intensity=light)
    plotter.add_light(light)

    if show_grid:
        plotter.show_grid(
            color='black',
            grid='back',
            location='outer',
            xtitle='X (µm)',
            ytitle='Y (µm)',
            ztitle='Z (µm)',
            font_family='arial',
            use_2d=False,
            use_3d_text=False,
            font_size=12,
            n_zlabels=2,
            all_edges=True
        )

    if interactive:
        plotter.show(title=window_title, jupyter_backend='client')
    else:
        # Save the plot to a buffer
        buffer = io.BytesIO()
        plotter.screenshot(buffer)
        buffer.seek(0)

        # Display the saved plot using PIL
        img = Image.open(buffer)
        if crop_white:
            bg = Image.new(img.mode, img.size, (255, 255, 255))
            diff = ImageChops.difference(img, bg)

            bbox = diff.getbbox()

            if bbox is not None:
                img = img.crop((bbox[0], 0, bbox[2], img.height))

        if colorbar:
            cb = _create_colorbar(np.nanmin(surface.data), np.nanmax(surface.data), cmap, height=cbar_height)
            cb = cb.resize((int(cb.width * img.height / cb.height), img.height))

            composite = Image.new('RGB', (img.width + cb.width + cbar_pad, img.height), (255, 255, 255))
            composite.paste(img, (0, 0))
            composite.paste(cb, (img.width + cbar_pad, 0))
            img = composite

        return img