The SFLZ file format
====================

``.sflz`` (Surfalize LZ) is surfalize's native file format. It is designed to store one height
(topography) channel together with an arbitrary number of additional image layers and free-form
metadata, while keeping the file size as small as possible through optional compression and optional
integer quantization.

Design goals
------------

* **Small files.** Layer data may be compressed with zlib or LZMA, and the height channel may be
  quantized to a compact integer type.
* **Lossless when needed.** Storing the height channel in a floating point type reproduces the data
  bit-for-bit, including non-measured points (``NaN``).
* **Arbitrary channels.** Besides the height channel, any number of named image layers (grayscale or
  multi-channel, e.g. RGB) can be stored, each with its own dimensions and data type.
* **Free-form metadata.** Metadata is stored as a JSON document, so strings, numbers, booleans,
  nested structures and timestamps all round-trip.
* **Unit aware.** The lateral spacing and height values carry an explicit length unit (micrometers by
  default), converted to and from surfalize's internal micrometers on write and read.
* **Versioned.** The file carries a version string so the layout can evolve in the future without
  breaking existing readers. The current version is 1.0.

All multi-byte numbers are little-endian.

Overall structure
-----------------

.. code-block:: text

    +-------------------------------+
    | magic            "SFLZ"  (4)  |
    | version string  (10, ASCII)   |
    +-------------------------------+
    | file header     (33 bytes)    |
    +-------------------------------+
    | metadata blob   (JSON, UTF-8) |
    +-------------------------------+
    | layer 0 header + data         |   <- height channel ("topography")
    | layer 1 header + data         |   <- optional image layer
    | ...                           |
    +-------------------------------+

Header
------

The 4-byte magic ``SFLZ`` is followed by a 10-byte, space-padded ASCII version string (``"1.0"``).
The fixed-size file header follows:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``step_x``
     - float64
     - Pixel spacing in x, expressed in ``unit``.
   * - ``step_y``
     - float64
     - Pixel spacing in y, expressed in ``unit``.
   * - ``unit``
     - 8 bytes
     - Length unit of ``step_x``, ``step_y`` and the height channel (e.g. ``um``, ``nm``, ``mm``,
       ``m``). Defaults to ``um``.
   * - ``num_layers``
     - uint16
     - Total number of layers, including the height channel.
   * - ``num_metadata``
     - uint16
     - Number of top-level metadata entries (informational).
   * - ``compression_algorithm``
     - uint8
     - ``0`` = none, ``1`` = zlib, ``2`` = LZMA. Applies to every layer.
   * - ``metadata_size``
     - uint32
     - Length in bytes of the metadata blob that follows the header.

Metadata blob
-------------

Immediately after the header come ``metadata_size`` bytes of UTF-8 encoded JSON, representing the
``Surface.metadata`` dictionary. JSON-native types (str, int, float, bool, list, dict, ``null``) are
preserved. ``datetime`` objects are encoded as ``{"__datetime__": "<ISO 8601>"}`` and restored on
read. Any other type is stored as its string representation. If there is no metadata,
``metadata_size`` is ``0`` and the blob is empty.

Layers
------

Each layer consists of a variable-length header followed by its compressed data block. The header is:

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Field
     - Type
     - Description
   * - ``name_length``
     - uint16
     - Length of the layer name in bytes.
   * - ``name``
     - bytes
     - Layer name (UTF-8). The height channel is named ``topography``.
   * - ``is_height``
     - bool (uint8)
     - ``True`` for the height channel, ``False`` for image layers.
   * - ``width``
     - uint32
     - Number of columns.
   * - ``height``
     - uint32
     - Number of rows.
   * - ``channels``
     - uint8
     - Number of channels per pixel (``1`` for grayscale, ``3`` for RGB, ...).
   * - ``datatype``
     - 4 bytes
     - NumPy dtype string of the stored samples, e.g. ``<u4``, ``<f8`` or ``|u1``.
   * - ``scaled``
     - bool (uint8)
     - ``True`` if the values were quantized (see below); ``False`` if stored as-is.
   * - ``has_nan``
     - bool (uint8)
     - ``True`` if a NaN sentinel is present in a quantized layer.
   * - ``min_value``
     - float64
     - Minimum value used for quantization (unused when ``scaled`` is ``False``).
   * - ``max_value``
     - float64
     - Maximum value used for quantization (unused when ``scaled`` is ``False``).
   * - ``size``
     - uint32
     - Length in bytes of the compressed data block that follows.

The data block is the layer's raw sample buffer (row-major, shape
``(height, width)`` or ``(height, width, channels)``), compressed with the file's compression
algorithm.

Quantization and NaN handling
-----------------------------

When the height channel is written with an integer ``dtype`` (the default is ``<u4``), it is quantized:
the value range ``[min_value, max_value]`` is mapped linearly onto the integer range. On read, the
inverse mapping restores the floating point values. This is lossy, but with a 32-bit type the
resolution (range divided by ~4.3 billion) is finer than float32 for any realistic height range.

If the height data contains ``NaN`` (non-measured points), the largest integer value is reserved as a
sentinel and ``has_nan`` is set; the remaining range is used for the real values. On read, samples
equal to the sentinel are restored to ``NaN``.

To store the height channel **losslessly**, write it with a floating point ``dtype`` (``<f4`` or
``<f8``). In that case ``scaled`` is ``False``, the samples are stored verbatim, and ``NaN`` is
preserved natively without a sentinel.

Image layers are always stored losslessly in their own data type and are never quantized.

Usage
-----

.. code-block:: python

    from surfalize import Surface

    surface = Surface.load('measurement.vk4', read_image_layers=True)

    # Default: zlib-compressed, height quantized to uint32 (small, near-lossless)
    surface.save('measurement.sflz')

    # Maximum compression
    surface.save('measurement.sflz', compression='lzma')

    # Fully lossless height channel
    surface.save('measurement.sflz', dtype='<f8')

    # Smaller, more lossy height channel
    surface.save('measurement.sflz', dtype='<u2')

    # Height channel only, without image layers or metadata
    surface.save('measurement.sflz', save_image_layers=False, write_metadata=False)

    loaded = Surface.load('measurement.sflz', read_image_layers=True)

Writer options
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Argument
     - Default
     - Description
   * - ``compression``
     - ``'zlib'``
     - One of ``'none'``, ``'zlib'`` or ``'lzma'``.
   * - ``dtype``
     - ``'<u4'``
     - Storage dtype of the height channel. Integer types quantize; ``'<f4'``/``'<f8'`` are lossless.
   * - ``save_image_layers``
     - ``True``
     - Whether to write the additional image layers.
   * - ``write_metadata``
     - ``True``
     - Whether to write the metadata blob.
   * - ``unit``
     - ``'um'``
     - Length unit the lateral spacing and height values are stored in. The values are converted from
       surfalize's internal micrometers on write and back on read.
