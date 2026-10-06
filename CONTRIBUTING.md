# Contributing to surfalize

Thank you for considering a contribution to surfalize! Bug reports, feature requests, sample files for new file
formats and pull requests are all welcome. This guide explains how to get started and what we look for in a pull
request.

## Reporting bugs and requesting features

Please open an issue on [GitHub](https://github.com/fredericjs/surfalize/issues). For bugs, include

- the surfalize version (`import surfalize; print(surfalize.__version__)`) and your Python version,
- a minimal code example that reproduces the problem,
- the full error message and traceback,
- if the problem concerns reading a file, the file itself (if you are allowed to share it) and the name and version
  of the instrument or software that wrote it.

If you would like support for a new file format, a few sample files together with exports of the same measurements in
a format that surfalize already reads (e.g. `.sur`, `.sdf` or `.tif`) make it much easier to implement and validate a
reader.

## Setting up a development environment

Clone the repository and install surfalize in editable mode together with the test dependencies:

```commandline
git clone https://github.com/fredericjs/surfalize.git
cd surfalize
git checkout develop
pip install -e ".[tests]"
```

To build the documentation locally, install the `doc` extra (`pip install -e ".[doc]"`).

## Branches

- `main` contains the latest release.
- `develop` is the integration branch for the next release.

Please base your work on `develop` and open your pull request against `develop`, not `main`. Keep a pull request
focused on one topic; unrelated changes are easier to review as separate pull requests.

## Making changes

- **Tests:** Add tests for new features and bug fixes in `tests/`. Run the whole test suite with

  ```commandline
  python -m pytest
  ```

  The tests are currently not run automatically on pull requests, so please make sure they pass locally.
- **Code style:** Match the style of the surrounding code. Public functions and methods use
  [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)-style docstrings, and lines should not exceed
  120 characters.
- **Changelog:** Add a short entry describing your change to the `## Unreleased` section at the top of
  `CHANGELOG.md` (create the section if it does not exist yet).
- **Dependencies:** Avoid adding new required dependencies. If a feature needs an additional package, discuss it in an
  issue first; it may be possible to make it an optional dependency.

## Adding a file format

File readers and writers live in `surfalize/file/`, one module per format. Every module in this package is imported
automatically, so a new reader only needs to register itself:

```python
from .common import RawSurface, FileHandler

@FileHandler.register_reader(suffix='.abc', magic=b'ABC')
def read_abc(filehandle, read_image_layers=False, encoding='utf-8'):
    ...
    return RawSurface(data, step_x, step_y, metadata=metadata, image_layers=image_layers)
```

- `data` is a 2d float array of heights in micrometers, with rows along y. Non-measured points are `NaN`.
- `step_x` and `step_y` are the lateral pixel sizes in micrometers.
- `metadata` is a dict of acquisition parameters. If the file records a measurement date, store it as a `datetime`
  under the key `timestamp`.
- Image layers (e.g. intensity or RGB images) are only read when `read_image_layers` is `True`.
- `magic` (optional) is the byte sequence at the start of the file, which lets surfalize detect the format when the
  file suffix is missing or wrong.

Writers are registered analogously with `FileHandler.register_writer(suffix=...)`. Please describe at the top of the
module where the format information comes from (specification, other open source implementations or reverse
engineering), and add the format to the table of supported file formats in `README.md`.

**Test files:** Add at least one sample file named `test_1.<suffix>` to `tests/test_files/`. All files in that folder
are loaded by the generic file format tests, so it must only contain measurement files. Only add files you are allowed
to redistribute under the license of this project; if a file comes from a third party under another license, add its
attribution to `tests/THIRD_PARTY_TEST_FILES.md`. If possible, also add a test that checks the imported values against
a reference, for example an export of the same measurement from the vendor software.

## License

surfalize is licensed under the GNU General Public License v3.0. By submitting a pull request, you agree that your
contribution is published under the same license.
