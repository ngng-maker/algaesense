"""Raw sensor acquisition and schemas."""

"""
Named `logging_` (trailing underscore) rather than `logging` so that
`import jaxsr_calibration.logging_` never shadows Python's own built-in
`logging` standard-library module -- if this were named `logging`, any
code inside this package that did `import logging` (to use Python's actual
logging facilities) could accidentally import itself instead, depending on
how Python's import system resolves the name. The trailing underscore
sidesteps the ambiguity entirely.

This module holds schema.py (the pyarrow raw-record schemas and the
ExperimentMeta pydantic model). The actual acquisition functions that talk
to real hardware live on the Raspberry Pi (algaesense-edge package),
which imports and reuses these schemas rather than redefining them.
"""

from jaxsr_calibration.logging_.schema import (
    CAMERA_RAW_SCHEMA,
    VOC_RAW_SCHEMA,
    ExperimentMeta,
)

"""
Re-exporting the schema module's public names here lets callers write the
shorter `from jaxsr_calibration.logging_ import VOC_RAW_SCHEMA` instead of
reaching one level deeper into `jaxsr_calibration.logging_.schema`.
"""

__all__ = ["VOC_RAW_SCHEMA", "CAMERA_RAW_SCHEMA", "ExperimentMeta"]
