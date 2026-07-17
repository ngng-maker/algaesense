"""Sensor acquisition: reading the VOC PID sensor and recording/processing
camera video clips, writing both to disk in jaxsr_calibration's raw schemas.
"""

from algaesense_edge.acquisition.camera import (
    ClipFeatures,
    Picamera2CameraCapture,
    create_hardware_camera_capture,
    process_clip,
)
from algaesense_edge.acquisition.i2c import scan_i2c
from algaesense_edge.acquisition.voc import (
    Ads1115VOCSensorReader,
    Bme280TRHSensorReader,
    create_hardware_trh_reader,
    create_hardware_voc_reader,
)
from algaesense_edge.acquisition.writer import PartitionedParquetWriter

__all__ = [
    "Ads1115VOCSensorReader",
    "Bme280TRHSensorReader",
    "create_hardware_voc_reader",
    "create_hardware_trh_reader",
    "ClipFeatures",
    "Picamera2CameraCapture",
    "process_clip",
    "create_hardware_camera_capture",
    "PartitionedParquetWriter",
    "scan_i2c",
]
