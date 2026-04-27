"""Live recording of a glove session.

Writes a continuous "master tape" of DAQ sensor data to disk. Demo boundaries
are determined later at ingest time by matching GoPro MP4 wall-clock windows
against this stream.

See `recorder.py` for the writer, `__main__.py` for the CLI entry point.
"""
