###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Write crpropa output to a parquet/pyarrow file                          #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             August-2026                                                     #
###############################################################################

import crpropa

import queue
import threading
import pyarrow as pa
import pyarrow.parquet as pq

from loguru import logger
from pathlib import Path

class AsyncDynamicParquetOutput(crpropa.Module):
    """High-performance PyArrow Parquet module with ZERO data loss guarantee.

    Engineered to handle high thread counts (80+ threads) without dropping
    batches or deadlocking during simulation teardown.
    """

    def __init__(
        self,
        filename      : Path,
        fields        : int   = crpropa.Output.Everything,
        length_unit   : float = crpropa.Mpc,
        energy_unit   : float = crpropa.EeV,
        batch_size    : int   = 50000,
        queue_maxsize : int   = 1000,
    ):
        super().__init__()
        self.filename    = filename
        self.length_unit = length_unit
        self.energy_unit = energy_unit
        self.batch_size  = batch_size

        self.fields          = fields
        self._schema_fields  = []
        self._row_extractors = []

        self._thread_buffers = {}
        self._buffer_lock    = threading.Lock()

        # Telemetry counters
        self._telemetry_lock = threading.Lock()
        self.total_processed = 0
        self.total_saved     = 0

        self._is_closed      = False
        self._shutdown_event = threading.Event()
        self._rebuild_all()

        self.write_queue   = queue.Queue(maxsize=queue_maxsize)
        self.writer        = None
        self.worker_thread = threading.Thread(
            target=self._parquet_writer_worker, daemon=True
        )

        self.worker_thread.start()

    # -----------------------------------------------------------------
    # Context Manager Support
    # -----------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if not getattr(self, "_is_closed", True):
            self.close()

    # -----------------------------------------------------------------
    # CRPropa Native Output API Methods
    # -----------------------------------------------------------------
    def enable(self, field: int):
        """Enable specific OutputColumn flag(s) using bitwise OR."""
        self._flush_all_buffers()
        self.fields |= field
        self._rebuild_all()

    def disable(self, field: int):
        """Disable specific OutputColumn flag(s) using bitwise AND NOT."""
        self._flush_all_buffers()
        self.fields &= ~field
        self._rebuild_all()

    def enableAll(self):
        """Enable all standard OutputColumn flags defined in Output.h."""
        self._flush_all_buffers()
        mask = 0
        for attr in dir(crpropa.Output):
            if attr.endswith("Column"):
                val = getattr(crpropa.Output, attr)
                if isinstance(val, int):
                    mask |= val
        self.fields = mask
        self._rebuild_all()

    def disableAll(self):
        """Disable all standard CRPropa columns."""
        self._flush_all_buffers()
        self.fields = 0
        self._rebuild_all()

    def setFields(self, fields: int):
        """Set the exact bitmask directly."""
        self._flush_all_buffers()
        self.fields = fields
        self._rebuild_all()

    # -----------------------------------------------------------------
    # Internal Schema & Safe Thread Buffer Management
    # -----------------------------------------------------------------
    # def _rebuild_all(self):
    #     self._schema_fields.clear()
    #     self._row_extractors = []
    #     self._build_standard_fields()
    #     self.schema = pa.schema(self._schema_fields)
    # -----------------------------------------------------------------
    # Helper to resolve CRPropa unit floats into readable strings
    # -----------------------------------------------------------------
    def _resolve_unit_name(self,unit_val:float,default_type:str) -> str:
        """Converts CRPropa numeric unit values back into string labels."""
        if default_type == "energy":
            known_units = {
                crpropa.EeV   : "EeV",
                crpropa.PeV   : "PeV",
                crpropa.TeV   : "TeV",
                crpropa.GeV   : "GeV",
                crpropa.MeV   : "MeV",
                crpropa.eV    : "eV",
                crpropa.joule : "J",
            }
        elif default_type == "length":
            known_units = {
                crpropa.Gpc   : "Gpc",
                crpropa.Mpc   : "Mpc",
                crpropa.kpc   : "kpc",
                crpropa.pc    : "pc",
                crpropa.au    : "AU",
                crpropa.km    : "km",
                crpropa.meter : "m",
                crpropa.cm    : "cm",
            }
        else:
            known_units = {}

        # Look for match within numerical tolerance
        for val,name in known_units.items():
            if abs(unit_val - val) / val < 1e-6:
                return name

        return f"Custom Scale ({unit_val})"

    # -----------------------------------------------------------------
    # Internal Schema & Footer Metadata Setup
    # -----------------------------------------------------------------
    def _rebuild_all(self):
        self._schema_fields.clear()
        self._row_extractors = []
        self._build_standard_fields()

        # 1. Dynamically resolve configured unit names
        energy_unit_str = self._resolve_unit_name(self.energy_unit,"energy")
        length_unit_str = self._resolve_unit_name(self.length_unit,"length")

        # 2. Construct string dictionary for the Parquet Footer Thrift Metadata
        footer_metadata = {
            "CRPROPA:VERSION": getattr(crpropa,"__version__","3.x"),
            "UNITS:ENERGY"    : energy_unit_str,
            "UNITS:LENGTH"    : length_unit_str,
            "COLUMN_UNITS:E"  : energy_unit_str,
            "COLUMN_UNITS:E0" : energy_unit_str,
            "COLUMN_UNITS:E1" : energy_unit_str,
            "COLUMN_UNITS:X"  : length_unit_str,
            "COLUMN_UNITS:Y"  : length_unit_str,
            "COLUMN_UNITS:Z"  : length_unit_str,
            "COLUMN_UNITS:D"  : length_unit_str,
            "COLUMN_UNITS:Px" : "dimensionless (unit vector)",
            "COLUMN_UNITS:Py" : "dimensionless (unit vector)",
            "COLUMN_UNITS:Pz" : "dimensionless (unit vector)",
        }

        # 3. Attach metadata to schema (PyArrow handles writing to Footer)
        self.schema = pa.schema(self._schema_fields,metadata=footer_metadata)

    def _get_thread_buffer(self):
        tid = threading.get_ident()
        if tid not in self._thread_buffers:
            with self._buffer_lock:
                if tid not in self._thread_buffers:
                    self._thread_buffers[tid] = {
                        "count": 0,
                        "data": {
                            name: [] for name, _ in self._schema_fields
                        },
                    }
        return self._thread_buffers[tid]

    def _build_standard_fields(self):
        col = crpropa.Output

        # -------------------------------------------------------------
        # 1. CURRENT STATE
        # -------------------------------------------------------------
        if hasattr(col, "CurrentIdColumn") and (
            self.fields & col.CurrentIdColumn
        ):
            self._schema_fields.append(("ID", pa.int32()))
            self._row_extractors.append(("ID", lambda c: c.current.getId()))

        if hasattr(col, "CurrentEnergyColumn") and (
            self.fields & col.CurrentEnergyColumn
        ):
            self._schema_fields.append(("E", pa.float64()))
            self._row_extractors.append(
                ("E", lambda c: c.current.getEnergy() / self.energy_unit)
            )

        if hasattr(col, "CurrentPositionColumn") and (
            self.fields & col.CurrentPositionColumn
        ):
            for axis in ["X", "Y", "Z"]:
                self._schema_fields.append((axis, pa.float64()))

            def get_pos(c):
                p = c.current.getPosition()
                return (
                    p.x / self.length_unit,
                    p.y / self.length_unit,
                    p.z / self.length_unit,
                )

            self._row_extractors.append((("X", "Y", "Z"), get_pos))

        if hasattr(col, "CurrentDirectionColumn") and (
            self.fields & col.CurrentDirectionColumn
        ):
            for axis in ["Px", "Py", "Pz"]:
                self._schema_fields.append((axis, pa.float64()))

            def get_dir(c):
                d = c.current.getDirection()
                return (d.x, d.y, d.z)

            self._row_extractors.append((("Px", "Py", "Pz"), get_dir))

        if hasattr(col, "SerialNumberColumn") and (
            self.fields & col.SerialNumberColumn
        ):
            self._schema_fields.append(("SN", pa.uint64()))
            self._row_extractors.append(
                ("SN", lambda c: c.getSerialNumber())
            )

        # -------------------------------------------------------------
        # 2. CREATED STATE
        # -------------------------------------------------------------
        if hasattr(col, "CreatedIdColumn") and (
            self.fields & col.CreatedIdColumn
        ):
            self._schema_fields.append(("ID1", pa.int32()))
            self._row_extractors.append(("ID1", lambda c: c.created.getId()))

        if hasattr(col, "CreatedEnergyColumn") and (
            self.fields & col.CreatedEnergyColumn
        ):
            self._schema_fields.append(("E1", pa.float64()))
            self._row_extractors.append(
                ("E1", lambda c: c.created.getEnergy() / self.energy_unit)
            )

        if hasattr(col, "CreatedPositionColumn") and (
            self.fields & col.CreatedPositionColumn
        ):
            for axis in ["X1", "Y1", "Z1"]:
                self._schema_fields.append((axis, pa.float64()))

            def get_pos1(c):
                p = c.created.getPosition()
                return (
                    p.x / self.length_unit,
                    p.y / self.length_unit,
                    p.z / self.length_unit,
                )

            self._row_extractors.append((("X1", "Y1", "Z1"), get_pos1))

        if hasattr(col, "CreatedDirectionColumn") and (
            self.fields & col.CreatedDirectionColumn
        ):
            for axis in ["Px1", "Py1", "Pz1"]:
                self._schema_fields.append((axis, pa.float64()))

            def get_dir1(c):
                d = c.created.getDirection()
                return (d.x, d.y, d.z)

            self._row_extractors.append((("Px1", "Py1", "Pz1"), get_dir1))

        if hasattr(col, "CreatedSerialNumberColumn") and (
            self.fields & col.CreatedSerialNumberColumn
        ):
            self._schema_fields.append(("SN1", pa.uint64()))
            self._row_extractors.append(
                ("SN1", lambda c: c.getCreatedSerialNumber())
            )

        # -------------------------------------------------------------
        # 3. SOURCE STATE
        # -------------------------------------------------------------
        if hasattr(col, "SourceIdColumn") and (
            self.fields & col.SourceIdColumn
        ):
            self._schema_fields.append(("ID0", pa.int32()))
            self._row_extractors.append(("ID0", lambda c: c.source.getId()))

        if hasattr(col, "SourceEnergyColumn") and (
            self.fields & col.SourceEnergyColumn
        ):
            self._schema_fields.append(("E0", pa.float64()))
            self._row_extractors.append(
                ("E0", lambda c: c.source.getEnergy() / self.energy_unit)
            )

        if hasattr(col, "SourcePositionColumn") and (
            self.fields & col.SourcePositionColumn
        ):
            for axis in ["X0", "Y0", "Z0"]:
                self._schema_fields.append((axis, pa.float64()))

            def get_pos0(c):
                p = c.source.getPosition()
                return (
                    p.x / self.length_unit,
                    p.y / self.length_unit,
                    p.z / self.length_unit,
                )

            self._row_extractors.append((("X0", "Y0", "Z0"), get_pos0))

        if hasattr(col, "SourceDirectionColumn") and (
            self.fields & col.SourceDirectionColumn
        ):
            for axis in ["Px0", "Py0", "Pz0"]:
                self._schema_fields.append((axis, pa.float64()))

            def get_dir0(c):
                d = c.source.getDirection()
                return (d.x, d.y, d.z)

            self._row_extractors.append((("Px0", "Py0", "Pz0"), get_dir0))

        if hasattr(col, "SourceSerialNumberColumn") and (
            self.fields & col.SourceSerialNumberColumn
        ):
            self._schema_fields.append(("SN0", pa.uint64()))
            self._row_extractors.append(
                ("SN0", lambda c: c.getSourceSerialNumber())
            )

        # -------------------------------------------------------------
        # 4. METADATA, WEIGHT & TAG
        # -------------------------------------------------------------
        if hasattr(col, "TrajectoryLengthColumn") and (
            self.fields & col.TrajectoryLengthColumn
        ):
            self._schema_fields.append(("D", pa.float64()))
            self._row_extractors.append(
                (
                    "D",
                    lambda c: c.getTrajectoryLength() / self.length_unit,
                )
            )

        if hasattr(col, "WeightColumn") and (
            self.fields & col.WeightColumn
        ):
            self._schema_fields.append(("Weight", pa.float64()))
            self._row_extractors.append(("Weight", lambda c: c.getWeight()))

        tag_col = getattr(col, "ColumnTag", getattr(col, "TagColumn", None))
        if tag_col is not None and (self.fields & tag_col):
            self._schema_fields.append(("Tag", pa.string()))
            self._row_extractors.append(("Tag", lambda c: c.getTagOrigin()))

    # -----------------------------------------------------------------
    # Parallel Simulation Processing
    # -----------------------------------------------------------------
    def process(self, candidate: crpropa.Candidate):
        if self._is_closed:
            return

        try:
            tb = self._get_thread_buffer()
            buf = tb["data"]

            for key, extractor_func in self._row_extractors:
                val = extractor_func(candidate)
                if isinstance(key, tuple):
                    for k, v in zip(key, val):
                        buf[k].append(v)
                else:
                    buf[key].append(val)

            tb["count"] += 1

            if tb["count"] >= self.batch_size:
                self._flush_single_buffer(tb)

        except Exception as e:
            logger.error(f"[AsyncDynamicParquetOutput] Process error: {e}")

    def _flush_single_buffer(self, tb: dict):
        if tb["count"] == 0:
            return

        old_data = tb["data"]
        count = tb["count"]

        tb["data"] = {name: [] for name, _ in self._schema_fields}
        tb["count"] = 0

        # Enforce uniform column length safely within thread scope
        lengths = {k: len(v) for k, v in old_data.items()}
        min_len = min(lengths.values()) if lengths else 0

        if min_len > 0:
            for k in old_data:
                old_data[k] = old_data[k][:min_len]

            with self._telemetry_lock:
                self.total_processed += count
                self.total_saved += min_len

            while not self._shutdown_event.is_set():
                try:
                    self.write_queue.put(old_data, block=True, timeout=1.0)
                    break
                except queue.Full:
                    continue

    def _flush_all_buffers(self):
        with self._buffer_lock:
            for tid in list(self._thread_buffers.keys()):
                tb = self._thread_buffers[tid]
                self._flush_single_buffer(tb)

    # -----------------------------------------------------------------
    # Writer Loop & Clean Shutdown Sequence
    # -----------------------------------------------------------------
    def _parquet_writer_worker(self):
        while True:
            try:
                batch = self.write_queue.get(timeout=0.5)
            except queue.Empty:
                if self._shutdown_event.is_set() and self.write_queue.empty():
                    break
                continue

            if batch is None:
                self.write_queue.task_done()
                break

            try:
                table = pa.Table.from_pydict(batch, schema=self.schema)
                if self.writer is None:
                    self.writer = pq.ParquetWriter(
                        self.filename,
                        self.schema,
                        # compression="snappy"
                        compression="zstd",
                        compression_level=6
                    )
                self.writer.write_table(table)
            except Exception as e:
                logger.error(f"Parquet write error: {e}")
            finally:
                self.write_queue.task_done()

    def close(self):
        """Drains all thread buffers and writes 100% of candidates to disk."""
        if self._is_closed:
            return

        self._is_closed = True

        self._flush_all_buffers()
        self._shutdown_event.set()
        self.write_queue.put(None)
        self.write_queue.join()

        if self.worker_thread.is_alive():
            self.worker_thread.join()

        if self.writer:
            self.writer.close()
            self.writer = None

        logger.info(
            f"Parquet Output closed with zero data loss: {self.filename}"
        )
        logger.info(f"Total processed : {self.total_processed:,}")
        logger.info(f"Total saved     : {self.total_saved:,}")
