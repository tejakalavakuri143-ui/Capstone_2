
import time
import logging

from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {

    ".pdf",
    ".docx",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"

}

# ============================================================
# WATCHDOG HANDLER
# ============================================================

class InvoiceHandler(
    FileSystemEventHandler
):

    def __init__(
        self,
        on_files_ready
    ):

        self.on_files_ready = (
            on_files_ready
        )

        self._processing = set()

    def on_created(
        self,
        event
    ):

        if event.is_directory:
            return

        path = Path(
            event.src_path
        )

        # ----------------------------------------------------
        # FILE TYPE FILTER
        # ----------------------------------------------------

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        if path in self._processing:
            return

        self._processing.add(path)

        logger.info(
            "New invoice detected: %s",
            path.name
        )

        # ----------------------------------------------------
        # WAIT FOR FILE WRITE COMPLETION
        # ----------------------------------------------------

        time.sleep(2)

        try:

            logger.info(
                "Triggering invoice processing..."
            )

            # IMPORTANT:
            # send Path object
            # not string

            self.on_files_ready(
                [path]
            )

            logger.info(
                "Invoice processing completed."
            )

        except Exception as e:

            logger.exception(
                "Invoice processing failed: %s",
                e
            )

        finally:

            self._processing.discard(path)

# ============================================================
# MONITOR AGENT
# ============================================================

class InvoiceMonitor:

    def __init__(
        self,
        watch_dir,
        on_files_ready
    ):

        self.watch_dir = Path(
            watch_dir
        )

        self.on_files_ready = (
            on_files_ready
        )

        self._processing = set()

        self.observer = Observer()

    def start(
        self,
        process_existing=False
    ):

        self.watch_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        handler = InvoiceHandler(
            self.on_files_ready
        )

        self.observer.schedule(

            handler,

            str(self.watch_dir),

            recursive=False

        )

        self.observer.start()

        logger.info(
            "Watching folder: %s",
            self.watch_dir
        )

        if process_existing:
            existing = [
                path
                for path in sorted(self.watch_dir.iterdir())
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            if existing:
                self.on_files_ready(existing)

    def stop(self):

        self.observer.stop()

        self.observer.join()

        logger.info(
            "Invoice monitor stopped."
        )

