import logging
import typing
from pathlib import Path

import luigi

logger = logging.getLogger("luigi-interface")


class SingleInMixin:

    @property
    def input_path(self) -> Path:
        input_path = typing.cast(luigi.LocalTarget, self.input())
        return Path(input_path.path).resolve()


class MultiInMixin:

    @property
    def input_paths(self) -> list[Path]:
        return [
            Path(input_path.path).resolve()
            for input_path in typing.cast(list[luigi.LocalTarget], self.input())
        ]


class SingleOutMixin:

    @property
    def output_path(self) -> Path:
        output_path = typing.cast(luigi.LocalTarget, self.output())
        return Path(output_path.path).resolve()


class LoggerMixin:

    def log_info(self, message: str) -> None:
        logger.info(f"[{self.__class__.__name__}] {message}")
