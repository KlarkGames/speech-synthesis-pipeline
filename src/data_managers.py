from abc import ABC, abstractmethod
import os
import io
from lakefs_spec import LakeFSFileSystem
from contextlib import contextmanager
from pathlib import Path


class AbstractFileSystemManager(ABC):
    @abstractmethod
    def is_path_exists(self, path: str | os.PathLike[str]) -> bool: ...

    @abstractmethod
    def get_absolute_path(self, path: str | os.PathLike[str]) -> str: ...

    @property
    @abstractmethod
    def directory_name(self) -> str: ...

    @abstractmethod
    @contextmanager
    def open_file(self, path: str | os.PathLike[str], mode: str) -> io.BufferedIOBase: ...

    @abstractmethod
    @contextmanager
    def get_buffered_writer(self, path: str | os.PathLike[str]) -> io.BufferedWriter: ...

    @abstractmethod
    @contextmanager
    def get_buffered_reader(self, path: str | os.PathLike[str]) -> io.BufferedReader: ...


class LocalFileSystemManager:
    def __init__(self, path_to_directory: str | os.PathLike[str]):
        self._prefix_path = path_to_directory

    def is_path_exists(self, path: str | os.PathLike[str]) -> bool:
        return os.path.exists(self.get_absolute_path(path))

    def get_absolute_path(self, path: str | os.PathLike[str]) -> str:
        return os.path.join(self._prefix_path, path)

    @property
    def directory_name(self) -> str:
        return Path(self._prefix_path).stem

    @contextmanager
    def open_file(self, path: str | os.PathLike[str], mode: str) -> io.BufferedIOBase:
        try:
            file = open(self.get_absolute_path(path), mode)
            yield file
        finally:
            file.close()

    @contextmanager
    def get_buffered_writer(self, path: str | os.PathLike[str]) -> io.BufferedWriter:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            file = open(self.get_absolute_path(path), "wb")
            yield io.BufferedWriter(file)
        finally:
            file.close()

    @contextmanager
    def get_buffered_reader(self, path: str | os.PathLike[str]) -> io.BufferedReader:
        try:
            file = open(self.get_absolute_path(path), "rb")
            yield io.BufferedReader(io.BytesIO(file.read()))
        finally:
            file.close()


class LakeFSFileSystemManager:
    def __init__(
        self,
        lakefs_address: str,
        lakefs_port: str,
        lakefs_ACCESS_KEY: str,
        lakefs_SECRET_KEY: str,
        lakefs_repository_name: str,
        lakefs_branch_name: str,
    ):
        self._address = lakefs_address
        self._port = lakefs_port
        self._ACCESS_KEY = lakefs_ACCESS_KEY
        self._SECRET_KEY = lakefs_SECRET_KEY
        self._repository_name = lakefs_repository_name
        self._branch_name = lakefs_branch_name

        self._file_system = LakeFSFileSystem(
            host=f"{self._address}:{self._port}", username=self._ACCESS_KEY, password=self._SECRET_KEY
        )

        self._transaction = self._file_system.transaction(self._repository_name, self._branch_name)

    def is_path_exists(self, path: str | os.PathLike[str]) -> bool:
        return self._file_system.exists(path)

    def get_absolute_path(self, path: str | os.PathLike[str]):
        return f"lakefs://{os.path.join(self._repository_name, self._transaction.branch.id, path)}"

    @property
    def directory_name(self) -> str:
        if self._branch_name == "main":
            return self._repository_name
        else:
            return f"{self._repository_name}_{self._branch_name}"

    @contextmanager
    def open_file(self, path: str | os.PathLike[str], mode: str) -> io.BufferedIOBase:
        try:
            file = self._file_system.open(os.path.join(self._repository_name, self._branch_name, path), mode)
            yield file
        finally:
            file.close()

    @contextmanager
    def get_buffered_writer(self, path: str | os.PathLike[str]) -> io.BufferedWriter:
        try:
            file = self._file_system.open(os.path.join(self._repository_name, self._branch_name, path), "wb")
            yield io.BufferedWriter(file)
        finally:
            file.close()

    @contextmanager
    def get_buffered_reader(self, path: str | os.PathLike[str]) -> io.BufferedReader:
        try:
            file = self._file_system.open(os.path.join(self._repository_name, self._branch_name, path), "rb")
            yield io.BufferedReader(io.BytesIO(file.read()))
        finally:
            file.close()
