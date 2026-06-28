"""JSONL stream storage abstraction for large LingTai history/log files.

The filesystem store preserves the historical flat-file behavior. The NoKV
segmented store publishes immutable JSONL segment objects plus a replace-by-
generation manifest. The mirror store is the Feature 04 Phase 1 migration mode:
local JSONL remains canonical while NoKV receives a best-effort segmented copy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from ._fsutil import append_jsonl, atomic_write_text, iter_jsonl_records, tail_jsonl_records, utc_now_iso

STREAM_PATHS: dict[str, Path] = {
    "logs/events": Path("logs/events.jsonl"),
    "history/chat_history": Path("history/chat_history.jsonl"),
    "logs/token_ledger": Path("logs/token_ledger.jsonl"),
}


@dataclass(frozen=True)
class StreamPosition:
    stream: str
    seq: int
    backend: str
    source_file: str | None = None
    source_offset: int | None = None
    segment: str | None = None
    generation: int | None = None
    mirrored: bool = True


@dataclass(frozen=True)
class ExportResult:
    stream: str
    output_path: Path
    record_count: int
    bytes_written: int


@dataclass(frozen=True)
class StorageHealth:
    status: str
    backend: str
    streams: list[str]
    last_error: str | None = None
    last_error_stream: str | None = None
    updated_at: str | None = None

    def status_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "status": self.status,
            "backend": self.backend,
            "streams": list(self.streams),
        }
        if self.last_error:
            document["last_error"] = self.last_error
        if self.last_error_stream:
            document["last_error_stream"] = self.last_error_stream
        if self.updated_at:
            document["updated_at"] = self.updated_at
        return document


def normalize_stream(stream: str) -> str:
    normalized = stream.strip().strip("/")
    aliases = {
        "events": "logs/events",
        "chat_history": "history/chat_history",
        "token_ledger": "logs/token_ledger",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in STREAM_PATHS:
        raise ValueError(f"unsupported JSONL stream {stream!r}")
    return normalized


def _jsonl_line(record: Any, *, ensure_ascii: bool, default: Any | None = None) -> str:
    return json.dumps(record, ensure_ascii=ensure_ascii, default=default) + "\n"


def _count_records(path: Path) -> int:
    return sum(1 for _ in iter_jsonl_records(path))


def _content_from_result(result: Any) -> str:
    if isinstance(result, bytes):
        return result.decode("utf-8")
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("content", "text", "body"):
            value = result.get(key)
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if isinstance(value, str):
                return value
    return str(result)


def _safe_mirror_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: mirror write failed"


class FilesystemJsonlStreamStore:
    """Flat JSONL implementation preserving the current local file contract."""

    def __init__(self, agent_dir: str | Path, *, streams: list[str] | None = None) -> None:
        self.agent_dir = Path(agent_dir)
        self._streams = {normalize_stream(s) for s in streams} if streams else set(STREAM_PATHS)

    def handles(self, stream: str) -> bool:
        return normalize_stream(stream) in self._streams

    def path_for_stream(self, stream: str) -> Path:
        normalized = normalize_stream(stream)
        if normalized not in self._streams:
            raise ValueError(f"JSONL stream is not configured: {stream}")
        return self.agent_dir / STREAM_PATHS[normalized]

    def append(
        self,
        stream: str,
        record: dict,
        *,
        dedupe_key: str | None = None,
        ensure_ascii: bool = True,
        default: Any | None = None,
    ) -> StreamPosition:
        del dedupe_key
        normalized = normalize_stream(stream)
        path = self.path_for_stream(normalized)
        seq = _count_records(path) + 1
        source_offset = append_jsonl(path, record, ensure_ascii=ensure_ascii, default=default)
        return StreamPosition(
            stream=normalized,
            seq=seq,
            backend="filesystem",
            source_file=str(path),
            source_offset=source_offset,
        )

    def tail(self, stream: str, limit: int) -> list[dict]:
        return tail_jsonl_records(self.path_for_stream(stream), limit)

    def iter_range(
        self,
        stream: str,
        start_seq: int | None = None,
        end_seq: int | None = None,
    ) -> Iterator[dict]:
        for seq, record in enumerate(iter_jsonl_records(self.path_for_stream(stream)), 1):
            if start_seq is not None and seq < start_seq:
                continue
            if end_seq is not None and seq > end_seq:
                break
            if isinstance(record, dict):
                yield record

    def find(self, stream: str, **predicate: Any) -> list[dict]:
        return [
            record
            for record in self.iter_range(stream)
            if all(record.get(key) == value for key, value in predicate.items())
        ]

    def export_jsonl(self, stream: str, output_path: str | Path) -> ExportResult:
        source = self.path_for_stream(stream)
        target = Path(output_path)
        text = source.read_text(encoding="utf-8") if source.is_file() else ""
        atomic_write_text(target, text)
        return ExportResult(
            stream=normalize_stream(stream),
            output_path=target,
            record_count=sum(1 for line in text.splitlines() if line.strip()),
            bytes_written=len(text.encode("utf-8")),
        )

    def replace_from_file(self, stream: str, input_path: str | Path) -> ExportResult:
        source = Path(input_path)
        target = self.path_for_stream(stream)
        text = source.read_text(encoding="utf-8") if source.is_file() else ""
        if source.resolve() != target.resolve():
            atomic_write_text(target, text)
        return ExportResult(
            stream=normalize_stream(stream),
            output_path=target,
            record_count=sum(1 for line in text.splitlines() if line.strip()),
            bytes_written=len(text.encode("utf-8")),
        )

    def health(self) -> StorageHealth:
        return StorageHealth(
            status="ok",
            backend="filesystem",
            streams=sorted(self._streams),
        )


class NoKVSegmentedJsonlStreamStore:
    """Segmented JSONL stream store backed by an injected NoKV-like client."""

    def __init__(
        self,
        agent_dir: str | Path,
        stream_roots: Mapping[str, str],
        backend: Any,
        *,
        segment_max_records: int = 5000,
    ) -> None:
        self.agent_dir = Path(agent_dir)
        self._roots = {
            normalize_stream(stream): self._normalize_remote_root(root)
            for stream, root in stream_roots.items()
        }
        self._backend = backend
        self._segment_max_records = max(1, int(segment_max_records))
        self._last_error: str | None = None

    @staticmethod
    def _normalize_remote_root(root: str) -> str:
        return "/" + "/".join(part for part in root.split("/") if part)

    def handles(self, stream: str) -> bool:
        return normalize_stream(stream) in self._roots

    def _root(self, stream: str) -> str:
        normalized = normalize_stream(stream)
        if normalized not in self._roots:
            raise ValueError(f"JSONL stream is not configured for NoKV: {stream}")
        return self._roots[normalized]

    def _call(self, method_names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
        for name in method_names:
            method = getattr(self._backend, name, None)
            if not callable(method):
                continue
            if kwargs:
                try:
                    return method(*args, **kwargs)
                except TypeError:
                    return method(*args)
            return method(*args)
        raise RuntimeError("NoKV backend does not support any of: " + ", ".join(method_names))

    def _read_text(self, path: str) -> str:
        return _content_from_result(self._call(("read", "cat", "get"), path))

    def _write_text(self, path: str, text: str, *, metadata: dict | None = None) -> Any:
        return self._call(
            ("write", "put", "put_file", "put_artifact", "pipe_file"),
            path,
            text,
            metadata=metadata,
        )

    def _manifest_path(self, stream: str) -> str:
        return f"{self._root(stream)}/manifest.json"

    def _empty_manifest(self, stream: str) -> dict:
        return {
            "schema_version": 1,
            "stream": normalize_stream(stream),
            "agent_path": str(self.agent_dir),
            "record_count": 0,
            "first_seq": None,
            "last_seq": 0,
            "generation": 0,
            "segments": [],
            "updated_at": utc_now_iso().replace("+00:00", "Z"),
        }

    def _read_manifest(self, stream: str) -> dict:
        try:
            manifest = json.loads(self._read_text(self._manifest_path(stream)))
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return self._empty_manifest(stream)
        if not isinstance(manifest, dict):
            return self._empty_manifest(stream)
        return manifest

    def _write_manifest(self, stream: str, manifest: dict) -> None:
        manifest["updated_at"] = utc_now_iso().replace("+00:00", "Z")
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self._write_text(
            self._manifest_path(stream),
            payload,
            metadata={
                "lingtai_kind": "jsonl_stream_manifest",
                "stream": normalize_stream(stream),
                "generation": manifest.get("generation"),
            },
        )

    def _segment_name(self, generation: int, first_seq: int) -> str:
        return f"g{generation:018d}-{first_seq:018d}.jsonl"

    def _segment_path(self, stream: str, name: str) -> str:
        return f"{self._root(stream)}/segments/{name}"

    def append(
        self,
        stream: str,
        record: dict,
        *,
        dedupe_key: str | None = None,
        ensure_ascii: bool = True,
        default: Any | None = None,
    ) -> StreamPosition:
        normalized = normalize_stream(stream)
        manifest = self._read_manifest(normalized)
        dedupe = dict(manifest.get("dedupe_keys") or {})
        if dedupe_key and dedupe_key in dedupe:
            seq = int(dedupe[dedupe_key])
            return StreamPosition(
                stream=normalized,
                seq=seq,
                backend="nokv",
                generation=int(manifest.get("generation") or 0),
            )

        seq = int(manifest.get("last_seq") or 0) + 1
        generation = int(manifest.get("generation") or 0) + 1
        line = _jsonl_line(record, ensure_ascii=ensure_ascii, default=default)
        name = self._segment_name(generation, seq)
        payload_bytes = line.encode("utf-8")
        segment = {
            "name": name,
            "first_seq": seq,
            "last_seq": seq,
            "record_count": 1,
            "bytes": len(payload_bytes),
            "sha256": sha256(payload_bytes).hexdigest(),
        }
        self._write_text(
            self._segment_path(normalized, name),
            line,
            metadata={
                "lingtai_kind": "jsonl_stream_segment",
                "stream": normalized,
                "first_seq": seq,
                "last_seq": seq,
            },
        )
        segments = list(manifest.get("segments") or [])
        segments.append(segment)
        manifest.update(
            {
                "schema_version": 1,
                "stream": normalized,
                "agent_path": str(self.agent_dir),
                "record_count": int(manifest.get("record_count") or 0) + 1,
                "first_seq": manifest.get("first_seq") or 1,
                "last_seq": seq,
                "generation": generation,
                "segments": segments,
            }
        )
        if dedupe_key:
            dedupe[dedupe_key] = seq
            manifest["dedupe_keys"] = dedupe
        self._write_manifest(normalized, manifest)
        self._last_error = None
        return StreamPosition(
            stream=normalized,
            seq=seq,
            backend="nokv",
            segment=name,
            generation=generation,
        )

    def _iter_segment_lines(self, stream: str) -> Iterator[tuple[int, str]]:
        manifest = self._read_manifest(stream)
        for segment in manifest.get("segments") or []:
            first_seq = int(segment.get("first_seq") or 0)
            content = self._read_text(self._segment_path(stream, segment["name"]))
            for idx, line in enumerate(content.splitlines(), first_seq):
                if line.strip():
                    yield idx, line

    def tail(self, stream: str, limit: int) -> list[dict]:
        if limit <= 0:
            return []
        records = list(self.iter_range(stream))
        return records[-limit:]

    def iter_range(
        self,
        stream: str,
        start_seq: int | None = None,
        end_seq: int | None = None,
    ) -> Iterator[dict]:
        normalized = normalize_stream(stream)
        for seq, line in self._iter_segment_lines(normalized):
            if start_seq is not None and seq < start_seq:
                continue
            if end_seq is not None and seq > end_seq:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record

    def find(self, stream: str, **predicate: Any) -> list[dict]:
        return [
            record
            for record in self.iter_range(stream)
            if all(record.get(key) == value for key, value in predicate.items())
        ]

    def export_jsonl(self, stream: str, output_path: str | Path) -> ExportResult:
        normalized = normalize_stream(stream)
        target = Path(output_path)
        pieces: list[str] = []
        count = 0
        for _, line in self._iter_segment_lines(normalized):
            pieces.append(line.rstrip("\n"))
            count += 1
        text = ("\n".join(pieces) + "\n") if pieces else ""
        atomic_write_text(target, text)
        return ExportResult(
            stream=normalized,
            output_path=target,
            record_count=count,
            bytes_written=len(text.encode("utf-8")),
        )

    def replace_from_file(self, stream: str, input_path: str | Path) -> ExportResult:
        normalized = normalize_stream(stream)
        source = Path(input_path)
        lines = [
            line.rstrip("\n")
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] if source.is_file() else []
        old_manifest = self._read_manifest(normalized)
        generation = int(old_manifest.get("generation") or 0) + 1
        segments: list[dict] = []
        for chunk_index in range(0, len(lines), self._segment_max_records):
            chunk = lines[chunk_index:chunk_index + self._segment_max_records]
            first_seq = chunk_index + 1
            last_seq = first_seq + len(chunk) - 1
            name = self._segment_name(generation, first_seq)
            content = "\n".join(chunk) + "\n"
            payload_bytes = content.encode("utf-8")
            segment = {
                "name": name,
                "first_seq": first_seq,
                "last_seq": last_seq,
                "record_count": len(chunk),
                "bytes": len(payload_bytes),
                "sha256": sha256(payload_bytes).hexdigest(),
            }
            self._write_text(
                self._segment_path(normalized, name),
                content,
                metadata={
                    "lingtai_kind": "jsonl_stream_segment",
                    "stream": normalized,
                    "first_seq": first_seq,
                    "last_seq": last_seq,
                },
            )
            segments.append(segment)
        manifest = {
            "schema_version": 1,
            "stream": normalized,
            "agent_path": str(self.agent_dir),
            "record_count": len(lines),
            "first_seq": 1 if lines else None,
            "last_seq": len(lines),
            "generation": generation,
            "segments": segments,
        }
        self._write_manifest(normalized, manifest)
        text = ("\n".join(lines) + "\n") if lines else ""
        self._last_error = None
        return ExportResult(
            stream=normalized,
            output_path=source,
            record_count=len(lines),
            bytes_written=len(text.encode("utf-8")),
        )

    def health(self) -> StorageHealth:
        return StorageHealth(
            status="degraded" if self._last_error else "ok",
            backend="nokv",
            streams=sorted(self._roots),
            last_error=self._last_error,
        )


class MirrorJsonlStreamStore:
    """Phase 1 dual-write store: local primary plus best-effort NoKV mirror."""

    def __init__(
        self,
        primary: FilesystemJsonlStreamStore,
        mirror: NoKVSegmentedJsonlStreamStore,
        *,
        health_writer: Callable[[StorageHealth], None] | None = None,
    ) -> None:
        self.primary = primary
        self.mirror = mirror
        self._last_error: str | None = None
        self._last_error_stream: str | None = None
        self._health_writer = health_writer

    def _publish_health(self) -> None:
        if self._health_writer is None:
            return
        try:
            self._health_writer(self.health())
        except Exception:
            pass

    def _mark_ok(self) -> None:
        if self._last_error is None and self._last_error_stream is None:
            return
        self._last_error = None
        self._last_error_stream = None
        self._publish_health()

    def _mark_degraded(self, stream: str, exc: Exception) -> None:
        self._last_error = _safe_mirror_error(exc)
        self._last_error_stream = normalize_stream(stream)
        self._publish_health()

    def handles(self, stream: str) -> bool:
        normalized = normalize_stream(stream)
        return self.primary.handles(normalized)

    def append(
        self,
        stream: str,
        record: dict,
        *,
        dedupe_key: str | None = None,
        ensure_ascii: bool = True,
        default: Any | None = None,
    ) -> StreamPosition:
        position = self.primary.append(
            stream,
            record,
            dedupe_key=dedupe_key,
            ensure_ascii=ensure_ascii,
            default=default,
        )
        mirrored = True
        if self.mirror.handles(stream):
            try:
                self.mirror.append(
                    stream,
                    record,
                    dedupe_key=dedupe_key,
                    ensure_ascii=ensure_ascii,
                    default=default,
                )
                self._mark_ok()
            except Exception as exc:  # noqa: BLE001 - local primary must survive mirror failure
                mirrored = False
                self._mark_degraded(stream, exc)
        return StreamPosition(
            stream=position.stream,
            seq=position.seq,
            backend=position.backend,
            source_file=position.source_file,
            source_offset=position.source_offset,
            segment=position.segment,
            generation=position.generation,
            mirrored=mirrored,
        )

    def tail(self, stream: str, limit: int) -> list[dict]:
        return self.primary.tail(stream, limit)

    def iter_range(
        self,
        stream: str,
        start_seq: int | None = None,
        end_seq: int | None = None,
    ) -> Iterator[dict]:
        return self.primary.iter_range(stream, start_seq, end_seq)

    def find(self, stream: str, **predicate: Any) -> list[dict]:
        records = self.primary.find(stream, **predicate)
        if records or not self.mirror.handles(stream):
            return records
        try:
            records = self.mirror.find(stream, **predicate)
            self._mark_ok()
            return records
        except Exception as exc:  # noqa: BLE001 - mirror lookup is best effort
            self._mark_degraded(stream, exc)
            return []

    def export_jsonl(self, stream: str, output_path: str | Path) -> ExportResult:
        return self.primary.export_jsonl(stream, output_path)

    def replace_from_file(self, stream: str, input_path: str | Path) -> ExportResult:
        result = self.primary.replace_from_file(stream, input_path)
        if self.mirror.handles(stream):
            try:
                self.mirror.replace_from_file(stream, input_path)
                self._mark_ok()
            except Exception as exc:  # noqa: BLE001
                self._mark_degraded(stream, exc)
        return result

    def health(self) -> StorageHealth:
        return StorageHealth(
            status="degraded" if self._last_error else "ok",
            backend="mirror",
            streams=sorted(self.primary._streams),
            last_error=self._last_error,
            last_error_stream=self._last_error_stream,
            updated_at=utc_now_iso().replace("+00:00", "Z"),
        )
