"""Input data, method interface and encoded bytes."""

from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol

Category = Literal["A", "B", "C", "D", "E", "F", "G"]
Fidelity = Literal["native", "adapted"]


class ResourceLimitError(RuntimeError):
    """A declared codec/table/buffer limit, distinct from a measured zero saving."""


@dataclass(frozen=True)
class MethodSpec:
    id: str
    name: str
    category: Category
    fidelity: Fidelity
    original_technique: str
    adaptation: str = ""
    implementation_version: str = ""
    baseline_method_id: str = ""
    comparison_kind: str = ""


@dataclass(frozen=True)
class Component:
    id: str
    role: str
    path: str
    offset: int
    length: int
    storage_format: str


@dataclass(frozen=True)
class Target:
    id: str
    logical_shape: tuple[int, ...]
    components: tuple[Component, ...]
    structure: Mapping[str, object] = field(default_factory=dict)
    inference: Mapping[str, object] = field(default_factory=dict)

    @property
    def original_bytes(self) -> int:
        return sum(c.length for c in self.components)


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    revision: str
    coverage: Literal["subset", "complete"]
    targets: tuple[Target, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Support:
    applicable: bool
    reason: str


@dataclass(frozen=True)
class EncodedArtifact:
    """All bytes needed for decoding, including metadata and variable assets."""

    payload: bytes
    side_information: bytes = b""

    @property
    def total_bytes(self) -> int:
        return len(self.payload) + len(self.side_information)


class Method(Protocol):
    """One complete technique; decode uses only the artifact on a fresh instance."""

    spec: MethodSpec

    def check_support(self, target: Target, parameters: Mapping[str, object]) -> Support: ...

    def encode(self, target: Target, components: Mapping[str, bytes],
               parameters: Mapping[str, object]) -> EncodedArtifact: ...

    def decode(self, artifact: EncodedArtifact) -> Mapping[str, bytes]: ...
