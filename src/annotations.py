from dataclasses import dataclass


@dataclass
class Annotation:
    class_name: str
    color: tuple
