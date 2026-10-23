"""Vehicle damage detection utilities."""

TARGET_CLASSES = ("Crack", "Scratch", "Dent", "Broken Glass")
LABEL_TO_NAME = {index + 1: name for index, name in enumerate(TARGET_CLASSES)}

__all__ = ["TARGET_CLASSES", "LABEL_TO_NAME"]
