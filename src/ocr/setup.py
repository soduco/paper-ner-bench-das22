from setuptools import Extension, setup
from pybind11.setup_helpers import Pybind11Extension

sourcefiles = [
    "isri_tools_src/align.cpp",
    "isri_tools_src/Modules/accrpt.c",
    "isri_tools_src/Modules/charclass.c",
    "isri_tools_src/Modules/ci.c",
    "isri_tools_src/Modules/dist.c",
    "isri_tools_src/Modules/edorpt.c",
    "isri_tools_src/Modules/list.c",
    "isri_tools_src/Modules/sort.c",
    "isri_tools_src/Modules/stopword.c",
    "isri_tools_src/Modules/sync.c",
    "isri_tools_src/Modules/table.c",
    "isri_tools_src/Modules/text.c",
    "isri_tools_src/Modules/unicode.c",
    "isri_tools_src/Modules/util.c",
    "isri_tools_src/Modules/wacrpt.c",
    "isri_tools_src/Modules/word.c",
    ]

ext = Pybind11Extension("isri_tools", sourcefiles)

setup(
    ext_modules = [ext]
)
