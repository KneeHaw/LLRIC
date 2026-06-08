import os
import subprocess
from pathlib import Path
from pybind11 import Pybind11Extension, build_ext
from setuptools import find_packages, setup

# bash 
# /home/kneehaw/anaconda3/envs/MIT_ToyModel/bin/python /home/kneehaw/python_projects/LLRIC/isolated_TinyLIC/setup.py build_ext --inplace

cwd = Path(__file__).resolve().parent

package_name = "src"

def get_extensions():
    ext_dirs = cwd / package_name / "cpp_exts"
    ext_modules = []

    # Add rANS module
    # rans_lib_dir = cwd / "third_party/ryg_rans"
    rans_ext_dir = ext_dirs / "rans"
    rans_lib_dir = rans_ext_dir

    extra_compile_args = ["-std=c++17"]
    if os.getenv("DEBUG_BUILD", None):
        extra_compile_args += ["-O0", "-g", "-UNDEBUG"]
    else:
        extra_compile_args += ["-O3"]
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}.ans",
            sources=[str(s) for s in rans_ext_dir.glob("*.cpp")],
            language="c++",
            include_dirs=[rans_lib_dir, rans_ext_dir],
            extra_compile_args=extra_compile_args,
        )
    )

    # Add ops
    ops_ext_dir = ext_dirs / "ops"
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}._CXX",
            sources=[str(s) for s in ops_ext_dir.glob("*.cpp")],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    )

    return ext_modules


setup(name="src", ext_modules=get_extensions())