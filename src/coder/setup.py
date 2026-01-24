from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = dict(
    rans=Pybind11Extension(
        name=f"cbench.rans",
        sources=["cbench/csrc/rans/rans_interface.cpp",],
        language="c++",
        include_dirs=["cbench/csrc/rans"],
        extra_compile_args=['-std=c++17', '-g'], # enable debugging symbols
    ),
)

cmdclass = {
    'build_ext': build_ext,
}

# skip check requires to speed up installation, but may be needed for production
install_requires = [
]

def generate_pyi(module_name):
    from pybind11_stubgen import ModuleStubsGenerator

    module = ModuleStubsGenerator("cbench."+module_name)
    module.parse()
    module.write_setup_py = False

    with open("cbench/%s.pyi" % module_name, "w") as fp:
        fp.write("#\n# Automatically generated file, do not edit!\n#\n\n")
        fp.write("\n".join(module.to_lines()))

# TODO: 'CC=clang CXX=clang++' on macos ?
setup(
    name="cbench",
    version="0.1",
    description="Data Compression Benchmark",
    url="",
    packages=find_packages(),
    install_requires=install_requires,
    ext_modules=list(ext_modules.values()),
    cmdclass=cmdclass,
)

for module_name in ext_modules.keys():
    generate_pyi(module_name)