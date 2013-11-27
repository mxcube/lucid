from distutils.core import setup
import sys

setup(name = "lucid",version = "0.1",
      description = "Loop and uCrystals Identification", 
      author="E. Francois, J. Kieffer, M. Guijarro (ESRF)",
      package_dir={"lucid": "lucid"},
      packages = ["lucid"])

