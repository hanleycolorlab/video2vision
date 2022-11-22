FROM ubuntu:20.04

    ########################
    # Install apt packages #
    ########################

# Upgrade and install packages:
#
# ca-certificates: Certificates needed for pip to authenticate.
# fonts-inconsolata: Inconsolata font, used in notebooks.
# libraw-dev: For opening RAW files.
# make: Allows building makefiles; convenience for working with sphinx.
# python3-dev: Python with development headers.
# wget: Needed for fetching source code for stuff not handled by package.
#
# --no-install-recommends prevents installing anything but what is strictly
# necessary. apt-get clean and rm -rf /var/lib/apt/lists/* cleans out the cache
# to avoid saving two copies of the packages. See:
#
# https://www.augmentedmind.de/2022/02/06/optimize-docker-image-size/
#
# For details.

RUN apt update && apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    ca-certificates \
    fonts-inconsolata \
    libraw-dev \
    make \
    python3-dev \
    wget \
  && apt clean \
  && rm -rf /var/lib/apt/lists/*

    ###############
    # Install pip #
    ###############

# Install pip. We do this outside of apt-get to avoid leaving the system in an
# inconsistent state if we upgrade pip. The "-O -" flag means to write the file
# to stdout instead of saving it.

RUN wget -O - https://bootstrap.pypa.io/get-pip.py | python3

    ########################
    # Install video2vision #
    ########################

COPY . /opt/video2vision
RUN python3 -m pip install --no-cache-dir /opt/video2vision

# Install optional dependencies
RUN python3 -m pip install --no-cache-dir -r /opt/video2vision/requirements-optional.txt

    #####################
    # Finishing Touches #
    #####################

# Tricks Intel MKL into using AVX2 instructions on AMD Threadripper, as discussed at:
# https://github.com/flame/blis/issues/312
ENV MKL_DEBUG_CPU_TYPE=5

# Run tests.
RUN python3 /opt/video2vision/tests/run_all_tests.py
