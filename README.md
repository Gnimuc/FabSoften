# FabSoften

[![CI](https://github.com/Gnimuc/FabSoften/actions/workflows/CI.yml/badge.svg)](https://github.com/Gnimuc/FabSoften/actions/workflows/CI.yml)
[![Build Status](https://dev.azure.com/Gnimuc/FabSoften/_apis/build/status/Gnimuc.FabSoften?branchName=main)](https://dev.azure.com/Gnimuc/FabSoften/_build/latest?definitionId=1&branchName=main)
[![Windows](https://svgshare.com/i/ZhY.svg)](https://svgshare.com/i/ZhY.svg)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://gnimuc.github.io/FabSoften)

<!-- This project is mainly developed on Windows, and there is no intention to support other platforms at the moment. PRs are always welcome! -->

This project is an unofficial implementation of [FabSoften: face beautification via dynamic skin smoothing, guided feathering, and texture restoration](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Velusamy_FabSoften_Face_Beautification_via_Dynamic_Skin_Smoothing_Guided_Feathering_and_CVPRW_2020_paper.pdf). 

## Prerequisites

Before you begin, you'll install the following softwares on your system:

- [OpenCV 4](https://github.com/opencv/opencv): Open Source Computer Vision Library
- [doxygen](https://github.com/doxygen/doxygen)(optional): The de facto standard tool for generating documentation from annotated C++ sources
- [7zip](https://www.7-zip.org/)(optional): A file archiver with a high compression ratio

[7zip](https://www.7-zip.org/) is used for extracting [bzip2](https://en.wikipedia.org/wiki/Bzip2) files(e.g. `shape_predictor_68_face_landmarks.dat.bz2`). If you'd like to [manually provide those models](./models/README.md) without auto-downloading through CMake, there is no need to use 7zip. If not, make sure `7z -h` works on the command line, for example:

```powershell
C:\Users\Gnimuc>7z -h

7-Zip 19.00 (x64) : Copyright (c) 1999-2018 Igor Pavlov : 2019-02-21

Usage: 7z <command> [<switches>...] <archive_name> [<file_names>...] [@listfile]

<Commands>
  a : Add files to archive
  b : Benchmark
...
```

<!-- On Linux or macOS, you need `bzip2`. -->

## Build
On Windows, it's highly recommended to use `Powershell` or [Windows Terminal](https://aka.ms/terminal) to build this project:

<!-- I believe pro-Linux/macOS users are savvy enough to fix any problems on their own. ;) -->

```powershell
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/your_install_prefix_dir
cmake --build . --config Release
ctest --Release
```

## Examples

## LICENSE

FabSoften is primarily distributed under the terms of the MIT license.

All _code_ in this repository is released under the terms of the MIT license.

Those [assets](./assets), [models](./models), and [external dependencies](./external) are released under their licenses, respectively.
