# FabSoften

## Prerequisites
On Windows, [7zip](https://www.7-zip.org/) is needed for extracting [bzip2](https://en.wikipedia.org/wiki/Bzip2) files. Make sure `7z -h` works on the command line, for example:

```powershell
C:\Users\Gnimuc>7z -h

7-Zip 19.00 (x64) : Copyright (c) 1999-2018 Igor Pavlov : 2019-02-21

Usage: 7z <command> [<switches>...] <archive_name> [<file_names>...] [@listfile]

<Commands>
  a : Add files to archive
  b : Benchmark
  d : Delete files from archive
  e : Extract files from archive (without using directory names)
  h : Calculate hash values for files
  i : Show information about supported formats
  l : List contents of archive
  rn : Rename files in archive
  t : Test integrity of archive
  u : Update files to archive
  x : eXtract files with full paths
```

## Build
On Windows, it's highly recommended to use `Powershell` or [Windows Terminal](https://aka.ms/terminal) to build this project:

```powershell
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/opt
cmake --build . --config Release --target Install
```

The above example installs everything to the `opt` folder in your home directory(e.g. `C:\Users\your_user_name`) and this can be changed by editing the definition of `CMAKE_INSTALL_PREFIX` when configuring CMake.

Note that, `${CMAKE_INSTALL_PREFIX}\bin` may need to be added in the environment variable `Path`.

## Examples