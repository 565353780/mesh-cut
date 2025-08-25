if [ "$(uname)" = "Darwin" ]; then
  PROCESSOR_NUM=$(sysctl -n hw.physicalcpu)
elif [ "$(uname)" = "Linux" ]; then
  PROCESSOR_NUM=$(cat /proc/cpuinfo | grep "processor" | wc -l)
fi

export MAX_JOBS=${PROCESSOR_NUM}

export CC=$(which gcc)
export CXX=$(which g++)
echo "Using CC: $CC"
echo "Using CXX: $CXX"

COMPILE_MCUT=true
if [ "$(uname)" = "Darwin" ]; then
  if [ -f "./mesh_cut/Lib/mcut/build/bin/libmcut.dylib" ]; then
    COMPILE_MCUT=false
  fi
fi
if [ "$(uname)" = "Linux" ]; then
  if [ -f "./mesh_cut/Lib/mcut/build/bin/libmcut.so" ]; then
    COMPILE_MCUT=false
  fi
fi

if [ $COMPILE_MCUT = true ]; then
  cd ./mesh_cut/Lib/mcut/
  rm -rf build
  mkdir build
  cd build
  cmake ..
  make -j

  cd ../../../../
fi

pip uninstall cut-cpp -y

rm -rf build
rm -rf *.egg-info
rm *.so

# bear -- python setup.py build_ext --inplace
python setup.py build_ext --inplace
mv compile_commands.json build

pip install .
