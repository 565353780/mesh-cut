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

if [ ! -f "./mesh_graph_cut/Lib/mcut/build/bin/libmcut.so" ]; then
  cd ./mesh_graph_cut/Lib/mcut/
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
