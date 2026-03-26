cd ..
git clone https://github.com/565353780/mesh-sample.git
git clone https://github.com/565353780/diff-curvature.git
git clone https://github.com/565353780/hole-fill.git

pip install scikit-learn matplotlib shapely rtree \
  mapbox-earcut

cd mesh-sample
./setup.sh

cd ../diff-curvature
./setup.sh

cd ../hole-fill
./setup.sh

cd ../mesh-cut
./compile.sh
