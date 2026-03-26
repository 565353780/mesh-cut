cd ..
git clone git@github.com:565353780/mesh-sample.git
git clone git@github.com:565353780/diff-curvature.git
git clone git@github.com:565353780/hole-fill.git

pip install scikit-learn matplotlib shapely rtree \
  mapbox-earcut

cd mesh-sample
./dev_setup.sh

cd ../diff-curvature
./dev_setup.sh

cd ../hole-fill
./dev_setup.sh

cd ../mesh-cut
./compile.sh
