cd ..
git clone https://github.com/565353780/mesh-sample.git
git clone https://github.com/565353780/diff-curvature.git

pip install -U numpy open3d scipy scikit-learn matplotlib \
  tqdm

cd mesh-sample
./setup.sh

cd ../diff-curvature
./setup.sh

cd ../mesh-cut
./compile.sh
