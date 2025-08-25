cd ..
git clone git@github.com:565353780/mesh-sample.git
git clone git@github.com:565353780/diff-curvature.git

pip install -U numpy open3d scipy scikit-learn matplotlib \
  tqdm

cd mesh-sample
./dev_setup.sh

cd ../diff-curvature
./dev_setup.sh
