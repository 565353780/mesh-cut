cd ..
git clone git@github.com:565353780/diff-curvature.git

pip install -U numpy open3d scipy scikit-learn matplotlib \
  tqdm numba

cd diff-curvature
./dev_setup.sh
