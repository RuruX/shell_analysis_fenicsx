#!/bin/bash

# install package and dependencies
apt-get -y install python3-pip
pip3 install redbaron
apt-get -y install software-properties-common
add-apt-repository ppa:fenics-packages/fenics
apt-get -y update
apt-get -y install --no-install-recommends fenics
pip3 install -e .

# build docs
make -C docs clean
make -C docs html

# copy docs to temporary directory
docroot=`mktemp -d`
here=`pwd`
rsync -av "${here}/docs/_build/html/" "${docroot}"

# create git repository in temporary directory
pushd "${docroot}"
git init
git remote add deploy "https://token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git checkout -b gh-pages

# add .nojekyll so GitHub generates site from docs
touch .nojekyll

# add README for gh-pages branch
cat > README.md <<EOF
# shell_analysis_fenicsx

This branch contains the generated documentation pages for
[shell_analysis_fenicsx](https://rurux.github.io/shell_analysis_fenicsx).

EOF

# commit changes
git config --global user.name "${GITHUB_ACTOR}"
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"
git add .
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
msg="Updating Docs for commit ${GITHUB_SHA} from ${GITHUB_REF} by ${GITHUB_ACTOR}"
git commit -am "${msg}"

# update gh-pages
git push deploy gh-pages --force
