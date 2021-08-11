#!/usr/bin/env bash

set -x

if [[ "${TRIGGER_FEEDSTOCK}" != "yes" ]]; then
    exit 0;
fi
if [[ "${TRAVIS_PULL_REQUEST}" != "false" ]]; then
    echo "Testing a pull request, feedstock is not triggered.";
    exit 0;
fi
if [[ "${GH_TOKEN}" == "" ]]; then
    echo "Testing a fork, feedstock is not triggered.";
    exit 0;
fi


cd $PYTHON_SOURCE_DIR;
git clean -dfx;
export ver=`git describe --tags`
if [[ $ver == "v"* ]]
then
  ver=${ver:1};
fi

export symengine_ver=`cat symengine_version.txt`
if [[ $symengine_ver == "v"* ]]
then
  symengine_ver=${symengine_ver:1};
fi

export commit=`git rev-parse HEAD`

git config --global user.name "Isuru Fernando"
git config --global user.email "isuruf@gmail.com"

set +x
git clone "https://${GH_TOKEN}@github.com/symengine/python-symengine-feedstock.git" feedstock -q
set -x

cd feedstock
if [[ "${TRAVIS_TAG}" != "" ]]; then
    git checkout tagged
else
    echo "Testing merge. Not triggering feedstock"
    exit 0
    # git checkout dev
fi

sed -ie '1,2d' recipe/meta.yaml
sed -i '1s/^/{% set version = "'${ver}'" %}\n/' recipe/meta.yaml
sed -i '1s/^/{% set commit = "'${commit}'" %}\n/' recipe/meta.yaml
sed -i 's/^    - symengine     [0-9].*/    - symengine     '${symengine_ver}'/' recipe/meta.yaml
git add recipe/meta.yaml
git commit -m "Update symengine version to ${ver}"
git push -q

