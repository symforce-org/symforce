#!/bin/bash
#
# Prints the list of authors who contributed to a given release.
# You can copy & paste the result into a Markdown document.
#
# To obtain the list of authors who contributed to the v0.3.0 release:
#
# bin/release_authors.sh v0.2.0 v0.3.0
#
# To obtain the list of authors who contributed to upcoming (not yet tagged)
# v0.4.0 release:
#
# bin/release_authors.sh v0.3.0 master

set -e

echo "People who contributed to the release:"
git log --reverse --format="* %aN  " $1..$2 | awk ' !x[$0]++'
