#!/bin/bash

# Generates release notes from PR titles.
# Arguments are the same for release_authors.sh

git log $1...$2 > notes.txt
cat notes.txt | grep "Merge pull request" -A 2 | sed '/^--/d' | sed '/^    $/d' > notes2.txt
rm notes.txt
while read p; do
  if [[ "$p" == "Merge pull request"* ]]; then
    pr_num=$(echo $p | cut -b 21- | cut -b -4)
  else
    echo "- $p - #$pr_num" >> notes.txt
  fi
done <notes2.txt
rm notes2.txt

cat notes.txt
rm notes.txt
