#!/bin/bash

NAME=`basename "$0"`

if [ ! -f ".clang-format" ]; then
    echo ".clang-format file not found!"
    exit 1
fi

CLANG_FORMAT="clang-format"

which "clang-format-3.9" > /dev/null && CLANG_FORMAT="clang-format-3.9"
which "clang-format-3.8" > /dev/null && CLANG_FORMAT="clang-format-3.8"

FILES=`git ls-files | grep -E "\.(cpp|h|hpp|c)$" | grep -Ev "symengine/utilities" | grep -Ev "cmake/"`

for FILE in $FILES; do
    if [ "$NAME" != "pre-commit" ]; then
        # if this is not a pre-commit hook format code inplace
        $CLANG_FORMAT -i $FILE
    else
        staged_file=`git show :$FILE`
        formatted_file=`cat << EOF | $CLANG_FORMAT
$staged_file
EOF`
        if [ "$staged_file" != "$formatted_file" ]; then
            actual_file=`cat $FILE`
            if [ "$actual_file" != "$staged_file" ]; then
                echo "WARNING: $FILE is not formatted properly. Cannot fix formatting as there are unstaged changes"
            else
                echo "Fixing formatting of $FILE automatically"
                $CLANG_FORMAT -i $FILE
                git add $FILE
            fi
        fi
    fi
done

