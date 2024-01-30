#!/bin/bash

# Check if commit message is provided as argument, otherwise use default message.
if [ -z "$1" ]; then
    msg="Automatic commit and push."
else
    msg="$1"
fi

# Commit and push changes.
git add .
git commit -m "$msg"
git push

# Exit with success.
exit 0