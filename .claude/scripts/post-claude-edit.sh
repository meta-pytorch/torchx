#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Consume stdin (required for hooks)
stdin_data=$(cat)

# Extract file path from JSON
file_path=$(echo "$stdin_data" | jq -r '.tool_input.file_path // empty')

if [[ -z "$file_path" ]]; then
    exit 0
fi

# Check if file is in .claude directory
if [[ "$file_path" == *"/.claude/"* ]] && [[ -f "$file_path" ]]; then
    # Run linting deterministically
    echo "Auto-linting: $file_path"
    arc f "$file_path" 2>/dev/null || true
    arc lint -a "$file_path" 2>/dev/null || true

    # Output review reminder for non-automatable checks
    cat << 'EOF'
Review .claude/ for: duplicates, verbosity, misplaced content, fb-* naming.
EOF
fi
