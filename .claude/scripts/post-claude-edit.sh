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
    echo "Auto-linting: $file_path"

    # Detect repo type and use appropriate linter
    if git rev-parse --git-dir >/dev/null 2>&1; then
        # Git checkout - use lintrunner
        if command -v lintrunner >/dev/null 2>&1; then
            lintrunner -a --paths "$file_path" 2>/dev/null || true
        elif command -v uv >/dev/null 2>&1; then
            uv run lintrunner -a --paths "$file_path" 2>/dev/null || true
        fi
    elif hg root >/dev/null 2>&1; then
        # Hg checkout (fbcode) - use arc
        arc f "$file_path" 2>/dev/null || true
        arc lint -a "$file_path" 2>/dev/null || true
    fi

    # Output review reminder for non-automatable checks
    cat << 'EOF'
Review .claude/ for: duplicates, verbosity, misplaced content.
EOF
fi
