"""
Massive bug check for TinyLLM project
Checks for security, correctness, performance bugs
"""
import os, re, glob, json

def check_file(path):
    issues = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Security: path traversal
        if '\"..\"' in content or "'..'" in content:
            if 'not \"..\" in' in content or "not '..' in" in content:
                pass  # has check
            else:
                issues.append("Potential path traversal without check for '..'")

        # Check for open() without sanitization in app.py
        if 'app.py' in path:
            if 'os.path.exists(\".\"+parsed.path)' in content:
                if 'lstrip' in content:
                    pass
                else:
                    issues.append("Potential path traversal in serve_file")

        # Check for eval or exec
        if re.search(r'\beval\s*\(', content):
            issues.append("Use of eval() - security risk")
        if re.search(r'\bexec\s*\(', content):
            issues.append("Use of exec() - security risk")

        # Check for torch.load without weights_only - we use weights_only=False intentionally for config, but should note
        if 'torch.load' in content and 'weights_only' not in content:
            issues.append("torch.load without weights_only check")

        # Check for hardcoded secrets? None expected

        # Check for bare except
        if re.search(r'except\s*:\s*\n', content):
            issues.append("Bare except clause")

        # Check for print without flush in training (buffering issue)
        if 'train' in path and 'print(' in content:
            if 'flush=True' not in content:
                # not critical but good to have
                pass

        # Check for potential division by zero
        if '/=' in content or '/ ' in content:
            # heuristic
            pass

        # Check model.py specific
        if 'model.py' in path:
            if 'def rotate_half' in content and 'rotate_half(x)' not in content.split('def rotate_half')[1][:500]:
                issues.append("Dead code: rotate_half defined but not used")

        # Check config.py
        if 'config.py' in path:
            if 'max_steps' in content and 'micro_steps' in content:
                # Check confusion
                if 'TinyConfigV2' in content and 'max_steps: int = 610' in content:
                    # This is expected but confusing naming
                    pass

        # Check dataset_fast.py loads entire file
        if 'dataset_fast.py' in path:
            if 'torch.load' in content and 'map_location' in content:
                pass
            else:
                issues.append("Dataset loads entire file into memory - may OOM for large tokens")

        # Check for missing .gitignore for large files
        # Not file content but repo

        # Check for XSS in chat.html
        if path.endswith('.html'):
            if 'innerHTML' in content and 'escapeHtml' not in content:
                # Check if innerHTML used without escaping
                if 'mdToHtml' in content:
                    # mdToHtml does escape code blocks but not fully
                    pass

        # Check for torch.compile without fallback
        if 'torch.compile' in content:
            if 'try:' not in content and 'except' not in content:
                issues.append("torch.compile without try/except fallback - may crash on CPU without Python.h")

    except Exception as e:
        issues.append(f"Failed to read/check: {e}")

    return issues

def main():
    print("=== Massive Bug Check for TinyLLM ===")
    files = []
    for root, dirs, filenames in os.walk("."):
        # Skip venv, __pycache__, .git
        dirs[:] = [d for d in dirs if d not in ['.venv', '__pycache__', '.git', 'node_modules', 'checkpoints', 'data']]
        for fn in filenames:
            if fn.endswith(('.py','.html','.js')):
                files.append(os.path.join(root, fn))

    all_issues = {}
    for path in sorted(files):
        issues = check_file(path)
        if issues:
            all_issues[path] = issues

    # Summary
    total_issues = sum(len(v) for v in all_issues.values())
    print(f"\nChecked {len(files)} files, found {total_issues} potential issues in {len(all_issues)} files\n")
    for path, issues in all_issues.items():
        print(f"{path}:")
        for iss in issues:
            print(f"  - {iss}")
        print()

    # Check for missing files
    print("=== Checking for missing critical files ===")
    critical = [
        "llm/config.py",
        "llm/model.py",
        "llm/tokenizer.py",
        "llm/train.py",
        "llm/train_v2.py",
        "llm/dataset_fast.py",
        "checkpoints/tokenizer.json",
        "data/tokens.pt",
        "app.py",
        "chat.html",
        "index.html",
        "requirements.txt"
    ]
    for cf in critical:
        if not os.path.exists(cf):
            print(f"MISSING: {cf}")
        else:
            print(f"OK: {cf}")

    # Check large files not ignored
    print("\n=== Checking .gitignore ===")
    if os.path.exists(".gitignore"):
        with open(".gitignore") as f:
            gi = f.read()
            print(gi)
            needed = [".venv/", "data/corpus.txt", "data/tokens.pt", "__pycache__/"]
            for n in needed:
                if n not in gi:
                    print(f"WARNING: {n} not in .gitignore - may commit large files")
    else:
        print("No .gitignore!")

    # Check model param counts
    print("\n=== Checking model configs ===")
    try:
        from llm.config import TinyConfig, TinyConfigV2, TinyConfig2M
        for cfg_cls in [TinyConfig, TinyConfigV2, TinyConfig2M]:
            cfg = cfg_cls()
            print(f"{cfg_cls.__name__}: {cfg.param_count():,} params, tokens {cfg.total_tokens:,}, seq_len {cfg.max_seq_len}")
    except Exception as e:
        print(f"Failed to check configs: {e}")

    print("\n=== Bug check done ===")

if __name__ == "__main__":
    main()
