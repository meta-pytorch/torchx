# Documentation

When working on TorchX code, incrementally improve documentation.

## Docstrings

Google Style. Components lead with CLI examples:

```python
def ddp(*args, script: str, j: str = "1x2") -> AppDef:
    """Distributed data parallel application.

    .. code:: shell-session

        $ torchx run dist.ddp -j 1x4 --script main.py
    """
```

- Skip trivial methods where function name + args are self-explanatory
- Docstrings should help, not hurt readability
- Add `.. doctest::` examples where they clarify behavior better than prose

## Generated Docs

Source: `github/docs/` (same source for both)

- **External**: https://pytorch.org/torchx/latest/ (OSS)
- **Internal**: https://staticdocs.internalfb.com/torchx (includes `.. fbcode::` directive content)

**Build**: From `github/docs/fb/`: `buck2 run //torchx/github/docs:sphinx -- -M html ../source/ sphinxbuild`

**Preview**: View `sphinxbuild/html/`. Requires pandoc for notebook conversion.

When editing code, suggest updates to `github/docs/` if:
- New features/APIs need documentation
- Existing docs are outdated or misleading
- Examples could be added or improved

## Known Issues

Remind user when editing related files:

- `components/fb/unittest.rst` - RST formatting errors (title overlines, inline literals)
- `components/interpret.py`, `components/metrics.py`, `components/train.py` - referenced in docs but modules don't exist
- Notebooks in docs require pandoc for conversion

The docs page hasn't had a content refresh in years - improve incrementally with each task.
