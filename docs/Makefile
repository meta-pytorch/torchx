
# Minimal makefile for Sphinx documentation
# Usage:
#    make html
#

# You can set these variables from the command line.
SPHINXOPTS    = -W
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = torchx
SOURCEDIR     = source
BUILDDIR      = build
VERSION       := $(shell python -c "import torchx; print(torchx.__version__)")

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@echo "Deleting build directory"
	rm -rf "$(BUILDDIR)"

.PHONY: help Makefile clean livehtml

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)/$(VERSION)" $(SPHINXOPTS) $(O)

# optional live version
livehtml:
	sphinx-autobuild --watch ../torchx --watch ../examples --re-ignore ".*(examples_.*|.new|source/.*(Dockerfile|.py))" "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
