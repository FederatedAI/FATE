# Image for build FATE's documents

This image is modified from [mkdocs-meterial](https://squidfunk.github.io/mkdocs-material/) with some plugins embeded.

Usage

Mount the folder where your mkdocs.yml resides as a volume into /docs:

- Start development server on http://localhost:8000

```console
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs sagewei0/mkdocs
```

- Build documentation

```console
docker run --rm -it -v ${PWD}:/docs sagewei/mkdocs build
```

- Deploy documentation to GitHub Pages

```console
docker run --rm -it -v ~/.ssh:/root/.ssh -v ${PWD}:/docs sagewei0/mkdocs gh-deploy 
```
