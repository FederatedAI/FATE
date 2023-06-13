mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
.DEFAULT_GOAL:=help

##@ Dev
.PHONY: install-rust
install-rust_paillier: ## Install rust_paillier.
	@echo "install fate_utils"
	@cd ${mkfile_dir} && \
		. venv/bin/activate.sh && \
		maturin develop --release -m rust/fate_utils/Cargo.toml --target-dir build

##@ Build
.PHONY: build-rust
build-rust: ## Build fate_utils.
	@echo "build fate_utils"
	@cd ${mkfile_dir} && \
		. ${mkfile_dir}venv/bin/activate && \
		maturin build --release -m rust/fate_utils/crates/fate_utils/Cargo.toml --out dist --target-dir build

.PHONY: build-fate
build-fate: ## Build fate
	@echo "build fate"
	@cd ${mkfile_dir}/python && \
		python3 setup.py sdist --formats=gztar -d ../dist

.PHONY: build ## Build all
build: build-rust_paillier build-fate
	@echo "build all"

##@ Generate
.PHONY: gen-osx-proto
gen-osx-proto: ## Generate osx protobuf.
	@cd ${mkfile_dir} && \
		python3 -m grpc_tools.protoc --proto_path=proto/ \
		--python_out=python/fate/arch/federation/osx/ \
		--grpc_python_out=python/fate/arch/federation/osx/ \
		proto/osx.proto

.PHONY: gen-task-jsonschema
gen-task-jsonschema: ## Generate task jsonschema.
	@cd ${mkfile_dir}/python && \
		python3 -c "from fate.components.spec.task import TaskConfigSpec; print(TaskConfigSpec.schema_json(indent=2))" >> \
		${mkfile_dir}/schemas/task.schema.json

##@ Clean
.PHONY: clean
clean: ## Clean unused files.
	@cd ${mkfile_dir} && \
		rm -rf dist/ build/ rust/tensor/rust_paillier/target && \
		find python -name '*.pyc' -exec rm -f {} \; && \
		find python -name '__pycache__' -exec rm -rf {} \+ && \
		find python -name '*.egg-info' -exec rm -rf {} \+

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
