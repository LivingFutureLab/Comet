.PHONY: build clean style

style:
	ruff format --check pyproject.toml .

build:
	python -m build --wheel

clean:
	@echo "Cleaning build artifacts..."
	
	@if [ -d "build" ]; then \
		rm -rf build; \
		echo "Removed build/"; \
	fi
	
	@if [ -d "dist" ]; then \
		rm -rf dist; \
		echo "Removed dist/"; \
	fi
	
	@if ls ./*.egg-info >/dev/null 2>&1; then \
		echo "Removing egg-info directories:"; \
		for dir in ./*.egg-info; do \
			rm -rf "$$dir"; \
			echo "  $$dir"; \
		done; \
	fi
	
	@echo "Cleanup complete!"
