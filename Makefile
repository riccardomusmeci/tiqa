SHELL = /bin/bash

coverage_percentage = 25
pkg_name = tiqa
src_pkg = src/$(pkg_name)

.PHONY: all install update clean clean_deep check test test_fast full coverage ci
all:
	make clean
	make install
	make doc

install:
	poetry install --all-extras
	make full

update:
	-rm poetry.lock
	make install

clean:
	-rm -rf htmlcov
	-rm -rf .benchmarks
	-rm -rf .mypy_cache
	-rm -rf .pytest_cache
	-rm -rf docs/_build
	-rm -rf docs/source/_autosummary
	-rm -rf prof
	-rm -rf build
	-rm -rf .eggs
	-rm -rf dist
	-rm -rf *.egg-info
	-find . -not -path "./.git/*" -name logs -exec rm -rf {} \;
	-rm -rf logs
	-rm -rf tests/logs
	-rm .coverage
	-rm -rf .ruff_cache
	-find . -not -path "./.git/*" -name '.benchmarks' -exec rm -rf {} \;
	-find tests -depth -type d -empty -delete

clean_deep:
	make clean
	-rm -rf $(shell poetry env info -p)	# delete current virtualenv
	# Clear all poetry caches
	for source in $(shell poetry cache list); do poetry cache clear $$source --all -n; done;

check:
	poetry run ruff check --diff .
	poetry run black --check --diff .
	poetry run docformatter --check --diff src tests
	poetry run mypy .

format:
	poetry run ruff check --show-fixes .
	poetry run black .
	poetry run docformatter --in-place src tests
	poetry run mypy .

test:
	poetry run pytest --cov=$(src_pkg) --cov-branch --cov-report=term-missing --cov-fail-under=$(coverage_percentage) tests

test_fast:
	# Make use of pytest-xdist to run tests in parallel
	poetry run pytest --cov=$(src_pkg) --cov-branch --cov-report=term-missing --cov-fail-under=$(coverage_percentage) -n auto tests

full:
	make check
	make test

coverage:
	make test
	poetry run coverage report -m
	poetry run coverage html
