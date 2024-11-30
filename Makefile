hello:
	echo "hello world"

hellopy:
	python -c """print('hello world py')"""

help: ## show this helper
	pwsh -Command """$file = "MakeFile" $h = Get-content $file | select-string "##" Write-Output $h"""

stylecheck:  ## check for pep 8 formating errors.
	python -m flake8 ./pipelines --builtins 'dbutils, sc, spark' \
			--per-file-ignores '__init__.py:F401' \
			--max-complexity 10 \
			--max-line-length 150

typecheck: ## check for pep 484 typehints errors.
	python -m mypy ./pipelines/\
		--disallow-untyped-defs \
		--check-untyped-defs \
		--ignore-missing-imports \
		--warn-unused-configs \
		--disable-error-code name-defined

doccheck: ## check for pep 257 Docstrings errors, in Google format.
	python -m pydocstyle ./pipelines --convention google

importformat: ## Order and format your imports.
	python -m isort ./pipelines -m 3

unittest: ## Run unit tests inside tests folder.
	python -m pytest --verbose ./tests/

all: ##all: Run all tests 
	make stylecheck 
	make typecheck 
	make doccheck 
	make importformat
	make unittest