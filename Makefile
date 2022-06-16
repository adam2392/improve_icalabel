
CODESPELL_SKIPS ?= "*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,*.hist,empty_file,*.orig,*.js,*.map,*.ipynb,searchindex.dat,plot_*.rst,*.rst.txt,*.html,gdf_encodes.txt"
CODESPELL_DIRS ?= mnestudy/ docs/ examples/

codespell:  # running manually
	@codespell -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@codespell -i 0 -q 7 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle mnestudy

black:
	@if command -v black > /dev/null; then \
		echo "Running black"; \
		black mnestudy; \
	else \
		echo "black not found, please install it!"; \
		exit 1; \
	fi;
	@echo "black passed"

isort:
	@if command -v isort > /dev/null; then \
		echo "Running isort"; \
		isort mnestudy docs examples; \
	else \
		echo "isort not found, please install it!"; \
		exit 1; \
	fi;
	@echo "isort passed"

check:
	@$(MAKE) -k black pydocstyle codespell-error check-manifest

run-checks:
	isort --check .
	black --check mnestudy
	check-manifest
	@$(MAKE) codespell-error
	

test:
	pytest --verbose --junitxml=test-results/junit.xml --cov-report=xml --cov=mnestudy "mnestudy/**/test_*.py"
	
# CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tests/ mnestudy/
