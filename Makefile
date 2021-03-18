
clean-pyc:
	find archngv -type f -name "*.py[co]" -o -name __pychache__ -exec rm -rf {} +

clean-cpp:
	find archngv -type f -name '*.c' -delete -o -name '*.o' -delete -o -name '*.so' -delete -o -name '*.cpp' -delete

clean-general:
	find archngv -type f -name .DS_Store -delete -o -name .idea -delete -o -name '*~' -delete

clean-build:
	rm -rf build dist *.egg-info


.PHONY: clean
clean: clean-build clean-general clean-cpp clean-pyc

.PHONY: install
install: clean
	pip3 install setuptools pip --upgrade
	pip3 install numpy cython "snakemake<6"
	pip3 install --use-deprecated=legacy-resolver --index-url https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple/ -e .[all]
