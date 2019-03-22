
clean-pyc:
	find . -type f -name "*.py[co]" -o -name __pychache__ -exec rm -rf {} +

clean-cpp:
	find . -type f -name '*.c' -o -name '*.o' -o -name '*.so' -o -name '*.cpp' -delete

clean-general:
	find . -type f -name .DS_Store -o -name .idea -o -name '*~' -delete

clean-build:
	rm -rf build dist *.egg-info


.PHONY: clean
clean: clean-build clean-general clean-cpp clean-pyc

.PHONY: install
install: clean
	pip install -e .
