default: tests

all: linter tests docs dist

linter:
	flake8 demucs

tests:
	python3 -m demucs.separate -n demucs_unittest test.mp3
	python3 -m demucs.separate -n demucs_unittest --mp3 test.mp3

dist:
	python3 setup.py sdist

clean:
	rm -r dist build *.egg-info


.PHONY: linter tests dist
