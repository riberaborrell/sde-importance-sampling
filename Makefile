help:
	@echo "---------------HELP-----------------"
	@echo "To create venv with required packages type 'make venv'"
	@echo "To create venv with developer packages type 'make develop'"
	@echo "To run the tests of the project type 'make tests'"
	@echo "To clean the virtual environment type 'make clean'"
	@echo "------------------------------------"

venv:
	python3.9 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install --upgrade setuptools
	venv/bin/pip install -e .

develop: venv
	venv/bin/pip install -e .[test]

tests: venv develop
	venv/bin/pytest 

clean:
	rm -rf venv
	rm -rf *.egg-info
