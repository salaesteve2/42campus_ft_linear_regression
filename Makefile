
VENV := .venv

# default target, when make executed without arguments
all: venv

$(VENV)/bin/activate:
	python3 -m pip install --user virtualenv
	python3 -m virtualenv $(VENV)
	./$(VENV)/bin/pip install pandas
	./$(VENV)/bin/pip install matplotlib
	./$(VENV)/bin/python -m pip install --upgrade pip
	./$(VENV)/bin/pip install scikit-learn
	./$(VENV)/bin/pip install joblib

venv: $(VENV)/bin/activate

run: venv
	./$(VENV)/bin/python3 main.py

check: venv
	./$(VENV)/bin/python3 check.py

clean:
	rm -rf $(VENV)
	rm -f linear_model.pkl
	rm -f parameters.json
	find . -type f -name '*.pyc' -delete

.PHONY: all venv run clean check