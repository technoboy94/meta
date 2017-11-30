CC = g++
CFLAGS = -I`python -c 'from distutils.sysconfig import *; print (get_python_inc())'` -std=c++11 -L/usr/lib/ -L/usr/local/lib -lpython3.6m -lboost_python3 -lboost_serialization -g
CFLAGS_SO = -I`python -c 'from distutils.sysconfig import *; print (get_python_inc())'` -std=c++11 -L/usr/lib/ -L/usr/local/lib -lpython3.6m -lboost_python3 -lboost_serialization -O2

install:
	$(CC) model.cpp -o hpylm $(CFLAGS_SO)

test: ## LLDB用.
	$(CC) test.cpp $(CFLAGS)

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.DEFAULT_GOAL := help
