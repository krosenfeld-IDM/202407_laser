# NOTE that there are hard-coded filenames here that are also variable in settings.py. That's an open issue.
# Ideally those filenames would be gotten from settings.py but that's mixing makefiles and python in a way 
# That might suggest we move to snakemake.
# # Define variables
PYTHON := python3
CC := g++
CFLAGS := -shared -fPIC -O3 -mavx2 -flto -fpermissive -fopenmp
#CFLAGS := -shared -fPIC -g -march=native -flto -fpermissive -fopenmp
# ALSO NO: -finline-functions
# NO: -floop-parallelize-all
# neutral: -funroll-loops
# NO: -ffast-math
#CFLAGS := -shared -fPIC -O0 -g
CREATE_POP_SCRIPT := python3 -m idmlaser.utils.create_pop_as_csv
POP_SEEDED_CSV := pop_seeded.csv
EULA_DB := eula.db
UPDATE_AGES_SO := update_ages.so
MODELED_POP_CSV := modeled_pop.csv
EULA_POP_CSV := eula_pop.csv

# Default target
all: $(MODELED_POP_CSV) fits.npy 

# Rule to build update_ages.so from update_ages.c
$(UPDATE_AGES_SO): update_ages.cpp
	$(CC) $(CFLAGS) -o $@ $<

$(MODELED_POP_CSV): demographics_settings.py
	$(CREATE_POP_SCRIPT)

eula_pops_20yr_noaging.csv: eula_binned.csv
	# eula_binned.csv -> eula_pops_20yr_noaging.csv
	$(PYTHON) -m idmlaser.utils.eula_precalc eula_pops_20yr_noaging.csv

fits.npy: eula_pops_20yr_noaging.csv
	# eula_pops_20yr_noaging.csv -> fits.npy
	$(PYTHON) -m idmlaser.utils.fit_eula_pops eula_pops_20yr_noaging.csv

run:
	python3 -m idmlaser.measles

setup:
	./setup.sh

# Clean target to remove generated files
clean:
	rm -f $(UPDATE_AGES_SO) $(EULA_DB) $(POP_SEEDED_CSV) $(MODELED_POP_CSV)

test:
	PYTHONPATH=$PYTHONPATH:../src python3 -m unittest discover -s ../tests -p 'test_*.py' -v

# Phony targets
.PHONY: all clean

